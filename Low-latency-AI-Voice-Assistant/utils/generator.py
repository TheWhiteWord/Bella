from dataclasses import dataclass
from typing import List, Tuple
import os
from pathlib import Path

import torch
import torchaudio
from huggingface_hub import hf_hub_download
from .models import Model, ModelArgs  # Fix import path to use local models.py
from moshi.models import loaders
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer

# Configure local cache directory
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(CACHE_DIR, exist_ok=True)

@dataclass
class Segment:
    speaker: int
    text: str
    # (num_samples,), sample_rate = 24_000
    audio: torch.Tensor

def get_local_model_path(filename: str) -> str:
    """Get the path to a local model file, searching in common locations"""
    # Check direct path in models directory
    direct_path = os.path.join(CACHE_DIR, filename)
    if os.path.exists(direct_path):
        return direct_path
    
    # Check in Hugging Face cache structure
    hf_path = os.path.join(CACHE_DIR, "models--sesame--csm-1b", "snapshots")
    if os.path.exists(hf_path):
        for snapshot in os.listdir(hf_path):
            model_path = os.path.join(hf_path, snapshot, filename)
            if os.path.exists(model_path):
                return model_path
    
    return None

def load_llama3_tokenizer():
    """Load Llama tokenizer from local files."""
    snapshot_path = os.path.join(CACHE_DIR, 
                               "models--meta-llama--Llama-3.2-1B", 
                               "snapshots",
                               "4e20de362430cd3b72f300e6b0f18e50e7166e08")
    
    if not os.path.isdir(snapshot_path):
        raise FileNotFoundError("Could not find local Llama tokenizer files")
        
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot_path,
        local_files_only=True  # Only use local files
    )

    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],
    )

    return tokenizer


class Generator:
    def __init__(
        self,
        model: Model,
        max_seq_len: int = 2048
    ):
        self._model = model
        self._model.setup_caches(1)
        self.max_seq_len = max_seq_len

        self._text_tokenizer = load_llama3_tokenizer()

        device = next(model.parameters()).device
        mimi_weight = hf_hub_download(
            loaders.DEFAULT_REPO, 
            loaders.MIMI_NAME, 
            cache_dir=CACHE_DIR,
            token=os.getenv("HUGGING_FACE_TOKEN")
        )
        # Move mimi model to half precision
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            mimi = loaders.get_mimi(mimi_weight, device=device)
            mimi.set_num_codebooks(32)
        self._audio_tokenizer = mimi

        self.sample_rate = mimi.sample_rate
        self.device = device

    def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_tokens = []
        frame_masks = []

        text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
        text_frame = torch.zeros(len(text_tokens), 33).long()
        text_frame_mask = torch.zeros(len(text_tokens), 33).bool()
        text_frame[:, -1] = torch.tensor(text_tokens)
        text_frame_mask[:, -1] = True

        frame_tokens.append(text_frame.to(self.device))
        frame_masks.append(text_frame_mask.to(self.device))

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_tokens = []
        frame_masks = []

        # (K, T)
        audio = audio.to(self.device)
        audio_tokens = self._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
        # add EOS frame
        eos_frame = torch.zeros(audio_tokens.size(0), 1).to(self.device)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

        audio_frame = torch.zeros(audio_tokens.size(1), 33).long().to(self.device)
        audio_frame_mask = torch.zeros(audio_tokens.size(1), 33).bool().to(self.device)
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :-1] = True

        frame_tokens.append(audio_frame)
        frame_masks.append(audio_frame_mask)

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (seq_len, 33), (seq_len, 33)
        """
        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio)

        return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat([text_masks, audio_masks], dim=0)

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.7,
        topk: int = 25,
    ) -> torch.Tensor:
        """Generate audio with optimized CUDA memory management."""
        self._model.reset_caches()

        max_audio_frames = int(max_audio_length_ms / 80)
        tokens, tokens_mask = [], []
        
        # Process context with memory efficiency
        for segment in context:
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
                tokens.append(segment_tokens)
                tokens_mask.append(segment_tokens_mask)

        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)

        # Move tensors to GPU efficiently
        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0), device=self.device).unsqueeze(0)

        effective_max_seq_len = self.max_seq_len - max_audio_frames
        if curr_tokens.size(1) >= effective_max_seq_len:
            raise ValueError(f"Inputs too long, must be below max_seq_len - max_audio_frames: {effective_max_seq_len}")

        # Generate frames with optimized memory management
        samples = []
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            for _ in range(max_audio_frames):
                sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
                if torch.all(sample == 0):
                    break  # eos

                samples.append(sample)

                curr_tokens = torch.cat([sample, torch.zeros(1, 1, device=self.device).long()], dim=1).unsqueeze(1)
                curr_tokens_mask = torch.cat(
                    [torch.ones_like(sample).bool(), torch.zeros(1, 1, device=self.device).bool()], dim=1
                ).unsqueeze(1)
                curr_pos = curr_pos[:, -1:] + 1

                # Periodic memory cleanup
                if len(samples) % 50 == 0:
                    torch.cuda.empty_cache()

        # Process final audio with optimized memory
        with torch.amp.autocast(device_type='cuda', enabled=False):
            stacked_samples = torch.stack(samples)
            audio = self._audio_tokenizer.decode(stacked_samples.permute(1, 2, 0)).squeeze(0).squeeze(0)
            audio = audio.to(dtype=torch.float32)
            
            # Clear intermediate tensors
            del samples, stacked_samples
            torch.cuda.empty_cache()

            # Normalize with headroom
            max_val = torch.max(torch.abs(audio))
            if max_val > 0:
                audio = (audio / max_val) * 0.95

        return audio


def load_csm_1b(ckpt_path: str = "ckpt.pt", device: str = "cuda", max_seq_len: int = 2048) -> Generator:
    """Load CSM-1B model, prioritizing local files and CUDA optimization"""
    # Convert device string to proper torch.device object
    if isinstance(device, str):
        device = torch.device("cuda:0" if device == "cuda" else device)
    
    # Check CUDA availability and set optimal CUDA settings
    if device.type == "cuda" and torch.cuda.is_available():
        # Set optimal CUDA settings for RTX 4070 Ti SUPER (16GB VRAM)
        torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of available VRAM
        torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
        torch.backends.cuda.matmul.allow_tf32 = True  # Enable TensorFloat-32
        torch.backends.cudnn.allow_tf32 = True  # Enable TF32 for cuDNN
        print("CUDA optimizations enabled for high-end GPU")
    else:
        print("CUDA requested but not available, falling back to CPU")
        device = torch.device("cpu")
    
    # First check if the provided path exists
    if os.path.exists(ckpt_path):
        model_path = ckpt_path
    else:
        # Try to find the model in local cache
        model_path = get_local_model_path("ckpt.pt")
        if not model_path:
            print(f"Model not found locally, attempting to download...")
            model_path = hf_hub_download(
                repo_id="sesame/csm-1b",
                filename="ckpt.pt",
                cache_dir=CACHE_DIR,
                token=os.getenv("HUGGING_FACE_TOKEN")
            )
            # Create symlink for easier access
            symlink_path = os.path.join(CACHE_DIR, "ckpt.pt")
            if not os.path.exists(symlink_path):
                os.symlink(model_path, symlink_path)

    model_args = ModelArgs(
        backbone_flavor="llama-1B",
        decoder_flavor="llama-100M",
        text_vocab_size=128256,
        audio_vocab_size=2051,
        audio_num_codebooks=32,
        max_seq_len=max_seq_len
    )
    
    # Initialize model with proper device handling
    print(f"Loading model to {device} with bfloat16 and optimized settings...")
    
    # Load state dict first
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    
    # Create model instance and register buffers before loading state dict
    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
        model = Model(model_args)
        model.device = device  # Ensure device is properly set
        
        # Pre-register buffers before loading state dict
        backbone_mask = torch.tril(torch.ones(model.backbone.max_seq_len, model.backbone.max_seq_len, 
                                            dtype=torch.bool, device=device))
        decoder_mask = torch.tril(torch.ones(model_args.audio_num_codebooks, model_args.audio_num_codebooks, 
                                           dtype=torch.bool, device=device))
        model.register_buffer("backbone_causal_mask", backbone_mask)
        model.register_buffer("decoder_causal_mask", decoder_mask)
        
        # Now load the state dict
        model.load_state_dict(state_dict, strict=False)  # Use strict=False to allow missing buffers
        
        # Move model to device and initialize caches
        model = model.to(device=device, dtype=torch.bfloat16, non_blocking=True)
        model.setup_caches(1)  # Initialize caches after moving to device
    
    # Initialize generator with device-aware model
    generator = Generator(model, max_seq_len=max_seq_len)
    return generator
