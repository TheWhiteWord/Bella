from dataclasses import dataclass
import torch
import torch.nn as nn
import torchtune
from torchtune.models import llama3_2
from flash_attn import flash_attn_func, flash_attn_kvpacked_func


def llama3_2_1B(max_seq_len: int = 4096) -> torchtune.modules.transformer.TransformerDecoder:
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=16,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=2048,
        max_seq_len=max_seq_len,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )


def llama3_2_100M(max_seq_len: int = 4096) -> torchtune.modules.transformer.TransformerDecoder:
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=4,
        num_heads=8,
        num_kv_heads=2,
        embed_dim=1024,
        max_seq_len=max_seq_len,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )


FLAVORS = {
    "llama-1B": llama3_2_1B,
    "llama-100M": llama3_2_100M,
}


def _prepare_transformer(model):
    embed_dim = model.tok_embeddings.embedding_dim
    model.tok_embeddings = nn.Identity()
    model.output = nn.Identity()
    return model, embed_dim


def _create_causal_mask(seq_len: int, device: torch.device):
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


def _index_causal_mask(mask: torch.Tensor, input_pos: torch.Tensor):
    """
    Args:
        mask: (max_seq_len, max_seq_len)
        input_pos: (batch_size, seq_len)

    Returns:
        (batch_size, seq_len, max_seq_len)
    """
    r = mask[input_pos, :]
    return r


def _multinomial_sample_one_no_sync(probs):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def sample_topk(logits: torch.Tensor, topk: int, temperature: float):
    logits = logits / temperature

    filter_value: float = -float("Inf")
    indices_to_remove = logits < torch.topk(logits, topk)[0][..., -1, None]
    scores_processed = logits.masked_fill(indices_to_remove, filter_value)
    scores_processed = torch.nn.functional.log_softmax(scores_processed, dim=-1)
    probs = torch.nn.functional.softmax(scores_processed, dim=-1)

    sample_token = _multinomial_sample_one_no_sync(probs)
    return sample_token


@dataclass
class ModelArgs:
    backbone_flavor: str
    decoder_flavor: str
    text_vocab_size: int
    audio_vocab_size: int
    audio_num_codebooks: int
    max_seq_len: int = 4096  # Added default sequence length


class CUDAKVCache:
    """Wrapper to ensure KV cache stays on CUDA"""
    def __init__(self, original_cache):
        self.original_cache = original_cache
        self.device = None

    def to(self, device):
        """Move cache to specified device"""
        self.device = device
        if hasattr(self.original_cache, 'cache_pos'):
            self.original_cache.cache_pos = self.original_cache.cache_pos.to(device)
        if hasattr(self.original_cache, 'k_cache'):
            self.original_cache.k_cache = self.original_cache.k_cache.to(device)
        if hasattr(self.original_cache, 'v_cache'):
            self.original_cache.v_cache = self.original_cache.v_cache.to(device)
        return self

    def __getattr__(self, name):
        """Forward all other attributes to original cache"""
        attr = getattr(self.original_cache, name)
        if torch.is_tensor(attr) and self.device is not None:
            attr = attr.to(self.device)
        return attr


def _patch_kv_cache_update(cache, device):
    """Patch the KV cache update method to ensure CUDA consistency"""
    # Check if this is a KVCache object with an update method
    if hasattr(cache, 'update'):
        original_update = cache.update

        def wrapped_update(k, v):
            # Ensure inputs are on the correct device
            if hasattr(cache, 'cache_pos'):
                cache.cache_pos = cache.cache_pos.to(device)
            if hasattr(cache, 'k_cache'):
                cache.k_cache = cache.k_cache.to(device)
            if hasattr(cache, 'v_cache'):
                cache.v_cache = cache.v_cache.to(device)
            
            # Move inputs to correct device
            k = k.to(device)
            v = v.to(device)
            
            return original_update(k, v)
        
        cache.update = wrapped_update
    # If it's a tensor, just move it to the correct device
    elif isinstance(cache, torch.Tensor):
        cache = cache.to(device)
    
    return cache

class CUDADeviceContext:
    """Context manager to ensure tensors stay on the correct device"""
    def __init__(self, model, device=None):
        self.model = model
        # Allow device to be explicitly specified or default to CUDA if available
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(f"cuda:0" if device == "cuda" else device)
        self.device = device
        self.original_device = None
        self._original_kv_cache_states = {}

    def __enter__(self):
        # Store original device
        if torch.cuda.is_available():
            self.original_device = torch.cuda.current_device()
            torch.cuda.set_device(self.device)
            
            # Store and move KV cache states
            for name, module in self.model.named_modules():
                if hasattr(module, 'cache'):
                    self._original_kv_cache_states[name] = {
                        'cache_pos': getattr(module.cache, 'cache_pos', None),
                        'k_cache': getattr(module.cache, 'k_cache', None),
                        'v_cache': getattr(module.cache, 'v_cache', None)
                    }
                    # Move cache tensors to correct device
                    for attr in ['cache_pos', 'k_cache', 'v_cache']:
                        if hasattr(module.cache, attr):
                            tensor = getattr(module.cache, attr)
                            if tensor is not None and tensor.device != self.device:
                                setattr(module.cache, attr, tensor.to(self.device))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_device is not None:
            # Restore original KV cache states
            for name, module in self.model.named_modules():
                if name in self._original_kv_cache_states:
                    if hasattr(module, 'cache'):
                        for attr, tensor in self._original_kv_cache_states[name].items():
                            if tensor is not None:
                                setattr(module.cache, attr, tensor)
            
            # Restore original device
            torch.cuda.set_device(self.original_device)
            torch.cuda.synchronize()  # Ensure all operations are complete

    def ensure_tensor_device(self, tensor):
        """Ensure a tensor is on the correct device"""
        if tensor is not None and hasattr(tensor, 'device'):
            if tensor.device != self.device:
                with torch.cuda.device(self.device):
                    return tensor.to(self.device, non_blocking=True)
        return tensor

    def ensure_cache_device(self, cache):
        """Ensure KV cache tensors are on the correct device"""
        if cache is not None:
            with torch.cuda.device(self.device):
                for attr in ['cache_pos', 'k_cache', 'v_cache']:
                    if hasattr(cache, attr):
                        tensor = getattr(cache, attr)
                        if tensor is not None and tensor.device != self.device:
                            setattr(cache, attr, tensor.to(self.device, non_blocking=True))
        return cache

class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        
        # Set device based on CUDA availability
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Optimize CUDA settings for RTX 4070 Ti
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.kv_cache_fp16 = True
            self.chunk_size = 512
            
        # Initialize components with explicit device placement
        with torch.cuda.device(self.device):
            # Initialize model components
            self.backbone, backbone_dim = _prepare_transformer(FLAVORS[args.backbone_flavor](args.max_seq_len))
            self.decoder, decoder_dim = _prepare_transformer(FLAVORS[args.decoder_flavor](args.max_seq_len))

            # Use memory-efficient embeddings
            dtype = torch.bfloat16
            self.text_embeddings = nn.Embedding(args.text_vocab_size, backbone_dim).to(dtype).to(self.device)
            self.audio_embeddings = nn.Embedding(args.audio_vocab_size * args.audio_num_codebooks, backbone_dim).to(dtype).to(self.device)

            self.projection = nn.Linear(backbone_dim, decoder_dim, bias=False).to(dtype).to(self.device)
            self.codebook0_head = nn.Linear(backbone_dim, args.audio_vocab_size, bias=False).to(dtype).to(self.device)
            self.audio_head = nn.Parameter(torch.empty(args.audio_num_codebooks - 1, decoder_dim, args.audio_vocab_size, dtype=dtype, device=self.device))

            # Move backbone and decoder to device
            self.backbone = self.backbone.to(self.device)
            self.decoder = self.decoder.to(self.device)
            
            # Initialize caches on device immediately
            self.setup_caches(1)

            # Initialize Flash Attention with optimal settings
            self.use_flash_attn = True
            if self.use_flash_attn:
                try:
                    from flash_attn import flash_attn_func
                    from flash_attn.flash_attn_interface import flash_attn_varlen_kvpacked_func
                    self.flash_attn = flash_attn_func
                    self.flash_attn_kv = flash_attn_varlen_kvpacked_func
                    print("Flash Attention 2.0 enabled with optimized settings")
                except ImportError:
                    print("Flash Attention not available, falling back to standard attention")
                    self.use_flash_attn = False

    def setup_caches(self, max_batch_size: int) -> None:
        """Setup KV caches with device consistency."""
        device = self.device if hasattr(self, 'device') else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16

        with torch.cuda.device(device):
            # Initialize backbone and decoder caches
            self.backbone.setup_caches(max_batch_size, dtype)
            self.decoder.setup_caches(max_batch_size, dtype, decoder_max_seq_len=self.args.audio_num_codebooks)

            # Pre-compute and move attention masks to device
            backbone_mask = torch.tril(torch.ones(self.backbone.max_seq_len, self.backbone.max_seq_len, 
                                                dtype=torch.bool, device=device))
            decoder_mask = torch.tril(torch.ones(self.args.audio_num_codebooks, self.args.audio_num_codebooks, 
                                               dtype=torch.bool, device=device))
            
            self.register_buffer("backbone_causal_mask", backbone_mask)
            self.register_buffer("decoder_causal_mask", decoder_mask)

            # Initialize KV caches for all attention modules
            def init_module_caches(module):
                if hasattr(module, 'cache'):
                    # Handle both tensor and KVCache objects
                    if isinstance(module.cache, torch.Tensor):
                        module.cache = module.cache.to(device)
                    else:
                        # Initialize cache components on device
                        if hasattr(module.cache, 'cache_pos'):
                            module.cache.cache_pos = torch.zeros(max_batch_size, dtype=torch.long, device=device)
                        if hasattr(module.cache, 'k_cache'):
                            shape = list(module.cache.k_cache.shape) if module.cache.k_cache is not None else [1, module.num_heads, 0, module.head_dim]
                            module.cache.k_cache = torch.zeros(shape, dtype=dtype, device=device)
                        if hasattr(module.cache, 'v_cache'):
                            shape = list(module.cache.v_cache.shape) if module.cache.v_cache is not None else [1, module.num_heads, 0, module.head_dim]
                            module.cache.v_cache = torch.zeros(shape, dtype=dtype, device=device)
                        
                        # Patch the update method only for KVCache objects
                        module.cache = _patch_kv_cache_update(module.cache, device)

            # Apply initialization to all modules recursively
            self.backbone.apply(init_module_caches)
            self.decoder.apply(init_module_caches)

            # Synchronize to ensure all CUDA operations are complete
            torch.cuda.synchronize()

    def init_kv_cache(self, device=None):
        """Initialize KV cache tensors on device"""
        if device is None:
            device = self.device
            
        def init_cache(module):
            if hasattr(module, 'cache'):
                # Initialize cache components on device
                if hasattr(module.cache, 'cache_pos'):
                    if module.cache.cache_pos is None:
                        module.cache.cache_pos = torch.zeros(1, dtype=torch.long, device=device)
                    else:
                        module.cache.cache_pos = module.cache.cache_pos.to(device)
                        
                # Initialize k_cache and v_cache if they exist
                for cache_name in ['k_cache', 'v_cache']:
                    if hasattr(module.cache, cache_name):
                        cache = getattr(module.cache, cache_name)
                        if cache is None:
                            shape = [1, getattr(module, 'num_heads', 1), 0, getattr(module, 'head_dim', 64)]
                            setattr(module.cache, cache_name, torch.zeros(shape, device=device))
                        else:
                            setattr(module.cache, cache_name, cache.to(device))

                # Ensure the cache object itself is device-aware
                if hasattr(module.cache, 'to'):
                    module.cache = module.cache.to(device)
        
        # Apply initialization to all modules
        self.apply(init_cache)
        torch.cuda.synchronize()  # Ensure all transfers are complete

    def _ensure_cache_device(self, device=None):
        """Ensure all cache-related tensors are on the correct device."""
        if device is None:
            device = next(self.parameters()).device
            
        def move_cache_tensors(module):
            if hasattr(module, 'cache'):
                if isinstance(module.cache, torch.Tensor):
                    module.cache = module.cache.to(device)
                else:
                    # For KVCache objects
                    for attr_name in ['cache_pos', 'k_cache', 'v_cache']:
                        if hasattr(module.cache, attr_name):
                            curr_tensor = getattr(module.cache, attr_name)
                            if curr_tensor is not None and curr_tensor.device != device:
                                setattr(module.cache, attr_name, curr_tensor.to(device))
        
        # Apply to all modules
        self.backbone.apply(move_cache_tensors)
        self.decoder.apply(move_cache_tensors)

    def _prepare_input_tensors(self, tokens, tokens_mask, input_pos):
        """Ensure all input tensors are on the correct device and have proper dtype."""
        device = next(self.parameters()).device
        return (
            tokens.to(device=device, non_blocking=True),
            tokens_mask.to(device=device, non_blocking=True),
            input_pos.to(device=device, non_blocking=True)
        )

    @torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
    def generate_frame(
        self,
        tokens: torch.Tensor,
        tokens_mask: torch.Tensor,
        input_pos: torch.Tensor,
        temperature: float,
        topk: int,
    ) -> torch.Tensor:
        """Generate a frame with optimized memory usage."""
        with CUDADeviceContext(self) as ctx:
            # Ensure inputs are on correct device
            tokens = ctx.ensure_tensor_device(tokens)
            tokens_mask = ctx.ensure_tensor_device(tokens_mask)
            input_pos = ctx.ensure_tensor_device(input_pos)

            assert self.backbone.caches_are_enabled(), "backbone caches are not enabled"
            curr_backbone_mask = _index_causal_mask(self.backbone_causal_mask, input_pos)

            # Process in optimized chunks
            chunk_size = getattr(self, 'chunk_size', 512)
            num_chunks = (tokens.size(1) + chunk_size - 1) // chunk_size

            # Pre-allocate memory
            h_chunks = []
            torch.cuda.empty_cache()

            # Use dedicated CUDA stream for better performance
            with torch.cuda.stream(torch.cuda.Stream()):
                for i in range(num_chunks):
                    start_idx = i * chunk_size
                    end_idx = min((i + 1) * chunk_size, tokens.size(1))
                    
                    chunk_embeds = self._embed_tokens(tokens[:, start_idx:end_idx])
                    chunk_mask = tokens_mask[:, start_idx:end_idx].unsqueeze(-1)
                    chunk_h = chunk_embeds * chunk_mask
                    chunk_h = chunk_h.sum(dim=2)
                    
                    # Process through backbone
                    chunk_h = self.backbone(
                        chunk_h,
                        input_pos=input_pos[:, start_idx:end_idx],
                        mask=curr_backbone_mask[:, start_idx:end_idx]
                    )
                    
                    h_chunks.append(chunk_h)

                    if i % 4 == 0:  # Periodic cleanup
                        torch.cuda.empty_cache()

                # Process final chunks
                h = torch.cat(h_chunks, dim=1)
                last_h = h[:, -1, :]

                # Generate first codebook
                c0_logits = self.codebook0_head(last_h)
                c0_sample = sample_topk(c0_logits, topk, temperature)
                c0_embed = self._embed_audio(0, c0_sample)

                curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1)
                curr_sample = c0_sample.clone()
                curr_pos = torch.arange(0, curr_h.size(1), device=ctx.device).unsqueeze(0).repeat(curr_h.size(0), 1)

                # Memory cleanup
                del h, h_chunks, chunk_h
                torch.cuda.empty_cache()

                # Reset decoder caches
                self.decoder.reset_caches()

                # Generate remaining codebooks
                for i in range(1, self.args.audio_num_codebooks):
                    curr_decoder_mask = _index_causal_mask(self.decoder_causal_mask, curr_pos)
                    
                    decoder_h = self.decoder(
                        self.projection(curr_h),
                        input_pos=curr_pos,
                        mask=curr_decoder_mask
                    )
                    
                    ci_logits = torch.mm(decoder_h[:, -1, :], self.audio_head[i - 1])
                    ci_sample = sample_topk(ci_logits, topk, temperature)
                    ci_embed = self._embed_audio(i, ci_sample)
                    curr_h = ci_embed
                    curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
                    curr_pos = curr_pos[:, -1:] + 1

                    if i % 8 == 0:  # Periodic cleanup
                        del decoder_h, ci_logits
                        torch.cuda.empty_cache()

            return curr_sample

    def reset_caches(self):
        """Reset KV caches and clear CUDA memory"""
        self.backbone.reset_caches()
        self.decoder.reset_caches()
        torch.cuda.empty_cache()

    def _embed_audio(self, codebook: int, tokens: torch.Tensor) -> torch.Tensor:
        return self.audio_embeddings(tokens + codebook * self.args.audio_vocab_size)

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            text_embeds = self.text_embeddings(tokens[:, :, -1]).unsqueeze(-2)
            
            audio_tokens = tokens[:, :, :-1] + (
                self.args.audio_vocab_size * torch.arange(self.args.audio_num_codebooks, device=tokens.device)
            )
            audio_embeds = self.audio_embeddings(audio_tokens.view(-1)).reshape(
                tokens.size(0), tokens.size(1), self.args.audio_num_codebooks, -1
            )

            return torch.cat([audio_embeds, text_embeds], dim=-2)

    def cleanup_tensors(self):
        """Clean up intermediate tensors and CUDA cache"""
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Ensure all CUDA ops are complete

    def move_to_device(self, device: torch.device):
        """Ensure all model components are on the same device"""
        self.to(device)
        if hasattr(self, 'text_embeddings'):
            self.text_embeddings = self.text_embeddings.to(device)
        if hasattr(self, 'audio_embeddings'):
            self.audio_embeddings = self.audio_embeddings.to(device)
        if hasattr(self, 'backbone'):
            self.backbone = self.backbone.to(device)
        if hasattr(self, 'decoder'):
            self.decoder = self.decoder.to(device)
        
        # Move any cached tensors
        for module in self.modules():
            if hasattr(module, 'cache'):
                if isinstance(module.cache, torch.Tensor):
                    module.cache = module.cache.to(device)
                elif hasattr(module.cache, 'to'):
                    module.cache = module.cache.to(device)
