from dataclasses import dataclass

import torch
import torch.nn as nn
import torchtune
from torchtune.models import llama3_2


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


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        # Pass max_seq_len to the backbone and decoder
        self.backbone, backbone_dim = _prepare_transformer(FLAVORS[args.backbone_flavor](args.max_seq_len))
        self.decoder, decoder_dim = _prepare_transformer(FLAVORS[args.decoder_flavor](args.max_seq_len))

        # Use memory-efficient embeddings
        self.text_embeddings = nn.Embedding(args.text_vocab_size, backbone_dim)
        self.audio_embeddings = nn.Embedding(args.audio_vocab_size * args.audio_num_codebooks, backbone_dim)

        self.projection = nn.Linear(backbone_dim, decoder_dim, bias=False)
        self.codebook0_head = nn.Linear(backbone_dim, args.audio_vocab_size, bias=False)
        self.audio_head = nn.Parameter(torch.empty(args.audio_num_codebooks - 1, decoder_dim, args.audio_vocab_size))

    def setup_caches(self, max_batch_size: int) -> torch.Tensor:
        """Setup KV caches and return a causal mask."""
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device

        with device:
            self.backbone.setup_caches(max_batch_size, dtype)
            self.decoder.setup_caches(max_batch_size, dtype, decoder_max_seq_len=self.args.audio_num_codebooks)

        # Pre-compute attention masks
        self.register_buffer(
            "backbone_causal_mask",
            torch.tril(torch.ones(self.backbone.max_seq_len, self.backbone.max_seq_len, dtype=torch.bool, device=device))
        )
        self.register_buffer(
            "decoder_causal_mask",
            torch.tril(torch.ones(self.args.audio_num_codebooks, self.args.audio_num_codebooks, dtype=torch.bool, device=device))
        )

    @torch.cuda.amp.autocast()  # Enable automatic mixed precision
    def generate_frame(
        self,
        tokens: torch.Tensor,
        tokens_mask: torch.Tensor,
        input_pos: torch.Tensor,
        temperature: float,
        topk: int,
    ) -> torch.Tensor:
        """
        Args:
            tokens: (batch_size, seq_len, audio_num_codebooks+1)
            tokens_mask: (batch_size, seq_len, audio_num_codebooks+1)
            input_pos: (batch_size, seq_len)
            temperature: temperature for sampling
            topk: top-k for sampling

        Returns:
            (batch_size, audio_num_codebooks) sampled tokens
        """
        b, s, _ = tokens.size()

        assert self.backbone.caches_are_enabled(), "backbone caches are not enabled"
        curr_backbone_mask = _index_causal_mask(self.backbone_causal_mask, input_pos)
        
        # Use bfloat16 for token processing
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            embeds = self._embed_tokens(tokens)
            masked_embeds = embeds * tokens_mask.unsqueeze(-1)
            h = masked_embeds.sum(dim=2)
            h = self.backbone(h, input_pos=input_pos, mask=curr_backbone_mask)

            last_h = h[:, -1, :]
            c0_logits = self.codebook0_head(last_h)
            c0_sample = sample_topk(c0_logits, topk, temperature)
            c0_embed = self._embed_audio(0, c0_sample)

            curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1)
            curr_sample = c0_sample.clone()
            curr_pos = torch.arange(0, curr_h.size(1), device=curr_h.device).unsqueeze(0).repeat(curr_h.size(0), 1)

            # Free memory
            del h, embeds, masked_embeds
            torch.cuda.empty_cache()

        # Decoder caches must be reset every frame.
        self.decoder.reset_caches()
        
        # Process decoder tokens while maintaining integer type
        for i in range(1, self.args.audio_num_codebooks):
            curr_decoder_mask = _index_causal_mask(self.decoder_causal_mask, curr_pos)
            
            # Use bfloat16 for processing
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                decoder_h = self.decoder(self.projection(curr_h), input_pos=curr_pos, mask=curr_decoder_mask)
                ci_logits = torch.mm(decoder_h[:, -1, :], self.audio_head[i - 1])
                
            # Ensure integer type for token sampling
            ci_sample = sample_topk(ci_logits, topk, temperature).long()
            
            # Back to bfloat16 for embeddings
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                ci_embed = self._embed_audio(i, ci_sample)
                curr_h = ci_embed
            
            curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
            curr_pos = curr_pos[:, -1:] + 1
            
            # Clear unnecessary tensors
            del decoder_h, ci_logits
            if i % 8 == 0:  # Periodically clear cache
                torch.cuda.empty_cache()

        return curr_sample.long()  # Ensure integer type for output

    def reset_caches(self):
        """Reset KV caches and clear CUDA memory"""
        self.backbone.reset_caches()
        self.decoder.reset_caches()
        torch.cuda.empty_cache()

    def _embed_audio(self, codebook: int, tokens: torch.Tensor) -> torch.Tensor:
        return self.audio_embeddings(tokens + codebook * self.args.audio_vocab_size)

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        text_embeds = self.text_embeddings(tokens[:, :, -1]).unsqueeze(-2)

        audio_tokens = tokens[:, :, :-1] + (
            self.args.audio_vocab_size * torch.arange(self.args.audio_num_codebooks, device=tokens.device)
        )
        audio_embeds = self.audio_embeddings(audio_tokens.view(-1)).reshape(
            tokens.size(0), tokens.size(1), self.args.audio_num_codebooks, -1
        )

        return torch.cat([audio_embeds, text_embeds], dim=-2)
