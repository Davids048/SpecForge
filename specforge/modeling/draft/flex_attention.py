import torch
import torch._dynamo as dynamo
from torch.nn.attention.flex_attention import (
    create_block_mask,
    flex_attention,
    or_masks,
)
from transformers.utils import is_torchdynamo_compiling

dynamo.config.recompile_limit = 64


# Reference Implementation https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/flex_attention.py
class WrappedFlexAttention:
    """
    We are doing a singleton class so that flex attention is compiled once when it's first called.
    """

    _instance = None
    _is_flex_compiled = False
    _compiled_flex_attention = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            # Create a new instance if one doesn't already exist
            cls._instance = super().__new__(cls)
        return cls._instance

    @torch.compiler.disable(recursive=False)
    def __init__(self):
        """
        Initialize or update the singleton instance.
        """
        if not self._is_flex_compiled:
            # Enable dynamic shapes to handle different input sizes
            self._compiled_flex_attention = torch.compile(
                flex_attention,
                # mode="max-autotune-no-cudagraphs",
            )
            self._is_flex_compiled = True

    def __call__(self):
        return self._compiled_flex_attention


def compile_friendly_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    # First call initialise singleton wrapper object, second call invokes the object method to return compiled flex attention
    # Do not use compiled version if already compiling forward (it raises issues)
    flex_attention_compiled = (
        WrappedFlexAttention()() if not is_torchdynamo_compiling() else flex_attention
    )
    return flex_attention_compiled(
        query,
        key,
        value,
        **kwargs,
    )


class WrappedCreateBlockMask:
    _instance = None
    _is_create_block_mask_compiled = False
    _compiled_create_block_mask = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @torch.compiler.disable(recursive=False)
    def __init__(self):
        if not self._is_create_block_mask_compiled:
            self._compiled_create_block_mask = torch.compile(create_block_mask)
            self._is_create_block_mask_compiled = True

    def __call__(self):
        return self._compiled_create_block_mask


def compile_friendly_create_block_mask(
    mask_mod,
    B,
    H,
    Q_LEN,
    KV_LEN,
    device,
):
    create_block_mask_compiled = (
        WrappedCreateBlockMask()()
        if not is_torchdynamo_compiling()
        else create_block_mask
    )
    return create_block_mask_compiled(
        mask_mod,
        B,
        H,
        Q_LEN,
        KV_LEN,
        device,
    )


def generate_eagle3_mask(
    seq_lengths: torch.Tensor, Q_LEN: int, KV_LEN: int, lck: int = 0
):

    def causal_mask(b, h, q_idx, kv_idx):
        # Causal will keep shrinking by 1 diagnol due to appended suffix
        # Shirnk the causal by diagnol
        causal_mask = q_idx >= kv_idx
        padding_mask = (kv_idx < seq_lengths[b]) & (q_idx < seq_lengths[b])
        return causal_mask & padding_mask

    def suffix_mask(b, h, q_idx, kv_idx):
        suffix_mask = kv_idx >= Q_LEN
        padding_mask = kv_idx % Q_LEN < seq_lengths[b]
        diagnol_mask = (kv_idx - q_idx) % Q_LEN == 0
        return suffix_mask & padding_mask & diagnol_mask

    mask_mod = or_masks(causal_mask, suffix_mask)
    mask_mod.__name__ = f"eagle3_mask_Q_{Q_LEN}_KV_{KV_LEN}_lck_{lck}"
    return mask_mod


def generate_one_step_jacobi_mask(
    seq_lengths: torch.Tensor,  # [batch_size], original valid lengths
    Q_LEN: int,                  # input_len per block (e.g., 128)
    num_blocks: int              # self.length + 2
):
    """
    Generate attention mask for one-step Jacobi drafting where all blocks are processed in parallel.

    This creates a block-structured attention pattern where:
    - Block 0 (context): standard causal attention
    - Blocks 1+ (seed, mask tokens): diagonal attention to same relative position in previous blocks

    The mask replicates what _parallel_drafting does iteratively but in a single attention operation.
    """
    def mask_mod(b, h, q_idx, kv_idx):
        # Determine which block query and key belong to
        q_block = q_idx // Q_LEN
        kv_block = kv_idx // Q_LEN

        # Relative positions within blocks
        q_rel = q_idx % Q_LEN
        kv_rel = kv_idx % Q_LEN

        # Calculate effective sequence length for this query's block
        # Mirrors: lck = q_block, padding_offset = lck - 1 if lck > 1 else 0
        # Use torch.clamp to avoid data-dependent control flow (required for vmap)
        padding_offset = torch.clamp(q_block - 1, min=0)
        effective_seq_len = seq_lengths[b] - padding_offset

        # ============ Causal mask (context block) ============
        # Standard causal attention within the context block
        causal_attention = (kv_block == 0) & (q_rel >= kv_rel)
        causal_padding = (kv_rel < effective_seq_len) & (q_rel < effective_seq_len)
        causal_mask = causal_attention & causal_padding

        # ============ Diagonal mask (suffix blocks) ============
        # Attend to same relative position in blocks 1..q_block
        diagonal_attention = (kv_block > 0) & (kv_block <= q_block) & (kv_rel == q_rel)
        # Key position must be valid in original sequence
        # Query position must be valid in effective (offset-adjusted) sequence
        diagonal_padding = (kv_rel < seq_lengths[b]) & (q_rel < effective_seq_len)
        diagonal_mask = diagonal_attention & diagonal_padding

        return causal_mask | diagonal_mask

    mask_mod.__name__ = f"one_step_jacobi_Q_{Q_LEN}_blocks_{num_blocks}"
    return mask_mod
