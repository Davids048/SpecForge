from typing import Optional, Callable
from typing_extensions import Unpack, Tuple
import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    Qwen3Config,
    Qwen3PreTrainedModel,
    Qwen3MLP,
    GradientCheckpointingLayer,
    FlashAttentionKwargs,
    rotate_half,
    eager_attention_forward,
    ALL_ATTENTION_FUNCTIONS,
)
from transformers import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache
# from .utils import build_target_layer_ids, extract_context_feature, sample
from specforge.modeling.draft.base import JacobiDraftModel
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from specforge.modeling.draft.flex_attention import (
    compile_friendly_create_block_mask,
    compile_friendly_flex_attention,
    generate_eagle3_mask,
)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def build_target_layer_ids(num_target_layers: int, num_draft_layers: int):
    if num_draft_layers == 1:
        return [(num_target_layers // 2)]
    start = 1
    end = num_target_layers - 3
    span = end - start
    target_layer_ids = [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]
    return target_layer_ids

class Qwen3AttentionBase(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

class Qwen3DFlashAttention(Qwen3AttentionBase):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        attn_fn: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attn_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class Qwen3JacobiFlexAttention(Qwen3AttentionBase):
    def forward(
        self,
        hidden_states: Optional[torch.Tensor],
        target_hidden: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert (target_hidden is not None and hidden_states is None) or (target_hidden is None and hidden_states is not None)
        forward_hidden = target_hidden if target_hidden is not None else hidden_states
        bsz, q_len, _ = forward_hidden.size()


        num_past_seen_tokens = past_key_values.get_seq_length() if past_key_values else 0

        query_states = self.q_proj(forward_hidden)
        key_states = self.k_proj(forward_hidden)
        value_states = self.v_proj(forward_hidden)

        query_states = query_states.view(
            bsz, q_len, self.num_attention_heads, self.head_dim
        )
        query_states = self.q_norm(query_states).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )
        key_states = self.q_norm(key_states).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        cos, sin = position_embeddings
        cos, sin = cos.to(query_states.device), sin.to(query_states.device)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_values is not None:
            cache_position: torch.Tensor = torch.arange(
                num_past_seen_tokens, num_past_seen_tokens + q_len,  device=forward_hidden.device
            )

            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

            key_cache, value_cache = past_key_values.update(
                key_states,
                value_states,
                layer_idx=self.layer_idx,
                cache_kwargs=cache_kwargs,
            )

        seq_lengths = attention_mask.sum(dim=-1)
        lck = num_past_seen_tokens // q_len # This acts as an offset, indicating which "round" of input we are forwarding.

        # lck is the number of times we processed a whole sequence length.
        # the first time is context, second time are input ids, third and after are the mask section.
        # There are tokens exceeding the training sequence starting after the 3rd iteration. So we do not attend to that section.
        padding_offset = lck - 1 if lck > 1 else 0
        seq_lengths -= padding_offset

        # NOTE: Copied from specforge/modeling/draft/llama3_eagle.py
        # TODO: Remove the usage of uncompiled create_block_mask after
        # https://github.com/pytorch/pytorch/issues/160018
        if q_len <= 128:
            create_block_mask_func = create_block_mask
            flex_attention_func = flex_attention
        else:
            create_block_mask_func = compile_friendly_create_block_mask
            flex_attention_func = compile_friendly_flex_attention
        block_mask = create_block_mask_func(
            mask_mod=generate_eagle3_mask(
                seq_lengths=seq_lengths,
                Q_LEN=q_len,
                KV_LEN=key_cache.shape[-2],
                lck=lck,
            ),
            B=bsz,
            H=1,  # Rely on broadcast
            Q_LEN=q_len,
            KV_LEN=key_cache.shape[-2],
            device=query_states.device,
        )
        attn_output = flex_attention_func(
            query=query_states,
            key=key_cache.contiguous(),
            value=value_cache.contiguous(),
            block_mask=block_mask,
            enable_gqa=True,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.head_dim * self.num_attention_heads)
        attn_output = self.o_proj(attn_output)
        return (attn_output,)


class Qwen3DFlashDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int, attention_backend="flash_attn"):
        super().__init__()
        self.hidden_size = config.hidden_size

        if attention_backend == "flash_attn":
            self.self_attn = Qwen3DFlashAttention(config=config, layer_idx=layer_idx)
        elif attention_backend == "flex_attention":
            self.self_attn = Qwen3JacobiFlexAttention(config=config, layer_idx=layer_idx)
        else:
            raise ValueError(f"Unknown attention backend: {attention_backend}")

        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        target_hidden: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]] | None:
        if target_hidden is not None and hidden_states is None:
            store_target = True
        else:
            store_target = False

        if store_target:
            residual = None
        else:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]

        if store_target:
            # this round's input is used as hidden in next layer. we force it to be none to keep forwarding only target hidden.
            return None
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class Qwen3ForCausalLMJacobi(JacobiDraftModel):
    config_class = Qwen3Config
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]
    def __init__(self, config, quant_config=None, attention_backend="sdpa") -> None:
        super().__init__(config)
        self.config = config

        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.num_layers = getattr(config, 'num_hidden_layers', 1)
        self.num_target_layers = getattr(config, 'num_target_layers', 3)
        self.layers = nn.ModuleList(
            [Qwen3DFlashDecoderLayer(config, layer_idx, attention_backend) for layer_idx in range(config.num_hidden_layers)]
        )
        self.target_layer_ids = build_target_layer_ids(config.num_target_layers, config.num_hidden_layers)
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        # Use num_target_layers directly (not len(target_layer_ids)) since target always extracts 3 layers
        # TODO: (ds8) Change this to use dynamic target layer size.
        self.fc = nn.Linear(self.num_target_layers * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.block_size = config.block_size
        # lm_head uses full vocab size - weights loaded from target model
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

        # NOTE: Compatibility, not used.
        # create vocab buffers
        # TODO: (ds8) make the check more robust.
        assert config.vocab_size == config.draft_vocab_size, "Jacobi model's vocab_size and draft_vocab_size must be the same!"
        self.draft_vocab_size = config.draft_vocab_size
        assert self.draft_vocab_size == self.vocab_size
        t2d = torch.ones(self.vocab_size, dtype=torch.bool)
        d2t = torch.zeros(self.draft_vocab_size, dtype=torch.int64)
        self.register_buffer("t2d", t2d)
        self.register_buffer("d2t", d2t)

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden)) # [BS, INPUT SEQ, NDRaft * HD] -> [BS, seq len, HD]
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed the input ids."""
        return self.embed_tokens(input_ids)

    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # eagle 3 requires hidden states from 3 layers
        assert hidden_states.size(-1) == self.config.hidden_size * self.num_target_layers
        return self.norm(self.fc(hidden_states))

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute the logits of the draft model using target's lm_head."""
        return self.lm_head(self.norm(hidden_states))

    def backbone(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """The backbone of the draft model."""
        raise NotImplementedError("backbone not yet implemented for Qwen3ForCausalLMJacobi")

    def parallel_forward(
        self,
        target_hidden: Optional[torch.Tensor],
        hidden_states: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: DynamicCache,
        use_cache: bool = False,
    ):
        """Parallel forward for Jacobi drafting.
        Each call forward a section of length input_ids, corresponding to one position in the jacobi block.

        target_hidden: already fused target hidden
        hidden_states: the regular block to forward. len: input length. from noise Embedding
        position ids: pos of either target hidden or noise_embedding
        """
        forward_hidden = target_hidden if target_hidden is not None else hidden_states
        position_embeddings = self.rotary_emb(forward_hidden, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                target_hidden=target_hidden,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings
            )
        return hidden_states

