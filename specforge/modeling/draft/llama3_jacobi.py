import torch
from typing import Optional
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.cache_utils import Cache
import torch.nn as nn

from .base import Eagle3DraftModel
from .llama3_eagle import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
)

# TODO: Change this to use Jacobi's own draft model.
# The base model defines:
# -prepare mask
# - freeze embedding
# - load embedding
# - load vocab mapping
class LlamaForCausalLMJacobi(Eagle3DraftModel):
    config_class = LlamaConfig
    def __init__(self, config, quant_config=None, attention_backend="sdpa") -> None:
        super().__init__(config)
        self.config = config

        self.vocab_size = config.vocab_size
        self.draft_vocab_size = config.draft_vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        num_layers = getattr(config, 'num_hidden_layers', 1)
        # TODO: CHANGE to use jacobi parallel decodeer layer.
        self.midlayer = nn.ModuleList([
            LlamaDecoderLayer(config, attention_backend=attention_backend)
            for _ in range(num_layers)
        ])

        if hasattr(config, "target_hidden_size"):
            self.fc = torch.nn.Linear(
                config.target_hidden_size * 3, config.hidden_size, bias=False
            )
        else:
            self.fc = torch.nn.Linear(
                config.hidden_size * 3, config.hidden_size, bias=False
            )

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(
            config.hidden_size, config.draft_vocab_size, bias=False
        )

        # create vocab buffers
        t2d = torch.ones(self.vocab_size, dtype=torch.bool)
        d2t = torch.zeros(self.draft_vocab_size, dtype=torch.int64)
        self.register_buffer("t2d", t2d)
        self.register_buffer("d2t", d2t)



    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Placeholder: Embed the input ids.
        """
        raise NotImplementedError("embed_input_ids not yet implemented for Jacobi model")

    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Placeholder: Project the concatenated hidden states from the high, medium and low layers to the target hidden size.
        """
        raise NotImplementedError("project_hidden_states not yet implemented for Jacobi model")

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Placeholder: Compute the logits of the draft model.
        """
        raise NotImplementedError("compute_logits not yet implemented for Jacobi model")

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
        """
        Placeholder: The backbone of the draft model.
        """
        raise NotImplementedError("backbone not yet implemented for Jacobi model")
