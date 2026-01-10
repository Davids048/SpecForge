from .base import Eagle3DraftModel, JacobiDraftModel
from .llama3_eagle import LlamaForCausalLMEagle3
from .qwen3_jacobi import Qwen3ForCausalLMJacobi

__all__ = [
    "Eagle3DraftModel",
    "JacobiDraftModel",
    "LlamaForCausalLMEagle3",
    "Qwen3ForCausalLMJacobi",
]
