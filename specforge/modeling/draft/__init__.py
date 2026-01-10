from .base import Eagle3DraftModel, JacobiDraftModel
from .llama3_eagle import LlamaForCausalLMEagle3
from .llama3_jacobi import LlamaForCausalLMJacobi

__all__ = [
    "Eagle3DraftModel",
    "JacobiDraftModel",
    "LlamaForCausalLMEagle3",
    "LlamaForCausalLMJacobi",
]
