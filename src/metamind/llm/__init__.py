from .base import LLMClient, LLMConfig
from .openai_client import OpenAIResponsesClient

__all__ = ["LLMClient", "LLMConfig", "OpenAIResponsesClient"]


def make_llm_client(cfg: LLMConfig | None = None) -> LLMClient | None:
    """Best-effort LLM client factory.
    Returns None when LLM is not enabled or credentials/deps are missing.
    """
    cfg = cfg or LLMConfig()
    if not cfg.enabled:
        return None
    try:
        return OpenAIResponsesClient(cfg)  # type: ignore
    except Exception:
        return None
