"""LiteLLM constants."""

import litellm

LITELLM_DEFAULT_THINKING_EFFORT: str = "medium"
LITELLM_DEFAULT_THINKING_SUMMARY: str = "auto"

# every provider name LiteLLM recognizes; the closed set Settings.set_provider validates
# against
LITELLM_PROVIDER_NAMES: frozenset[str] = frozenset(
    member.value for member in litellm.LlmProviders
)
