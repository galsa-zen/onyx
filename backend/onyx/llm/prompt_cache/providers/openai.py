"""OpenAI provider adapter for prompt caching.

Requests to GPT-5.6+ models also carry explicit prompt_cache_breakpoints on
every tool result (`mark_tool_results`), applied when the request is built.
"""

from collections.abc import Sequence
from functools import lru_cache

from pydantic import JsonValue

from onyx.llm.constants import LlmProviderNames
from onyx.llm.interfaces import LLMConfig
from onyx.llm.model_request import (
    ChatCompletionMessage,
    ResponsesInputText,
    ToolMessage,
)
from onyx.llm.prompt_cache.models import CacheMetadata
from onyx.llm.prompt_cache.providers.base import PromptCacheProvider
from onyx.llm.prompt_cache.utils import prepare_messages_with_cacheable_transform

PROMPT_CACHE_BREAKPOINT: dict[str, JsonValue] = {"mode": "explicit"}
PROMPT_CACHE_OPTIONS: dict[str, JsonValue] = {"mode": "explicit"}


def uses_openai_explicit_cache_breakpoint(llm_config: LLMConfig) -> bool:
    """Whether requests for this model carry explicit prompt_cache_breakpoints."""
    if llm_config.model_provider != LlmProviderNames.OPENAI:
        return False
    return _supports_breakpoint(llm_config.deployment_name, llm_config.model_name)


@lru_cache(maxsize=256)
def _supports_breakpoint(deployment_name: str | None, model_name: str) -> bool:
    from litellm.integrations.anthropic_cache_control_hook import (
        supports_openai_prompt_cache_breakpoint,
    )

    return any(
        supports_openai_prompt_cache_breakpoint(name)
        for name in (deployment_name, model_name)
        if name
    )


def mark_tool_results(
    messages: Sequence[ChatCompletionMessage],
) -> Sequence[ChatCompletionMessage]:
    """Mark every tool result in the request with an explicit breakpoint.

    A breakpoint is part of the cached prefix, so an entry is reused only if
    the same message still carries the marker. Marking each tool result
    (not only the last) keeps earlier bytes identical from step to step.
    Only tool output can carry the marker through LiteLLM: user content must
    be chat-completions ``text`` (whose extra keys the bridge drops), system
    messages become ``instructions``, and assistant output does not accept
    one. Before the first tool result, implicit caching applies.
    """
    result = list(messages)
    for index, message in enumerate(result):
        if not isinstance(message, ToolMessage):
            continue
        text = (
            message.content
            if isinstance(message.content, str)
            else "".join(part.text for part in message.content)
        )
        result[index] = message.model_copy(
            update={
                "content": [
                    ResponsesInputText(
                        text=text, prompt_cache_breakpoint=PROMPT_CACHE_BREAKPOINT
                    )
                ]
            }
        )
    return result


def has_prompt_cache_breakpoint(messages: Sequence[ChatCompletionMessage]) -> bool:
    return any(
        isinstance(message, ToolMessage)
        and not isinstance(message.content, str)
        and any(part.prompt_cache_breakpoint for part in message.content)
        for message in messages
    )


class OpenAIPromptCacheProvider(PromptCacheProvider):
    """OpenAI adapter for prompt caching (implicit caching)."""

    def supports_caching(self) -> bool:
        """OpenAI supports automatic prompt caching."""
        return True

    def prepare_messages_for_caching(
        self,
        cacheable_prefix: list[ChatCompletionMessage] | None,
        suffix: list[ChatCompletionMessage],
        continuation: bool,
        cache_metadata: CacheMetadata | None,  # noqa: ARG002
    ) -> list[ChatCompletionMessage]:
        """Prepare messages for OpenAI caching.

        OpenAI handles caching automatically, so we just normalize and combine
        the messages. The provider will automatically cache prefixes >1024 tokens.

        Args:
            cacheable_prefix: Optional cacheable prefix
            suffix: Non-cacheable suffix
            continuation: Whether to append suffix to last prefix message
            cache_metadata: Cache metadata (ignored for implicit caching)

        Returns:
            Combined messages ready for LLM API call
        """
        # No transformation needed for OpenAI (implicit caching)
        return prepare_messages_with_cacheable_transform(
            cacheable_prefix=cacheable_prefix,
            suffix=suffix,
            continuation=continuation,
            transform_cacheable=None,
        )

    def extract_cache_metadata(
        self,
        response: dict,  # noqa: ARG002
        cache_key: str,  # noqa: ARG002
    ) -> CacheMetadata | None:
        """Extract cache metadata from OpenAI response.

        OpenAI responses may include cached_tokens in the usage field.
        For implicit caching, we don't need to store much metadata.

        Args:
            response: OpenAI API response dictionary
            cache_key: Cache key used for this request

        Returns:
            CacheMetadata if extractable, None otherwise
        """
        # For implicit caching, OpenAI handles everything automatically
        # We could extract cached_tokens from response.get("usage", {}).get("cached_tokens")
        # but for now, we don't need to store metadata for implicit caching
        return None

    def get_cache_ttl_seconds(self) -> int:
        """Get cache TTL for OpenAI (1 hour max)."""
        return 3600
