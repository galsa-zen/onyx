"""Requests to GPT-5.6+ OpenAI models mark every tool result with an explicit
prompt_cache_breakpoint and send prompt_cache_options, with chat-history
caching off (the default)."""

from collections.abc import Mapping
from typing import Any
from unittest.mock import patch

import pytest
from litellm import ModelResponse as LiteLLMModelResponse

from onyx.llm.factory import get_llm
from onyx.llm.models import (
    AssistantMessage,
    GenerationRequest,
    Message,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)

_RESPONSE = LiteLLMModelResponse(
    id="r",
    created=1,
    choices=[
        {"message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}
    ],
)


def _turn() -> list[Message]:
    return [
        UserMessage(content="Question"),
        AssistantMessage(
            content=[ToolCall(id="c1", name="internal_search", arguments={})]
        ),
        ToolResultMessage(
            content="Evidence", tool_call_id="c1", tool_name="internal_search"
        ),
        AssistantMessage(
            content=[ToolCall(id="c2", name="internal_search", arguments={})]
        ),
        ToolResultMessage(
            content="More evidence", tool_call_id="c2", tool_name="internal_search"
        ),
        UserMessage(content="<system-reminder>Cite sources.</system-reminder>"),
    ]


def _sent(provider: str, model: str, messages: list[Message]) -> Mapping[str, Any]:
    client = get_llm(
        provider=provider,
        model=model,
        deployment_name=None,
        max_input_tokens=100_000,
    )
    with (
        patch("onyx.llm.multi_llm._env_injection_enabled", return_value=False),
        patch("litellm.completion", return_value=_RESPONSE) as completion,
    ):
        client.invoke(GenerationRequest(messages=messages))
    return completion.call_args.kwargs


def _tool_contents(sent: Mapping[str, Any]) -> list[Any]:
    return [m["content"] for m in sent["messages"] if m["role"] == "tool"]


def test_marks_every_tool_result_without_history_caching() -> None:
    sent = _sent("openai", "gpt-5.6-luna", _turn())
    assert _tool_contents(sent) == [
        [
            {
                "type": "input_text",
                "text": text,
                "prompt_cache_breakpoint": {"mode": "explicit"},
            }
        ]
        for text in ("Evidence", "More evidence")
    ]
    assert sent["prompt_cache_options"] == {"mode": "explicit"}
    # The moving reminder itself stays plain text.
    assert sent["messages"][-1]["content"] == (
        "<system-reminder>Cite sources.</system-reminder>"
    )


@pytest.mark.parametrize(
    ("provider", "model"), [("openai", "gpt-4.1"), ("anthropic", "claude-sonnet-5")]
)
def test_other_models_keep_plain_tool_results(provider: str, model: str) -> None:
    sent = _sent(provider, model, _turn())
    assert all(isinstance(content, str) for content in _tool_contents(sent))
    assert "prompt_cache_options" not in sent


def test_no_tool_result_keeps_implicit_caching() -> None:
    sent = _sent("openai", "gpt-5.6-luna", [UserMessage(content="Hello")])
    assert "prompt_cache_options" not in sent
