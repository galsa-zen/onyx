"""The code mode store keeps values that history compaction hides from the model."""

from onyx.agents.models import AgentState, CompactionCheckpoint
from onyx.agents.runtime import Agent
from onyx.llm.cancellation import CancellationSignal
from onyx.llm.interfaces import GenerationContext
from onyx.llm.models import (
    AssistantMessage,
    GenerationRequest,
    Message,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from onyx.tools.models import ToolContext
from onyx.tools.tool_implementations.code_mode.cells import CodeCells
from onyx.tools.tool_implementations.code_mode.code_mode_tool import CodeModeTool
from onyx.tools.tool_implementations.code_mode.models import (
    CodeModeExecutionResult,
    StoreCommit,
)
from onyx.tools.tool_implementations.code_mode.sandbox import SandboxLimits
from onyx.tracing.flows import LLMFlow
from tests.unit.onyx.agents.fakes import EchoTool, FakeModelClient

EARLIER_TURN: list[Message] = [
    UserMessage(id="question", content="Remember a value"),
    AssistantMessage(
        id="store-call",
        content=[
            ToolCall(id="store", name=CodeModeTool.NAME, arguments={"code": "..."})
        ],
    ),
    ToolResultMessage(
        id="store-result",
        tool_call_id="store",
        tool_name=CodeModeTool.NAME,
        content="Stored: k.",
        details=CodeModeExecutionResult(
            cell_id=1,
            code="store('k', 'kept')",
            status="completed",
            commit=StoreCommit(ordinal=1, writes={"k": "kept"}),
        ),
    ),
    AssistantMessage(id="answer", content=[TextContent(text="Saved.")]),
]


def _load_in_new_turn(
    history: list[Message], checkpoint: CompactionCheckpoint | None
) -> tuple[str, list[GenerationRequest]]:
    requests: list[GenerationRequest] = []

    def reply(request: GenerationRequest, _: CancellationSignal) -> AssistantMessage:
        requests.append(request)
        if len(requests) == 1:
            return AssistantMessage(
                content=[
                    ToolCall(
                        id="load",
                        name=CodeModeTool.NAME,
                        arguments={"code": "load('k')"},
                    )
                ]
            )
        return AssistantMessage(content=[TextContent(text="Done.")])

    agent = Agent(
        FakeModelClient(reply),
        tools=[
            CodeModeTool([EchoTool()], cells=CodeCells(), limits=SandboxLimits()).bind(
                ToolContext
            )
        ],
        generation_context=GenerationContext(flow=LLMFlow.CHAT_RESPONSE),
        state=AgentState(messages=history, checkpoint=checkpoint),
    )
    run = agent.start(max_steps=2, messages=[UserMessage(content="Load it")])
    run.result()
    assert run.wait_for_idle(timeout=5)
    loaded = next(
        message
        for message in requests[-1].messages
        if isinstance(message, ToolResultMessage) and message.tool_call_id == "load"
    )
    assert isinstance(loaded.details, CodeModeExecutionResult)
    return loaded.details.result or "", requests


def test_store_keeps_values_written_before_the_compaction_boundary() -> None:
    checkpoint = CompactionCheckpoint(
        summary="The user saved a value.", covered_through_message_id="answer"
    )
    value, requests = _load_in_new_turn(EARLIER_TURN, checkpoint)
    assert value == "'kept'"
    # The model does not see the compacted store call.
    assert not any(
        isinstance(message, ToolResultMessage) for message in requests[0].messages
    )
