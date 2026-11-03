"""Code mode programs in the agent runtime: the store and restore."""

import pytest

from onyx.agents.events import AgentEvent, ToolEndEvent
from onyx.agents.models import (
    AgentState,
    ExecutionStatus,
    RunState,
    RunStatus,
    ToolCallContext,
)
from onyx.agents.runtime import Agent
from onyx.agents.tools import (
    HumanToolAnswer,
    InputDecision,
    InputMode,
    PendingToolInput,
    ToolInvocation,
    ToolOutcome,
)
from onyx.llm.cancellation import CancellationSignal
from onyx.llm.interfaces import GenerationContext
from onyx.llm.models import (
    AssistantMessage,
    GenerationRequest,
    TextContent,
    ToolCall,
    ToolResult,
    ToolResultMessage,
    UserMessage,
)
from onyx.tools.models import ToolContext
from onyx.tools.tool_implementations.code_mode.cells import (
    CodeCells,
)
from onyx.tools.tool_implementations.code_mode.code_mode_tool import (
    RESTORED_RUNNING_MESSAGE,
    CodeModeTool,
)
from onyx.tracing.flows import LLMFlow
from tests.unit.onyx.agents.fakes import FakeModelClient
from tests.unit.onyx.tools.code_mode.fakes import LIMITS, FakeTool

WRITE = ToolCall(
    id="write", name=CodeModeTool.NAME, arguments={"code": "store('n', 1)"}
)
READ = ToolCall(
    id="read",
    name=CodeModeTool.NAME,
    arguments={"code": "await echo(value='x')\nprint(load('n') + 1)"},
)


def _answer_with_last_result(
    request: GenerationRequest, _signal: CancellationSignal
) -> AssistantMessage:
    results = [
        message
        for message in request.messages
        if isinstance(message, ToolResultMessage)
    ]
    if results:
        return AssistantMessage(content=[TextContent(text=results[-1].text)])
    return AssistantMessage(content=[WRITE, READ])


def _approve_read(context: ToolCallContext) -> PendingToolInput | None:
    if context.call.id != READ.id:
        return None
    return PendingToolInput(
        request_id="run-read", prompt="Run?", mode=InputMode.EXECUTE
    )


def _agent(
    echo_calls: list[str], *, gate: bool, state: AgentState | None = None
) -> Agent:
    """A fresh agent with a fresh code mode tool, as a new process builds it."""

    def echo(invocation: ToolInvocation, _context: ToolContext) -> ToolOutcome:
        echo_calls.append(invocation.call_id)
        return ToolResult(content="echoed")

    tool = CodeModeTool([FakeTool("echo", echo)], cells=CodeCells(), limits=LIMITS)
    return Agent(
        FakeModelClient(_answer_with_last_result),
        generation_context=GenerationContext(flow=LLMFlow.CHAT_RESPONSE),
        agent_id="agent",
        state=state,
        tools=[tool.bind(ToolContext)],
        before_tool_call=_approve_read if gate else None,
    )


def _suspended_before_read(echo_calls: list[str]) -> tuple[AgentState, RunState]:
    agent = _agent(echo_calls, gate=True)
    run = agent.start(max_steps=3, messages=[UserMessage(content="Count")])
    assert run.wait_until_settled(timeout=10).status == RunStatus.SUSPENDED
    assert run.wait_for_idle(5)
    captured = run.capture()
    return captured.agent_state, captured.run_state


def _saved_result(state: RunState, call_id: str) -> ToolResultMessage:
    result = state.steps[0].tools[call_id].result
    assert result is not None
    return result


@pytest.mark.parametrize("restored", [False, True])
def test_a_program_reads_an_earlier_program_of_the_same_step(restored: bool) -> None:
    echo_calls: list[str] = []
    if restored:
        history, saved = _suspended_before_read(echo_calls)
        run = _agent(echo_calls, gate=True, state=history).resume(saved)
        run.submit(
            HumanToolAnswer(request_id="run-read", decision=InputDecision.APPROVE)
        )
    else:
        run = _agent(echo_calls, gate=False).start(
            max_steps=3, messages=[UserMessage(content="Count")]
        )
    output = run.result(timeout=10).output.text
    assert run.wait_for_idle(5)
    assert "Execution completed" in output
    assert "Output:\n2" in output


def test_a_program_saved_while_running_is_not_run_again_on_restore() -> None:
    """The runtime can resume a run whose owner was lost while a program ran.

    Chat saves checkpoints only at idle suspension, so this state comes from
    another runtime caller. A program's calls can have effects, so it never runs
    twice.
    """
    echo_calls: list[str] = []
    history, saved = _suspended_before_read(echo_calls)
    # Built by hand: the owner was lost while the program ran, so no answer is
    # pending any more.
    assert saved.progress is not None
    saved.progress.pending_tool_calls = {}
    assert saved.steps[0].tools[READ.id].status == ExecutionStatus.RUNNING

    events: list[AgentEvent] = []
    agent = _agent(echo_calls, gate=True, state=history)
    run = agent.resume(saved, on_event=events.append)
    output = run.result(timeout=10).output.text
    assert run.wait_for_idle(5)

    assert echo_calls == []
    assert output == RESTORED_RUNNING_MESSAGE
    snapshot = run.snapshot()
    assert snapshot.steps[0].tools[READ.id].status == ExecutionStatus.ERROR
    saved_result = _saved_result(snapshot, READ.id)
    assert (saved_result.is_error, saved_result.details) == (True, None)
    assert any(
        isinstance(event, ToolEndEvent) and event.tool_call.id == READ.id
        for event in events
    )
