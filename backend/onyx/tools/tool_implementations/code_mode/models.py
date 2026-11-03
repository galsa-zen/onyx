"""Typed details for code mode: one model-written program that calls other tools."""

from collections.abc import Callable
from typing import Annotated, Literal, Self, get_args

from pydantic import BaseModel, Field, JsonValue, SerializeAsAny

from onyx.context.search.models import SearchDocsResponse
from onyx.llm.models import ToolResultMessage
from onyx.tools.models import (
    CustomToolCallSummary,
    FileReadResult,
    LlmBashExecutionResult,
    LlmPythonExecutionResult,
    MemoryUpdated,
    ToolExecutionError,
)
from onyx.tools.tool_implementations.images.models import FinalImageGenerationResponse
from onyx.tools.tool_implementations.search.models import SearchPlanningResult

# Durable details of the tools that programs can call.
CodeModeCallDetails = Annotated[
    SearchDocsResponse
    | CustomToolCallSummary
    | FinalImageGenerationResponse
    | FileReadResult
    | LlmBashExecutionResult
    | LlmPythonExecutionResult
    | MemoryUpdated
    | ToolExecutionError,
    Field(discriminator="type"),
]
CODE_MODE_CALL_DETAIL_TYPES: tuple[type[BaseModel], ...] = get_args(
    get_args(CodeModeCallDetails)[0]
)


# How a program ended.
SandboxStatus = Literal["completed", "failed", "timed_out", "cancelled"]
# How a cell ended: `terminated` cells were stopped by wait_code or by the end of
# the response.
ClosedStatus = Literal[SandboxStatus, "terminated"]
# The state of a cell at the end of an observation: `running` cells continue.
CodeModeStatus = Literal[ClosedStatus, "running"]
# How a call ended: `running` calls had not ended when the observation ended;
# `unfinished` calls were still running when the program ended.
CodeModeCallState = Literal["running", "ok", "error", "unfinished"]


class CodeModeCall(BaseModel):
    """One tool call made by the program, with the result the program received."""

    index: int
    # Tool call id: the id of the `execute_code` call that started the cell, then
    # `:<index>`. Traces, saved history and the UI all use it.
    id: str
    tool_name: str
    # Tool row of the called tool; the UI uses it to reconnect custom and MCP tools.
    tool_id: int | None = None
    # Name of the Python function the program called; differs when the tool name is not
    # a valid identifier.
    function_name: str
    arguments: dict[str, JsonValue]
    # Saved calls keep a bounded copy of the text; the program received all of it.
    content: str = ""
    state: CodeModeCallState
    # Saved search details keep citation numbers but no documents; chat/response.py
    # removes the documents on save and looks them up again on restore.
    details: CodeModeCallDetails | None = None
    # Set while the turn runs: the planning decisions of a search call. Saving
    # (chat/response.py) clears it.
    metadata: SearchPlanningResult | None = None
    # Set by saving (chat/response.py): ids of `details.displayed_docs`. Restoring
    # rebuilds `details.displayed_docs` from them. None while the turn runs.
    displayed_document_ids: list[str] | None = None

    @property
    def is_error(self) -> bool:
        return self.state in ("error", "unfinished")

    def as_result(self) -> ToolResultMessage:
        """This call as the result of an ordinary tool call."""
        return ToolResultMessage(
            tool_call_id=self.id,
            tool_name=self.tool_name,
            content=self.content,
            details=self.details,
            is_error=self.is_error,
            metadata=self.metadata,
        )


class StoreCommit(BaseModel):
    """The store() writes of one cell that ended by itself (completed, or stopped
    at an uncaught exception), None deleting the key, and the commit's place in
    the completion order of the conversation branch."""

    ordinal: int
    writes: dict[str, JsonValue]


class CellWindow(BaseModel):
    """What a cell did in one time window."""

    stdout: str = ""
    # Calls that ended in the window, and calls still running at its end.
    calls: list[CodeModeCall] = Field(default_factory=list)
    # Set on the window in which the cell ended by itself with store() writes. Later
    # programs read the store as the fold of commits over the conversation, in
    # ordinal order (code_mode/store.py).
    commit: StoreCommit | None = None
    # Calls the cell had started by the end of the window.
    tool_call_count: int = 0


class CellClosing(CellWindow):
    """What a cell did after its last observation, recorded when the response ended."""

    # `interrupted`: the server stopped, so what the cell did after is not known.
    reason: Literal["response_ended", "interrupted"]
    status: ClosedStatus


class CodeModeExecutionResult(CellWindow):
    """Saved result of one observation of a cell (an `execute_code` or `wait_code`
    call); replay renders everything from it.

    Its window covers this observation; `closing` covers the cell after it.
    """

    type: Literal["code_mode_execution"] = "code_mode_execution"
    cell_id: int
    # 0 for the `execute_code` call that started the cell, then 1, 2, ...
    observation: int = 0
    # Set on observation 0 only.
    code: str = ""
    status: CodeModeStatus
    # repr() of the final expression, when it is not None.
    result: str | None = None
    error: str | None = None
    duration_seconds: float = 0.0
    # Set when the response ended after this observation reported a running cell.
    closing: CellClosing | None = None

    def windows(self) -> list[CellWindow]:
        """The observation's window, then its closing record's."""
        return [self] if self.closing is None else [self, self.closing]

    def all_calls(self) -> list[CodeModeCall]:
        """The calls of every window, in window order."""
        return [call for window in self.windows() for call in window.calls]

    def map_calls(self, transform: Callable[[CodeModeCall], CodeModeCall]) -> Self:
        """A copy with `transform` applied to the calls of every window."""
        closing = self.closing
        return self.model_copy(
            update={
                "calls": [transform(call) for call in self.calls],
                "closing": None
                if closing is None
                else closing.model_copy(
                    update={"calls": [transform(call) for call in closing.calls]}
                ),
            }
        )


# Progress details, sent through ToolInvocation.update while the program runs.
# They are not saved; the final CodeModeExecutionResult holds the same information.


class CodeModeCallStarted(BaseModel):
    """A call the program started; it has no result yet."""

    type: Literal["code_mode_call_started"] = "code_mode_call_started"
    call: CodeModeCall


class CodeModeCallProgress(BaseModel):
    """Progress details of an inner tool, forwarded unchanged."""

    type: Literal["code_mode_call_progress"] = "code_mode_call_progress"
    index: int
    content: str = ""
    details: SerializeAsAny[BaseModel] | None = None


class CodeModeCallFinished(BaseModel):
    type: Literal["code_mode_call_finished"] = "code_mode_call_finished"
    call: CodeModeCall


class CodeModeStdout(BaseModel):
    """New text the program printed."""

    type: Literal["code_mode_stdout"] = "code_mode_stdout"
    text: str


# What a cell reports to its observer while it runs.
CellEvent = (
    CodeModeCallStarted | CodeModeCallProgress | CodeModeCallFinished | CodeModeStdout
)
