import { isToolComplete } from "@/app/app/message/messageComponents/toolDisplayHelpers";
import {
  CODE_MODE_TOOL_TYPE,
  Packet,
  PacketType,
} from "@/app/app/services/streamingModels";

const parent = { turn_index: 0, tab_index: 0 };
const nested = { ...parent, sub_turn_index: 0 };

const codeModeRunning: Packet[] = [
  {
    placement: parent,
    obj: {
      type: PacketType.TOOL_CALL_ARGUMENT_DELTA,
      tool_type: CODE_MODE_TOOL_TYPE,
      tool_id: "call-1",
      argument_deltas: { code: "await search(query='x')" },
    },
  },
  {
    placement: parent,
    obj: { type: PacketType.CODE_MODE_START, code: "await search(query='x')" },
  },
  { placement: nested, obj: { type: PacketType.SEARCH_TOOL_START } },
  { placement: nested, obj: { type: PacketType.SECTION_END } },
];

describe("isToolComplete", () => {
  test("a code mode step is not complete after a nested SECTION_END", () => {
    expect(isToolComplete(codeModeRunning)).toBe(false);
  });

  test("a code mode step is complete after its parent SECTION_END", () => {
    expect(
      isToolComplete([
        ...codeModeRunning,
        { placement: parent, obj: { type: PacketType.SECTION_END } },
      ])
    ).toBe(true);
  });

  test("a research agent is complete only after its parent SECTION_END", () => {
    const research: Packet[] = [
      {
        placement: parent,
        obj: { type: PacketType.RESEARCH_AGENT_START, research_task: "x" },
      },
      { placement: nested, obj: { type: PacketType.SEARCH_TOOL_START } },
      { placement: nested, obj: { type: PacketType.SECTION_END } },
    ];

    expect(isToolComplete(research)).toBe(false);
    expect(
      isToolComplete([
        ...research,
        { placement: parent, obj: { type: PacketType.SECTION_END } },
      ])
    ).toBe(true);
  });

  test("a nested bash error does not end a running coding agent", () => {
    const codingAgent: Packet[] = [
      {
        placement: parent,
        obj: { type: PacketType.CODING_AGENT_START, query: "fix", repo: "r" },
      },
      {
        placement: nested,
        obj: { type: PacketType.BASH_TOOL_START, cmd: "ls" },
      },
      { placement: nested, obj: { type: PacketType.ERROR, message: "exit 1" } },
    ];

    expect(isToolComplete(codingAgent)).toBe(false);
    expect(
      isToolComplete([
        ...codingAgent,
        {
          placement: parent,
          obj: { type: PacketType.CODING_AGENT_FINAL, answer: "done" },
        },
        { placement: parent, obj: { type: PacketType.SECTION_END } },
      ])
    ).toBe(true);
  });

  test("a tool without nested calls is complete after its SECTION_END", () => {
    expect(
      isToolComplete([
        { placement: parent, obj: { type: PacketType.SEARCH_TOOL_START } },
        { placement: parent, obj: { type: PacketType.SECTION_END } },
      ])
    ).toBe(true);
  });
});
