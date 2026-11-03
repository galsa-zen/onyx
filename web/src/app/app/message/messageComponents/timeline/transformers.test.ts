import {
  expandCodeModeSteps,
  groupStepsByTurn,
  isCodeModeWorking,
  transformPacketGroups,
} from "@/app/app/message/messageComponents/timeline/transformers";
import {
  createInitialState,
  processPackets,
} from "@/app/app/message/messageComponents/timeline/hooks/packetProcessor";
import { Packet, PacketType } from "@/app/app/services/streamingModels";

function at(turn: number, tab = 0, sub?: number) {
  return { turn_index: turn, tab_index: tab, sub_turn_index: sub };
}

function observation(
  turn: number,
  cellId: number | null,
  body: Packet[],
  status?: "running" | "completed"
): Packet[] {
  return [
    {
      placement: at(turn),
      obj: {
        type: PacketType.CODE_MODE_START,
        code: "",
        cell_id: turn === 0 ? null : cellId,
      },
    },
    ...body,
    ...(status
      ? [
          {
            placement: at(turn),
            obj: {
              type: PacketType.CODE_MODE_DELTA,
              stdout: "",
              status,
              cell_id: cellId,
            },
          } as Packet,
          {
            placement: at(turn),
            obj: { type: PacketType.SECTION_END },
          } as Packet,
        ]
      : []),
  ];
}

function search(turn: number, sub: number, done: boolean): Packet[] {
  return [
    {
      placement: at(turn, 0, sub),
      obj: { type: PacketType.SEARCH_TOOL_START },
    },
    ...(done
      ? [
          {
            placement: at(turn, 0, sub),
            obj: { type: PacketType.SECTION_END },
          } as Packet,
        ]
      : []),
  ];
}

/** Timeline steps of a stream, grouped as the packet processor groups them. */
function stepsOf(packets: Packet[]) {
  return transformPacketGroups(
    processPackets(createInitialState(0), packets).toolGroups
  );
}

function expand(packets: Packet[]) {
  return groupStepsByTurn(expandCodeModeSteps(stepsOf(packets)));
}

test("each call of a program is its own step, without the program", () => {
  const turns = expand(
    observation(
      0,
      1,
      [...search(0, 0, true), ...search(0, 1, true)],
      "completed"
    )
  );

  expect(turns.map((turn) => turn.steps.map((step) => step.key))).toEqual([
    ["0-0-0"],
    ["0-0-1"],
  ]);
  for (const turn of turns) {
    expect(turn.isParallel).toBe(false);
    expect(turn.steps[0]!.packets[0]!.obj.type).toBe(
      PacketType.SEARCH_TOOL_START
    );
  }
});

test("a later observation replaces a running call in place", () => {
  const turns = expand([
    ...observation(0, 1, search(0, 0, false), "running"),
    { placement: at(1), obj: { type: PacketType.SEARCH_TOOL_START } },
    { placement: at(1), obj: { type: PacketType.SECTION_END } },
    ...observation(2, 1, search(2, 0, true), "completed"),
  ]);

  expect(turns.map((turn) => turn.steps.map((step) => step.key))).toEqual([
    ["0-0-0"],
    ["1-0"],
  ]);
  const call = turns[0]!.steps[0]!;
  expect(call.groupKey).toBe("2-0");
  expect(call.packets.some((p) => p.obj.type === PacketType.SECTION_END)).toBe(
    true
  );
});

test("a program without calls has no step", () => {
  const first = observation(0, 1, [], "running");
  expect(expand([...first, ...observation(1, 1, [])])).toEqual([]);

  const withCall = [...first, ...observation(1, 1, search(1, 0, true))];
  expect(expand(withCall).map((turn) => turn.steps.map((s) => s.key))).toEqual([
    ["0-0-0"],
  ]);
});

test("a program works between calls until it returns", () => {
  const working = (packets: Packet[]) => isCodeModeWorking(stepsOf(packets));

  expect(working(observation(0, 1, []))).toBe(true);
  expect(working(observation(0, 1, search(0, 0, false)))).toBe(false);
  expect(working(observation(0, 1, search(0, 0, true)))).toBe(true);
  expect(working(observation(0, 1, [], "completed"))).toBe(false);
});

test("direct parallel calls keep their shared turn", () => {
  const turns = expand([
    ...search(0, 0, true).map((p) => ({ ...p, placement: at(0, 0) })),
    ...search(0, 0, true).map((p) => ({ ...p, placement: at(0, 1) })),
  ]);

  expect(turns).toHaveLength(1);
  expect(turns[0]!.isParallel).toBe(true);
});
