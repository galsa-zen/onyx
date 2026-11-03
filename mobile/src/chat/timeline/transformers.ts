// GroupedPacket → step → turn-group transforms + parallel detection. Port of web's transformers.

import { GroupedPacket } from "@/chat/messageProcessor";
import { PacketType } from "@/chat/streamingModels";
import {
  isCodeModePackets,
  isGroupEndPacket,
  isNestedPacket,
  NestedCall,
} from "@/chat/timeline/packetHelpers";

export interface TransformedStep {
  key: string;
  turnIndex: number;
  tabIndex: number;
  packets: GroupedPacket["packets"];
  // Tool calls nested under the step, ordered by sub_turn_index
  nestedCalls: NestedCall[];
}

export interface TurnGroup {
  turnIndex: number;
  steps: TransformedStep[];
  isParallel: boolean;
}

export function transformPacketGroup(group: GroupedPacket): TransformedStep {
  return {
    key: `${group.turn_index}-${group.tab_index}`,
    turnIndex: group.turn_index,
    tabIndex: group.tab_index,
    packets: group.packets,
    nestedCalls: group.nestedCalls,
  };
}

export function transformPacketGroups(
  groups: GroupedPacket[],
): TransformedStep[] {
  return groups.map(transformPacketGroup);
}

export function groupStepsByTurn(steps: TransformedStep[]): TurnGroup[] {
  const turnMap = new Map<number, TransformedStep[]>();

  for (const step of steps) {
    const existing = turnMap.get(step.turnIndex);
    if (existing) {
      existing.push(step);
    } else {
      turnMap.set(step.turnIndex, [step]);
    }
  }

  const result: TurnGroup[] = [];
  const sortedTurnIndices = Array.from(turnMap.keys()).sort((a, b) => a - b);

  for (const turnIndex of sortedTurnIndices) {
    const stepsForTurn = turnMap.get(turnIndex)!;
    stepsForTurn.sort((a, b) => a.tabIndex - b.tabIndex);

    result.push({
      turnIndex,
      steps: stepsForTurn,
      isParallel: stepsForTurn.length > 1,
    });
  }

  return result;
}

/** The cell that a code mode step observes, once the stream names it. */
function codeModeCellId(step: TransformedStep): number | null {
  for (const packet of step.packets) {
    if (isNestedPacket(packet)) continue;
    const obj = packet.obj;
    if (
      (obj.type === PacketType.CODE_MODE_START ||
        obj.type === PacketType.CODE_MODE_DELTA) &&
      obj.cell_id != null
    ) {
      return obj.cell_id;
    }
  }
  return null;
}

/**
 * Shows code mode as the tool calls it makes. Each nested call of a program becomes its own
 * step, as a direct call would; the program itself has no step. A cell reports a call in every
 * observation until the call ends, so the latest report replaces earlier ones in place. Steps get
 * sequential turn indices in stream order; direct steps of one turn keep a shared index.
 */
export function expandCodeModeSteps(
  steps: TransformedStep[],
): TransformedStep[] {
  const out: TransformedStep[] = [];
  // Where each call's step sits in `out`, and the turn index it got first.
  const slotByKey = new Map<string, { index: number; turnIndex: number }>();
  const directTurns = new Map<number, number>();
  // The key of each cell's first step; the keys of its calls derive from it.
  const cellKeys = new Map<number, string>();
  let nextTurn = 0;

  for (const step of steps) {
    if (!isCodeModePackets(step.packets)) {
      let turn = directTurns.get(step.turnIndex);
      if (turn === undefined) {
        turn = nextTurn++;
        directTurns.set(step.turnIndex, turn);
      }
      out.push({ ...step, turnIndex: turn });
      continue;
    }

    const cellId = codeModeCellId(step);
    const cellKey =
      (cellId === null ? undefined : cellKeys.get(cellId)) ?? step.key;
    if (cellId !== null) cellKeys.set(cellId, cellKey);

    for (const call of step.nestedCalls) {
      const key = `${cellKey}-${call.subTurnIndex}`;
      const slot = slotByKey.get(key);
      const callStep = {
        key,
        tabIndex: 0,
        packets: call.packets,
        nestedCalls: [],
      };
      if (slot === undefined) {
        const turnIndex = nextTurn++;
        slotByKey.set(key, { index: out.length, turnIndex });
        out.push({ ...callStep, turnIndex });
      } else {
        out[slot.index] = { ...callStep, turnIndex: slot.turnIndex };
      }
    }
  }

  return out;
}

/** Whether a code mode program is working between tool calls: the latest step is a program that
 * has not returned and has no call in flight. Only the timeline header shows this. */
export function isCodeModeWorking(steps: TransformedStep[]): boolean {
  const last = steps.at(-1);
  if (!last || !isCodeModePackets(last.packets)) return false;
  if (last.packets.some(isGroupEndPacket)) return false;
  return last.nestedCalls.every((call) => call.complete);
}
