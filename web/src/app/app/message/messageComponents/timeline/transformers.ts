import { GroupedPacket } from "./hooks/packetProcessor";
import { PacketType } from "@/app/app/services/streamingModels";
import {
  isCodeModePackets,
  isGroupEndPacket,
  isNestedPacket,
  NestedCall,
} from "./packetHelpers";

/**
 * Transformed step data ready for rendering
 */
export interface TransformedStep {
  /** Unique key for React rendering */
  key: string;
  /** Key of the packet group the step's packets come from */
  groupKey: string;
  /** Turn index from packet placement */
  turnIndex: number;
  /** Tab index for parallel tools */
  tabIndex: number;
  /** Raw packets for content rendering */
  packets: GroupedPacket["packets"];
  /** Tool calls nested under the step, ordered by sub_turn_index */
  nestedCalls: NestedCall[];
}

/**
 * Group steps by turn_index for detecting parallel tools
 */
export interface TurnGroup {
  turnIndex: number;
  steps: TransformedStep[];
  /** True if multiple steps have the same turn_index (parallel execution) */
  isParallel: boolean;
}

/**
 * Transform a single GroupedPacket into step data
 */
export function transformPacketGroup(group: GroupedPacket): TransformedStep {
  const key = `${group.turn_index}-${group.tab_index}`;
  return {
    key,
    groupKey: key,
    turnIndex: group.turn_index,
    tabIndex: group.tab_index,
    packets: group.packets,
    nestedCalls: group.nestedCalls,
  };
}

/**
 * Transform all packet groups into step data
 */
export function transformPacketGroups(
  groups: GroupedPacket[]
): TransformedStep[] {
  return groups.map(transformPacketGroup);
}

/**
 * Group transformed steps by turn_index to detect parallel tools
 *
 * @example
 * // Input: TransformedStep[]
 * // ┌──────────────────────────────────────────┐
 * // │ [0] key="0-0" turnIndex=0 tabIndex=0     │
 * // │ [1] key="0-1" turnIndex=0 tabIndex=1     │
 * // │ [2] key="1-0" turnIndex=1 tabIndex=0     │
 * // └──────────────────────────────────────────┘
 * //
 * // Step 1: Build Map<turnIndex, TransformedStep[]>
 * // ┌─────────────────────────────────────────────┐
 * // │ turnMap = {                                 │
 * // │   0 → [step(0-0), step(0-1)]               │
 * // │   1 → [step(1-0)]                          │
 * // │ }                                          │
 * // └─────────────────────────────────────────────┘
 * //
 * // Step 2: Sort turn indices & steps by tabIndex
 * //
 * // Step 3: Build TurnGroup[] with isParallel flag
 * // ┌─────────────────────────────────────────────┐
 * // │ Output: TurnGroup[]                         │
 * // ├─────────────────────────────────────────────┤
 * // │ [0] turnIndex=0                             │
 * // │     steps=[0-0, 0-1]                        │
 * // │     isParallel=true  ← 2 steps = parallel   │
 * // │                                             │
 * // │ [1] turnIndex=1                             │
 * // │     steps=[1-0]                             │
 * // │     isParallel=false ← 1 step = sequential  │
 * // └─────────────────────────────────────────────┘
 */
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

/** Whether the program of a code mode step has started running; before that
 * the model is still writing it. */
function programStarted(step: TransformedStep): boolean {
  return step.packets.some(
    (packet) =>
      !isNestedPacket(packet) && packet.obj.type === PacketType.CODE_MODE_START
  );
}

export interface CodeModeTimeline {
  steps: TransformedStep[];
  /** A program is running between tool calls: it has started, has not
   * returned, and none of its cell's calls is in flight. The program has no
   * step of its own, so only the timeline header shows this. */
  codeModeWorking: boolean;
}

// Steps built for nested calls, reused while the call object, key and
// position are unchanged, so memoized step renderers skip them.
const callSteps = new WeakMap<NestedCall, TransformedStep>();

function callStep(
  call: NestedCall,
  key: string,
  groupKey: string,
  turnIndex: number
): TransformedStep {
  const cached = callSteps.get(call);
  if (
    cached &&
    cached.key === key &&
    cached.groupKey === groupKey &&
    cached.turnIndex === turnIndex
  ) {
    return cached;
  }
  const step: TransformedStep = {
    key,
    groupKey,
    turnIndex,
    tabIndex: 0,
    packets: call.packets,
    nestedCalls: [],
  };
  callSteps.set(call, step);
  return step;
}

/**
 * Shows code mode as the tool calls it makes. Each nested call of a program
 * becomes its own step, as a direct call would; the program itself has no
 * step. A cell reports a call in every observation until the call ends, so
 * the latest report of a call replaces the earlier ones in place. Steps get
 * sequential turn indices in stream order; direct steps of one turn keep a
 * shared index, so parallel calls stay grouped.
 */
export function expandCodeModeSteps(
  steps: TransformedStep[]
): CodeModeTimeline {
  const out: TransformedStep[] = [];
  // Where each call's step sits in `out`, the turn index it got first, its
  // cell, and whether its latest report has ended.
  const slotByKey = new Map<
    string,
    { index: number; turnIndex: number; cellKey: string; complete: boolean }
  >();
  const directTurns = new Map<number, number>();
  // The key of each cell's first step; the keys of its calls derive from it.
  const cellKeys = new Map<number, string>();
  // Cell keys of observations whose program is running.
  const runningCells = new Set<string>();
  let nextTurn = 0;

  for (const step of steps) {
    if (!isCodeModePackets(step.packets)) {
      let turn = directTurns.get(step.turnIndex);
      if (turn === undefined) {
        turn = nextTurn++;
        directTurns.set(step.turnIndex, turn);
      }
      out.push(turn === step.turnIndex ? step : { ...step, turnIndex: turn });
      continue;
    }

    const cellId = codeModeCellId(step);
    const cellKey =
      (cellId === null ? undefined : cellKeys.get(cellId)) ?? step.key;
    if (cellId !== null) cellKeys.set(cellId, cellKey);
    if (programStarted(step) && !step.packets.some(isGroupEndPacket)) {
      runningCells.add(cellKey);
    } else {
      runningCells.delete(cellKey);
    }

    for (const call of step.nestedCalls) {
      const key = `${cellKey}-${call.subTurnIndex}`;
      const slot = slotByKey.get(key);
      if (slot === undefined) {
        const turnIndex = nextTurn++;
        slotByKey.set(key, {
          index: out.length,
          turnIndex,
          cellKey,
          complete: call.complete,
        });
        out.push(callStep(call, key, step.groupKey, turnIndex));
      } else {
        slot.complete = call.complete;
        out[slot.index] = callStep(call, key, step.groupKey, slot.turnIndex);
      }
    }
  }

  const slots = Array.from(slotByKey.values());
  const codeModeWorking = Array.from(runningCells).some((cellKey) =>
    slots.every((slot) => slot.cellKey !== cellKey || slot.complete)
  );
  return { steps: out, codeModeWorking };
}
