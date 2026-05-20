import { type StressState, createStress } from './StressAccumulator.js';

/**
 * A destructible chunk of solid bed elevation. Lives in the grid as a region
 * of cells. While `isStatic`, it raises bed elevation by `wallTop − floorY`.
 * When fractured, its cells are released and the chunk becomes a dynamic body.
 */
export interface WallChunkInit {
  id: number;
  /** World-space AABB cells the chunk currently occupies. */
  cells: ReadonlyArray<{ i: number; j: number }>;
  /** Wall top elevation while intact. */
  wallTopY: number;
  /** Floor elevation (the bed value once the chunk is gone). */
  floorY: number;
  /** Center of mass in world space (X, Y, Z). */
  com: readonly [number, number, number];
  /** Stress budget in N·s. */
  stressBudget: number;
}

export class WallChunk {
  readonly id: number;
  readonly cells: ReadonlyArray<{ i: number; j: number }>;
  readonly wallTopY: number;
  readonly floorY: number;
  com: readonly [number, number, number];
  readonly stress: StressState;
  isStatic = true;

  constructor(init: WallChunkInit) {
    this.id = init.id;
    this.cells = init.cells;
    this.wallTopY = init.wallTopY;
    this.floorY = init.floorY;
    this.com = init.com;
    this.stress = createStress(init.stressBudget);
  }
}
