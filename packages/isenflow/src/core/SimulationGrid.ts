import type { GridDescriptor } from '../utils/gridMath.js';

export interface SimulationGridOptions extends GridDescriptor {
  /** Initial uniform water depth, meters. */
  initialDepth?: number;
  /** Optional CPU-side seed of the bed elevation (length = width * height). */
  initialBed?: Float32Array;
}

/**
 * CPU mirror of the simulation grid descriptor and any seed buffers.
 * The authoritative state lives on the GPU inside `VirtualPipesSolver`.
 */
export class SimulationGrid {
  readonly width: number;
  readonly height: number;
  readonly dx: number;
  readonly origin: readonly [number, number];
  readonly initialDepth: number;
  readonly bedSeed: Float32Array;

  constructor(opts: SimulationGridOptions) {
    this.width = opts.width;
    this.height = opts.height;
    this.dx = opts.dx;
    this.origin = opts.origin;
    this.initialDepth = opts.initialDepth ?? 0;
    const cells = this.width * this.height;
    this.bedSeed = opts.initialBed
      ? new Float32Array(opts.initialBed)
      : new Float32Array(cells);
    if (this.bedSeed.length !== cells) {
      throw new Error(
        `[isenflow] initialBed length ${this.bedSeed.length} ≠ width*height ${cells}`,
      );
    }
  }

  get cells(): number {
    return this.width * this.height;
  }

  worldExtent(): { width: number; height: number } {
    return { width: this.width * this.dx, height: this.height * this.dx };
  }
}
