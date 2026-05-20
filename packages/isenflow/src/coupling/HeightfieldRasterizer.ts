import type { VirtualPipesSolver } from '../core/VirtualPipesSolver.js';
import type { Aabb2 } from '../utils/gridMath.js';
import { clipAabbToGrid } from '../utils/gridMath.js';

export interface BoxObstacle {
  aabb: Aabb2;
  /** Elevation that water must surmount to enter this cell, in meters. */
  topY: number;
}

/**
 * Bakes static and dynamic obstacles into the solver's bed texture, per spec §4.
 *
 * Static geometry is written once at scene load. Dynamic geometry (debris,
 * vehicles, floating crates) is rasterized each frame.
 */
export class HeightfieldRasterizer {
  constructor(public readonly solver: VirtualPipesSolver) {}

  /** Stamp a single rectangular obstacle into the bed by taking max(bed, topY). */
  bakeBoxObstacle(box: BoxObstacle, terrainElevation = 0): void {
    const g = this.solver.grid;
    const range = clipAabbToGrid(g, box.aabb);
    const w = range.i1 - range.i0 + 1;
    const h = range.j1 - range.j0 + 1;
    if (w <= 0 || h <= 0) return;
    const values = new Float32Array(w * h);
    // Take per-cell max with current terrain. (Naive: assumes terrain flat in region.)
    for (let i = 0; i < values.length; i++) values[i] = Math.max(terrainElevation, box.topY);
    this.solver.writeBedRegion({ x: range.i0, y: range.j0, w, h }, values);
  }

  /** Stamp a doorway (low height = ground) over a previously baked wall. */
  bakeDoorway(aabb: Aabb2, sillHeight: number): void {
    const g = this.solver.grid;
    const range = clipAabbToGrid(g, aabb);
    const w = range.i1 - range.i0 + 1;
    const h = range.j1 - range.j0 + 1;
    if (w <= 0 || h <= 0) return;
    const values = new Float32Array(w * h);
    values.fill(sillHeight);
    this.solver.writeBedRegion({ x: range.i0, y: range.j0, w, h }, values);
  }

  /**
   * Bake a heightmap (e.g. terrain mesh sampled to a Float32Array) into the
   * bed's "terrain" channel for the whole grid.
   */
  bakeTerrainHeightmap(map: Float32Array): void {
    if (map.length !== this.solver.grid.cells) {
      throw new Error('bakeTerrainHeightmap: length mismatch');
    }
    this.solver.writeBedRegion(
      { x: 0, y: 0, w: this.solver.grid.width, h: this.solver.grid.height },
      map,
    );
  }
}
