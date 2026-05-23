import type { VirtualPipesSolver } from '../core/VirtualPipesSolver.js';
import type { SweSolver } from '../core/SweSolver.js';
import type { Aabb2 } from '../utils/gridMath.js';
import { clipAabbToGrid } from '../utils/gridMath.js';

/**
 * Either solver type — both expose the same write/clear helpers consumed
 * by the rasterizer.  Lets demos pick either VP (legacy) or KP (new).
 */
type AnySolver = VirtualPipesSolver | SweSolver;

export interface BoxObstacle {
  aabb: Aabb2;
  /** Elevation that water must surmount to enter this cell, in meters. */
  topY: number;
}

export interface DynamicBodyDescriptor {
  /** Stable chunk id (≥1; 0 is reserved for "no chunk"). */
  chunkId: number;
  /** World-space (X, Z) AABB of the body's footprint at this frame. */
  aabb: Aabb2;
  /** World-space top Y of the body. (Currently informational; buoyancy uses COM±halfY.) */
  topY: number;
  /** Half-extent along Y. The shader uses (com.y ± halfY) for buoyancy. */
  halfY: number;
  /** Linear velocity in world space (used by drag forces). */
  linvel: readonly [number, number, number];
  /** World-space centre of mass (used to convert per-cell forces to torques). */
  com: readonly [number, number, number];
}

/**
 * Bakes static and dynamic obstacles into the solver's bed buffer (spec §4).
 *
 * Static geometry: written once. Dynamic geometry: re-rasterized each frame
 * via `bakeDynamicBodies`, which also clears the previous frame's footprint
 * before stamping the new one to keep the bed consistent.
 */
export class HeightfieldRasterizer {
  private readonly lastFootprints = new Map<number, { x: number; y: number; w: number; h: number }>();

  constructor(public readonly solver: AnySolver) {}

  /**
   * Stamp a single rectangular obstacle into the bed by taking max with the
   * provided terrain elevation. The total bed channel is set; terrain is
   * preserved-or-raised to match. (For dynamic bodies use `bakeDynamicBodies`.)
   */
  bakeBoxObstacle(box: BoxObstacle, terrainElevation = 0): void {
    const g = this.solver.grid;
    const range = clipAabbToGrid(g, box.aabb);
    const w = range.i1 - range.i0 + 1;
    const h = range.j1 - range.j0 + 1;
    if (w <= 0 || h <= 0) return;
    const values = new Float32Array(w * h);
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

  /**
   * Re-rasterize a set of dynamic bodies into the solver's chunk-id, chunk-vel,
   * total-bed and chunk-COM buffers. Cells previously occupied by the body
   * (according to the rasterizer's bookkeeping) are cleared first so a moving
   * body does not leave behind a permanent bed bump.
   *
   * Call this once per frame BEFORE `solver.step()` and `accumulateForces()`.
   */
  bakeDynamicBodies(bodies: Iterable<DynamicBodyDescriptor>): void {
    const g = this.solver.grid;
    // 1) Clear last frame's footprints (chunk id, chunk vel, total bed).
    const seen = new Set<number>();
    for (const body of bodies) {
      seen.add(body.chunkId);
    }
    for (const [id, prev] of this.lastFootprints) {
      // For ids the caller still owns, the new bake will overwrite, but we
      // need to reset the "leftover" cells (cells in the old footprint that
      // are NOT in the new one). To keep this simple and correct: always
      // clear the prior footprint then re-bake. Performance is fine because
      // typical demos have <16 small dynamic bodies.
      this.clearFootprint(prev);
      if (!seen.has(id)) this.lastFootprints.delete(id);
    }
    // 2) Stamp current frame. Dynamic bodies do NOT raise the bed —
    //    instead they mark chunkId/chunkVel and the shader uses
    //    chunkCOMs (com.xyz, halfY in .w) to compute proper buoyancy.
    const coms = new Float32Array(this.solver.maxChunks * 4);
    for (const body of bodies) {
      const range = clipAabbToGrid(g, body.aabb);
      const w = Math.max(0, range.i1 - range.i0 + 1);
      const h = Math.max(0, range.j1 - range.j0 + 1);
      if (w === 0 || h === 0) {
        this.lastFootprints.set(body.chunkId, { x: range.i0, y: range.j0, w: 0, h: 0 });
      } else {
        const region = { x: range.i0, y: range.j0, w, h };
        this.lastFootprints.set(body.chunkId, region);
        this.solver.writeChunkIdRegion(region, body.chunkId);
        this.solver.writeChunkVelRegion(region, body.linvel[0], body.linvel[1], body.linvel[2]);
      }
      const idx = body.chunkId * 4;
      if (idx + 3 < coms.length) {
        coms[idx + 0] = body.com[0];
        coms[idx + 1] = body.com[1];
        coms[idx + 2] = body.com[2];
        coms[idx + 3] = body.halfY;
      }
    }
    this.solver.writeChunkCOMs(coms);
  }

  /**
   * Forget all dynamic-body bookkeeping and zero the chunk-id / chunk-vel
   * buffers. Use when restarting a demo.
   */
  resetDynamicBodies(): void {
    this.lastFootprints.clear();
    this.solver.clearChunkIds();
    this.solver.clearChunkVel();
  }

  private clearFootprint(region: { x: number; y: number; w: number; h: number }): void {
    if (region.w === 0 || region.h === 0) return;
    this.solver.writeChunkIdRegion(region, 0);
    this.solver.writeChunkVelRegion(region, 0, 0, 0);
    // We no longer raise the bed under dynamic bodies, so nothing to restore.
  }
}
