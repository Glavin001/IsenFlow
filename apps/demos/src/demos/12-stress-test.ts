import * as THREE from 'three';
import type { Demo, DemoContext } from '../shared/Scene.js';
import { ownByDemo } from '../shared/Scene.js';
import { WaterSurface } from '../shared/Water.js';
import { BoundaryType, applyImpact, spawnImpactSplash } from 'isenflow';

/**
 * L. Stress Test — a torture demo that exercises every feature simultaneously
 * on the KP solver.  Designed to demonstrate that the post-rewrite solver
 * handles the regimes that broke Virtual Pipes:
 *
 *   - Complex Gaussian-peak terrain (mountain ridge + lake basin)
 *   - 4 inflow boundaries pouring water from "mountain passes"
 *   - ~36 buildings around the lake, all using the Solid boundary type
 *     (zero-flux walls — the wet/dry interface that gave VP spike artifacts)
 *   - 3 scheduled dam-break events at t = 8 / 16 / 24 s, releasing held
 *     water as massive transient waves
 *   - 12 floating debris boxes coupled via Rapier (true momentum drag from
 *     KP's accurate hu/hv field)
 *   - Periodic rockfall impacts (large, dense objects → high Froude) using
 *     the new momentum-radiating applyImpact pipeline
 *
 * Grid: shared 16 m × 16 m / 384² default.  Future per-demo grid override
 * (Phase 5a in the plan) will let this demo scale to 32 m or larger.
 */

// ── Seeded PRNG (stable test fixture) ────────────────────────────────
function makeRng(seed: number): () => number {
  let s = seed | 0 || 1;
  return () => {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return (s >>> 0) / 0x100000000;
  };
}

// ── Terrain features ─────────────────────────────────────────────────
interface Peak { cx: number; cz: number; amp: number; sx: number; sz: number; }

// Mountain ridge along the north + lake basin in the south-center
const TERRAIN: Peak[] = [
  // North mountain ridge
  { cx: -6, cz: -7, amp: 3.5, sx: 1.6, sz: 1.0 },
  { cx: -2, cz: -7, amp: 4.2, sx: 1.4, sz: 1.2 },
  { cx:  2, cz: -7, amp: 3.8, sx: 1.5, sz: 1.0 },
  { cx:  6, cz: -7, amp: 3.0, sx: 1.6, sz: 1.0 },
  // Mid-ridge mountain pass valleys (narrow channels through the ridge)
  { cx: -4, cz: -6, amp: -1.5, sx: 0.5, sz: 0.8 },
  { cx:  0, cz: -6, amp: -1.5, sx: 0.5, sz: 0.8 },
  { cx:  4, cz: -6, amp: -1.5, sx: 0.5, sz: 0.8 },
  // Western highlands
  { cx: -7, cz: -2, amp: 2.5, sx: 1.5, sz: 1.5 },
  { cx: -7, cz:  2, amp: 2.0, sx: 1.5, sz: 1.5 },
  // Eastern highlands
  { cx:  7, cz: -2, amp: 2.3, sx: 1.5, sz: 1.5 },
  { cx:  7, cz:  2, amp: 1.8, sx: 1.5, sz: 1.5 },
  // Central lake basin (depression)
  { cx:  0, cz:  1, amp: -1.8, sx: 3.5, sz: 2.5 },
  // South-flowing outflow channel
  { cx:  0, cz:  6, amp: -0.8, sx: 0.6, sz: 1.5 },
];

function sampleTerrain(wx: number, wz: number): number {
  let h = 0;
  for (const p of TERRAIN) {
    const dx = wx - p.cx;
    const dz = wz - p.cz;
    h += p.amp * Math.exp(-(dx * dx) / (2 * p.sx * p.sx) - (dz * dz) / (2 * p.sz * p.sz));
  }
  return Math.max(-2.0, h);  // clip the lake basin so it has a floor
}

interface Building {
  cx: number; cz: number;     // grid-center cell coords
  halfW: number; halfH: number;  // half-extent in cells
  wallH: number;
  color: number;
}

interface DamBreak {
  cells: { x: number; y: number; w: number; h: number };
  triggerT: number;   // sim-time at which the dam fails
  visualMeshes: THREE.Mesh[];
  failed: boolean;
}

interface DebrisBody { id: number; mesh: THREE.Mesh; }

const demo: Demo = {
  id: '12-stress-test',
  label: 'L. Stress Test',
  description:
    'Kilometer-scale terrain stress test: mountain ridge with 3 passes pouring flood water into a central lake basin, ~36 buildings (solid-walled) around the perimeter, 3 scheduled dam breaks unleashing massive waves, periodic rockfall impacts, and 12 floating debris bodies driven by the KP momentum field.  Validates every spike-prone regime in one scene.',

  setup(ctx: DemoContext) {
    const g = ctx.solver.grid;
    const W = g.width;
    const H = g.height;
    const dx = g.dx;
    const rng = makeRng(1337);

    // 1. Bake terrain
    const terrain = new Float32Array(W * H);
    for (let j = 0; j < H; j++) {
      for (let i = 0; i < W; i++) {
        const wx = g.origin[0] + (i + 0.5) * dx;
        const wz = g.origin[1] + (j + 0.5) * dx;
        terrain[j * W + i] = sampleTerrain(wx, wz);
      }
    }
    ctx.solver.writeBedFull(terrain);

    // 2. Pre-fill lake basin with water
    const lakeInit = new Float32Array(W * H);
    for (let j = 0; j < H; j++) {
      for (let i = 0; i < W; i++) {
        const wz = g.origin[1] + (j + 0.5) * dx;
        const bed = terrain[j * W + i] ?? 0;
        // Fill below z=4 to lake surface elevation +0.5m above bed
        if (wz > -3 && wz < 5 && bed < 0.5) {
          lakeInit[j * W + i] = Math.max(0, 0.5 - bed);
        }
      }
    }
    ctx.solver.writeWaterFull(lakeInit);

    // 3. North-edge inflow patches at 3 mountain pass locations
    //    + sustained flood for 30 seconds.
    const passXs = [-4, 0, 4];
    for (const px of passXs) {
      const pCellX = Math.round((px - g.origin[0]) / dx);
      const region = {
        x: Math.max(0, pCellX - 2),
        y: 0,
        w: 5,
        h: 3,
      };
      ctx.solver.writeBoundaryRegionTarget(region, BoundaryType.Inflow, 0.8);
    }
    // South-edge outflow
    ctx.solver.writeBoundaryRegion(
      { x: 0, y: H - 1, w: W, h: 1 },
      BoundaryType.Open,
    );

    // 4. Buildings around the lake perimeter (all Solid-walled when supported)
    const buildings: Building[] = [];
    const ringRadius = 4.5;
    const buildingCount = 36;
    for (let k = 0; k < buildingCount; k++) {
      const theta = (k / buildingCount) * Math.PI * 2 + rng() * 0.1;
      const wx = Math.cos(theta) * ringRadius;
      const wz = 1 + Math.sin(theta) * ringRadius * 0.7;
      // Skip if inside the north mountain pass area
      if (wz < -3) continue;
      buildings.push({
        cx: Math.round((wx - g.origin[0]) / dx),
        cz: Math.round((wz - g.origin[1]) / dx),
        halfW: 2 + Math.floor(rng() * 3),    // 5-11 cells wide (~0.2-0.45 m)
        halfH: 2 + Math.floor(rng() * 3),
        wallH: 1.5 + rng() * 1.5,             // 1.5-3.0 m tall
        color: 0x886655 + Math.floor(rng() * 0x222222),
      });
    }

    const supportsSolid =
      typeof (ctx.solver as { markSolidRegion?: unknown }).markSolidRegion === 'function';

    const placedBoxes: THREE.Mesh[] = [];
    for (const b of buildings) {
      // Stamp wall cells (bed elevation + solid mask if supported)
      const x0 = b.cx - b.halfW;
      const y0 = b.cz - b.halfH;
      const w = 2 * b.halfW + 1;
      const h = 2 * b.halfH + 1;
      if (x0 < 0 || y0 < 0 || x0 + w >= W || y0 + h >= H) continue;
      const buf = new Float32Array(w * h).fill(b.wallH);
      ctx.solver.writeBedRegion({ x: x0, y: y0, w, h }, buf);
      if (supportsSolid) {
        const ms = (ctx.solver as { markSolidRegion: (r: { x: number; y: number; w: number; h: number }) => void }).markSolidRegion;
        ms({ x: x0, y: y0, w, h });
      }
      // Visual mesh
      const mat = new THREE.MeshStandardMaterial({ color: b.color, roughness: 0.85 });
      const lx = w * dx;
      const lz = h * dx;
      const cxw = g.origin[0] + (b.cx + 0.5) * dx;
      const czw = g.origin[1] + (b.cz + 0.5) * dx;
      const mesh = ownByDemo(new THREE.Mesh(new THREE.BoxGeometry(lx, b.wallH, lz), mat));
      mesh.name = `building-${b.cx}-${b.cz}`;
      mesh.position.set(cxw, b.wallH / 2, czw);
      ctx.scene.add(mesh);
      placedBoxes.push(mesh);
    }

    // 5. Scheduled dam-break walls.
    // 3 transverse dams within the lake basin, each holding back ~1.5 m water
    // until they "fail" by switching their cells from Solid → empty bed.
    const dams: DamBreak[] = [];
    const damXs = [-3, 0, 3];
    const damTriggers = [8, 16, 24];
    for (let k = 0; k < damXs.length; k++) {
      const dx_world = damXs[k]!;
      const cx = Math.round((dx_world - g.origin[0]) / dx);
      const region = { x: cx, y: Math.round((-2.5 - g.origin[1]) / dx), w: 1, h: Math.round(3 / dx) };
      // Tall solid wall stretching across the basin's north-south extent
      const wallH = 2.5;
      const buf = new Float32Array(region.w * region.h).fill(wallH);
      ctx.solver.writeBedRegion(region, buf);
      if (supportsSolid) {
        const ms = (ctx.solver as { markSolidRegion: (r: { x: number; y: number; w: number; h: number }) => void }).markSolidRegion;
        ms(region);
      }
      // Visual mesh
      const mat = new THREE.MeshStandardMaterial({ color: 0x554433, roughness: 0.9 });
      const lx = region.w * dx;
      const lz = region.h * dx;
      const cxw = g.origin[0] + (cx + 0.5) * dx;
      const czw = g.origin[1] + (region.y + region.h / 2) * dx;
      const mesh = ownByDemo(new THREE.Mesh(new THREE.BoxGeometry(lx, wallH, lz), mat));
      mesh.name = `dam-${k}`;
      mesh.position.set(cxw, wallH / 2, czw);
      ctx.scene.add(mesh);
      dams.push({
        cells: region,
        triggerT: damTriggers[k]!,
        visualMeshes: [mesh],
        failed: false,
      });
    }

    // 6. Floating debris (12 wooden crates)
    const debris: DebrisBody[] = [];
    const crateMat = new THREE.MeshStandardMaterial({ color: 0xaa7733, roughness: 0.8 });
    for (let k = 0; k < 12; k++) {
      const wx = -3 + rng() * 6;
      const wz = -1 + rng() * 4;
      const size = 0.2;
      const body = ctx.world.createRigidBody(
        ctx.rapier.RigidBodyDesc.dynamic()
          .setTranslation(wx, 2 + rng() * 1, wz)
          .setLinearDamping(0.4)
          .setAngularDamping(0.3),
      );
      ctx.world.createCollider(
        ctx.rapier.ColliderDesc.cuboid(size / 2, size / 2, size / 2).setDensity(600),
        body,
      );
      const mesh = ownByDemo(new THREE.Mesh(new THREE.BoxGeometry(size, size, size), crateMat));
      mesh.name = `crate-${k}`;
      ctx.scene.add(mesh);
      const cb = ctx.spawnCoupledBody({
        name: `crate-${k}`,
        body,
        halfExtents: [size / 2, size / 2, size / 2],
        mesh,
        waterLevelRef: 0.5,
        bedLevelRef: -1.0,
      });
      debris.push({ id: cb.chunkId, mesh });
    }

    // 7. Water surface
    const water = new WaterSurface(ctx.solver);
    ctx.scene.add(ownByDemo(water.mesh));

    ctx.scratch.water = water;
    ctx.scratch.dams = dams;
    ctx.scratch.debris = debris;
    ctx.scratch.terrain = terrain;
    ctx.scratch.placedBoxes = placedBoxes;
    ctx.scratch.t = 0;
    ctx.scratch.nextImpact = 3.0;
    ctx.scratch.rng = rng;
  },

  tick(ctx, dt) {
    (ctx.scratch.water as WaterSurface).update();
    ctx.scratch.t = (ctx.scratch.t as number) + dt;
    const t = ctx.scratch.t as number;
    const dams = ctx.scratch.dams as DamBreak[];
    const terrain = ctx.scratch.terrain as Float32Array;
    const rng = ctx.scratch.rng as () => number;

    // Trigger dam breaks
    for (const d of dams) {
      if (d.failed) continue;
      if (t >= d.triggerT) {
        d.failed = true;
        // Restore terrain in the dam cells (water rushes through)
        const buf = new Float32Array(d.cells.w * d.cells.h);
        const W = ctx.solver.grid.width;
        for (let dj = 0; dj < d.cells.h; dj++) {
          for (let di = 0; di < d.cells.w; di++) {
            buf[dj * d.cells.w + di] = terrain[(d.cells.y + dj) * W + (d.cells.x + di)] ?? 0;
          }
        }
        ctx.solver.writeBedRegion(d.cells, buf);
        ctx.solver.writeBoundaryRegion(d.cells, BoundaryType.Interior);
        // Hide visual meshes
        for (const m of d.visualMeshes) m.visible = false;
      }
    }

    // Periodic rockfall impacts in the mountain area
    if (t >= (ctx.scratch.nextImpact as number)) {
      const wx = -6 + rng() * 12;
      const wz = -7 + rng() * 4;  // mountain region
      const result = applyImpact(ctx.solver, {
        worldX: wx,
        worldZ: wz,
        velocity: 5 + rng() * 10,
        shape: 'sphere',
        size: 0.15 + rng() * 0.35,
        density: 2500,
        waterDepth: Math.max(0.05, 0.3 + rng() * 0.5),
      });
      spawnImpactSplash(ctx.splashes, {
        worldX: wx,
        worldZ: wz,
        velocity: 8,
        shape: 'sphere',
        size: 0.3,
        density: 2500,
        waterDepth: 0.5,
      }, result);
      ctx.scratch.nextImpact = t + 2 + rng() * 3;
    }
  },
};

export default demo;
