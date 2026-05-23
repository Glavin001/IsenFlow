/**
 * Demo 10 — Mountain Terrain
 *
 * A procedural mountain range built from a multi-peak Gaussian heightfield.
 * Four major summits surround a central lake basin. Mountain springs feed
 * continuous inflow at the peaks; water flows through valleys, cascades over
 * ridgelines, and drains from the open edges. Floating debris bobs in the lake.
 *
 * Key technique: `HeightfieldRasterizer.bakeTerrainHeightmap` stamps the full
 * 384×384 bed in one call; a separate 64×64 Rapier heightfield collider (stored
 * column-major per the rapier3d API) provides physics for the floating objects.
 */
import * as THREE from 'three';
import type { Demo } from '../shared/Scene.js';
import { ownByDemo } from '../shared/Scene.js';
import { WaterSurface } from '../shared/Water.js';
import { BoundaryType } from 'isenflow';

// ---------------------------------------------------------------------------
// World constants
// ---------------------------------------------------------------------------

const WORLD_SIZE = 16; // matches the 384-cell grid in Scene.ts

/** Water surface level for the initial lake fill (metres above datum). */
const LAKE_SURFACE = 1.8;

/** Minimum terrain elevation — the basin floor. */
const VALLEY_FLOOR = 0.15;

// ---------------------------------------------------------------------------
// Mountain peaks (normalised [0,1] × [0,1] → world [-8,8] × [-8,8])
// ---------------------------------------------------------------------------

interface Peak {
  nx: number; // 0..1 in X
  nz: number; // 0..1 in Z
  amp: number; // height in metres
  sig: number; // Gaussian sigma (in normalised coords)
}

const PEAKS: readonly Peak[] = [
  { nx: 0.17, nz: 0.19, amp: 5.8, sig: 0.10 }, // NW summit
  { nx: 0.83, nz: 0.17, amp: 6.8, sig: 0.09 }, // NE summit (tallest)
  { nx: 0.14, nz: 0.80, amp: 5.2, sig: 0.10 }, // SW summit
  { nx: 0.81, nz: 0.82, amp: 6.3, sig: 0.09 }, // SE summit
];

// ---------------------------------------------------------------------------
// Terrain function
// ---------------------------------------------------------------------------

/** Returns terrain elevation in metres for normalised position (nx, nz). */
function terrainAt(nx: number, nz: number): number {
  let elev = 0;
  for (const p of PEAKS) {
    const dx = nx - p.nx;
    const dz = nz - p.nz;
    elev += p.amp * Math.exp(-(dx * dx + dz * dz) / (2 * p.sig * p.sig));
  }
  // Central bowl that creates the lake basin
  const cx = nx - 0.5;
  const cz = nz - 0.5;
  elev -= 2.4 * Math.exp(-(cx * cx + cz * cz) / 0.07);
  return Math.max(VALLEY_FLOOR, elev);
}

// ---------------------------------------------------------------------------
// Heightmap builders
// ---------------------------------------------------------------------------

/** Row-major heightmap for the IsenFlow bed (j * width + i). */
function buildBedMap(width: number, height: number): Float32Array {
  const h = new Float32Array(width * height);
  for (let j = 0; j < height; j++) {
    for (let i = 0; i < width; i++) {
      h[j * width + i] = terrainAt(i / (width - 1), j / (height - 1));
    }
  }
  return h;
}

/**
 * Column-major heightmap for the Rapier heightfield collider.
 *
 * Rapier's `ColliderDesc.heightfield(nrows, ncols, heights, scale)` expects
 * heights stored column-major: `heights[row + col * nrows]`.
 * Here col → X axis, row → Z axis, matching our world convention.
 */
function buildRapierHeights(nrows: number, ncols: number): Float32Array {
  const h = new Float32Array(nrows * ncols);
  for (let col = 0; col < ncols; col++) {
    const nx = col / (ncols - 1);
    for (let row = 0; row < nrows; row++) {
      const nz = row / (nrows - 1);
      h[row + col * nrows] = terrainAt(nx, nz);
    }
  }
  return h;
}

// ---------------------------------------------------------------------------
// Visual terrain mesh
// ---------------------------------------------------------------------------

/**
 * Build a PlaneGeometry (rotated to XZ) with vertex Y from the heightmap and
 * biome-based vertex colours.
 *
 * Vertex `j * res + i` maps to world (X = -8 + i*16/(res-1), Z = -8 + j*16/(res-1)),
 * matching the column-i / row-j convention used in buildBedMap.
 */
function buildTerrainMesh(res: number, worldSize: number, bed: Float32Array): THREE.Mesh {
  const geo = new THREE.PlaneGeometry(worldSize, worldSize, res - 1, res - 1);
  geo.rotateX(-Math.PI / 2);

  const pos = geo.attributes['position'] as THREE.BufferAttribute;
  for (let v = 0; v < pos.count; v++) {
    pos.setY(v, bed[v] ?? 0);
  }
  pos.needsUpdate = true;
  geo.computeVertexNormals();

  // Biome vertex colours by elevation
  const colors = new Float32Array(pos.count * 3);
  for (let v = 0; v < pos.count; v++) {
    const y = bed[v] ?? 0;
    let r: number, g: number, b: number;
    if (y < 0.35) {
      r = 0.56; g = 0.50; b = 0.35; // sandy basin
    } else if (y < 1.4) {
      r = 0.30; g = 0.52; b = 0.24; // grass / meadow
    } else if (y < 3.2) {
      r = 0.45; g = 0.40; b = 0.35; // rocky slope
    } else if (y < 5.0) {
      r = 0.33; g = 0.31; b = 0.30; // dark cliff
    } else {
      r = 0.88; g = 0.91; b = 0.94; // snow cap
    }
    colors[v * 3 + 0] = r;
    colors[v * 3 + 1] = g;
    colors[v * 3 + 2] = b;
  }
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

  const mat = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 0.88,
    metalness: 0.0,
  });
  return new THREE.Mesh(geo, mat);
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

const demo: Demo = {
  id: '10-mountain-terrain',
  label: 'J. Mountain Terrain',
  description:
    'A procedural mountain range with four snow-capped peaks. Springs feed a central lake; water carves valleys and flows over ridgelines. Floating debris rides the currents.',

  setup(ctx) {
    const g = ctx.solver.grid;
    const W = g.width;
    const H = g.height;

    // ── 1. Bake terrain heightmap into the SWE bed ────────────────────
    const bedMap = buildBedMap(W, H);
    ctx.rasterizer.bakeTerrainHeightmap(bedMap);

    // ── 2. Seed water: fill basin up to the lake surface level ────────
    const waterInit = new Float32Array(W * H);
    for (let j = 0; j < H; j++) {
      for (let i = 0; i < W; i++) {
        const bed = bedMap[j * W + i]!;
        waterInit[j * W + i] = Math.max(0, LAKE_SURFACE - bed);
      }
    }
    ctx.solver.writeWaterFull(waterInit);

    // ── 3. Open boundaries on all four edges (water drains freely) ────
    ctx.solver.writeBoundaryRegionTarget({ x: 0, y: 0, w: W, h: 1 }, BoundaryType.Open, 0);
    ctx.solver.writeBoundaryRegionTarget({ x: 0, y: H - 1, w: W, h: 1 }, BoundaryType.Open, 0);
    ctx.solver.writeBoundaryRegionTarget({ x: 0, y: 0, w: 1, h: H }, BoundaryType.Open, 0);
    ctx.solver.writeBoundaryRegionTarget({ x: W - 1, y: 0, w: 1, h: H }, BoundaryType.Open, 0);

    // ── 4. Mountain springs: Inflow boundary near each peak ───────────
    //
    // BoundaryType.Inflow with target depth acts as a floor: the solver
    // maintains at least `targetDepth` metres of water at these cells,
    // continuously injecting water that then flows downhill.
    const SPRING_RADIUS = 5; // cells
    const SPRING_DEPTH = 0.6; // metres above terrain
    for (const p of PEAKS) {
      const pi = Math.round(p.nx * (W - 1));
      const pj = Math.round(p.nz * (H - 1));
      const x0 = Math.max(0, pi - SPRING_RADIUS);
      const y0 = Math.max(0, pj - SPRING_RADIUS);
      const x1 = Math.min(W - 1, pi + SPRING_RADIUS);
      const y1 = Math.min(H - 1, pj + SPRING_RADIUS);
      ctx.solver.writeBoundaryRegionTarget(
        { x: x0, y: y0, w: x1 - x0 + 1, h: y1 - y0 + 1 },
        BoundaryType.Inflow,
        SPRING_DEPTH,
      );
    }

    // ── 5. Visual terrain mesh (128×128 resolution) ───────────────────
    const MESH_RES = 128;
    const meshBed = buildBedMap(MESH_RES, MESH_RES);
    const terrainMesh = ownByDemo(buildTerrainMesh(MESH_RES, WORLD_SIZE, meshBed));
    terrainMesh.name = 'terrain';
    ctx.scene.add(terrainMesh);

    // ── 6. Rapier heightfield collider (64×64 resolution) ─────────────
    //
    // Lower resolution than the visual mesh to keep Rapier's triangle count
    // manageable while still accurately blocking objects from falling through.
    const RAPIER_RES = 64;
    const rapierHeights = buildRapierHeights(RAPIER_RES, RAPIER_RES);
    const terrainBody = ctx.world.createRigidBody(
      ctx.rapier.RigidBodyDesc.fixed().setTranslation(0, 0, 0),
    );
    ctx.world.createCollider(
      ctx.rapier.ColliderDesc.heightfield(
        RAPIER_RES, // nrows (Z axis)
        RAPIER_RES, // ncols (X axis)
        rapierHeights,
        { x: WORLD_SIZE, y: 1, z: WORLD_SIZE },
      ),
      terrainBody,
    );

    // ── 7. Floating log in the central lake ───────────────────────────
    {
      const hx = 0.50, hy = 0.09, hz = 0.17;
      const mesh = ownByDemo(new THREE.Mesh(
        new THREE.BoxGeometry(hx * 2, hy * 2, hz * 2),
        new THREE.MeshStandardMaterial({ color: 0x7a4a1e, roughness: 0.9 }),
      ));
      mesh.name = 'log';
      mesh.position.set(-0.6, LAKE_SURFACE + 0.05, 0.4);
      ctx.scene.add(mesh);

      const body = ctx.world.createRigidBody(
        ctx.rapier.RigidBodyDesc.dynamic()
          .setTranslation(-0.6, LAKE_SURFACE + 0.05, 0.4)
          .setLinearDamping(1.5)
          .setAngularDamping(4.0),
      );
      ctx.world.createCollider(
        ctx.rapier.ColliderDesc.cuboid(hx, hy, hz).setDensity(500),
        body,
      );
      ctx.spawnCoupledBody({
        name: 'log',
        body,
        halfExtents: [hx, hy, hz],
        mesh,
        waterLevelRef: LAKE_SURFACE,
        bedLevelRef: VALLEY_FLOOR,
      });
    }

    // ── 8. Floating barrel ────────────────────────────────────────────
    {
      const hx = 0.13, hy = 0.16, hz = 0.13;
      const mesh = ownByDemo(new THREE.Mesh(
        new THREE.CylinderGeometry(hx, hx, hy * 2, 12),
        new THREE.MeshStandardMaterial({ color: 0x4a2e10, roughness: 0.85 }),
      ));
      mesh.name = 'barrel';
      mesh.position.set(0.9, LAKE_SURFACE + 0.10, -0.6);
      ctx.scene.add(mesh);

      const body = ctx.world.createRigidBody(
        ctx.rapier.RigidBodyDesc.dynamic()
          .setTranslation(0.9, LAKE_SURFACE + 0.10, -0.6)
          .setLinearDamping(1.5)
          .setAngularDamping(5.0),
      );
      ctx.world.createCollider(
        ctx.rapier.ColliderDesc.cuboid(hx, hy, hz).setDensity(700),
        body,
      );
      ctx.spawnCoupledBody({
        name: 'barrel',
        body,
        halfExtents: [hx, hy, hz],
        mesh,
        waterLevelRef: LAKE_SURFACE,
        bedLevelRef: VALLEY_FLOOR,
      });
    }

    // ── 9. Floating crate ─────────────────────────────────────────────
    {
      const hx = 0.18, hy = 0.18, hz = 0.18;
      const mesh = ownByDemo(new THREE.Mesh(
        new THREE.BoxGeometry(hx * 2, hy * 2, hz * 2),
        new THREE.MeshStandardMaterial({ color: 0xaa7733, roughness: 0.85 }),
      ));
      mesh.name = 'crate';
      mesh.position.set(-0.3, LAKE_SURFACE + 0.10, -0.8);
      ctx.scene.add(mesh);

      const body = ctx.world.createRigidBody(
        ctx.rapier.RigidBodyDesc.dynamic()
          .setTranslation(-0.3, LAKE_SURFACE + 0.10, -0.8)
          .setLinearDamping(1.5)
          .setAngularDamping(4.5),
      );
      ctx.world.createCollider(
        ctx.rapier.ColliderDesc.cuboid(hx, hy, hz).setDensity(600),
        body,
      );
      ctx.spawnCoupledBody({
        name: 'crate',
        body,
        halfExtents: [hx, hy, hz],
        mesh,
        waterLevelRef: LAKE_SURFACE,
        bedLevelRef: VALLEY_FLOOR,
      });
    }

    // ── 10. Camera for landscape overview ─────────────────────────────
    ctx.camera.position.set(20, 16, 20);
    ctx.controls.target.set(0, 1.5, 0);
    ctx.controls.update();

    // ── 11. Water surface ─────────────────────────────────────────────
    const water = new WaterSurface(ctx.solver);
    ctx.scene.add(ownByDemo(water.mesh));
    ctx.scratch.water = water;
    ctx.scratch.t = 0;
  },

  tick(ctx, dt) {
    (ctx.scratch.water as WaterSurface).update();
    ctx.scratch.t = (ctx.scratch.t as number) + dt;
  },
};

export default demo;
