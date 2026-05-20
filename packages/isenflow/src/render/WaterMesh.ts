/**
 * Plane mesh whose vertex Y is read from the water+bed textures.
 *
 * Uses Three.js's WebGPURenderer via TSL. Imported lazily so non-render
 * code paths don't pull in three.
 */
import type * as THREE from 'three';

export interface WaterMeshOptions {
  width: number;       // cells
  height: number;      // cells
  dx: number;          // cell size, m
  origin: readonly [number, number];
}

/**
 * Build a `THREE.Mesh` whose vertices sample the (bed + water) textures
 * in the vertex shader and displace along Y.
 *
 * The texture binding is left to the caller (see WaterMaterial).
 */
export function createWaterMesh(
  THREEns: typeof THREE,
  opts: WaterMeshOptions,
  material: THREE.Material,
): THREE.Mesh {
  const geom = new THREEns.PlaneGeometry(
    opts.width * opts.dx,
    opts.height * opts.dx,
    opts.width,
    opts.height,
  );
  // PlaneGeometry is XY by default; we want XZ.
  geom.rotateX(-Math.PI / 2);
  const mesh = new THREEns.Mesh(geom, material);
  mesh.position.set(
    opts.origin[0] + (opts.width * opts.dx) / 2,
    0,
    opts.origin[1] + (opts.height * opts.dx) / 2,
  );
  mesh.frustumCulled = false;
  return mesh;
}
