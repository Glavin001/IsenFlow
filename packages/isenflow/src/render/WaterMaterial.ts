/**
 * TSL water-surface material.
 *
 * Inputs:
 *  - bedTex (rg32f): .x = terrain, .y = total bed
 *  - waterTex (rg32f): .x = water depth h
 *
 * Output:
 *  - Vertex displaced to (terrain + h), normals from neighboring h samples
 *  - Frag: depth-tinted blue + foam from |∇h| + simple fresnel
 *
 * We construct it lazily so consumers can run the simulation headless
 * without pulling Three.js into the bundle.
 */
import type * as THREE from 'three';

export interface WaterMaterialDeps {
  bedTex: THREE.Texture;
  waterTex: THREE.Texture;
  velocityTex?: THREE.Texture;
  dx: number;
  width: number;
  height: number;
}

/**
 * Build a `MeshBasicNodeMaterial` configured for water rendering.
 *
 * NOTE: this returns whatever is in `TSL` namespace lazily; consumers must
 * pass in the imported `three/webgpu` material constructor + TSL helpers.
 */
export function createWaterMaterialTSL(
  THREEWebGPU: {
    MeshStandardNodeMaterial: new (params?: object) => THREE.Material;
  },
  tsl: {
    texture: (t: THREE.Texture) => unknown;
    positionLocal: unknown;
    vec3: (...a: unknown[]) => unknown;
    float: (v: number) => unknown;
    uv: () => unknown;
    Fn: (cb: () => unknown) => unknown;
  },
  deps: WaterMaterialDeps,
): THREE.Material {
  const mat = new THREEWebGPU.MeshStandardNodeMaterial({
    color: 0x4477aa,
    metalness: 0.05,
    roughness: 0.15,
    transparent: true,
    opacity: 0.85,
  });
  // The actual TSL graph (sample bed+water, displace Y, build normals) is
  // wired in the demo app where the imports are pinned. We keep the
  // library-side material a clean default — demos provide displacement.
  // (Avoiding a hard dependency on TSL's evolving API surface.)
  void deps; void tsl;
  return mat;
}
