/**
 * Ocean-style water-surface renderer.
 *
 * The plane geometry is displaced on the CPU every few frames by reading
 * the GPU water depth back from the solver. The shading is a custom TSL
 * NodeMaterial modeled on three.js's `WaterMesh`, but driven by the
 * solver-displaced plane's own normals so SWE waves stay visually obvious.
 */
import * as THREE from 'three';
import { NodeMaterial } from 'three/webgpu';
import {
  Fn,
  Discard,
  texture,
  uniform,
  varying,
  attribute,
  positionWorld,
  cameraPosition,
  transformedNormalWorld,
  reflect,
  normalize,
  pmremTexture,
  vec2,
  vec3,
  vec4,
  float,
  max,
  dot,
  pow,
  mix,
  add,
  sub,
  div,
  mul,
  smoothstep,
} from 'three/tsl';
import type { VirtualPipesSolver, SweSolver } from 'isenflow';

/** Accepts either solver — they expose the same readWater/readBed/grid API. */
type AnySolver = VirtualPipesSolver | SweSolver;

const READBACK_ERROR_THRESHOLD = 5;

export class WaterSurface {
  readonly mesh: THREE.Mesh;
  readonly geom: THREE.PlaneGeometry;
  readonly material: NodeMaterial;
  readonly simpleMaterial: THREE.MeshStandardMaterial;
  private _oceanStyle = true;
  private positions: Float32BufferAttribute;
  private depthAttr: Float32BufferAttribute;

  private framesSinceRead = 0;
  private reading = false;
  private consecutiveErrors = 0;
  private lastErrorMessage = '';
  /** True once the first successful readback has populated vertex Y values. */
  hasFreshData = false;
  /** Raw GPU readback from last successful update (interleaved [h, h_prev]). */
  lastWaterData: Float32Array | null = null;
  /** Raw GPU readback from last successful update (interleaved [terrain, total]). */
  lastBedData: Float32Array | null = null;

  // Material uniforms (kept as fields so they can be tweaked from outside).
  private readonly uSunDirection = uniform(new THREE.Vector3(0.7071, 0.7071, 0));
  private readonly uSunColor = uniform(new THREE.Color(0xffffff));
  private readonly uWaterColor = uniform(new THREE.Color(0x001e0f));
  private readonly uAlpha = uniform(0.92);
  private readonly uSize = uniform(2.0);
  private readonly uDetailStrength = uniform(0.35);
  private readonly uHasEnv = uniform(0);
  private readonly uTime = uniform(0);

  private readonly envTexture: THREE.Texture;
  private envNode: ReturnType<typeof pmremTexture> | null = null;
  private _pendingEnv: THREE.Texture | null = null;
  private _lastWiredEnv: THREE.Texture | null = null;

  constructor(private readonly solver: AnySolver) {
    const g = solver.grid;
    this.geom = new THREE.PlaneGeometry(g.width * g.dx, g.height * g.dx, g.width, g.height);
    this.geom.rotateX(-Math.PI / 2);
    this.positions = this.geom.attributes.position as Float32BufferAttribute;
    this.depthAttr = new THREE.Float32BufferAttribute(
      new Float32Array((g.width + 1) * (g.height + 1)), 1,
    ) as Float32BufferAttribute;
    this.geom.setAttribute('waterDepth', this.depthAttr);

    // Tiled animated normal map (vendored locally — no network at runtime).
    const loader = new THREE.TextureLoader();
    const normals = loader.load('textures/waternormals.jpg');
    normals.wrapS = THREE.RepeatWrapping;
    normals.wrapT = THREE.RepeatWrapping;

    // Placeholder env texture until the demo wires the real PMREM target in.
    // Must be a valid DataTexture (not empty) so pmremTexture can build.
    this.envTexture = new THREE.DataTexture(new Uint8Array([128, 128, 255, 255]), 1, 1);
    this.envTexture.needsUpdate = true;

    this.material = this.buildMaterial(normals);
    this.simpleMaterial = new THREE.MeshStandardMaterial({
      color: 0x3870c8,
      transparent: true,
      opacity: 0.85,
      roughness: 0.2,
      metalness: 0.05,
      side: THREE.DoubleSide,
    });
    this.mesh = new THREE.Mesh(this.geom, this.material);
    this.mesh.name = 'water';
    this.mesh.position.set(
      g.origin[0] + (g.width * g.dx) / 2,
      0,
      g.origin[1] + (g.height * g.dx) / 2,
    );
    this.mesh.frustumCulled = false;

    // Pick up the scene's sun direction + environment automatically. Demos
    // don't need to know about these uniforms; they just configure the scene.
    this.mesh.onBeforeRender = (_renderer, scene) => {
      const s = scene as THREE.Scene;
      const sun = s.userData?.sun as THREE.Vector3 | undefined;
      if (sun) this.uSunDirection.value.copy(sun).normalize();
      if (s.environment && s.environment !== this._lastWiredEnv) {
        this._lastWiredEnv = s.environment;
        this.setEnvironment(s.environment);
      }
    };
  }

  /** Replace the env texture used for the reflection lobe. */
  setEnvironment(envTexture: THREE.Texture | null): void {
    // PMREMNode exposes a value setter that resets its internal cache, so
    // the next render will re-derive the convolved cubemap from the new
    // texture. We treat null as "no env" → fall back to a flat tint.
    if (envTexture) {
      this._pendingEnv = envTexture;
      if (this.envNode) {
        (this.envNode as unknown as { value: THREE.Texture }).value = envTexture;
      }
      this.uHasEnv.value = 1;
    } else {
      this._pendingEnv = null;
      this.uHasEnv.value = 0;
    }
  }

  /** Set the sun direction (world space, normalized). */
  setSunDirection(dir: THREE.Vector3): void {
    this.uSunDirection.value.copy(dir).normalize();
  }

  private buildMaterial(waterNormals: THREE.Texture): NodeMaterial {
    const material = new NodeMaterial();
    const normalsTex = texture(waterNormals);

    // Two animated taps at different scales, remapped to [-1, 1].
    // The solver geometry already carries the real wave shape, so two taps
    // provide enough surface detail without the cost of the WaterMesh's four.
    const getNoise = Fn(([uv]: [ReturnType<typeof vec2>]) => {
      const offset = this.uTime;
      const uv0 = add(div(uv, 103), vec2(div(offset, 17), div(offset, 29))).toVar();
      const uv1 = div(uv, 107).sub(vec2(div(offset, -19), div(offset, 31))).toVar();
      const s0 = normalsTex.uv(uv0);
      const s1 = normalsTex.uv(uv1);
      return s0.add(s1).sub(1);
    });

    // Per-vertex water depth drives alpha fade at shoreline edges.
    const vDepth = varying(attribute('waterDepth'), 'vDepth');

    material.fragmentNode = Fn(() => {
      Discard(vDepth.lessThanEqual(0.0));
      const edgeAlpha = smoothstep(float(0.0), float(0.02), vDepth);

      // Mesh-derived world-space normal: this is the solver-shaped surface
      // (waves, dam-break front, etc.), recomputed every readback via
      // PlaneGeometry.computeVertexNormals.
      const baseNormal = normalize(transformedNormalWorld);

      // Tiled animated detail normal sampled by world XZ.
      const noise = getNoise(positionWorld.xz.mul(this.uSize));
      const detailNormal = noise.xzy.mul(vec3(1.5, 1.0, 1.5));
      const surfaceNormal = normalize(baseNormal.add(detailNormal.mul(this.uDetailStrength)));

      const worldToEye = cameraPosition.sub(positionWorld);
      const eyeDirection = normalize(worldToEye);

      // Sun specular + diffuse — straight out of WaterMesh.
      const reflection = normalize(reflect(this.uSunDirection.negate(), surfaceNormal));
      const direction = max(0.0, dot(eyeDirection, reflection));
      const specularLight = pow(direction, 100).mul(this.uSunColor).mul(2.0);
      const diffuseLight = max(dot(this.uSunDirection, surfaceNormal), 0.0)
        .mul(this.uSunColor)
        .mul(0.5);

      // Env-map reflection. pmremTexture wants a world-space direction; we
      // reflect the eye→fragment vector about the perturbed normal. Build
      // the PMREMNode here (it captures reflectDir) and stash it so
      // setEnvironment can swap the source texture later.
      const reflectDir = reflect(eyeDirection.negate(), surfaceNormal);
      const initialEnv = this._pendingEnv ?? this.envTexture;
      const envNode = pmremTexture(initialEnv, reflectDir, float(0));
      this.envNode = envNode;
      const envSample = envNode.mul(this.uHasEnv);

      // Fresnel + scatter blend (Schlick), same shape as WaterMesh.
      const theta = max(dot(eyeDirection, surfaceNormal), 0.0);
      const rf0 = float(0.3);
      const reflectance = mul(pow(float(1.0).sub(theta), 5.0), float(1.0).sub(rf0)).add(rf0);
      const scatter = max(0.0, dot(surfaceNormal, eyeDirection)).mul(this.uWaterColor);
      const albedo = mix(
        this.uSunColor.mul(diffuseLight).mul(0.3).add(scatter),
        envSample.mul(specularLight).add(envSample.mul(0.9)).add(vec3(0.1)),
        reflectance,
      );

      return vec4(albedo, this.uAlpha.mul(edgeAlpha));
    })();

    material.transparent = true;
    material.side = THREE.DoubleSide;
    return material;
  }

  /** Toggle between the ocean NodeMaterial and a simple flat material. */
  setOceanStyle(enabled: boolean): void {
    this._oceanStyle = enabled;
    this.mesh.material = enabled ? this.material : this.simpleMaterial;
  }

  get oceanStyle(): boolean {
    return this._oceanStyle;
  }

  /** Advance the shader animation clock by the given amount (seconds). */
  advanceTime(dt: number): void {
    this.uTime.value += dt * 0.15;
  }

  /** Pull the current water depth from the GPU every ~3 frames. */
  update(): void {
    if (this.reading) return;
    this.framesSinceRead++;
    if (this.framesSinceRead < 3) return;
    this.framesSinceRead = 0;
    this.reading = true;
    Promise.all([this.solver.readWater(), this.solver.readBed()])
      .then(([w, b]) => {
        this.lastWaterData = w;
        this.lastBedData = b;
        const g = this.solver.grid;
        const pos = this.positions;
        const nx = g.width + 1;

        // Optional 1-2-1 spatial smoothing of h (NOT the simulation —
        // this is purely a render-side filter that hides any residual
        // numerical roughness without affecting physics).  Applied
        // only in ocean-style mode; raw mesh exposes the unsmoothed
        // field for debugging.
        const hSmooth = this._oceanStyle ? smoothHField(w, g.width, g.height) : null;
        const sampleH = (ci: number, cj: number): number => {
          const idx2 = cj * g.width + ci;
          if (hSmooth) return hSmooth[idx2] ?? 0;
          return w[idx2 * 2] ?? 0;
        };

        for (let j = 0; j <= g.height; j++) {
          for (let i = 0; i <= g.width; i++) {
            const ci = Math.min(g.width - 1, i);
            const cj = Math.min(g.height - 1, j);
            const idx = cj * g.width + ci;
            const h = sampleH(ci, cj);
            const bedTotal = b[idx * 2 + 1] ?? 0;
            const vi = j * nx + i;
            if (h > 0.001) {
              pos.setY(vi, bedTotal + h);
              this.depthAttr.setX(vi, h);
            } else {
              pos.setY(vi, bedTotal);
              this.depthAttr.setX(vi, 0);
            }
          }
        }
        this.depthAttr.needsUpdate = true;
        pos.needsUpdate = true;
        this.geom.computeVertexNormals();
        // Clamp normals so any residual narrow spike can't catch the
        // sharp specular lobe.  Limits min |normal.y| to 0.34 (~70°
        // max surface slope).  Cheap operation on 384² ≈ 150K vertices.
        clampNormals(this.geom);
        this.consecutiveErrors = 0;
        this.hasFreshData = true;
      })
      .catch((err) => {
        this.consecutiveErrors++;
        this.lastErrorMessage = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
        // eslint-disable-next-line no-console
        console.warn(
          `[WaterSurface] readback failed (${this.consecutiveErrors}× in a row): ${this.lastErrorMessage}`,
        );
        if (this.consecutiveErrors >= READBACK_ERROR_THRESHOLD) {
          this.showFatalBanner();
        }
      })
      .finally(() => {
        this.reading = false;
      });
  }

  /** Last surfaced error message — useful for tests. */
  get readbackError(): { count: number; message: string } {
    return { count: this.consecutiveErrors, message: this.lastErrorMessage };
  }

  private showFatalBanner(): void {
    const el = (typeof document !== 'undefined' && document.getElementById('err')) as HTMLElement | null;
    if (!el) return;
    el.style.display = 'block';
    el.textContent =
      `WaterSurface: ${this.consecutiveErrors} consecutive GPU readback failures.\n` +
      `Last error: ${this.lastErrorMessage}\n` +
      `(WebGPU device may have been lost; check browser console.)`;
  }
}

type Float32BufferAttribute = THREE.BufferAttribute & { array: Float32Array };

/**
 * 1-2-1 separable smoothing on the cell-centered h field.  Input `water`
 * is the (h, h_prev) interleaved readback; output is a fresh Float32Array
 * of length W*H with smoothed h values.  Pure CPU, ~0.5 ms at 384².
 *
 * This is purely a RENDER-side filter — the simulation state is untouched.
 * Hides residual numerical roughness without affecting physics.
 */
function smoothHField(water: Float32Array, W: number, H: number): Float32Array {
  const src = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) src[i] = water[i * 2] ?? 0;

  // X-pass: each cell = 0.25·left + 0.5·self + 0.25·right
  const xPass = new Float32Array(W * H);
  for (let j = 0; j < H; j++) {
    for (let i = 0; i < W; i++) {
      const c = j * W + i;
      const l = i > 0 ? src[c - 1]! : src[c]!;
      const r = i < W - 1 ? src[c + 1]! : src[c]!;
      xPass[c] = 0.25 * l + 0.5 * src[c]! + 0.25 * r;
    }
  }
  // Y-pass on x-passed result
  const yPass = new Float32Array(W * H);
  for (let j = 0; j < H; j++) {
    for (let i = 0; i < W; i++) {
      const c = j * W + i;
      const d = j > 0 ? xPass[c - W]! : xPass[c]!;
      const u = j < H - 1 ? xPass[c + W]! : xPass[c]!;
      yPass[c] = 0.25 * d + 0.5 * xPass[c]! + 0.25 * u;
    }
  }
  return yPass;
}

/**
 * Clamp vertex normals so |n.y| ≥ 0.34 (max surface slope ≈ 70°).
 * Any residual single-cell spike that escapes the solver's positivity
 * preservation still produces a normal that's at-most-mildly tilted, so
 * the high-exponent specular lobe can't paint a bright vertical highlight.
 */
function clampNormals(geom: THREE.PlaneGeometry): void {
  const n = geom.attributes.normal as Float32BufferAttribute | undefined;
  if (!n) return;
  const arr = n.array;
  const MIN_Y = 0.34;
  for (let i = 0; i < arr.length; i += 3) {
    const y = arr[i + 1]!;
    if (y >= MIN_Y) continue;
    if (y <= -MIN_Y) continue;  // back-facing (unlikely for water) — leave alone
    // Preserve the horizontal direction, scale Y up to MIN_Y, renormalize.
    const x = arr[i + 0]!;
    const z = arr[i + 2]!;
    const newY = MIN_Y;
    // Rescale (x, z) so the resulting unit vector has y = MIN_Y
    const horizLen = Math.hypot(x, z);
    const targetHoriz = Math.sqrt(Math.max(0, 1 - newY * newY));
    if (horizLen > 1e-6) {
      const scale = targetHoriz / horizLen;
      arr[i + 0] = x * scale;
      arr[i + 2] = z * scale;
    }
    arr[i + 1] = newY;
  }
  n.needsUpdate = true;
}
