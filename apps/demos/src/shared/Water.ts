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
  positionLocal,
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
  time,
} from 'three/tsl';
import type { VirtualPipesSolver } from 'isenflow';

const READBACK_ERROR_THRESHOLD = 5;

export class WaterSurface {
  readonly mesh: THREE.Mesh;
  readonly geom: THREE.PlaneGeometry;
  readonly material: NodeMaterial;
  private positions: Float32BufferAttribute;

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

  private readonly envTexture: THREE.Texture;
  private envNode: ReturnType<typeof pmremTexture> | null = null;
  private _pendingEnv: THREE.Texture | null = null;
  private _lastWiredEnv: THREE.Texture | null = null;

  constructor(private readonly solver: VirtualPipesSolver) {
    const g = solver.grid;
    this.geom = new THREE.PlaneGeometry(g.width * g.dx, g.height * g.dx, g.width, g.height);
    this.geom.rotateX(-Math.PI / 2);
    this.positions = this.geom.attributes.position as Float32BufferAttribute;

    // Tiled animated normal map (vendored locally — no network at runtime).
    const loader = new THREE.TextureLoader();
    const normals = loader.load('textures/waternormals.jpg');
    normals.wrapS = THREE.RepeatWrapping;
    normals.wrapT = THREE.RepeatWrapping;

    // Placeholder env texture until the demo wires the real PMREM target in.
    // We keep a stable Texture reference so PMREMNode.value swaps cleanly.
    this.envTexture = new THREE.Texture();

    this.material = this.buildMaterial(normals);
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

    // Four animated taps, summed and remapped to [-1, 1] — same trick as WaterMesh.
    const getNoise = Fn(([uv]: [ReturnType<typeof vec2>]) => {
      const offset = time;
      const uv0 = add(div(uv, 103), vec2(div(offset, 17), div(offset, 29))).toVar();
      const uv1 = div(uv, 107).sub(vec2(div(offset, -19), div(offset, 31))).toVar();
      const uv2 = add(div(uv, vec2(8907.0, 9803.0)), vec2(div(offset, 101), div(offset, 97))).toVar();
      const uv3 = sub(div(uv, vec2(1091.0, 1027.0)), vec2(div(offset, 109), div(offset, -113))).toVar();
      const s0 = normalsTex.uv(uv0);
      const s1 = normalsTex.uv(uv1);
      const s2 = normalsTex.uv(uv2);
      const s3 = normalsTex.uv(uv3);
      return s0.add(s1).add(s2).add(s3).mul(0.5).sub(1);
    });

    // Forward local Y to the fragment so we can discard the dry-cell sentinel.
    const vCellY = varying(positionLocal.y, 'vCellY');

    material.fragmentNode = Fn(() => {
      Discard(vCellY.lessThan(-0.5));

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
      const envNode = pmremTexture(initialEnv, reflectDir);
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

      return vec4(albedo, this.uAlpha);
    })();

    material.transparent = true;
    material.side = THREE.DoubleSide;
    return material;
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
        for (let j = 0; j <= g.height; j++) {
          for (let i = 0; i <= g.width; i++) {
            const ci = Math.min(g.width - 1, i);
            const cj = Math.min(g.height - 1, j);
            const idx = (cj * g.width + ci) * 2;
            const h = w[idx] ?? 0;
            const bedTotal = b[idx + 1] ?? 0;
            const y = h > 0.005 ? bedTotal + h : -1000;
            pos.setY(j * nx + i, y);
          }
        }
        pos.needsUpdate = true;
        this.geom.computeVertexNormals();
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
