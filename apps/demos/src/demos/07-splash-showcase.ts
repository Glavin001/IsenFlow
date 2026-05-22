import * as THREE from 'three';
import type { Demo } from '../shared/Scene.js';
import { ownByDemo } from '../shared/Scene.js';
import { WaterSurface } from '../shared/Water.js';

/**
 * Wave Tank — multiple oscillating sources, a curved breakwater, and a
 * submerged island create rich interference, diffraction, and refraction
 * patterns in a shallow basin.
 */
const demo: Demo = {
  id: '07-splash',
  label: 'G. Wave Tank',
  description:
    'Three pulsing wave sources, a curved breakwater, and a submerged island. Watch interference fringes, diffraction through the gap, and refraction over the shallows.',
  setup(ctx) {
    const g = ctx.solver.grid;
    const W = g.width;
    const H = g.height;
    const dx = g.dx;

    // Basin depth — 0.8 m gives c ≈ 2.8 m/s
    const baseDepth = 0.8;
    ctx.solver.writeWaterFull(new Float32Array(W * H).fill(baseDepth));

    // ---------- Perimeter walls (closed, reflective) ----------
    const WALL_H = 1.5;
    const T = 2; // cells thick
    const fillBed = (x: number, y: number, w: number, h: number, elev: number) => {
      ctx.solver.writeBedRegion({ x, y, w, h }, new Float32Array(w * h).fill(elev));
      ctx.solver.writeBoundaryRegion({ x, y, w, h }, 1); // Closed
    };
    fillBed(0, 0, W, T, WALL_H);
    fillBed(0, H - T, W, T, WALL_H);
    fillBed(0, 0, T, H, WALL_H);
    fillBed(W - T, 0, T, H, WALL_H);

    // Visual perimeter
    const wallMat = new THREE.MeshStandardMaterial({ color: 0x607080, roughness: 0.8 });
    const worldW = W * dx;
    const worldH = H * dx;
    const bt = T * dx;
    const mkWall = (x: number, z: number, lx: number, lz: number, name: string) => {
      const m = ownByDemo(new THREE.Mesh(new THREE.BoxGeometry(lx, WALL_H, lz), wallMat));
      m.name = name;
      m.position.set(x, WALL_H / 2, z);
      ctx.scene.add(m);
    };
    mkWall(0, g.origin[1] + bt / 2, worldW, bt, 'wallN');
    mkWall(0, g.origin[1] + worldH - bt / 2, worldW, bt, 'wallS');
    mkWall(g.origin[0] + bt / 2, 0, bt, worldH, 'wallW');
    mkWall(g.origin[0] + worldW - bt / 2, 0, bt, worldH, 'wallE');

    // ---------- Curved breakwater with gap (diffraction barrier) ----------
    // Arc of cells at ~60% of width, spanning the middle ~60% of height,
    // with a gap at centre for waves to squeeze through.
    const breakwaterMat = new THREE.MeshStandardMaterial({ color: 0x8a7558, roughness: 1 });
    const bwI0 = Math.round(W * 0.58);
    const gapHalf = Math.max(4, Math.round(0.8 / dx)); // ~0.8 m gap
    const bwJCenter = Math.round(H * 0.50);
    const bwExtent = Math.round(H * 0.35); // half-span in cells
    const bwH = 1.2;
    const bwThick = Math.max(2, Math.round(0.15 / dx));

    for (let dj = -bwExtent; dj <= bwExtent; dj++) {
      // Skip the gap
      if (Math.abs(dj) < gapHalf) continue;
      // Gentle arc: offset i by a parabola so the wall curves toward the sources
      const frac = dj / bwExtent;
      const arc = Math.round(bwExtent * 0.15 * (1 - frac * frac));
      const ci = bwI0 - arc;
      const cj = bwJCenter + dj;
      if (cj < T || cj >= H - T || ci < T || ci + bwThick >= W - T) continue;
      fillBed(ci, cj, bwThick, 1, bwH);
    }

    // Visual breakwater — two arc segments (above and below gap)
    for (const sign of [-1, 1]) {
      const segLen = (bwExtent - gapHalf) * dx;
      if (segLen <= 0) continue;
      const segCenterJ = bwJCenter + sign * (gapHalf + (bwExtent - gapHalf) / 2);
      const m = ownByDemo(new THREE.Mesh(
        new THREE.BoxGeometry(bwThick * dx, bwH, segLen),
        breakwaterMat,
      ));
      m.name = `breakwater${sign > 0 ? 'S' : 'N'}`;
      m.position.set(
        g.origin[0] + bwI0 * dx,
        bwH / 2,
        g.origin[1] + segCenterJ * dx,
      );
      ctx.scene.add(m);
    }

    // ---------- Submerged island (refraction lens) ----------
    // Circular shallow region on the east side — raises bed to reduce local
    // depth, slowing waves and bending them (refraction).
    const islandCI = Math.round(W * 0.75);
    const islandCJ = Math.round(H * 0.45);
    const islandR = Math.max(12, Math.round(2.0 / dx)); // ~2.0 m radius (bigger)
    const islandPeak = 0.70; // bed rises to 0.70 m → water only 0.10 m deep on top

    const islandBuf = new Float32Array((2 * islandR + 1) * (2 * islandR + 1));
    for (let dj = 0; dj <= 2 * islandR; dj++) {
      for (let di = 0; di <= 2 * islandR; di++) {
        const dr = Math.hypot(di - islandR, dj - islandR) / islandR;
        // Smooth cosine-bell profile
        islandBuf[dj * (2 * islandR + 1) + di] =
          dr < 1.0 ? islandPeak * 0.5 * (1 + Math.cos(Math.PI * dr)) : 0;
      }
    }
    const ix0 = Math.max(T, islandCI - islandR);
    const iy0 = Math.max(T, islandCJ - islandR);
    const ix1 = Math.min(W - T, islandCI + islandR + 1);
    const iy1 = Math.min(H - T, islandCJ + islandR + 1);
    // Write only the in-bounds portion
    const cropW = ix1 - ix0;
    const cropH = iy1 - iy0;
    const cropBuf = new Float32Array(cropW * cropH);
    for (let j = 0; j < cropH; j++) {
      for (let i = 0; i < cropW; i++) {
        const si = (ix0 - (islandCI - islandR)) + i;
        const sj = (iy0 - (islandCJ - islandR)) + j;
        cropBuf[j * cropW + i] = islandBuf[sj * (2 * islandR + 1) + si] ?? 0;
      }
    }
    ctx.solver.writeBedRegion({ x: ix0, y: iy0, w: cropW, h: cropH }, cropBuf);

    // Visual island — sandy shoal, visible through the water
    const islandMat = new THREE.MeshStandardMaterial({
      color: 0xc4a86a,
      roughness: 1.0,
      transparent: true,
      opacity: 0.75,
    });
    const islandMesh = ownByDemo(new THREE.Mesh(
      new THREE.CylinderGeometry(islandR * dx * 0.8, islandR * dx, islandPeak, 32),
      islandMat,
    ));
    islandMesh.name = 'island';
    islandMesh.position.set(
      g.origin[0] + islandCI * dx,
      islandPeak / 2,
      g.origin[1] + islandCJ * dx,
    );
    ctx.scene.add(islandMesh);

    // ---------- Water surface ----------
    const water = new WaterSurface(ctx.solver);
    ctx.scene.add(ownByDemo(water.mesh));

    // ---------- Wave sources ----------
    // Three sources on the west side in a triangular arrangement:
    //   • Two outer sources pulse in phase  → constructive midline
    //   • One centre source at a different freq → beating / moiré
    const sources = [
      { i: Math.round(W * 0.18), j: Math.round(H * 0.30), freq: 1.5, amp: 0.25, phase: 0 },
      { i: Math.round(W * 0.18), j: Math.round(H * 0.70), freq: 2.2, amp: 0.18, phase: 0 },
      { i: Math.round(W * 0.12), j: Math.round(H * 0.50), freq: 1.2, amp: 0.35, phase: 0 },
    ];
    const srcRadii = [
      Math.max(2, Math.round(0.30 / dx)),  // top — medium
      Math.max(2, Math.round(0.25 / dx)),  // bottom — small
      Math.max(2, Math.round(0.45 / dx)),  // center — large (big buoy, big waves)
    ];

    // Visual markers — sized proportional to wave amplitude
    const markerMat = new THREE.MeshStandardMaterial({ color: 0xff4444, roughness: 0.5 });
    const markerRadii = [0.25, 0.18, 0.35]; // bigger buoy = bigger waves
    const markers: THREE.Mesh[] = [];
    for (let si = 0; si < sources.length; si++) {
      const s = sources[si]!;
      const r = markerRadii[si] ?? 0.15;
      const marker = ownByDemo(new THREE.Mesh(
        new THREE.SphereGeometry(r, 16, 12),
        markerMat,
      ));
      marker.position.set(
        g.origin[0] + s.i * dx,
        baseDepth,
        g.origin[1] + s.j * dx,
      );
      marker.name = 'waveSource';
      ctx.scene.add(marker);
      markers.push(marker);
    }

    ctx.scratch.water = water;
    ctx.scratch.t = 0;
    ctx.scratch.baseDepth = baseDepth;
    ctx.scratch.sources = sources;
    ctx.scratch.srcRadii = srcRadii;
    ctx.scratch.markers = markers;
  },

  tick(ctx, dt) {
    (ctx.scratch.water as WaterSurface).update();
    ctx.scratch.t = (ctx.scratch.t as number) + dt;
    const t = ctx.scratch.t as number;

    const g = ctx.solver.grid;
    const baseDepth = ctx.scratch.baseDepth as number;
    const sources = ctx.scratch.sources as { i: number; j: number; freq: number; amp: number; phase: number }[];
    const markers = ctx.scratch.markers as THREE.Mesh[];
    const srcRadii = ctx.scratch.srcRadii as number[];
    const markerRadii = [0.25, 0.18, 0.35];

    for (let si = 0; si < sources.length; si++) {
      const s = sources[si]!;
      const R = srcRadii[si]!;
      const diam = 2 * R + 1;
      const wave = Math.sin(2 * Math.PI * s.freq * t + s.phase);

      // Oscillate bed elevation under the buoy — like a piston pushing
      // water up/down. The solver naturally propagates the displaced water.
      // Peak bed rise = amp * 2 (displaces more water when up).
      const bedPeak = s.amp * 2 * Math.max(0, wave); // only push up, not below floor
      const buf = new Float32Array(diam * diam);
      for (let dj = 0; dj < diam; dj++) {
        for (let di = 0; di < diam; di++) {
          const dr = Math.hypot(di - R, dj - R) / R;
          // Smooth cosine-bell profile so edges blend naturally
          buf[dj * diam + di] = dr < 1.0
            ? bedPeak * 0.5 * (1 + Math.cos(Math.PI * dr))
            : 0;
        }
      }
      ctx.solver.writeBedRegion(
        { x: s.i - R, y: s.j - R, w: diam, h: diam },
        buf,
      );

      // Buoy floats at water surface: bed + water depth at center ≈ bedPeak + baseDepth
      const marker = markers[si];
      if (marker) {
        const buoyR = markerRadii[si] ?? 0.15;
        const surfaceY = bedPeak + baseDepth;
        marker.position.y = surfaceY - buoyR * 0.3; // sit partially submerged
      }
    }
  },
};

export default demo;
