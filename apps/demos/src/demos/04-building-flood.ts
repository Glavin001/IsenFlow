import * as THREE from 'three';
import type { Demo } from '../shared/Scene.js';
import { ownByDemo } from '../shared/Scene.js';
import { WaterSurface } from '../shared/Water.js';

interface Building {
  /** Cell-space centre. */
  cx: number;
  cz: number;
  /** Half-extent in cells (square footprint). */
  halfW: number;
  /** Wall height (m). */
  wallH: number;
  /** Door specification, or `null` for a fully watertight building. */
  door: { side: 'N' | 'S' | 'E' | 'W'; width: number } | null;
  name: string;
}

const demo: Demo = {
  id: '04-building-flood',
  label: 'D. Building Flood',
  description:
    'Walled village. A breach in the south perimeter floods the plaza; water rushes around three buildings, finds doorways, and pools inside. Border walls hold the floodwater in.',
  setup(ctx) {
    const g = ctx.solver.grid;
    const W = g.width;
    const H = g.height;

    const wallMat = new THREE.MeshStandardMaterial({ color: 0xc0a070 });
    const borderMat = new THREE.MeshStandardMaterial({ color: 0x8a7558, roughness: 1 });

    // === Border walls (perimeter) ============================================
    const BORDER_H = 4.0;
    const BORDER_THICKNESS = 2; // cells
    const fillRow = (h: number) => new Float32Array(W).fill(h);
    const fillCol = (h: number) => new Float32Array(H).fill(h);
    const borderH = BORDER_H;

    // North border
    ctx.solver.writeBedRegion(
      { x: 0, y: 0, w: W, h: BORDER_THICKNESS },
      (() => {
        const a = new Float32Array(W * BORDER_THICKNESS);
        a.fill(borderH);
        return a;
      })(),
    );
    // East border
    ctx.solver.writeBedRegion(
      { x: W - BORDER_THICKNESS, y: 0, w: BORDER_THICKNESS, h: H },
      (() => {
        const a = new Float32Array(BORDER_THICKNESS * H);
        a.fill(borderH);
        return a;
      })(),
    );
    // West border
    ctx.solver.writeBedRegion(
      { x: 0, y: 0, w: BORDER_THICKNESS, h: H },
      (() => {
        const a = new Float32Array(BORDER_THICKNESS * H);
        a.fill(borderH);
        return a;
      })(),
    );

    // South border with a centred breach (~1.5 m wide).
    const breachWidthM = 1.5;
    const breachW = Math.max(4, Math.round(breachWidthM / g.dx));
    const breachI0 = Math.floor(W / 2 - breachW / 2);
    const breachI1 = breachI0 + breachW;
    const south = new Float32Array(W * BORDER_THICKNESS);
    for (let j = 0; j < BORDER_THICKNESS; j++) {
      for (let i = 0; i < W; i++) {
        south[j * W + i] = i >= breachI0 && i < breachI1 ? 0 : borderH;
      }
    }
    ctx.solver.writeBedRegion({ x: 0, y: H - BORDER_THICKNESS, w: W, h: BORDER_THICKNESS }, south);

    // Visualise borders as a single thin frame.
    const worldW = W * g.dx;
    const worldH = H * g.dx;
    const borderThick = BORDER_THICKNESS * g.dx;
    const mkBorder = (sx: number, sz: number, lx: number, lz: number, name: string) => {
      const m = ownByDemo(new THREE.Mesh(new THREE.BoxGeometry(lx, BORDER_H, lz), borderMat));
      m.name = name;
      m.position.set(sx, BORDER_H / 2, sz);
      ctx.scene.add(m);
    };
    mkBorder(0, g.origin[1] + borderThick / 2, worldW, borderThick, 'borderN');
    mkBorder(g.origin[0] + worldW - borderThick / 2, 0, borderThick, worldH, 'borderE');
    mkBorder(g.origin[0] + borderThick / 2, 0, borderThick, worldH, 'borderW');
    // South: two segments straddling the breach.
    const breachStartX = g.origin[0] + breachI0 * g.dx;
    const breachEndX = g.origin[0] + breachI1 * g.dx;
    const sSegLeftLen = breachStartX - g.origin[0];
    const sSegRightLen = g.origin[0] + worldW - breachEndX;
    if (sSegLeftLen > 0) {
      mkBorder(
        g.origin[0] + sSegLeftLen / 2,
        g.origin[1] + worldH - borderThick / 2,
        sSegLeftLen,
        borderThick,
        'borderSl',
      );
    }
    if (sSegRightLen > 0) {
      mkBorder(
        breachEndX + sSegRightLen / 2,
        g.origin[1] + worldH - borderThick / 2,
        sSegRightLen,
        borderThick,
        'borderSr',
      );
    }

    // === Buildings ============================================================
    // Three buildings of varying size, offset to channel water around them.
    const buildings: Building[] = [
      // Big house: centred-ish, north of breach. ~1.2 m wide door faces
      // south (toward the flood). Door width is in cells (≈ width·dx m).
      { cx: Math.round(W * 0.55), cz: Math.round(H * 0.45), halfW: Math.round(W * 0.10), wallH: 1.5, door: { side: 'S', width: 28 }, name: 'houseBig' },
      // Smaller house (west of centre) — fully sealed; water flows around it.
      { cx: Math.round(W * 0.28), cz: Math.round(H * 0.55), halfW: Math.round(W * 0.07), wallH: 1.3, door: null, name: 'houseW' },
      // Tiny shed (north-east corner) — fully sealed.
      { cx: Math.round(W * 0.78), cz: Math.round(H * 0.25), halfW: Math.round(W * 0.05), wallH: 1.0, door: null, name: 'shedNE' },
    ];

    const placeBuilding = (b: Building) => {
      const { cx, cz, halfW, wallH, door } = b;
      const span = 2 * halfW + 1;
      const writeWall = (x: number, y: number, w: number, h: number, gapStart?: number, gapEnd?: number) => {
        const buf = new Float32Array(w * h);
        for (let j = 0; j < h; j++) {
          for (let i = 0; i < w; i++) {
            const isGap = gapStart !== undefined && gapEnd !== undefined && i >= gapStart && i < gapEnd;
            buf[j * w + i] = isGap ? 0 : wallH;
          }
        }
        ctx.solver.writeBedRegion({ x, y, w, h }, buf);
      };
      const doorMid = Math.floor(span / 2);
      const doorHalf = door ? Math.max(1, Math.floor(door.width / 2)) : 0;
      const doorOnSide = (side: Building['door'] extends infer _ ? 'N' | 'S' | 'E' | 'W' : never): boolean =>
        door !== null && door.side === side;
      // North wall (top side, smaller j)
      writeWall(
        cx - halfW,
        cz - halfW,
        span,
        1,
        doorOnSide('N') ? doorMid - doorHalf : undefined,
        doorOnSide('N') ? doorMid + doorHalf : undefined,
      );
      // South wall
      writeWall(
        cx - halfW,
        cz + halfW,
        span,
        1,
        doorOnSide('S') ? doorMid - doorHalf : undefined,
        doorOnSide('S') ? doorMid + doorHalf : undefined,
      );
      // West wall (smaller i)
      const westBuf = new Float32Array(span);
      for (let j = 0; j < span; j++) {
        const isGap = doorOnSide('W') && j >= doorMid - doorHalf && j < doorMid + doorHalf;
        westBuf[j] = isGap ? 0 : wallH;
      }
      ctx.solver.writeBedRegion({ x: cx - halfW, y: cz - halfW, w: 1, h: span }, westBuf);
      // East wall
      const eastBuf = new Float32Array(span);
      for (let j = 0; j < span; j++) {
        const isGap = doorOnSide('E') && j >= doorMid - doorHalf && j < doorMid + doorHalf;
        eastBuf[j] = isGap ? 0 : wallH;
      }
      ctx.solver.writeBedRegion({ x: cx + halfW, y: cz - halfW, w: 1, h: span }, eastBuf);

      // Visual: render each wall as oriented boxes; if this wall is the
      // door side, render it as TWO segments straddling the gap so the
      // door is visible.
      const worldX = g.origin[0] + cx * g.dx;
      const worldZ = g.origin[1] + cz * g.dx;
      const sideLen = span * g.dx;
      const doorLen = doorHalf * 2 * g.dx;
      const mk = (
        x: number,
        z: number,
        lx: number,
        lz: number,
        name: string,
      ) => {
        const m = ownByDemo(new THREE.Mesh(new THREE.BoxGeometry(lx, wallH, lz), wallMat));
        m.name = name;
        m.position.set(x, wallH / 2, z);
        ctx.scene.add(m);
      };
      const splitH = (centerX: number, centerZ: number, axis: 'x' | 'z', baseName: string) => {
        // axis = 'x' → wall runs along world X (N/S walls); 'z' → along world Z (E/W).
        const segLen = (sideLen - doorLen) / 2;
        if (segLen <= 0) {
          mk(centerX, centerZ, axis === 'x' ? sideLen : g.dx, axis === 'x' ? g.dx : sideLen, baseName);
          return;
        }
        if (axis === 'x') {
          mk(centerX - sideLen / 2 + segLen / 2, centerZ, segLen, g.dx, `${baseName}-l`);
          mk(centerX + sideLen / 2 - segLen / 2, centerZ, segLen, g.dx, `${baseName}-r`);
        } else {
          mk(centerX, centerZ - sideLen / 2 + segLen / 2, g.dx, segLen, `${baseName}-l`);
          mk(centerX, centerZ + sideLen / 2 - segLen / 2, g.dx, segLen, `${baseName}-r`);
        }
      };
      const wallN_C = [worldX, worldZ - halfW * g.dx] as const;
      const wallS_C = [worldX, worldZ + halfW * g.dx] as const;
      const wallW_C = [worldX - halfW * g.dx, worldZ] as const;
      const wallE_C = [worldX + halfW * g.dx, worldZ] as const;
      if (doorOnSide('N')) splitH(wallN_C[0], wallN_C[1], 'x', `${b.name}-wallN`);
      else mk(wallN_C[0], wallN_C[1], sideLen, g.dx, `${b.name}-wallN`);
      if (doorOnSide('S')) splitH(wallS_C[0], wallS_C[1], 'x', `${b.name}-wallS`);
      else mk(wallS_C[0], wallS_C[1], sideLen, g.dx, `${b.name}-wallS`);
      if (doorOnSide('W')) splitH(wallW_C[0], wallW_C[1], 'z', `${b.name}-wallW`);
      else mk(wallW_C[0], wallW_C[1], g.dx, sideLen, `${b.name}-wallW`);
      if (doorOnSide('E')) splitH(wallE_C[0], wallE_C[1], 'z', `${b.name}-wallE`);
      else mk(wallE_C[0], wallE_C[1], g.dx, sideLen, `${b.name}-wallE`);
    };

    for (const b of buildings) placeBuilding(b);

    // === Water surface ========================================================
    const water = new WaterSurface(ctx.solver);
    ctx.scene.add(ownByDemo(water.mesh));
    ctx.scratch.water = water;
    ctx.scratch.t = 0;
    ctx.scratch.tideH = 0;
    ctx.scratch.breachI0 = breachI0;
    ctx.scratch.breachI1 = breachI1;
    // Big-house door for tests.
    ctx.scratch.cx = buildings[0]!.cx;
    ctx.scratch.cz = buildings[0]!.cz;
    ctx.scratch.halfW = buildings[0]!.halfW;
    ctx.scratch.doorI0 = buildings[0]!.cx - 2;
    ctx.scratch.doorI1 = buildings[0]!.cx + 2;
    ctx.scratch.doorJ = buildings[0]!.cz + buildings[0]!.halfW;
  },
  tick(ctx, dt) {
    (ctx.scratch.water as WaterSurface).update();
    ctx.scratch.t = (ctx.scratch.t as number) + dt;
    const g = ctx.solver.grid;
    // Outside the south breach we hold a rising "tide" head; this drives an
    // unbounded flood through the gap into the walled town.
    const tideH = Math.min(2.5, 0.4 * (ctx.scratch.t as number));
    ctx.scratch.tideH = tideH;
    const breachI0 = ctx.scratch.breachI0 as number;
    const breachI1 = ctx.scratch.breachI1 as number;
    const breachW = breachI1 - breachI0;
    if (breachW > 0) {
      const flood = new Float32Array(breachW).fill(tideH);
      ctx.solver.writeWaterRegion({ x: breachI0, y: g.height - 1, w: breachW, h: 1 }, flood);
      ctx.solver.writeWaterRegion({ x: breachI0, y: g.height - 2, w: breachW, h: 1 }, flood);
    }
  },
};

export default demo;
