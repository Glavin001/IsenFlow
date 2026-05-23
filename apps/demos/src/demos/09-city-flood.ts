import * as THREE from 'three';
import type { Demo } from '../shared/Scene.js';
import { ownByDemo } from '../shared/Scene.js';
import { WaterSurface } from '../shared/Water.js';
import { BoundaryType } from 'isenflow';

/* ------------------------------------------------------------------ */
/*  Building descriptor                                                */
/* ------------------------------------------------------------------ */

interface Building {
  /** Cell-space centre X. */
  cx: number;
  /** Cell-space centre Z. */
  cz: number;
  /** Half-extent in cells (square footprint). */
  halfW: number;
  /** Half-extent in cells along Z (rectangular if different from halfW). */
  halfH?: number;
  /** Wall / obstacle height (m). */
  wallH: number;
  /** Optional opening: side, width (cells), sill height (m, 0 = ground-level door). */
  opening: { side: 'N' | 'S' | 'E' | 'W'; width: number; sillH: number } | null;
  color: number;
  name: string;
}

/* ------------------------------------------------------------------ */
/*  City layout generator                                              */
/* ------------------------------------------------------------------ */

/** Deterministic pseudo-random (splitmix32) seeded from an integer. */
function prng(seed: number) {
  let s = seed | 0;
  return () => {
    s |= 0; s = (s + 0x9e3779b9) | 0;
    let t = s ^ (s >>> 16); t = Math.imul(t, 0x21f0aaad);
    t = t ^ (t >>> 15); t = Math.imul(t, 0x735a2d97);
    t = t ^ (t >>> 15);
    return (t >>> 0) / 4294967296;
  };
}

const SIDES: Array<'N' | 'S' | 'E' | 'W'> = ['N', 'S', 'E', 'W'];

function generateCity(W: number, H: number): Building[] {
  const buildings: Building[] = [];
  let id = 0;
  const rng = prng(42);
  const margin = 6; // keep buildings away from grid edges

  // ================================================================
  // 1. DOWNTOWN CORE — 6×5 grid of tall towers (up to 30)
  // ================================================================
  const dtSpacing = Math.round(W * 0.065);
  const dtStartX = Math.round(W * 0.12);
  const dtStartZ = Math.round(H * 0.08);
  const dtCols = 6;
  const dtRows = 5;

  for (let row = 0; row < dtRows; row++) {
    for (let col = 0; col < dtCols; col++) {
      const cx = dtStartX + col * dtSpacing;
      const cz = dtStartZ + row * dtSpacing;
      // skip every other cell for street gaps
      if ((row + col) % 3 === 2) continue;
      if (cx + 16 >= W - margin || cz + 16 >= H - margin) continue;

      const r = rng();
      const wallH = 1.6 + r * 2.4; // 1.6 – 4.0 m
      const halfW = Math.round(W * (0.018 + r * 0.018));

      let opening: Building['opening'] = null;
      const v = (row * dtCols + col) % 7;
      if (v === 0) opening = { side: 'S', width: Math.max(3, Math.round(halfW * 0.7)), sillH: 0 };
      else if (v === 1) opening = { side: 'E', width: Math.max(3, Math.round(halfW * 0.5)), sillH: 0.6 + rng() * 0.6 };
      else if (v === 2) opening = { side: 'W', width: Math.max(4, Math.round(halfW * 0.9)), sillH: 0 };
      else if (v === 3) opening = { side: 'N', width: Math.max(3, Math.round(halfW * 0.6)), sillH: 1.0 };
      // v 4-6 => sealed

      const lum = Math.round(0x66 + rng() * 0x33);
      const color = (lum << 16) | ((lum - 0x10) << 8) | (lum + 0x10);
      buildings.push({ cx, cz, halfW, wallH, opening, color, name: `tower${id++}` });
    }
  }

  // ================================================================
  // 2. MIDTOWN — mixed-use, medium height, slightly irregular
  // ================================================================
  const mtStartZ = Math.round(H * 0.42);
  const mtRows = 3;
  const mtCols = 8;
  const mtSpacingX = Math.round(W * 0.055);
  const mtSpacingZ = Math.round(H * 0.07);

  for (let row = 0; row < mtRows; row++) {
    for (let col = 0; col < mtCols; col++) {
      const jitter = Math.round((rng() - 0.5) * 6);
      const cx = Math.round(W * 0.08) + col * mtSpacingX + jitter;
      const cz = mtStartZ + row * mtSpacingZ + Math.round((rng() - 0.5) * 4);
      if (cx < margin || cx + 12 >= W - margin || cz < margin || cz + 12 >= H - margin) continue;

      const wallH = 1.2 + rng() * 1.6;
      const halfW = Math.round(W * (0.015 + rng() * 0.012));
      const halfH = Math.round(H * (0.012 + rng() * 0.015));

      let opening: Building['opening'] = null;
      if (rng() < 0.5) {
        const side = SIDES[Math.floor(rng() * 4)]!;
        const sillH = rng() < 0.4 ? 0 : 0.3 + rng() * 0.8;
        const doorW = Math.max(3, Math.round((side === 'N' || side === 'S' ? halfW : halfH) * (0.4 + rng() * 0.4)));
        opening = { side, width: doorW, sillH };
      }

      buildings.push({
        cx, cz, halfW, halfH, wallH, opening,
        color: 0x99887a + Math.round(rng() * 0x202020),
        name: `midtown${id++}`,
      });
    }
  }

  // ================================================================
  // 3. RESIDENTIAL NEIGHBORHOODS — two strips of small houses
  // ================================================================
  for (let strip = 0; strip < 2; strip++) {
    const baseZ = Math.round(H * (0.64 + strip * 0.12));
    const cols = 10;
    for (let col = 0; col < cols; col++) {
      const cx = Math.round(W * 0.06) + col * Math.round(W * 0.09) + Math.round((rng() - 0.5) * 4);
      const cz = baseZ + (col % 2 === 0 ? 0 : Math.round(H * 0.03)) + Math.round((rng() - 0.5) * 3);
      if (cx < margin || cx + 10 >= W - margin || cz < margin || cz + 10 >= H - margin) continue;

      const wallH = 0.8 + rng() * 0.8;
      const halfW = Math.round(W * (0.012 + rng() * 0.010));

      let opening: Building['opening'] = null;
      const v = col % 4;
      if (v === 0) opening = { side: 'N', width: Math.max(2, Math.round(halfW * 0.6)), sillH: 0 };
      else if (v === 1) opening = { side: 'S', width: Math.max(2, Math.round(halfW * 0.5)), sillH: 0.3 };
      else if (v === 2) opening = { side: 'W', width: Math.max(2, Math.round(halfW * 0.7)), sillH: 0 };
      // v === 3 => sealed

      buildings.push({
        cx, cz, halfW, wallH, opening,
        color: 0xc0a070 + Math.round(rng() * 0x202010),
        name: `house${id++}`,
      });
    }
  }

  // ================================================================
  // 4. INDUSTRIAL DISTRICT — wide warehouses, east side
  // ================================================================
  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 2; col++) {
      const cx = Math.round(W * (0.76 + col * 0.10));
      const cz = Math.round(H * 0.10) + row * Math.round(H * 0.18);
      if (cx + 24 >= W - margin || cz + 18 >= H - margin) continue;

      const halfW = Math.round(W * (0.030 + rng() * 0.025));
      const halfH = Math.round(H * (0.020 + rng() * 0.018));

      let opening: Building['opening'] = null;
      if (rng() < 0.6) {
        opening = { side: 'W', width: Math.max(4, Math.round(halfH * 0.5)), sillH: 0 };
      }

      buildings.push({
        cx, cz, halfW, halfH,
        wallH: 1.0 + rng() * 0.8,
        opening,
        color: 0x556655 + Math.round(rng() * 0x111111),
        name: `warehouse${id++}`,
      });
    }
  }

  // ================================================================
  // 5. CIVIC BUILDINGS — large footprint, prominent
  // ================================================================
  // City hall (centre-north, tall, door facing south)
  buildings.push({
    cx: Math.round(W * 0.38), cz: Math.round(H * 0.06),
    halfW: Math.round(W * 0.04), halfH: Math.round(H * 0.03),
    wallH: 3.8,
    opening: { side: 'S', width: Math.max(6, Math.round(W * 0.025)), sillH: 0 },
    color: 0xbbbbcc,
    name: `cityHall${id++}`,
  });

  // Hospital (centre-east, medium, wide west entrance)
  buildings.push({
    cx: Math.round(W * 0.68), cz: Math.round(H * 0.38),
    halfW: Math.round(W * 0.035), halfH: Math.round(H * 0.05),
    wallH: 2.8,
    opening: { side: 'W', width: Math.max(6, Math.round(H * 0.03)), sillH: 0 },
    color: 0xccccdd,
    name: `hospital${id++}`,
  });

  // School (south-west, low, north-facing door)
  buildings.push({
    cx: Math.round(W * 0.15), cz: Math.round(H * 0.82),
    halfW: Math.round(W * 0.035), halfH: Math.round(H * 0.025),
    wallH: 1.5,
    opening: { side: 'N', width: Math.max(5, Math.round(W * 0.02)), sillH: 0 },
    color: 0xddccaa,
    name: `school${id++}`,
  });

  // ================================================================
  // 6. PARK PAVILIONS — open structures
  // ================================================================
  const pavilionSpots = [
    { x: 0.50, z: 0.55 },
    { x: 0.30, z: 0.75 },
    { x: 0.62, z: 0.80 },
  ];
  for (const spot of pavilionSpots) {
    buildings.push({
      cx: Math.round(W * spot.x), cz: Math.round(H * spot.z),
      halfW: Math.round(W * 0.02), wallH: 1.0,
      opening: { side: SIDES[Math.floor(rng() * 4)]!, width: Math.round(W * 0.025), sillH: 0 },
      color: 0x88aa88,
      name: `pavilion${id++}`,
    });
  }

  // ================================================================
  // 7. PERIMETER WALLS with gaps — channels flood through specific streets
  // ================================================================
  // North levee segments (two segments with a gap in the middle)
  const leveeH = 2.0;
  const leveeHalf = 2;
  const gapCentre = Math.round(W * 0.50);
  const gapHalf = Math.round(W * 0.04);
  // Left segment
  buildings.push({
    cx: Math.round((margin + gapCentre - gapHalf) / 2),
    cz: Math.round(H * 0.03),
    halfW: Math.round((gapCentre - gapHalf - margin) / 2),
    halfH: leveeHalf,
    wallH: leveeH,
    opening: null,
    color: 0x888877,
    name: `leveeNL${id++}`,
  });
  // Right segment
  buildings.push({
    cx: Math.round((gapCentre + gapHalf + W - margin) / 2),
    cz: Math.round(H * 0.03),
    halfW: Math.round((W - margin - gapCentre - gapHalf) / 2),
    halfH: leveeHalf,
    wallH: leveeH,
    opening: null,
    color: 0x888877,
    name: `leveeNR${id++}`,
  });

  // South levee (continuous, traps water)
  buildings.push({
    cx: Math.round(W * 0.50),
    cz: Math.round(H * 0.96),
    halfW: Math.round(W * 0.46),
    halfH: leveeHalf,
    wallH: leveeH,
    opening: null,
    color: 0x888877,
    name: `leveeS${id++}`,
  });

  // ================================================================
  // 8. BOLLARDS & STREET FURNITURE — many small obstacles
  // ================================================================
  // Row along main east-west avenue
  for (let i = 0; i < 12; i++) {
    const t = 0.08 + i * 0.075;
    if (t > 0.92) continue;
    buildings.push({
      cx: Math.round(W * t), cz: Math.round(H * 0.505),
      halfW: 1, wallH: 0.5,
      opening: null,
      color: 0x444444,
      name: `bollardAve${id++}`,
    });
  }
  // Row along main north-south boulevard
  for (let i = 0; i < 10; i++) {
    const t = 0.10 + i * 0.08;
    if (t > 0.90) continue;
    buildings.push({
      cx: Math.round(W * 0.495), cz: Math.round(H * t),
      halfW: 1, wallH: 0.5,
      opening: null,
      color: 0x444444,
      name: `bollardBlvd${id++}`,
    });
  }
  // Diagonal chicane in south-east
  for (let i = 0; i < 8; i++) {
    buildings.push({
      cx: Math.round(W * (0.55 + i * 0.045)),
      cz: Math.round(H * (0.60 + i * 0.03)),
      halfW: 2, wallH: 0.7,
      opening: null,
      color: 0x555555,
      name: `chicane${id++}`,
    });
  }

  // ================================================================
  // 9. SCATTERED SHEDS & GARAGES — tiny, random
  // ================================================================
  for (let i = 0; i < 15; i++) {
    const cx = Math.round(margin + rng() * (W - 2 * margin));
    const cz = Math.round(margin + rng() * (H - 2 * margin));
    // skip if too close to grid centre streets
    if (Math.abs(cx - W / 2) < W * 0.06 && Math.abs(cz - H / 2) < H * 0.06) continue;

    buildings.push({
      cx, cz,
      halfW: Math.round(2 + rng() * 3),
      wallH: 0.6 + rng() * 0.6,
      opening: rng() < 0.4 ? { side: SIDES[Math.floor(rng() * 4)]!, width: 2, sillH: 0 } : null,
      color: 0x887766 + Math.round(rng() * 0x222222),
      name: `shed${id++}`,
    });
  }

  // ================================================================
  // 10. L-SHAPED & U-SHAPED COMPOUND BUILDINGS (multi-wing)
  // ================================================================
  // L-shape near centre: two wings
  const lCx = Math.round(W * 0.42);
  const lCz = Math.round(H * 0.28);
  buildings.push({
    cx: lCx, cz: lCz,
    halfW: Math.round(W * 0.03), halfH: Math.round(H * 0.015),
    wallH: 2.2, opening: null, color: 0x9988aa, name: `Lwing1_${id++}`,
  });
  buildings.push({
    cx: lCx + Math.round(W * 0.03), cz: lCz + Math.round(H * 0.025),
    halfW: Math.round(W * 0.012), halfH: Math.round(H * 0.03),
    wallH: 2.2, opening: { side: 'W', width: 4, sillH: 0 }, color: 0x9988aa, name: `Lwing2_${id++}`,
  });

  // U-shape in industrial area: three wings forming a courtyard
  const uCx = Math.round(W * 0.72);
  const uCz = Math.round(H * 0.72);
  const uWingW = Math.round(W * 0.015);
  const uWingLen = Math.round(H * 0.04);
  // Left wing
  buildings.push({
    cx: uCx - Math.round(W * 0.03), cz: uCz,
    halfW: uWingW, halfH: uWingLen,
    wallH: 1.8, opening: null, color: 0x778877, name: `Uwing1_${id++}`,
  });
  // Right wing
  buildings.push({
    cx: uCx + Math.round(W * 0.03), cz: uCz,
    halfW: uWingW, halfH: uWingLen,
    wallH: 1.8, opening: null, color: 0x778877, name: `Uwing2_${id++}`,
  });
  // Connecting back wall
  buildings.push({
    cx: uCx, cz: uCz - uWingLen,
    halfW: Math.round(W * 0.035), halfH: uWingW,
    wallH: 1.8, opening: null, color: 0x778877, name: `Uback_${id++}`,
  });

  return buildings;
}

/* ------------------------------------------------------------------ */
/*  Place a building into the solver bed + Three.js scene              */
/* ------------------------------------------------------------------ */

function placeBuilding(ctx: Parameters<Demo['setup']>[0], b: Building) {
  const g = ctx.solver.grid;
  const halfH = b.halfH ?? b.halfW;
  const spanW = 2 * b.halfW + 1;
  const spanH = 2 * halfH + 1;

  // Solid-mask aware solver?  KP's SweSolver exposes markSolidRegion; the
  // legacy VP solver does not.  When available, mark NON-GAP wall cells as
  // Solid (zero flux through the wall faces — eliminates the wet/dry spike
  // pathology that VP suffered).  Gap cells stay as raised bed (sill).
  const supportsSolid = typeof (ctx.solver as { markSolidRegion?: unknown }).markSolidRegion === 'function';

  const writeWall = (
    x: number, y: number, w: number, h: number,
    axis: 'x' | 'y',
    gapStart?: number, gapEnd?: number, sillH = 0,
  ) => {
    const buf = new Float32Array(w * h);
    for (let j = 0; j < h; j++) {
      for (let i = 0; i < w; i++) {
        const along = axis === 'x' ? i : j;
        const isGap = gapStart !== undefined && gapEnd !== undefined
          && along >= gapStart && along < gapEnd;
        buf[j * w + i] = isGap ? sillH : b.wallH;
      }
    }
    ctx.solver.writeBedRegion({ x, y, w, h }, buf);

    // Stamp Solid on the non-gap cells so KP treats them as impermeable
    // walls (not just tall water).  Sill cells (gap) stay as bed only.
    if (supportsSolid) {
      const solver = ctx.solver as { markSolidRegion: (r: { x: number; y: number; w: number; h: number }) => void };
      if (gapStart === undefined || gapEnd === undefined) {
        solver.markSolidRegion({ x, y, w, h });
      } else {
        // Mark the two non-gap segments along the wall axis
        if (axis === 'x') {
          if (gapStart > 0) solver.markSolidRegion({ x, y, w: gapStart, h });
          if (gapEnd < w) solver.markSolidRegion({ x: x + gapEnd, y, w: w - gapEnd, h });
        } else {
          if (gapStart > 0) solver.markSolidRegion({ x, y, w, h: gapStart });
          if (gapEnd < h) solver.markSolidRegion({ x, y: y + gapEnd, w, h: h - gapEnd });
        }
      }
    }
  };

  const opening = b.opening;
  const doorHalf = opening ? Math.max(1, Math.floor(opening.width / 2)) : 0;
  const onSide = (s: string) => opening !== null && opening.side === s;

  // North wall
  {
    const midW = Math.floor(spanW / 2);
    writeWall(b.cx - b.halfW, b.cz - halfH, spanW, 1, 'x',
      onSide('N') ? midW - doorHalf : undefined,
      onSide('N') ? midW + doorHalf : undefined,
      onSide('N') ? opening!.sillH : 0);
  }
  // South wall
  {
    const midW = Math.floor(spanW / 2);
    writeWall(b.cx - b.halfW, b.cz + halfH, spanW, 1, 'x',
      onSide('S') ? midW - doorHalf : undefined,
      onSide('S') ? midW + doorHalf : undefined,
      onSide('S') ? opening!.sillH : 0);
  }
  // West wall
  {
    const midH = Math.floor(spanH / 2);
    writeWall(b.cx - b.halfW, b.cz - halfH, 1, spanH, 'y',
      onSide('W') ? midH - doorHalf : undefined,
      onSide('W') ? midH + doorHalf : undefined,
      onSide('W') ? opening!.sillH : 0);
  }
  // East wall
  {
    const midH = Math.floor(spanH / 2);
    writeWall(b.cx + b.halfW, b.cz - halfH, 1, spanH, 'y',
      onSide('E') ? midH - doorHalf : undefined,
      onSide('E') ? midH + doorHalf : undefined,
      onSide('E') ? opening!.sillH : 0);
  }

  // --- Visual mesh --------------------------------------------------
  const mat = new THREE.MeshStandardMaterial({ color: b.color, roughness: 0.85 });
  const worldX = g.origin[0] + b.cx * g.dx;
  const worldZ = g.origin[1] + b.cz * g.dx;
  const sideLenX = spanW * g.dx;
  const sideLenZ = spanH * g.dx;

  const mk = (x: number, z: number, lx: number, lz: number, h: number, name: string) => {
    const m = ownByDemo(new THREE.Mesh(new THREE.BoxGeometry(lx, h, lz), mat));
    m.name = name;
    m.position.set(x, h / 2, z);
    ctx.scene.add(m);
  };

  const doorLenW = doorHalf * 2 * g.dx;
  const doorLenH = doorHalf * 2 * g.dx;

  // Helper: render a wall as one or two segments (split if door is on that side).
  const renderWallX = (cx_: number, cz_: number, totalLen: number, side: string, baseName: string) => {
    if (onSide(side)) {
      const seg = (totalLen - doorLenW) / 2;
      if (seg > 0.01) {
        mk(cx_ - totalLen / 2 + seg / 2, cz_, seg, g.dx, b.wallH, `${baseName}-l`);
        mk(cx_ + totalLen / 2 - seg / 2, cz_, seg, g.dx, b.wallH, `${baseName}-r`);
      }
      // Render sill block if elevated opening
      if (opening && opening.sillH > 0.01) {
        mk(cx_, cz_, doorLenW, g.dx, opening.sillH, `${baseName}-sill`);
      }
    } else {
      mk(cx_, cz_, totalLen, g.dx, b.wallH, baseName);
    }
  };

  const renderWallZ = (cx_: number, cz_: number, totalLen: number, side: string, baseName: string) => {
    if (onSide(side)) {
      const seg = (totalLen - doorLenH) / 2;
      if (seg > 0.01) {
        mk(cx_, cz_ - totalLen / 2 + seg / 2, g.dx, seg, b.wallH, `${baseName}-l`);
        mk(cx_, cz_ + totalLen / 2 - seg / 2, g.dx, seg, b.wallH, `${baseName}-r`);
      }
      if (opening && opening.sillH > 0.01) {
        mk(cx_, cz_, g.dx, doorLenH, opening.sillH, `${baseName}-sill`);
      }
    } else {
      mk(cx_, cz_, g.dx, totalLen, b.wallH, baseName);
    }
  };

  renderWallX(worldX, worldZ - halfH * g.dx, sideLenX, 'N', `${b.name}-wN`);
  renderWallX(worldX, worldZ + halfH * g.dx, sideLenX, 'S', `${b.name}-wS`);
  renderWallZ(worldX - b.halfW * g.dx, worldZ, sideLenZ, 'W', `${b.name}-wW`);
  renderWallZ(worldX + b.halfW * g.dx, worldZ, sideLenZ, 'E', `${b.name}-wE`);
}

/* ------------------------------------------------------------------ */
/*  Demo definition                                                    */
/* ------------------------------------------------------------------ */

const demo: Demo = {
  id: '09-city-flood',
  label: 'I. City Flood',
  description:
    'Massive procedural city: ~100 buildings across 10 districts — downtown skyscrapers, midtown mixed-use, ' +
    'residential neighborhoods, industrial warehouses, civic buildings, L/U-shaped compounds, parks, ' +
    'levees with flood gaps, bollard chicanes, and scattered sheds. A rising flood from the west ' +
    'navigates a 6-street grid with a diagonal drainage canal.',
  setup(ctx) {
    const g = ctx.solver.grid;
    const W = g.width;
    const H = g.height;

    // --- Generate and place city ------------------------------------
    const buildings = generateCity(W, H);
    for (const b of buildings) placeBuilding(ctx, b);

    // --- Streets: depressed bed channels flood through the city ------
    const streetDepth = -0.06;
    const streetW = Math.round(W * 0.03); // ~12 cells wide

    // Main east-west avenue (centre)
    const aveJ = Math.round(H * 0.49);
    ctx.solver.writeBedRegion(
      { x: 0, y: aveJ, w: W, h: streetW },
      new Float32Array(W * streetW).fill(streetDepth),
    );

    // Secondary east-west street (north of centre)
    const ave2J = Math.round(H * 0.35);
    ctx.solver.writeBedRegion(
      { x: 0, y: ave2J, w: W, h: streetW },
      new Float32Array(W * streetW).fill(streetDepth),
    );

    // Tertiary east-west street (south residential area)
    const ave3J = Math.round(H * 0.68);
    ctx.solver.writeBedRegion(
      { x: 0, y: ave3J, w: W, h: streetW },
      new Float32Array(W * streetW).fill(streetDepth),
    );

    // Main north-south boulevard
    const blvdI = Math.round(W * 0.48);
    ctx.solver.writeBedRegion(
      { x: blvdI, y: 0, w: streetW, h: H },
      new Float32Array(streetW * H).fill(streetDepth),
    );

    // Secondary north-south street (west)
    const blvd2I = Math.round(W * 0.25);
    ctx.solver.writeBedRegion(
      { x: blvd2I, y: 0, w: streetW, h: H },
      new Float32Array(streetW * H).fill(streetDepth),
    );

    // Tertiary north-south street (east, through industrial)
    const blvd3I = Math.round(W * 0.72);
    ctx.solver.writeBedRegion(
      { x: blvd3I, y: 0, w: streetW, h: H },
      new Float32Array(streetW * H).fill(streetDepth),
    );

    // Drainage canal running diagonally (south-west to centre) — deeper
    const canalDepth = -0.15;
    const canalW = Math.round(W * 0.015);
    for (let step = 0; step < 20; step++) {
      const t = step / 20;
      const ci = Math.round(W * (0.05 + t * 0.40));
      const cj = Math.round(H * (0.90 - t * 0.40));
      if (ci + canalW >= W || cj + canalW >= H) continue;
      ctx.solver.writeBedRegion(
        { x: ci, y: cj, w: canalW, h: canalW },
        new Float32Array(canalW * canalW).fill(canalDepth),
      );
    }

    // --- Boundary conditions ----------------------------------------
    // Inflow on west edge (flood source)
    ctx.solver.writeBoundaryRegionTarget(
      { x: 0, y: 0, w: 2, h: H },
      BoundaryType.Inflow,
      0,
    );
    // Sea boundary on east edge (active drainage — pins depth to 0)
    ctx.solver.writeBoundaryRegionTarget(
      { x: W - 2, y: 0, w: 2, h: H },
      BoundaryType.Sea,
      0,
    );

    // --- Water surface ----------------------------------------------
    const water = new WaterSurface(ctx.solver);
    ctx.scene.add(ownByDemo(water.mesh));

    // --- Camera framing for the city --------------------------------
    // Pull camera back to frame the whole city
    ctx.camera.position.set(-2, 22, 20);
    ctx.controls.target.set(0, 0, 0);
    ctx.controls.update();

    ctx.scratch.water = water;
    ctx.scratch.t = 0;
    ctx.scratch.tideH = 0;
    ctx.scratch.buildingCount = buildings.length;
  },

  tick(ctx, dt) {
    (ctx.scratch.water as WaterSurface).update();
    const g = ctx.solver.grid;
    ctx.scratch.t = (ctx.scratch.t as number) + dt;

    // Rising flood: ramps over ~12 s of sim time, caps at 2.5 m.
    const tideH = Math.min(2.5, 0.2 * (ctx.scratch.t as number));
    ctx.scratch.tideH = tideH;

    ctx.solver.writeBoundaryRegionTarget(
      { x: 0, y: 0, w: 2, h: g.height },
      BoundaryType.Inflow,
      tideH,
    );
  },
};

export default demo;
