/**
 * CPU reference implementation of the Virtual-Pipes SWE solver.
 *
 * This is the algebraic oracle against which GPU kernels are verified.
 * Every formula is a literal transcription from the cited references:
 *
 *   - lisyarus/webgpu-shallow-water (MIT)
 *     https://github.com/lisyarus/webgpu-shallow-water
 *     Blog: http://lisyarus.github.io/blog/posts/simulating-water-over-terrain.html
 *     "Full simulation code" section — flux update + water integration loops.
 *
 *   - Mei, Decaudin, Hu (2007), "Fast Hydraulic Erosion Simulation and
 *     Visualization on GPU", §3.2.1 eqs. 2-5 (flux), §3.2.2 eqs. 8-9
 *     (velocity reconstruction with explicit /2).
 *     http://www-evasion.imag.fr/Publications/2007/MDH07/FastErosion_PG07.pdf
 *
 *   - Dagenais et al. (2018), "Real-Time Virtual Pipes Simulation and Modeling
 *     for Small-Scale Shallow Water" (VRIPHYS 2018), eq. (1):
 *     ζ = ω^Δt  (dt-independent damping convention, ω = 0.5 recommended).
 *
 *   - SWASHES §1 eqs. 1-2: Manning friction term (semi-implicit).
 *
 * Storage layout matches the GPU buffers (cell-centered, row-major idx = j*W + i):
 *   water[idx*2+0] = h,  water[idx*2+1] = h_prev
 *   bed[idx*2+0]   = terrain,  bed[idx*2+1] = total
 *   fluxLR[idx*2+0]= left,  fluxLR[idx*2+1] = right
 *   fluxUD[idx*2+0]= down,  fluxUD[idx*2+1] = up
 *   velocity[idx*2+0] = u,  velocity[idx*2+1] = v
 */

export interface CpuPipesParams {
  width: number;
  height: number;
  dx: number;
  dt: number;
  gravity: number;
  /** Fraction of flux retained per second (Dagenais 2018 ω). Default 0.5. */
  damping: number;
  pipeArea: number;
  pipeLen: number;
  /** Manning roughness coefficient (default 0.03 for natural channels). */
  manningN: number;
  /** Boundary type per cell (0=Interior, 1=Closed, 2=Open, 3=Sponge, 4=Inflow, 5=Sea). */
  boundary: Uint32Array;
  /** Per-cell boundary target depth (used by Inflow/Sea). */
  boundaryTargetH: Float32Array;
}

export interface CpuPipesState {
  water: Float32Array;    // 2 per cell
  bed: Float32Array;      // 2 per cell
  fluxLR: Float32Array;   // 2 per cell
  fluxUD: Float32Array;   // 2 per cell
  velocity: Float32Array; // 2 per cell
}

function idx(i: number, j: number, w: number): number {
  return j * w + i;
}

function inBounds(i: number, j: number, w: number, h: number): boolean {
  return i >= 0 && j >= 0 && i < w && j < h;
}

function hAt(water: Float32Array, i: number, j: number, w: number, h: number): number {
  if (!inBounds(i, j, w, h)) return 0;
  return water[idx(i, j, w) * 2]!;
}

function bedTotalAt(bed: Float32Array, i: number, j: number, w: number, h: number): number {
  if (!inBounds(i, j, w, h)) return 1e6;
  return bed[idx(i, j, w) * 2 + 1]!;
}

/**
 * One SWE sub-step: compute fluxes → update water + velocity.
 *
 * Mutates `state` in place.
 */
export function cpuStep(state: CpuPipesState, params: CpuPipesParams): void {
  cpuComputeFluxes(state, params);
  cpuUpdateWater(state, params);
}

/**
 * Compute-fluxes kernel (Mei 2007 eqs. 2-5, Dagenais 2018 eq. 1 for damping,
 * SWASHES §1 for Manning friction).
 */
export function cpuComputeFluxes(state: CpuPipesState, params: CpuPipesParams): void {
  const { width: W, height: H, dx, dt, gravity, damping, manningN, boundary } = params;
  const { bed, water, fluxLR, fluxUD } = state;

  // Dagenais 2018 eq. (1): dt-independent damping ζ = ω^dt
  const zeta = Math.pow(damping, dt);

  for (let j = 0; j < H; j++) {
    for (let i = 0; i < W; i++) {
      const ci = idx(i, j, W);
      const bSelf = bed[ci * 2 + 1]!;
      const hSelf = water[ci * 2]!;
      const etaSelf = bSelf + hSelf;

      // Open boundary (type 2): use transmissive condition at grid edges
      const bt = boundary[ci]!;
      const isOpen = bt === 2;
      const etaL = (isOpen && !inBounds(i - 1, j, W, H)) ? etaSelf : bedTotalAt(bed, i - 1, j, W, H) + hAt(water, i - 1, j, W, H);
      const etaR = (isOpen && !inBounds(i + 1, j, W, H)) ? etaSelf : bedTotalAt(bed, i + 1, j, W, H) + hAt(water, i + 1, j, W, H);
      const etaD = (isOpen && !inBounds(i, j - 1, W, H)) ? etaSelf : bedTotalAt(bed, i, j - 1, W, H) + hAt(water, i, j - 1, W, H);
      const etaU = (isOpen && !inBounds(i, j + 1, W, H)) ? etaSelf : bedTotalAt(bed, i, j + 1, W, H) + hAt(water, i, j + 1, W, H);

      const dhL = etaSelf - etaL;
      const dhR = etaSelf - etaR;
      const dhD = etaSelf - etaD;
      const dhU = etaSelf - etaU;

      // Depth-dependent pipe area: A = h_avg · dx, L = dx → A/L = h_avg.
      // Gives correct SWE wave speed c = √(g·h) instead of c = √(g·dx).
      const hL_nbr = hAt(water, i - 1, j, W, H);
      const hR_nbr = hAt(water, i + 1, j, W, H);
      const hD_nbr = hAt(water, i, j - 1, W, H);
      const hU_nbr = hAt(water, i, j + 1, W, H);

      const hpipeL = Math.max(0.01, 0.5 * (hSelf + hL_nbr));
      const hpipeR = Math.max(0.01, 0.5 * (hSelf + hR_nbr));
      const hpipeD = Math.max(0.01, 0.5 * (hSelf + hD_nbr));
      const hpipeU = Math.max(0.01, 0.5 * (hSelf + hU_nbr));

      // Mei 2007 eqs. 2-5: flux update with damping, accel = g · h_pipe
      let fL = Math.max(0, fluxLR[ci * 2]! * zeta + dt * gravity * hpipeL * dhL);
      let fR = Math.max(0, fluxLR[ci * 2 + 1]! * zeta + dt * gravity * hpipeR * dhR);
      let fD = Math.max(0, fluxUD[ci * 2]! * zeta + dt * gravity * hpipeD * dhD);
      let fU = Math.max(0, fluxUD[ci * 2 + 1]! * zeta + dt * gravity * hpipeU * dhU);

      // Manning friction (SWASHES §1 eqs. 1-2, semi-implicit)
      if (manningN > 0) {
        const applyManning = (flux: number, hPipe: number): number => {
          const qAbs = Math.abs(flux);
          const speed = qAbs / (dx * hPipe);
          const cf = gravity * manningN * manningN / Math.pow(hPipe, 4.0 / 3.0);
          return flux / (1.0 + dt * cf * speed);
        };
        fL = applyManning(fL, hpipeL);
        fR = applyManning(fR, hpipeR);
        fD = applyManning(fD, hpipeD);
        fU = applyManning(fU, hpipeU);
      }

      // Outflow scaling to prevent over-drain
      const totalOut = (fL + fR + fD + fU) * dt;
      const volumeAvailable = Math.max(0, hSelf) * dx * dx;
      const K = hSelf <= 0 ? 0 : Math.min(1, volumeAvailable / Math.max(totalOut, 1e-9));
      fL *= K;
      fR *= K;
      fD *= K;
      fU *= K;
      if (bt === 1) {
        fL = 0; fR = 0; fD = 0; fU = 0;
      }

      fluxLR[ci * 2] = fL;
      fluxLR[ci * 2 + 1] = fR;
      fluxUD[ci * 2] = fD;
      fluxUD[ci * 2 + 1] = fU;
    }
  }
}

/**
 * Update-water kernel (Mei 2007 eqs. 8-9 for velocity reconstruction,
 * with the explicit divide-by-2).
 */
export function cpuUpdateWater(state: CpuPipesState, params: CpuPipesParams): void {
  const { width: W, height: H, dx, dt, boundary, boundaryTargetH } = params;
  const { water, fluxLR, fluxUD, velocity } = state;

  const getLR = (i: number, j: number): [number, number] => {
    if (!inBounds(i, j, W, H)) return [0, 0];
    const ci = idx(i, j, W);
    return [fluxLR[ci * 2]!, fluxLR[ci * 2 + 1]!];
  };
  const getUD = (i: number, j: number): [number, number] => {
    if (!inBounds(i, j, W, H)) return [0, 0];
    const ci = idx(i, j, W);
    return [fluxUD[ci * 2]!, fluxUD[ci * 2 + 1]!];
  };

  for (let j = 0; j < H; j++) {
    for (let i = 0; i < W; i++) {
      const ci = idx(i, j, W);
      const [selfL, selfR] = getLR(i, j);
      const [selfD, selfU] = getUD(i, j);

      const [, nbrL_R] = getLR(i - 1, j);
      const [nbrR_L] = getLR(i + 1, j);
      const [, nbrD_U] = getUD(i, j - 1);
      const [nbrU_D] = getUD(i, j + 1);

      const outflow = selfL + selfR + selfD + selfU;
      const inflow = nbrL_R + nbrR_L + nbrD_U + nbrU_D;

      const area = dx * dx;
      const dV = (inflow - outflow) * dt;
      const dH = dV / area;

      const hOld = water[ci * 2]!;
      let newH = Math.max(0, hOld + dH);

      // Boundary conditions
      const bt = boundary[ci]!;
      if (bt === 4) {
        // Inflow: pin to target depth (floor)
        newH = Math.max(newH, boundaryTargetH[ci]!);
      } else if (bt === 5) {
        // Sea: pin to target depth
        newH = boundaryTargetH[ci]!;
      } else if (bt === 3) {
        // Sponge: damp toward neighbor mean
        const mean = (hAt(water, i - 1, j, W, H) +
                      hAt(water, i + 1, j, W, H) +
                      hAt(water, i, j - 1, W, H) +
                      hAt(water, i, j + 1, W, H)) * 0.25;
        newH = newH + 0.1 * (mean - newH); // mix(newH, mean, 0.1)
      }

      water[ci * 2] = newH;
      water[ci * 2 + 1] = hOld;

      // Mei 2007 eq. 8: velocity reconstruction WITH explicit /2
      const netX = (selfR - selfL + nbrL_R - nbrR_L) * 0.5;
      const netZ = (selfU - selfD + nbrD_U - nbrU_D) * 0.5;
      const hAvg = Math.max(0.05, 0.5 * (hOld + newH));
      const u = netX / (dx * hAvg);
      const v = netZ / (dx * hAvg);
      velocity[ci * 2] = u;
      velocity[ci * 2 + 1] = v;
    }
  }
}

/**
 * Create a fresh state with the given dimensions. All arrays are zero-filled.
 */
export function createCpuState(width: number, height: number): CpuPipesState {
  const cells = width * height;
  return {
    water: new Float32Array(cells * 2),
    bed: new Float32Array(cells * 2),
    fluxLR: new Float32Array(cells * 2),
    fluxUD: new Float32Array(cells * 2),
    velocity: new Float32Array(cells * 2),
  };
}

/**
 * Create default params with sensible defaults.
 */
export function createCpuParams(
  width: number,
  height: number,
  opts?: Partial<Omit<CpuPipesParams, 'width' | 'height' | 'boundary' | 'boundaryTargetH'>>,
): CpuPipesParams {
  const dx = opts?.dx ?? 1;
  const cells = width * height;
  return {
    width,
    height,
    dx,
    dt: opts?.dt ?? 1 / 240,
    gravity: opts?.gravity ?? 9.81,
    damping: opts?.damping ?? 0.5,
    pipeArea: opts?.pipeArea ?? dx * dx,
    pipeLen: opts?.pipeLen ?? dx,
    manningN: opts?.manningN ?? 0,
    boundary: new Uint32Array(cells),
    boundaryTargetH: new Float32Array(cells),
  };
}
