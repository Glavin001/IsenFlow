// Vite's ?raw import shape — works for both library and demos.
import common from './common.wgsl?raw';
import computeFluxes from './compute_fluxes.wgsl?raw';
import updateWater from './update_water.wgsl?raw';
import applyDisplacement from './apply_displacement.wgsl?raw';
import foldHDelta from './fold_h_delta.wgsl?raw';
import accumulateForces from './accumulate_forces.wgsl?raw';
import zeroForces from './zero_forces.wgsl?raw';
import snapshotBed from './snapshot_bed.wgsl?raw';

// Kurganov–Petrova SWE shaders (the production scheme post-Phase-7).
import kpCommonSrc from './kp/kp_common.wgsl?raw';
import kpSlopesSrc from './kp/kp_slopes.wgsl?raw';
import kpUpdateSrc from './kp/kp_update.wgsl?raw';
import kpRefreshSrc from './kp/kp_refresh_views.wgsl?raw';
import kpSyncSrc from './kp/kp_sync_state.wgsl?raw';

/** Concatenate `common.wgsl` with a kernel — replaces a real #include. */
function withCommon(src: string): string {
  return `${common}\n${src}`;
}

function withKpCommon(src: string): string {
  return `${common}\n${kpCommonSrc}\n${src}`;
}

export const ShaderSource = {
  // ---- Legacy Virtual-Pipes shaders (to be removed in Phase 7) ----
  computeFluxes: withCommon(computeFluxes),
  updateWater: withCommon(updateWater),
  // ---- Shared kernels (used by both VP and KP) ----
  applyDisplacement: withCommon(applyDisplacement),
  foldHDelta: withCommon(foldHDelta),
  accumulateForces: withCommon(accumulateForces),
  zeroForces: zeroForces, // no common required
  snapshotBed: withCommon(snapshotBed),
  // ---- Kurganov–Petrova kernels (new production scheme) ----
  kpSlopes:       withKpCommon(kpSlopesSrc),
  kpUpdate:       withKpCommon(kpUpdateSrc),
  kpRefreshViews: withCommon(kpRefreshSrc),
  kpSyncState:    withCommon(kpSyncSrc),
} as const;
