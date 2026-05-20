// Vite's ?raw import shape — works for both library and demos.
import common from './common.wgsl?raw';
import computeFluxes from './compute_fluxes.wgsl?raw';
import updateWater from './update_water.wgsl?raw';
import applyDisplacement from './apply_displacement.wgsl?raw';
import foldHDelta from './fold_h_delta.wgsl?raw';
import accumulateForces from './accumulate_forces.wgsl?raw';
import zeroForces from './zero_forces.wgsl?raw';
import snapshotBed from './snapshot_bed.wgsl?raw';

/** Concatenate `common.wgsl` with a kernel — replaces a real #include. */
function withCommon(src: string): string {
  return `${common}\n${src}`;
}

export const ShaderSource = {
  computeFluxes: withCommon(computeFluxes),
  updateWater: withCommon(updateWater),
  applyDisplacement: withCommon(applyDisplacement),
  foldHDelta: withCommon(foldHDelta),
  accumulateForces: withCommon(accumulateForces),
  zeroForces: zeroForces, // no common required
  snapshotBed: withCommon(snapshotBed),
} as const;
