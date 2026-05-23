// Public API barrel for `isenflow`.

export { acquireGPU, isWebGPUAvailable } from './core/GPUContext.js';
export type { GPUContext } from './core/GPUContext.js';
export { SimulationGrid } from './core/SimulationGrid.js';
export type { SimulationGridOptions } from './core/SimulationGrid.js';
export { VirtualPipesSolver } from './core/VirtualPipesSolver.js';
export type { SolverOptions } from './core/VirtualPipesSolver.js';
export { SweSolver } from './core/SweSolver.js';
export type { SweSolverOptions } from './core/SweSolver.js';

export { HeightfieldRasterizer } from './coupling/HeightfieldRasterizer.js';
export type { BoxObstacle, DynamicBodyDescriptor } from './coupling/HeightfieldRasterizer.js';
export { BodyTracker } from './coupling/BodyTracker.js';
export { ForceReadback } from './coupling/ForceReadback.js';
export type { ChunkForce } from './coupling/ForceReadback.js';
export { applyForcesToBodies, applyStabilizedForces, clampCoupledVelocities, createForceSmootherState } from './coupling/RapierBridge.js';
export type { RapierBodyLike, RapierBodyFull, RapierBodyMap, ApplyForcesOptions, CoupledBodyInfo, ForceSmootherState } from './coupling/RapierBridge.js';
export { computeCpuBuoyancy, computeVerticalDamping, estimateFootprintCells } from './coupling/CpuBuoyancy.js';
export { computeImpact, applyImpact, spawnImpactSplash } from './coupling/ImpactDisplacement.js';
export type { ImpactParams, ImpactResult, ImpactShape, ComputedImpact } from './coupling/ImpactDisplacement.js';

export { WallChunk } from './destruction/WallChunk.js';
export type { WallChunkInit } from './destruction/WallChunk.js';
export { createStress, tickStress } from './destruction/StressAccumulator.js';
export type { StressState } from './destruction/StressAccumulator.js';
export { FractureScheduler } from './destruction/FractureScheduler.js';

export { BoundaryType, isReflective, isAbsorbing, isSourceLike } from './boundaries/BoundaryConditions.js';
export type { BoundaryTypeValue } from './boundaries/BoundaryConditions.js';

export { SplashParticleSystem } from './particles/SplashParticleSystem.js';
export type { Particle, SpawnRequest } from './particles/SplashParticleSystem.js';

export { createWaterMesh } from './render/WaterMesh.js';
export type { WaterMeshOptions } from './render/WaterMesh.js';
export { createWaterMaterialTSL } from './render/WaterMaterial.js';
export type { WaterMaterialDeps } from './render/WaterMaterial.js';

// Math re-exports for convenience (also exported standalone via "isenflow/math").
export * from './math.js';
