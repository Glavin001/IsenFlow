/**
 * Tests that a floating body reaches stable equilibrium without sustained
 * oscillation. This catches underdamped buoyancy that causes visible vibration.
 *
 * Simulates a simple 1D vertical spring-damper (buoyancy + gravity + damping)
 * using the same discrete timestep as the real system (Rapier at 1/60s).
 */
import { describe, it, expect } from 'vitest';
import {
  computeCpuBuoyancy,
  computeVerticalDamping,
  estimateFootprintCells,
} from '../../src/coupling/CpuBuoyancy.js';

/** Simulate a floating body for N frames and return Y-position history. */
function simulateFloatingBody(opts: {
  mass: number;
  halfExtents: [number, number, number];
  waterLevel: number;
  bedLevel: number;
  dx: number;
  startY: number;
  frames: number;
  rapierDt?: number;
  gravity?: number;
  linearDamping?: number;
  vyDamping?: number;
}): { positions: number[]; velocities: number[] } {
  const {
    mass, halfExtents, waterLevel, bedLevel, dx,
    startY, frames,
    rapierDt = 1 / 60,
    gravity = 9.81,
    linearDamping = 2.0,
    vyDamping = 0.85,
  } = opts;
  const [halfX, halfY, halfZ] = halfExtents;
  const footprint = estimateFootprintCells(halfX, halfZ, dx);

  let y = startY;
  let vy = 0;
  const positions: number[] = [y];
  const velocities: number[] = [vy];

  for (let i = 0; i < frames; i++) {
    // 1) Compute buoyancy + damping (same as applyStabilizedForces)
    const buoyancy = computeCpuBuoyancy({
      comY: y,
      halfY,
      linvelY: vy,
      waterLevel,
      bedLevel,
      footprintCells: footprint,
      dx,
    });
    const vdamp = computeVerticalDamping(vy, buoyancy.submergedFraction, dx, footprint);
    const fy = buoyancy.fy + vdamp;

    // 2) Gravity (Rapier applies this)
    const fGravity = -mass * gravity;

    // 3) Integrate (semi-implicit Euler, like Rapier)
    const totalF = fy + fGravity;
    const accel = totalF / mass;
    vy += accel * rapierDt;

    // 4) Rapier linear damping: v *= 1 / (1 + dt * linearDamping)
    vy *= 1 / (1 + rapierDt * linearDamping);

    // 4b) Post-step vertical damping (from clampCoupledVelocities)
    vy *= vyDamping;

    // 5) Update position
    y += vy * rapierDt;

    positions.push(y);
    velocities.push(vy);
  }

  return { positions, velocities };
}

describe('Buoyancy vertical stability', () => {
  // Crate from demo 02: 0.4x0.2x0.4m, pine wood 400 kg/m³
  const mass = 0.4 * 0.2 * 0.4 * 400; // 12.8 kg
  const halfExtents: [number, number, number] = [0.2, 0.1, 0.2];
  const dx = 16 / 384; // ~0.042m
  const waterLevel = 0.8;
  const bedLevel = 0;

  it('body reaches equilibrium from above water', () => {
    const { positions } = simulateFloatingBody({
      mass, halfExtents, waterLevel, bedLevel, dx,
      startY: 1.5, // dropped from above
      frames: 300, // 5 seconds at 60fps
    });

    const final = positions[positions.length - 1]!;
    // Should settle near Archimedes equilibrium
    // For 400/1000 density, ~40% submerged → comY ≈ 0.82
    expect(final).toBeGreaterThan(0.75);
    expect(final).toBeLessThan(0.90);
  });

  it('body reaches equilibrium from below water', () => {
    const { positions } = simulateFloatingBody({
      mass, halfExtents, waterLevel, bedLevel, dx,
      startY: 0.3, // pushed below
      frames: 300,
    });

    const final = positions[positions.length - 1]!;
    expect(final).toBeGreaterThan(0.75);
    expect(final).toBeLessThan(0.90);
  });

  it('settled body has negligible vertical velocity (no vibration)', () => {
    const { velocities, positions } = simulateFloatingBody({
      mass, halfExtents, waterLevel, bedLevel, dx,
      startY: 1.2, // slight drop
      frames: 300,
    });

    // After settling (last 60 frames = 1 second), velocity should be near zero
    const lastSecond = velocities.slice(-60);
    const maxAbsVy = Math.max(...lastSecond.map(Math.abs));
    expect(maxAbsVy, 'vertical velocity in last second should be < 0.01 m/s').toBeLessThan(0.01);

    // Position should not oscillate: max - min in last second should be tiny
    const lastSecondPos = positions.slice(-60);
    const range = Math.max(...lastSecondPos) - Math.min(...lastSecondPos);
    expect(range, 'position range in last second should be < 1mm').toBeLessThan(0.001);
  });

  it('no overshoot beyond 2x equilibrium displacement', () => {
    // Drop from well above — body should not plunge below equilibrium
    // by more than the initial displacement
    const startY = 1.5;
    const { positions } = simulateFloatingBody({
      mass, halfExtents, waterLevel, bedLevel, dx,
      startY,
      frames: 300,
    });

    const equilibrium = positions[positions.length - 1]!;
    const minY = Math.min(...positions);
    // Body should not undershoot equilibrium by more than it was dropped
    const overshoot = equilibrium - minY;
    const drop = startY - equilibrium;
    expect(overshoot, 'overshoot should be less than initial drop').toBeLessThan(drop);
  });

  it('stable under water-level perturbation (rasterizer feedback)', () => {
    // Simulates the rasterizer feedback: body displaces water, so the
    // actual water level near the body oscillates around the reference.
    // We model this as a ±5% water level noise each frame.
    const footprint = estimateFootprintCells(halfExtents[0], halfExtents[2], dx);
    let y = 0.826; // near equilibrium
    let vy = 0;
    const rapierDt = 1 / 60;
    const positions: number[] = [];

    for (let i = 0; i < 300; i++) {
      // Perturbed water level (simulates rasterizer displacement)
      const perturbedWL = waterLevel + 0.04 * Math.sin(i * 0.5);

      const buoyancy = computeCpuBuoyancy({
        comY: y, halfY: halfExtents[1], linvelY: vy,
        waterLevel: perturbedWL, bedLevel, footprintCells: footprint, dx,
      });
      const vdamp = computeVerticalDamping(vy, buoyancy.submergedFraction, dx, footprint);
      const totalF = buoyancy.fy + vdamp - mass * 9.81;
      vy += (totalF / mass) * rapierDt;
      vy *= 1 / (1 + rapierDt * 2.0); // Rapier damping
      vy *= 0.85; // post-step damping
      y += vy * rapierDt;
      positions.push(y);
    }

    // Even with perturbation, body should stay within ±5cm of equilibrium
    const lastSecond = positions.slice(-60);
    const range = Math.max(...lastSecond) - Math.min(...lastSecond);
    expect(range, 'position range under perturbation < 5cm').toBeLessThan(0.05);
  });

  it('heavy body (concrete, 2400 kg/m³) sinks and settles on bed', () => {
    const concreteMass = 0.4 * 0.2 * 0.4 * 2400; // 76.8 kg
    const { positions } = simulateFloatingBody({
      mass: concreteMass, halfExtents, waterLevel, bedLevel, dx,
      startY: 1.0,
      frames: 300,
    });

    const final = positions[positions.length - 1]!;
    // Concrete is denser than water — should sink to near bed
    // (In real sim, bed collider stops it; here it goes negative)
    expect(final).toBeLessThan(0.3);
  });
});
