import { describe, it, expect } from 'vitest';
import { SplashParticleSystem } from '../../src/particles/SplashParticleSystem.js';

describe('SplashParticleSystem', () => {
  it('spawns particles with active flag', () => {
    const sys = new SplashParticleSystem(64);
    sys.spawn({ position: [0, 0, 0], intensity: 1, upwardSpeed: 5 });
    expect(sys.activeCount()).toBeGreaterThan(0);
  });

  it('expires particles after their lifetime elapses', () => {
    const sys = new SplashParticleSystem(64);
    sys.spawn({ position: [0, 5, 0], intensity: 1, upwardSpeed: 0 });
    for (let i = 0; i < 200; i++) sys.tick(0.05);
    expect(sys.activeCount()).toBe(0);
  });

  it('applies gravity to vy each tick', () => {
    const sys = new SplashParticleSystem(8);
    sys.spawn({ position: [0, 10, 0], intensity: 1, upwardSpeed: 10 });
    const start = sys.pool.find((p) => p.active)!;
    const vy0 = start.vy;
    sys.tick(1.0, 9.81);
    expect(start.vy).toBeCloseTo(vy0 - 9.81, 5);
  });
});
