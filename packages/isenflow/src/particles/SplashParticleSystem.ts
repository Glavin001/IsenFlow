/**
 * GPU-resident splash particle pool (spec §12).
 *
 * Pure decoration: no feedback to SWE. We allocate a fixed pool and recycle
 * particles when they expire. Both CPU spawn (called from displacement events
 * detected on the CPU mirror) and ballistic dynamics live here.
 *
 * For simplicity and to keep WebGPU bindings minimal, this v1 runs spawn +
 * integration on the CPU and uses a single instanced draw at render time.
 * It can be upgraded to a fully GPU pipeline later.
 */

export interface Particle {
  px: number; py: number; pz: number;
  vx: number; vy: number; vz: number;
  life: number;
  active: boolean;
}

export interface SpawnRequest {
  position: readonly [number, number, number];
  intensity: number; // 0..1
  upwardSpeed: number; // m/s, base
}

export class SplashParticleSystem {
  readonly pool: Particle[];
  private cursor = 0;

  constructor(public readonly capacity: number = 5000) {
    this.pool = Array.from({ length: capacity }, () => ({
      px: 0, py: 0, pz: 0, vx: 0, vy: 0, vz: 0, life: 0, active: false,
    }));
  }

  spawn(req: SpawnRequest): void {
    const count = Math.max(1, Math.floor(req.intensity * 30));
    for (let n = 0; n < count; n++) {
      const p = this.pool[this.cursor]!;
      this.cursor = (this.cursor + 1) % this.capacity;
      p.active = true;
      p.life = 1.0 + Math.random() * 0.5;
      p.px = req.position[0] + (Math.random() - 0.5) * 0.6;
      p.py = req.position[1];
      p.pz = req.position[2] + (Math.random() - 0.5) * 0.6;
      const angle = Math.random() * Math.PI * 2;
      const radial = Math.random() * req.upwardSpeed * 0.5;
      p.vx = Math.cos(angle) * radial;
      p.vz = Math.sin(angle) * radial;
      p.vy = req.upwardSpeed * (0.6 + Math.random() * 0.4);
    }
  }

  tick(dt: number, gravity = 9.81): void {
    for (const p of this.pool) {
      if (!p.active) continue;
      p.vy -= gravity * dt;
      p.px += p.vx * dt;
      p.py += p.vy * dt;
      p.pz += p.vz * dt;
      p.life -= dt;
      if (p.life <= 0 || p.py < -1) p.active = false;
    }
  }

  activeCount(): number {
    let n = 0;
    for (const p of this.pool) if (p.active) n++;
    return n;
  }
}
