/**
 * 3-buffer ring for asynchronous force/torque readback (spec §8).
 *
 * The solver stores forces in a single `forceAccum` buffer with stride 6×i32
 * per chunk: [fx, fy, fz, tx, ty, tz] in fixed-point. Each frame the solver
 * issues a `copyBufferToBuffer` from `forceAccum` into the next ring slot;
 * we mapAsync that slot N frames later, decoding to `ChunkForce[]` without
 * blocking the render thread.
 */
import { FIXED_POINT_SCALE } from '../utils/fixedPoint.js';

export interface ChunkForce {
  fx: number;
  fy: number;
  fz: number;
  tx: number;
  ty: number;
  tz: number;
}

interface RingSlot {
  buffer: GPUBuffer;
  state: 'idle' | 'recording' | 'mapping' | 'ready';
  ready?: ArrayBuffer;
}

const STRIDE_I32 = 6;

export class ForceReadback {
  private readonly slots: RingSlot[];
  private writeIdx = 0;
  /** Total number of force values harvested since construction (debug). */
  harvested = 0;

  constructor(
    private readonly device: GPUDevice,
    private readonly chunkCount: number,
    ringSize = 3,
  ) {
    const bytes = chunkCount * STRIDE_I32 * 4;
    this.slots = Array.from({ length: ringSize }, () => ({
      buffer: device.createBuffer({
        size: bytes,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      }),
      state: 'idle' as const,
    }));
  }

  /** Bytes per chunk in the source buffer. */
  static readonly STRIDE_BYTES = STRIDE_I32 * 4;

  /**
   * The next ring slot ready to receive a copy. Returns `null` if every
   * slot is mid-flight; in that case we just skip readback this frame
   * (one-frame latency is harmless).
   */
  acquireWriteSlot(): GPUBuffer | null {
    const slot = this.slots[this.writeIdx];
    if (!slot || slot.state !== 'idle') return null;
    slot.state = 'recording';
    this.writeIdx = (this.writeIdx + 1) % this.slots.length;
    return slot.buffer;
  }

  /**
   * Encode a copy from `forceAccum` into the next ring slot. Call once per
   * frame from inside the encoder, BEFORE submitting. Returns true if a
   * copy was queued, false if all slots were busy.
   */
  enqueueCopy(encoder: GPUCommandEncoder, forceAccum: GPUBuffer): boolean {
    const buf = this.acquireWriteSlot();
    if (!buf) return false;
    const bytes = this.chunkCount * STRIDE_I32 * 4;
    encoder.copyBufferToBuffer(forceAccum, 0, buf, 0, bytes);
    return true;
  }

  /**
   * Kick mapAsync on any slot in 'recording' state and harvest any 'ready'
   * slots into `ChunkForce[]`. Call once per frame.
   *
   * Returns the *most recently ready* harvest, or `null` if none completed
   * since the last poll. (Older readies are dropped — we only care about
   * the freshest forces.)
   */
  poll(): ChunkForce[] | null {
    let harvested: ChunkForce[] | null = null;
    for (const slot of this.slots) {
      if (slot.state === 'recording') {
        slot.state = 'mapping';
        slot.buffer.mapAsync(GPUMapMode.READ).then(
          () => {
            slot.ready = slot.buffer.getMappedRange().slice(0);
            slot.buffer.unmap();
            slot.state = 'ready';
          },
          () => {
            // Map failed (device lost / cancelled). Reset slot to idle so
            // the ring keeps moving.
            slot.state = 'idle';
          },
        );
      } else if (slot.state === 'ready' && slot.ready) {
        harvested = this.decode(slot.ready);
        slot.ready = undefined;
        slot.state = 'idle';
        this.harvested++;
      }
    }
    return harvested;
  }

  decode(buf: ArrayBuffer): ChunkForce[] {
    const i32 = new Int32Array(buf);
    const out: ChunkForce[] = new Array(this.chunkCount);
    for (let c = 0; c < this.chunkCount; c++) {
      const o = c * STRIDE_I32;
      out[c] = {
        fx: (i32[o + 0] ?? 0) / FIXED_POINT_SCALE,
        fy: (i32[o + 1] ?? 0) / FIXED_POINT_SCALE,
        fz: (i32[o + 2] ?? 0) / FIXED_POINT_SCALE,
        tx: (i32[o + 3] ?? 0) / FIXED_POINT_SCALE,
        ty: (i32[o + 4] ?? 0) / FIXED_POINT_SCALE,
        tz: (i32[o + 5] ?? 0) / FIXED_POINT_SCALE,
      };
    }
    return out;
  }

  destroy(): void {
    for (const s of this.slots) s.buffer.destroy();
  }
}
