/**
 * 3-buffer ring for asynchronous force readback (spec §8).
 *
 * Each frame we copy the GPU accumulator into the next staging buffer and
 * request a mapAsync. We never await on the render thread; instead we check
 * if the staging buffer from N frames ago is ready and consume it then.
 *
 * Results are decoded from fixed-point i32 → f32 N (or N·m).
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

export class ForceReadback {
  private readonly slots: RingSlot[];
  private writeIdx = 0;

  constructor(
    private readonly device: GPUDevice,
    private readonly chunkCount: number,
    ringSize = 3,
  ) {
    const bytes = chunkCount * 3 * 4 * 2; // force + torque
    this.slots = Array.from({ length: ringSize }, () => ({
      buffer: device.createBuffer({
        size: bytes,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      }),
      state: 'idle' as const,
    }));
  }

  /**
   * Encode a copy from the live force/torque buffers into the next ring slot.
   * Call once per frame from inside the encoder, BEFORE submitting.
   */
  enqueueCopy(encoder: GPUCommandEncoder, forceBuf: GPUBuffer, torqueBuf: GPUBuffer): void {
    const slot = this.slots[this.writeIdx];
    if (!slot) return;
    if (slot.state !== 'idle') return; // skip if not done with last cycle
    const half = this.chunkCount * 3 * 4;
    encoder.copyBufferToBuffer(forceBuf, 0, slot.buffer, 0, half);
    encoder.copyBufferToBuffer(torqueBuf, 0, slot.buffer, half, half);
    slot.state = 'recording';
    this.writeIdx = (this.writeIdx + 1) % this.slots.length;
  }

  /**
   * Kick mapAsync on any slot in 'recording' state and harvest any 'ready'
   * slots into ChunkForce[]. Call once per frame.
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
            slot.state = 'idle';
          },
        );
      } else if (slot.state === 'ready' && slot.ready) {
        harvested = this.decode(slot.ready);
        slot.ready = undefined;
        slot.state = 'idle';
      }
    }
    return harvested;
  }

  private decode(buf: ArrayBuffer): ChunkForce[] {
    const i32 = new Int32Array(buf);
    const out: ChunkForce[] = new Array(this.chunkCount);
    const halfFloats = this.chunkCount * 3;
    for (let c = 0; c < this.chunkCount; c++) {
      out[c] = {
        fx: i32[c * 3 + 0]! / FIXED_POINT_SCALE,
        fy: i32[c * 3 + 1]! / FIXED_POINT_SCALE,
        fz: i32[c * 3 + 2]! / FIXED_POINT_SCALE,
        tx: i32[halfFloats + c * 3 + 0]! / FIXED_POINT_SCALE,
        ty: i32[halfFloats + c * 3 + 1]! / FIXED_POINT_SCALE,
        tz: i32[halfFloats + c * 3 + 2]! / FIXED_POINT_SCALE,
      };
    }
    return out;
  }

  destroy(): void {
    for (const s of this.slots) s.buffer.destroy();
  }
}
