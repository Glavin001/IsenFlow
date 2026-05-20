/**
 * WebGPU device acquisition + capability probing.
 */

export interface GPUContext {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  readonly queue: GPUQueue;
  /** Adapter info if available (for debug logging). */
  readonly adapterInfo: GPUAdapterInfo | null;
}

export interface AcquireOpts {
  /** Required features; missing features cause the request to throw. */
  requiredFeatures?: GPUFeatureName[];
  /** Soft preference; passed through unchanged. */
  powerPreference?: GPUPowerPreference;
}

export async function acquireGPU(opts: AcquireOpts = {}): Promise<GPUContext> {
  if (typeof navigator === 'undefined' || !('gpu' in navigator)) {
    throw new Error(
      '[isenflow] WebGPU is not available. Ensure you are on a browser with WebGPU enabled.',
    );
  }
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: opts.powerPreference ?? 'high-performance',
  });
  if (!adapter) {
    throw new Error('[isenflow] navigator.gpu.requestAdapter() returned null.');
  }
  const device = await adapter.requestDevice({
    requiredFeatures: opts.requiredFeatures ?? [],
  });
  const adapterInfo = 'info' in adapter ? (adapter as GPUAdapter & { info: GPUAdapterInfo }).info : null;
  return { adapter, device, queue: device.queue, adapterInfo };
}

/** Returns true when WebGPU is present in the global scope. */
export function isWebGPUAvailable(): boolean {
  return typeof navigator !== 'undefined' && 'gpu' in navigator && navigator.gpu != null;
}
