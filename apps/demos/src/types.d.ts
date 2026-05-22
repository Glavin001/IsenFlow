// Ambient declarations for assets imported across the workspace.
declare module '*.wgsl?raw' {
  const src: string;
  export default src;
}
declare module '*.wgsl' {
  const src: string;
  export default src;
}

// Re-export the test-bridge type onto the global Window so that Playwright
// `page.evaluate(() => window.__isenflow_app...)` is type-safe.
import type { AppBridge } from './shared/testBridge.js';

declare global {
  interface Window {
    __isenflow_app?: AppBridge;
    __isenflow_test?: {
      ready: boolean;
      runConservation: (steps?: number) => Promise<{ drift: number; initial: number; final: number }>;
      runDamBreak: (steps?: number, depth?: number, cells?: number) => Promise<{ frontCellAtEnd: number; expectedCell: number }>;
      hasWebGPU: () => boolean;
      probeAdapter: () => Promise<{ available: boolean; reason?: string; adapter?: string }>;
      runForceAccumulator?: () => Promise<{ peakMagnitude: number; nonZeroChunks: number }>;
      runDisplacement?: () => Promise<{ totalHDelta: number; volumeBefore: number; volumeAfter: number }>;
    };
  }
}
