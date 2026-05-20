import { defineConfig } from 'vite';
import { resolve } from 'node:path';

export default defineConfig({
  build: {
    lib: {
      entry: {
        isenflow: resolve(__dirname, 'src/index.ts'),
        math: resolve(__dirname, 'src/math.ts'),
      },
      formats: ['es'],
    },
    sourcemap: true,
    rollupOptions: {
      external: ['three', 'three/tsl', 'three/webgpu', '@dimforge/rapier3d-compat'],
      output: {
        preserveModules: false,
      },
    },
    target: 'es2022',
  },
  assetsInclude: ['**/*.wgsl'],
});
