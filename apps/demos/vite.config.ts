import { defineConfig } from 'vite';

export default defineConfig({
  base: '/',
  server: { port: 5173, strictPort: false },
  preview: { port: 4173, strictPort: true },
  build: {
    target: 'es2022',
    sourcemap: true,
    rollupOptions: {
      input: {
        index: new URL('./index.html', import.meta.url).pathname,
        harness: new URL('./test-harness.html', import.meta.url).pathname,
      },
    },
  },
  optimizeDeps: { exclude: ['@dimforge/rapier3d-compat'] },
});
