import { createDemoContext, startLoop, switchDemo, type Demo, type DemoContext } from './shared/Scene.js';
import type { WaterSurface } from './shared/Water.js';

import damBreak from './demos/01-dam-break.js';
import riverboat from './demos/02-riverboat.js';
import sinkingVehicles from './demos/03-sinking-vehicles.js';
import buildingFlood from './demos/04-building-flood.js';
import tsunami from './demos/05-tsunami-overtopping.js';
import cascading from './demos/06-cascading-destruction.js';
import splashes from './demos/07-splash-showcase.js';
import impact from './demos/08-impact.js';
import cityFlood from './demos/09-city-flood.js';
import mountainRiver from './demos/10-mountain-river.js';
import rainStorm from './demos/11-rain-storm.js';

const demos: Demo[] = [
  damBreak,
  riverboat,
  sinkingVehicles,
  buildingFlood,
  tsunami,
  cascading,
  splashes,
  impact,
  cityFlood,
  mountainRiver,
  rainStorm,
];

async function main() {
  const app = document.getElementById('app')!;
  const canvas = document.createElement('canvas');
  app.appendChild(canvas);

  let ctx: DemoContext;
  try {
    ctx = await createDemoContext(canvas);
  } catch (err) {
    const e = err as Error;
    showFatal(e.message + '\n' + (e.stack ?? ''));
    return;
  }

  const select = document.getElementById('demoSelect') as HTMLSelectElement;
  const desc = document.getElementById('demoDescription')!;
  for (const d of demos) {
    const opt = document.createElement('option');
    opt.value = d.id;
    opt.textContent = d.label;
    select.appendChild(opt);
  }

  const $ = (id: string) => document.getElementById(id)!;
  const fpsEl = $('hud-fps');
  const frameMsEl = $('hud-frame-ms');
  const simMsEl = $('hud-sim-ms');
  const substepsEl = $('hud-substeps');
  const simHzEl = $('hud-sim-hz');
  const simRateEl = $('hud-sim-rate');
  const gridEl = $('hud-grid');
  const gpuEl = $('hud-gpu');
  const speedEl = $('hud-speed') as HTMLInputElement;
  const speedValEl = $('hud-speed-val');

  const g = ctx.solver.grid;
  gridEl.textContent = `${g.width}×${g.height} · dx=${g.dx}m`;
  gpuEl.textContent = ctx.adapterInfo.length > 30 ? ctx.adapterInfo.slice(0, 30) + '…' : ctx.adapterInfo;
  gpuEl.title = ctx.adapterInfo;

  const fmt = (n: number, d = 1) => (Number.isFinite(n) ? n.toFixed(d) : '--');
  speedEl.addEventListener('input', () => {
    ctx.simSpeed = parseFloat(speedEl.value);
    speedValEl.textContent = `${fmt(ctx.simSpeed, 2)}×`;
  });
  ctx.simSpeed = parseFloat(speedEl.value);
  speedValEl.textContent = `${fmt(ctx.simSpeed, 2)}×`;

  const pauseBtn = $('hud-pause') as HTMLButtonElement;
  let isPaused = false;
  let savedSimSpeed = ctx.simSpeed;

  const togglePause = () => {
    isPaused = !isPaused;
    if (isPaused) {
      savedSimSpeed = ctx.simSpeed || savedSimSpeed;
      ctx.simSpeed = 0;
      pauseBtn.textContent = 'Resume';
      speedEl.disabled = true;
    } else {
      ctx.simSpeed = savedSimSpeed;
      speedEl.value = String(savedSimSpeed);
      speedValEl.textContent = `${fmt(savedSimSpeed, 2)}×`;
      pauseBtn.textContent = 'Pause';
      speedEl.disabled = false;
    }
  };
  pauseBtn.addEventListener('click', togglePause);

  const oceanToggle = $('oceanToggle') as HTMLInputElement;
  oceanToggle.addEventListener('change', () => {
    const water = ctx.scratch.water as WaterSurface | undefined;
    if (water) water.setOceanStyle(oceanToggle.checked);
  });

  startLoop(ctx, (s) => {
    fpsEl.textContent = String(s.fps);
    frameMsEl.textContent = fmt(s.frameMsP50, 1);
    simMsEl.textContent = fmt(s.simStepMsP50, 2);
    substepsEl.textContent = String(s.substeps);
    simHzEl.textContent = fmt(s.simHz, 0);
    simRateEl.textContent = fmt(s.simRate, 2);
  });

  const pickFromHash = async () => {
    const id = (location.hash.replace('#', '') || demos[0]!.id);
    const demo = demos.find((d) => d.id === id) ?? demos[0]!;
    select.value = demo.id;
    desc.textContent = demo.description;
    try {
      await switchDemo(ctx, demo);
      const water = ctx.scratch.water as WaterSurface | undefined;
      if (water) water.setOceanStyle(oceanToggle.checked);
    } catch (err) {
      showFatal((err as Error).message);
    }
  };

  select.addEventListener('change', () => {
    location.hash = `#${select.value}`;
  });
  window.addEventListener('hashchange', () => { void pickFromHash(); });

  // Press "R" to reset the current demo without switching.
  window.addEventListener('keydown', (e) => {
    if ((e.target as HTMLElement)?.tagName === 'INPUT') return;
    if (e.key === 'r' || e.key === 'R') {
      if (isPaused) togglePause();
      void pickFromHash();
    }
    if (e.key === ' ') {
      e.preventDefault();
      togglePause();
    }
  });

  await pickFromHash();
}

function showFatal(message: string) {
  const el = document.getElementById('err')!;
  el.style.display = 'block';
  el.textContent = message;
}

main().catch((err) => showFatal((err as Error).message + '\n' + ((err as Error).stack ?? '')));
