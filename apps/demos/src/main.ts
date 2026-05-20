import { createDemoContext, startLoop, switchDemo, type Demo, type DemoContext } from './shared/Scene.js';

import damBreak from './demos/01-dam-break.js';
import riverboat from './demos/02-riverboat.js';
import sinkingVehicles from './demos/03-sinking-vehicles.js';
import buildingFlood from './demos/04-building-flood.js';
import tsunami from './demos/05-tsunami-overtopping.js';
import cascading from './demos/06-cascading-destruction.js';
import splashes from './demos/07-splash-showcase.js';

const demos: Demo[] = [
  damBreak,
  riverboat,
  sinkingVehicles,
  buildingFlood,
  tsunami,
  cascading,
  splashes,
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

  const fpsEl = document.getElementById('fps')!;
  startLoop(ctx, (fps) => (fpsEl.textContent = `${fps} fps`));

  const pickFromHash = async () => {
    const id = (location.hash.replace('#', '') || demos[0]!.id);
    const demo = demos.find((d) => d.id === id) ?? demos[0]!;
    select.value = demo.id;
    desc.textContent = demo.description;
    try {
      await switchDemo(ctx, demo);
    } catch (err) {
      showFatal((err as Error).message);
    }
  };

  select.addEventListener('change', () => {
    location.hash = `#${select.value}`;
  });
  window.addEventListener('hashchange', () => { void pickFromHash(); });

  await pickFromHash();
}

function showFatal(message: string) {
  const el = document.getElementById('err')!;
  el.style.display = 'block';
  el.textContent = message;
}

main().catch((err) => showFatal((err as Error).message + '\n' + ((err as Error).stack ?? '')));
