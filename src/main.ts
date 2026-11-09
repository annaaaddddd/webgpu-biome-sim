import Stats from 'stats.js';
import GUI from 'lil-gui';

import { initWebGPU, device } from './core/device';
import { initCamera, camera } from './core/camera';
import type { Track } from './core/track';
import { ClearTrack } from './tracks/clear';
import { ErosionTrack } from './tracks/erosion';
import { EcosystemTrack } from './tracks/ecosystem';

// Pick a track with ?track=<name>. Each M1 track registers itself here.
const tracks: Record<string, () => Track> = {
  clear: () => new ClearTrack(),
  erosion: () => new ErosionTrack(),
  ecosystem: () => new EcosystemTrack(),
};

await initWebGPU();
initCamera();

const stats = new Stats();
stats.showPanel(0);
document.body.appendChild(stats.dom);

const gui = new GUI();
const trackName = new URLSearchParams(location.search).get('track') ?? 'clear';
const track = tracks[trackName]();
await track.init(gui);

let last = performance.now();
function frame(now: number) {
  stats.begin();
  const dt = (now - last) / 1000;
  last = now;

  camera.update();
  const encoder = device.createCommandEncoder();
  track.frame(encoder, dt);
  device.queue.submit([encoder.finish()]);

  stats.end();
  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
