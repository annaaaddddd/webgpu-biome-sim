import Stats from 'stats.js';
import GUI from 'lil-gui';

import { initWebGPU, device, context } from './core/device';

await initWebGPU();

const stats = new Stats();
stats.showPanel(0);
document.body.appendChild(stats.dom);

const gui = new GUI();
const params = {
  clearColor: [0.1, 0.2, 0.3],
};
gui.addColor(params, 'clearColor');

function frame() {
  stats.begin();

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [
      {
        view: context.getCurrentTexture().createView(),
        clearValue: { r: params.clearColor[0], g: params.clearColor[1], b: params.clearColor[2], a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      },
    ],
  });
  pass.end();
  device.queue.submit([encoder.finish()]);

  stats.end();
  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
