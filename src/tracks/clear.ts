import type { GUI } from 'lil-gui';
import type { Track } from '../core/track';
import { context } from '../core/device';

// Placeholder track: clears the screen to a GUI-picked color.
export class ClearTrack implements Track {
  clearColor = [0.1, 0.2, 0.3];

  init(gui: GUI) {
    gui.addColor(this, 'clearColor');
  }

  frame(encoder: GPUCommandEncoder) {
    const [r, g, b] = this.clearColor;
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          clearValue: { r, g, b, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    });
    pass.end();
  }
}
