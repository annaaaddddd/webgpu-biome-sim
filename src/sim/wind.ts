import type { GUI } from 'lil-gui';
import { device } from '../core/device';
import { createBuffer } from '../core/gpu';

export const WIND_SPEED = 6; // world units per second at strength 1

// One wind for the whole scene: flowers sway with it, clouds drift along it and rain slants with it.
// Nothing else sets their direction or speed.
export class Wind {
  params = {
    direction: 20, // degrees, 0 blows toward +x
    strength: 1,   // 0 calm, 1 normal
  };
  dir = [1, 0];
  // Wind in common.wgsl: dir, offset, strength, time.
  readonly uniforms = createBuffer(GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 24);
  private offset = [0, 0];
  private time = 0;

  addGui(gui: GUI) {
    const folder = gui.addFolder('Wind');
    folder.add(this.params, 'direction', 0, 360, 1);
    folder.add(this.params, 'strength', 0, 2);
  }

  update(dt: number) {
    const angle = (this.params.direction * Math.PI) / 180;
    this.dir = [Math.cos(angle), Math.sin(angle)];
    // Accumulated, so turning the wind does not make the gusts jump.
    const moved = this.params.strength * WIND_SPEED * dt;
    this.offset[0] += this.dir[0] * moved;
    this.offset[1] += this.dir[1] * moved;
    this.time += dt;
    device.queue.writeBuffer(
      this.uniforms,
      0,
      new Float32Array([...this.dir, ...this.offset, this.params.strength, this.time]),
    );
  }
}
