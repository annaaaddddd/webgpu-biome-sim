import type { GUI } from 'lil-gui';
import { device, canvasFormat } from '../core/device';
import { camera } from '../core/camera';
import { HEIGHT_SCALE, WORLD_SIZE, MSAA_SAMPLES } from '../core/layout';
import { shaderModule } from '../core/gpu';
import { WIND_SPEED, type Wind } from './wind';
import cloudsWgsl from '../shaders/weather/clouds.wgsl?raw';

const spacing = 1.5; // vertical gap between layers, world units
const DRIFT = WIND_SPEED / 2; // clouds move at half the wind speed

// A few stacked layers of fbm clouds that drift over the map.
export class Clouds {
  params = {
    show: true,
    cover: 0.4,
    scale: 0.045,   // noise frequency; smaller = bigger clouds
    alpha: 0.85,
    layers: 3.0,
  };
  // Above the top of the rain streaks (see rain.ts).
  readonly height = HEIGHT_SCALE * 1.5;
  // Wider than the map so the edge fade happens off the map.
  private size = WORLD_SIZE * 2.2;
  private offset = [0, 0];

  // CloudParams in common.wgsl: offset (xz), cover, scale, alpha, height, size, layers, spacing.
  // Bind this in any shader that calls cloudDensity().
  readonly uniforms: GPUBuffer;
  private pipeline: GPURenderPipeline;
  private bindGroup: GPUBindGroup;

  constructor() {
    this.uniforms = device.createBuffer({
      size: 40,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const module = shaderModule(cloudsWgsl);
    this.pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module, entryPoint: 'vs_main' },
      fragment: {
        module,
        entryPoint: 'fs_main',
        targets: [{
          format: canvasFormat,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
          },
        }],
      },
      primitive: { cullMode: 'none' },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: false,
        depthCompare: 'less',
      },
      multisample: { count: MSAA_SAMPLES },
    });

    this.bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: camera.uniformBuffer } },
        { binding: 1, resource: { buffer: this.uniforms } },
      ],
    });
  }

  addGui(gui: GUI) {
    const folder = gui.addFolder('Clouds');
    folder.add(this.params, 'show');
    folder.add(this.params, 'cover', 0, 1);
    folder.add(this.params, 'scale', 0.01, 0.1);
    folder.add(this.params, 'alpha', 0, 1);
    folder.add(this.params, 'layers', 1, 5).step(1);
  }

  update(dt: number, wind: Wind) {
    const p = this.params;
    // Accumulate the drift so changing the wind does not make the layer jump.
    const drift = dt * DRIFT * wind.params.strength;
    this.offset[0] += wind.dir[0] * drift;
    this.offset[1] += wind.dir[1] * drift;
    device.queue.writeBuffer(this.uniforms, 0, new Float32Array([
      ...this.offset,
      p.cover,
      p.scale,
      p.alpha,
      this.height,
      this.size,
      p.layers,
      spacing,
      0, // pad to the struct's 40 bytes
    ]));
  }

  draw(pass: GPURenderPassEncoder) {
    if (!this.params.show) return;
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.draw(6, this.params.layers);
  }
}
