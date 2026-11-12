import type { GUI } from 'lil-gui';
import { device, canvasFormat } from '../core/device';
import { camera } from '../core/camera';
import { HEIGHT_SCALE, MSAA_SAMPLES } from '../core/layout';
import { shaderModule } from '../core/gpu';
import rainWgsl from '../shaders/weather/rain.wgsl?raw';

// Matches the rainRate slider's upper bound in erosion_sim.ts; full slider = maxDrops.
const RAIN_RATE_MAX = 0.005;

export class Rain {
  maxDrops = 20000;
  params = {
    speed: 25,
    length: 1.2,
    width: 0.04,
    alpha: 0.35,
    top: HEIGHT_SCALE * 1.3,
  };
  private time = 0;
  private count = 0;

  private uniforms: GPUBuffer;   // time, speed, length, width, alpha, top + pad
  private pipeline: GPURenderPipeline;
  private bindGroup: GPUBindGroup;

  constructor(height: GPUTexture) {
    this.uniforms = device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const module = shaderModule(rainWgsl);
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
        { binding: 1, resource: height.createView() },
        { binding: 2, resource: { buffer: this.uniforms } },
      ],
    });
  }

  addGui(gui: GUI) {
    const folder = gui.addFolder('Rain');
    folder.add(this.params, 'speed', 5, 60);
    folder.add(this.params, 'length', 0.2, 4);
    folder.add(this.params, 'width', 0.01, 0.2);
    folder.add(this.params, 'alpha', 0, 1);
  }

  update(dt: number, rainRate: number) {
    this.time += dt;
    this.count = Math.round(Math.min(rainRate / RAIN_RATE_MAX, 1) * this.maxDrops);
    const p = this.params;
    device.queue.writeBuffer(this.uniforms, 0, new Float32Array([
      this.time,
      p.speed,
      p.length,
      p.width,
      p.alpha,
      p.top,
    ]));
  }

  draw(pass: GPURenderPassEncoder) {
    if (this.count === 0) return;
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.draw(6, this.count);
  }
}
