import { device, canvas, canvasFormat } from './device';

// Pass a byte size for an empty buffer, or initial data to upload.
export function createBuffer(usage: GPUBufferUsageFlags, init: number | BufferSource) {
  if (typeof init === 'number') return device.createBuffer({ size: init, usage });
  const buffer = device.createBuffer({ size: init.byteLength, usage: usage | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(buffer, 0, init);
  return buffer;
}

export function createGridTexture(size: number, format: GPUTextureFormat, usage: GPUTextureUsageFlags) {
  return device.createTexture({ size: [size, size], format, usage });
}

export function createBindGroup(pipeline: GPUPipelineBase, resources: GPUBindingResource[]) {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: resources.map((resource, binding) => ({ binding, resource })),
  });
}

export function dispatchGrid(encoder: GPUCommandEncoder, pipeline: GPUComputePipeline, bindGroup: GPUBindGroup, size: number) {
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(size / 8, size / 8);
  pass.end();
}

// MSAA color + depth attachments that follow the canvas size.
export class RenderTargets {
  sampleCount = 4;
  color!: GPUTextureView;
  depth!: GPUTextureView;
  private textures: GPUTexture[] = [];
  private width = 0;
  private height = 0;

  update() {
    if (canvas.width === this.width && canvas.height === this.height) return;
    this.width = canvas.width;
    this.height = canvas.height;
    this.textures.forEach((t) => t.destroy());
    this.textures = [canvasFormat, 'depth24plus' as const].map((format) =>
      device.createTexture({
        size: [this.width, this.height],
        format,
        sampleCount: this.sampleCount,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      }),
    );
    [this.color, this.depth] = this.textures.map((t) => t.createView());
  }
}
