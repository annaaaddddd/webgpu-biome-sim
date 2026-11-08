import type { GUI } from 'lil-gui';
import type { Track } from '../core/track';
import { device, context, canvas, canvasFormat } from '../core/device';
import { camera } from '../core/camera';
import heightfieldSrc from '../shaders/erosion/heightfield.wgsl?raw';
import terrainSrc from '../shaders/erosion/terrain.wgsl?raw';

// Heightfield resolution (N x N texels, one mesh vertex per texel).
const N = 256;
// Two triangles per grid cell.
const indexCount = (N - 1) * (N - 1) * 6;

// Erosion track: generate an fBm heightfield on the GPU, render it as a mesh.
export class ErosionTrack implements Track {
  // --- GPU resources ---
  heightTex!: GPUTexture;
  heightfieldUniforms!: GPUBuffer;
  heightfieldPipeline!: GPUComputePipeline;
  heightfieldBindGroup!: GPUBindGroup;

  indexBuffer!: GPUBuffer;
  depthTex!: GPUTexture;
  terrainUniforms!: GPUBuffer;
  terrainPipeline!: GPURenderPipeline;
  terrainBindGroup!: GPUBindGroup;

  // --- GUI params ---
  frequency = 4;
  amplitude = 1;
  octaves = 5;
  seed = 0;
  heightScale = 40;

  init(gui: GUI) {

    // Pull the camera back so the whole 256-unit grid is in view.
    camera.distance = 300;
    camera.target[1] = 20;

    // --- Heightfield compute ---
    // r32float: written by the compute pass, read by the terrain vertex shader.
    this.heightTex = device.createTexture({
      size: [N, N],
      format: 'r32float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });

    // Params: 4 f32 = 16 bytes (frequency, amplitude, octaves, seed).
    this.heightfieldUniforms = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.heightfieldPipeline = device.createComputePipeline({
      layout: 'auto', // infer the bind group layout from the shader
      compute: { module: device.createShaderModule({ code: heightfieldSrc }), entryPoint: 'main' },
    });

    this.heightfieldBindGroup = device.createBindGroup({
      layout: this.heightfieldPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.heightfieldUniforms } },
        { binding: 1, resource: this.heightTex.createView() },
      ],
    });

    // --- Terrain render ---
    // Recreated in frame() when the canvas size changes.
    this.depthTex = device.createTexture({
      size: [canvas.width, canvas.height],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    // Only heightScale (4 bytes) is used; uniform buffers are padded to 16.
    this.terrainUniforms = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.indexBuffer = device.createBuffer({
      size: indexCount * 4,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });

    // Grid indices. Vertex i = z * N + x; each cell is split into two
    // triangles: (i, i+1, i+N) and (i+1, i+N+1, i+N).
    const indices = new Uint32Array(indexCount);
    let k = 0;
    for (let z = 0; z < N - 1; z++) {
      for (let x = 0; x < N - 1; x++) {
        const i = z * N + x;
        indices[k++] = i;      indices[k++] = i + 1;     indices[k++] = i + N;
        indices[k++] = i + 1;  indices[k++] = i + N + 1; indices[k++] = i + N;
      }
    }
    device.queue.writeBuffer(this.indexBuffer, 0, indices);

    const module = device.createShaderModule({ code: terrainSrc });
    this.terrainPipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module, entryPoint: 'vs_main', constants: { N } }, // sets `override N`
      fragment: { module, entryPoint: 'fs_main', targets: [{ format: canvasFormat }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: true,
        depthCompare: 'less'
      },
    });

    this.terrainBindGroup = device.createBindGroup({
      layout: this.terrainPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: camera.uniformBuffer } },
        { binding: 1, resource: this.heightTex.createView() },
        { binding: 2, resource: { buffer: this.terrainUniforms } }
      ],
    });


    // --- GUI ---
    // Noise params need a re-run of the compute pass; heightScale is read every frame.
    const regen = () => this.generateHeightfield();
    gui.add(this, 'frequency', 0.5, 16).onChange(regen);
    gui.add(this, 'amplitude', 0.5, 3).onChange(regen);
    gui.add(this, 'octaves', 1, 8).step(1).onChange(regen);
    gui.add(this, 'seed').onChange(regen);
    gui.add(this, 'heightScale', 1, 100);

    this.generateHeightfield();

  }

  // Upload params and run the heightfield compute pass once (not per frame).
  generateHeightfield() {
    const params = new Float32Array([
      this.frequency,
      this.amplitude,
      this.octaves,
      this.seed,
    ]);
    device.queue.writeBuffer(this.heightfieldUniforms, 0, params);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.heightfieldPipeline);
    pass.setBindGroup(0, this.heightfieldBindGroup);
    // One thread per texel, 8x8 workgroups (matches @workgroup_size).
    pass.dispatchWorkgroups(
      Math.ceil(N / 8),
      Math.ceil(N / 8),
    );
    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  frame(encoder: GPUCommandEncoder, _dt: number) {
    // Depth texture must match the canvas size; rebuild it after a resize.
    if (!this.depthTex || this.depthTex.width !== canvas.width || this.depthTex.height !== canvas.height) {
      this.depthTex?.destroy();
      this.depthTex = device.createTexture({
        size: [canvas.width, canvas.height],
        format: 'depth24plus',
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
    }
    device.queue.writeBuffer(this.terrainUniforms, 0, new Float32Array([this.heightScale]));
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.1, g: 0.2, b: 0.3, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
      depthStencilAttachment: {
        view: this.depthTex.createView(),
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });
    pass.setPipeline(this.terrainPipeline);
    pass.setBindGroup(0, this.terrainBindGroup);
    pass.setIndexBuffer(this.indexBuffer, 'uint32');
    pass.drawIndexed(indexCount); // no vertex buffer, see vs_main
    pass.end();
  }
}
