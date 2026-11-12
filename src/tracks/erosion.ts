import type { GUI } from 'lil-gui';
import type { Track } from '../core/track';
import { device, context, canvas, canvasFormat } from '../core/device';
import { camera } from '../core/camera';
import { GRID, WORLD_SIZE, HEIGHT_SCALE } from '../core/layout';
import { PassTimer } from '../sim/erosion_bench';

import terrainSrc from '../shaders/erosion/terrain.wgsl?raw';
import { ErosionSim } from '../sim/erosion_sim';

// Two triangles per grid cell.
const indexCount = (GRID - 1) * (GRID - 1) * 6;

// Erosion track: runs ErosionSim and draws its height and moisture output as a
// terrain mesh, with height/moisture debug views.
export class ErosionTrack implements Track {
  erosionSim!: ErosionSim;

  // --- GPU resources ---
  indexBuffer!: GPUBuffer;
  depthTex?: GPUTexture; // created on the first frame()
  terrainUniforms!: GPUBuffer;
  terrainPipeline!: GPURenderPipeline;
  terrainBindGroup!: GPUBindGroup;

  heightScale = HEIGHT_SCALE;
  view = 0; // 0 shaded, 1 height, 2 moisture (see fs_main in terrain.wgsl)

  init(gui: GUI) {
    // Pull the camera back so the whole grid is in view.
    camera.distance = 150;
    camera.target[1] = 8;

    // Shared height/moisture textures: ErosionSim writes them in its export
    // pass, the terrain shader reads them.
    const terrainTexDesc: GPUTextureDescriptor = {
      size: [GRID, GRID],
      format: 'r32float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    };
    const heightTex = device.createTexture(terrainTexDesc);
    const moistureTex = device.createTexture(terrainTexDesc);

    this.erosionSim = new ErosionSim(heightTex, moistureTex);

    // TerrainUniforms { heightScale, view }: 8 bytes, padded to 16.
    this.terrainUniforms = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.indexBuffer = device.createBuffer({
      size: indexCount * 4,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });

    // Grid indices. Vertex i = z * GRID + x; each cell is split into two
    // triangles: (i, i+1, i+GRID) and (i+1, i+GRID+1, i+GRID).
    const indices = new Uint32Array(indexCount);
    let k = 0;
    for (let z = 0; z < GRID - 1; z++) {
      for (let x = 0; x < GRID - 1; x++) {
        const i = z * GRID + x;
        indices[k++] = i;      indices[k++] = i + 1;        indices[k++] = i + GRID;
        indices[k++] = i + 1;  indices[k++] = i + GRID + 1; indices[k++] = i + GRID;
      }
    }
    device.queue.writeBuffer(this.indexBuffer, 0, indices);

    const module = device.createShaderModule({ code: terrainSrc });
    this.terrainPipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module, entryPoint: 'vs_main', constants: { GRID, WORLD_SIZE } }, // sets the `override`s
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
        { binding: 1, resource: this.erosionSim.height.createView() },
        { binding: 2, resource: { buffer: this.terrainUniforms } },
        { binding: 3, resource: this.erosionSim.moisture.createView() },
      ],
    });

    // --- GUI ---
    gui.add(this, 'heightScale', 1, 50);
    gui.add(this, 'view', { shaded: 0, height: 1, moisture: 2 });
    this.erosionSim.addGui(gui);

    // &steps=N keeps the GPU busy so its clock does not drift during a bench run.
    const steps = new URLSearchParams(location.search).get('steps');
    if (steps !== null) this.erosionSim.sim.stepsPerFrame = Number(steps);
    this.erosionSim.bench = PassTimer.fromUrl(ErosionSim.PASS_NAMES, {
      stepPasses: 6, // one sim step; export runs once per frame
      label: `erosion ${GRID}x${GRID}, ${this.erosionSim.sim.stepsPerFrame} steps/frame`,
      file: `erosion_timing_${GRID}.csv`,
    });
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
    device.queue.writeBuffer(this.terrainUniforms, 0, new Float32Array([this.heightScale, this.view]));

    // Sim steps + export, then draw the result in the same encoder.
    this.erosionSim.frame(encoder);

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
