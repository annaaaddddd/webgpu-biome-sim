import type { GUI } from 'lil-gui';
import type { Track } from '../core/track';
import { device, context, canvas, canvasFormat } from '../core/device';
import { camera } from '../core/camera';
import { GRID, WORLD_SIZE, HEIGHT_SCALE } from '../core/layout';

import heightfieldSrc from '../shaders/erosion/heightfield.wgsl?raw';
import terrainSrc from '../shaders/erosion/terrain.wgsl?raw';
import exportSrc from '../shaders/erosion/export.wgsl?raw';
import erosionPassesSrc from '../shaders/erosion/erosion.wgsl?raw';

// Two triangles per grid cell.
const indexCount = (GRID - 1) * (GRID - 1) * 6;
const cellSize = WORLD_SIZE / GRID;

// Erosion track: generate an fBm heightfield on the GPU, render it as a mesh.
export class ErosionTrack implements Track {
  // --- GPU resources ---
  // Both r32float in [0, 1]; these are what the ecosystem track reads.
  heightTex!: GPUTexture;
  moistureTex!: GPUTexture;
  heightfieldUniforms!: GPUBuffer;
  heightfieldPipeline!: GPUComputePipeline;
  heightfieldBindGroup!: GPUBindGroup;

  indexBuffer!: GPUBuffer;
  depthTex?: GPUTexture; // created on the first frame()
  terrainUniforms!: GPUBuffer;
  terrainPipeline!: GPURenderPipeline;
  terrainBindGroup!: GPUBindGroup;

  stateTex!: GPUTexture[];   // b: height, d: water depth, s: sediment
  fluxTex!: GPUTexture[];    // flux to 4 directions: left, right, top, bottom
  velocityTex!: GPUTexture;  // velocity (u, v)
  stateCur = 0;
  fluxCur = 0;

  exportPipeline!: GPUComputePipeline;
  exportBindGroups!: GPUBindGroup[];

  simUniforms!: GPUBuffer;
  rainPipeline!: GPUComputePipeline;
  rainBindGroups!: GPUBindGroup[];
  fluxPipeline!: GPUComputePipeline;
  fluxBindGroups!: GPUBindGroup[];
  waterPipeline!: GPUComputePipeline;
  waterBindGroups!: GPUBindGroup[];
  erodeDepositPipeline!: GPUComputePipeline;
  erodeDepositBindGroups!: GPUBindGroup[];
  transportPipeline!: GPUComputePipeline;
  transportBindGroups!: GPUBindGroup[];
  thermalPipeline!: GPUComputePipeline;
  thermalBindGroups!: GPUBindGroup[];

  // --- GUI params ---
  frequency = 4;
  octaves = 5;
  seed = 0;
  heightScale = HEIGHT_SCALE;
  view = 0; // 0 shaded, 1 height, 2 moisture (see fs_main in terrain.wgsl)

  // Simulation params. Uploaded every frame; field order must match `struct SimParams` in erosion.wgsl.
  sim = {
    dt: 0.05,        // simulation step, not frame time
    rainRate: 0.02,
    stepsPerFrame: 1,
    moistureScale: 2.0,
    pipeCoeff: 1.0,
    cellSize,
    Kc: 0.01,
    Ks: 0.3,
    Kd: 0.5,
    Ke: 0.5,
    minSlope: 0.05,
    maxDepth: 0.01,  // water depth at which erosion reaches full strength
    Kt: 0.5,
    talus: 0.01,     // height units; heights are in [0,1], so 0.01 is about 43 deg at 18x height scale
  };

  init(gui: GUI) {
    // Pull the camera back so the whole grid is in view.
    camera.distance = 150;
    camera.target[1] = 8;

    // --- Textures ---
    // Simulation state, ping-ponged between [0] and [1] (see `cur`).
    const stateAndFluxTexDesc: GPUTextureDescriptor = {
      size: [GRID, GRID],
      format: 'rgba32float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    };
    this.stateTex = [0, 1].map(() => device.createTexture(stateAndFluxTexDesc));
    this.fluxTex = [0, 1].map(() => device.createTexture(stateAndFluxTexDesc));

    this.velocityTex = device.createTexture({
      size: [GRID, GRID],
      format: 'rg32float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });

    // Outputs: written by the export pass, read by the terrain shaders.
    const terrainTexDesc: GPUTextureDescriptor = {
      size: [GRID, GRID],
      format: 'r32float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    };
    this.heightTex = device.createTexture(terrainTexDesc);
    this.moistureTex = device.createTexture(terrainTexDesc);

    // --- Buffers ---
    // Params: 3 f32 (frequency, octaves, seed), padded to 16 bytes.
    this.heightfieldUniforms = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
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

    // SimParams struct; grow the size as fields are added (16-byte multiples).
    this.simUniforms = device.createBuffer({
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // --- Pipelines ---
    // layout: 'auto' infers the bind group layout from the shader.
    this.heightfieldPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: heightfieldSrc }), entryPoint: 'main' },
    });

    this.exportPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: exportSrc }), entryPoint: 'main' },
    });

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

    const erosionModule = device.createShaderModule({ code: erosionPassesSrc });
    this.rainPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: erosionModule, entryPoint: 'rain' },
    });

    this.fluxPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: erosionModule, entryPoint: 'flux' },
    }); 

    this.waterPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: erosionModule, entryPoint: 'water' },
    }); 

    this.erodeDepositPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: erosionModule, entryPoint: 'erodeDeposit' },
    }); 

    this.transportPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: erosionModule, entryPoint: 'transport' },
    }); 

    this.thermalPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: erosionModule, entryPoint: 'thermal' },
    }); 

    // --- Bind groups ---
    this.heightfieldBindGroup = device.createBindGroup({
      layout: this.heightfieldPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.heightfieldUniforms } },
        { binding: 1, resource: this.stateTex[this.stateCur].createView() },
        { binding: 2, resource: this.fluxTex[this.fluxCur].createView() },
        { binding: 3, resource: this.velocityTex.createView() },
      ],
    });

    // One per ping-pong side; frame() picks exportBindGroups[cur].
    this.exportBindGroups = [0, 1].map((i) => device.createBindGroup({
      layout: this.exportPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.stateTex[i].createView() },
        { binding: 1, resource: this.heightTex.createView() },
        { binding: 2, resource: this.moistureTex.createView() },
        { binding: 3, resource: {buffer: this.simUniforms } },
      ],
    }));

    this.terrainBindGroup = device.createBindGroup({
      layout: this.terrainPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: camera.uniformBuffer } },
        { binding: 1, resource: this.heightTex.createView() },
        { binding: 2, resource: { buffer: this.terrainUniforms } },
        { binding: 3, resource: this.moistureTex.createView() },
      ],
    });

    this.rainBindGroups = [0, 1].map((i) => device.createBindGroup({
      layout: this.rainPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: {buffer: this.simUniforms } },
        { binding: 1, resource: this.stateTex[i].createView() },
        { binding: 2, resource: this.stateTex[1-i].createView() },
      ],
    }));

    this.fluxBindGroups = this.bindGroups4(this.fluxPipeline, (s, f) => [
      { binding: 0, resource: {buffer: this.simUniforms } },
      { binding: 1, resource: this.stateTex[s].createView() },
      { binding: 3, resource: this.fluxTex[f].createView() },
      { binding: 4, resource: this.fluxTex[1-f].createView() },
    ]);

    this.waterBindGroups = this.bindGroups4(this.waterPipeline, (s, f) => [
      { binding: 0, resource: {buffer: this.simUniforms } },
      { binding: 1, resource: this.stateTex[s].createView() },
      { binding: 2, resource: this.stateTex[1-s].createView() },
      { binding: 3, resource: this.fluxTex[f].createView() },
      { binding: 5, resource: this.velocityTex.createView() },
    ]);

    this.erodeDepositBindGroups = [0, 1].map((i) => device.createBindGroup({
      layout: this.erodeDepositPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: {buffer: this.simUniforms } },
        { binding: 1, resource: this.stateTex[i].createView() },
        { binding: 2, resource: this.stateTex[1-i].createView() },
        { binding: 6, resource: this.velocityTex.createView() },
      ],
    }));

    this.transportBindGroups = [0, 1].map((i) => device.createBindGroup({
      layout: this.transportPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: {buffer: this.simUniforms } },
        { binding: 1, resource: this.stateTex[i].createView() },
        { binding: 2, resource: this.stateTex[1-i].createView() },
        { binding: 6, resource: this.velocityTex.createView() },
      ],
    }));

    this.thermalBindGroups = [0, 1].map((i) => device.createBindGroup({
      layout: this.thermalPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: {buffer: this.simUniforms } },
        { binding: 1, resource: this.stateTex[i].createView() },
        { binding: 2, resource: this.stateTex[1-i].createView() },
      ],
    }));

    // --- GUI ---
    // Noise params need a re-run of the compute pass; heightScale is read every frame.
    const regen = () => this.generateHeightfield();
    gui.add(this, 'frequency', 0.5, 16).onChange(regen);
    gui.add(this, 'octaves', 1, 8).step(1).onChange(regen);
    gui.add(this, 'seed').onChange(regen);
    gui.add(this, 'heightScale', 1, 50);
    gui.add(this, 'view', { shaded: 0, height: 1, moisture: 2 });

    const simFolder = gui.addFolder('Simulation');
    simFolder.add(this.sim, 'dt', 0.001, 0.2);
    simFolder.add(this.sim, 'moistureScale', 1, 50);
    simFolder.add(this.sim, 'rainRate', 0, 2);
    simFolder.add(this.sim, 'stepsPerFrame', 0, 20).step(1);
    simFolder.add(this.sim, 'pipeCoeff', 0.1, 20);
    simFolder.add(this.sim, 'Kc', 0, 2.0);
    simFolder.add(this.sim, 'Ks', 0, 1.0);
    simFolder.add(this.sim, 'Kd', 0, 1.0);
    simFolder.add(this.sim, 'Ke', 0, 1);
    simFolder.add(this.sim, 'Kt', 0, 1);
    simFolder.add(this.sim, 'minSlope', 0.01, 0.1);
    simFolder.add(this.sim, 'maxDepth', 0.001, 0.05);
    simFolder.add(this.sim, 'talus', 0.001, 0.05);

    // --- Initial state ---
    this.generateHeightfield();
  }

  bindGroups4(pipeline: GPUComputePipeline, entries: (s: number, f: number) => GPUBindGroupEntry[]) {
    return [0, 1, 2, 3].map((k) => device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: entries(k >> 1, k & 1),
    }));
  }

  // One full-grid compute pass: one thread per texel, 8x8 workgroups
  // (matches @workgroup_size in the shaders).
  dispatch(encoder: GPUCommandEncoder, pipeline: GPUComputePipeline, bindGroup: GPUBindGroup) {
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(GRID / 8), Math.ceil(GRID / 8));
    pass.end();
  }

  // A simulation pass: reads side `cur`, writes the other side, then flips.
  step(encoder: GPUCommandEncoder, pipeline: GPUComputePipeline, bindGroup: GPUBindGroup, which: 'state' | 'flux') {
    this.dispatch(encoder, pipeline, bindGroup);

    if (which === 'state') this.stateCur = 1 - this.stateCur;
    else this.fluxCur = 1 - this.fluxCur;
  }

  // Upload noise params and regenerate the heightfield into side 0.
  // Resets the ping-pong so the simulation restarts from this terrain.
  generateHeightfield() {
    this.stateCur = 0;
    this.fluxCur = 0;
    device.queue.writeBuffer(this.heightfieldUniforms, 0, new Float32Array([this.frequency, this.octaves, this.seed]));
    const encoder = device.createCommandEncoder();
    this.dispatch(encoder, this.heightfieldPipeline, this.heightfieldBindGroup);
    device.queue.submit([encoder.finish()]);
  }

  // Copy the current state into the height/moisture textures. Reads only, no flip.
  export(encoder: GPUCommandEncoder) {
    this.dispatch(encoder, this.exportPipeline, this.exportBindGroups[this.stateCur]);
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
    device.queue.writeBuffer(this.simUniforms, 0, new Float32Array([
      this.sim.dt, 
      this.sim.rainRate, 
      this.sim.moistureScale, 
      this.sim.pipeCoeff, 
      cellSize,
      this.sim.Kc,
      this.sim.Ks,
      this.sim.Kd,
      this.sim.Ke,
      this.sim.Kt,
      this.sim.minSlope,
      this.sim.maxDepth,
      this.sim.talus,
    ]));

    for (let i = 0; i < this.sim.stepsPerFrame; i++) {
      this.step(encoder, this.rainPipeline, this.rainBindGroups[this.stateCur], 'state');
      this.step(encoder, this.fluxPipeline, this.fluxBindGroups[this.stateCur * 2 + this.fluxCur], 'flux');
      this.step(encoder, this.waterPipeline, this.waterBindGroups[this.stateCur * 2 + this.fluxCur], 'state');
      this.step(encoder, this.erodeDepositPipeline, this.erodeDepositBindGroups[this.stateCur], 'state');
      this.step(encoder, this.transportPipeline, this.transportBindGroups[this.stateCur], 'state');
      this.step(encoder, this.thermalPipeline, this.thermalBindGroups[this.stateCur], 'state');
    }
    this.export(encoder);

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
