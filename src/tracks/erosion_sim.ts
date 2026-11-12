import type { GUI } from 'lil-gui';
import { device } from '../core/device';
import { GRID, WORLD_SIZE } from '../core/layout';
import { PassTimer } from './erosion_bench';
import { shaderModule } from '../core/gpu';

import erosionPassesSrc from '../shaders/erosion/erosion.wgsl?raw';
import heightfieldSrc from '../shaders/erosion/heightfield.wgsl?raw';
import exportSrc from '../shaders/erosion/export.wgsl?raw';

const cellSize = WORLD_SIZE / GRID;

// Hydraulic + thermal erosion on a GRID x GRID heightfield (Mei et al. 2007,
// Jako & Toth 2011). Owns its internal state textures; each frame() runs
// stepsPerFrame sim steps and then exports into the caller's shared height and
// moisture textures (r32float, see core/layout.ts).
export class ErosionSim {
  // Timer slot names; the index is the `slot` passed to dispatch().
  static readonly PASS_NAMES = ['rain', 'flux', 'water', 'erodeDeposit', 'transport', 'thermal', 'export'];
  readonly height: GPUTexture;   // shared output, written by export()
  readonly moisture: GPUTexture; // shared output, smoothed over time by export()

  bench?: PassTimer;
  dirty = true;          // next frame() calls reset() to regenerate the heightfield
  seed = 0;
  // Uploaded to SimParams every frame; GUI-tunable via addGui().
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
    moistureTau: 2.0, // time constant for moisture to follow water depth (see export.wgsl)
  };

  heightfieldUniforms!: GPUBuffer;
  heightfieldPipeline!: GPUComputePipeline;
  heightfieldBindGroup!: GPUBindGroup;

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

  constructor(height: GPUTexture, moisture: GPUTexture) {
    this.height = height;
    this.moisture = moisture;

    // --- Textures ---
    // Simulation state, ping-ponged between [0] and [1]; stateCur/fluxCur mark
    // the side holding the latest data.
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

    // --- Buffers ---
    // Params: seed, padded to 16 bytes.
    this.heightfieldUniforms = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // SimParams in erosion.wgsl and export.wgsl: 14 f32 = 56 bytes, padded to 64.
    // Grow in 16-byte steps if fields are added.
    this.simUniforms = device.createBuffer({
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // --- Pipelines ---
    // layout: 'auto' infers the bind group layout from the shader.
    this.heightfieldPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: shaderModule(heightfieldSrc), entryPoint: 'main' },
    });

    this.exportPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: shaderModule(exportSrc), entryPoint: 'main' },
    });

    const erosionModule = shaderModule(erosionPassesSrc);
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
    // Writes side 0 of state/flux and clears moisture; reset() sets
    // stateCur/fluxCur back to 0 to match.
    this.heightfieldBindGroup = device.createBindGroup({
      layout: this.heightfieldPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.heightfieldUniforms } },
        { binding: 1, resource: this.stateTex[this.stateCur].createView() },
        { binding: 2, resource: this.fluxTex[this.fluxCur].createView() },
        { binding: 3, resource: this.velocityTex.createView() },
        { binding: 4, resource: moisture.createView() },
      ],
    });

    // Indexed by the side to read: export() picks exportBindGroups[stateCur].
    // The rain/erodeDeposit/transport/thermal groups below follow the same
    // pattern: read side i, write side 1 - i.
    this.exportBindGroups = [0, 1].map((i) => device.createBindGroup({
      layout: this.exportPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.stateTex[i].createView() },
        { binding: 1, resource: height.createView() },
        { binding: 2, resource: moisture.createView() },
        { binding: 3, resource: {buffer: this.simUniforms } },
      ],
    }));

    this.rainBindGroups = [0, 1].map((i) => device.createBindGroup({
      layout: this.rainPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: {buffer: this.simUniforms } },
        { binding: 1, resource: this.stateTex[i].createView() },
        { binding: 2, resource: this.stateTex[1-i].createView() },
      ],
    }));

    // flux and water read both state and flux, so they need all 4 side combos.
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
  }

  // `seed` is skipped when the caller already has its own seed control.
  addGui(gui: GUI, seed = true) {
    if (seed) gui.add(this, 'seed').onChange(() => (this.dirty = true));

    const simFolder = gui.addFolder('Simulation');
    simFolder.add(this.sim, 'dt', 0.001, 0.2);
    simFolder.add(this.sim, 'moistureScale', 1, 50);
    simFolder.add(this.sim, 'moistureTau', 0.1, 20);
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
  }

  // Bind groups for every (state side s, flux side f) pair, indexed by s * 2 + f.
  bindGroups4(pipeline: GPUComputePipeline, entries: (s: number, f: number) => GPUBindGroupEntry[]) {
    return [0, 1, 2, 3].map((k) => device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: entries(k >> 1, k & 1),
    }));
  }

  // One full-grid compute pass: one thread per texel, 8x8 workgroups
  // (matches @workgroup_size in the shaders).
  // `slot` (index into PASS_NAMES) times the pass when a bench run is active.
  dispatch(encoder: GPUCommandEncoder, pipeline: GPUComputePipeline, bindGroup: GPUBindGroup, slot?: number) {
    const pass = encoder.beginComputePass({ timestampWrites: this.bench?.writes(slot) });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(GRID / 8), Math.ceil(GRID / 8));
    pass.end();
  }

  // A simulation pass: reads the current side of `which`, writes the other
  // side, then flips stateCur or fluxCur.
  step(encoder: GPUCommandEncoder, pipeline: GPUComputePipeline, bindGroup: GPUBindGroup, which: 'state' | 'flux', slot?: number) {
    this.dispatch(encoder, pipeline, bindGroup, slot);

    if (which === 'state') this.stateCur = 1 - this.stateCur;
    else this.fluxCur = 1 - this.fluxCur;
  }

  // Copy the current state into the height/moisture textures. Reads only, no flip.
  export(encoder: GPUCommandEncoder) {
    this.dispatch(encoder, this.exportPipeline, this.exportBindGroups[this.stateCur], 6); // 6 = 'export'
  }

  // Fill state with fresh fBm terrain (no water, sediment, flux or velocity).
  reset(encoder: GPUCommandEncoder) {
    this.stateCur = 0;
    this.fluxCur = 0;
    device.queue.writeBuffer(this.heightfieldUniforms, 0, new Float32Array([this.seed]));
    this.dispatch(encoder, this.heightfieldPipeline, this.heightfieldBindGroup);
    this.dirty = false;
  }

  // Encode this frame's sim steps plus one export. Call once per frame.
  frame(encoder: GPUCommandEncoder) {
    if (this.dirty) this.reset(encoder);
    this.bench?.beginFrame();

    // Field order must match SimParams.
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
      this.sim.moistureTau,
      this.sim.dt * this.sim.stepsPerFrame,
    ]));

    for (let i = 0; i < this.sim.stepsPerFrame; i++) {
      // Only the first step of the frame is timed, so later steps do not overwrite it.
      const slot = (k: number) => (i === 0 ? k : undefined);
      this.step(encoder, this.rainPipeline, this.rainBindGroups[this.stateCur], 'state', slot(0));
      this.step(encoder, this.fluxPipeline, this.fluxBindGroups[this.stateCur * 2 + this.fluxCur], 'flux', slot(1));
      this.step(encoder, this.waterPipeline, this.waterBindGroups[this.stateCur * 2 + this.fluxCur], 'state', slot(2));
      this.step(encoder, this.erodeDepositPipeline, this.erodeDepositBindGroups[this.stateCur], 'state', slot(3));
      this.step(encoder, this.transportPipeline, this.transportBindGroups[this.stateCur], 'state', slot(4));
      this.step(encoder, this.thermalPipeline, this.thermalBindGroups[this.stateCur], 'state', slot(5));
    }
    this.export(encoder);

    if (this.sim.stepsPerFrame > 0) this.bench?.endFrame(encoder);
  }
}
