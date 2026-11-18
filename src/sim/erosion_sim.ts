import type { GUI } from 'lil-gui';
import { device } from '../core/device';
import { GRID, WORLD_SIZE } from '../core/layout';
import { PassTimer } from './erosion_bench';
import { MassProbe } from './erosion_probe';
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
  readonly moisture: GPUTexture; // shared output: height/noise baseline + wetTex, written by export()

  bench?: PassTimer;
  probe?: MassProbe;     // ?probe: logs grid totals once a second (see erosion_probe.ts)
  dirty = true;          // next frame() calls reset() to regenerate the heightfield
  erode = true;          // false freezes the terrain: rain, flow and evaporation keep running
  erosionSteps = 0;      // sim steps run with erosion on since the last reset; shows the terrain's age
  seed = 0;

  // Uploaded to SimParams every frame; GUI-tunable via addGui().
  sim = {
    dt: 0.05,        // simulation step, not frame time
    rainRate: 0.0003, // mean water depth settles at rainRate / Ke
    stepsPerFrame: 2,
    moistureScale: 10.0,
    pipeCoeff: 1.0,
    cellSize,
    Kc: 0.001,       // sediment capacity
    Ks: 0.1,
    Kd: 0.5,
    Ke: 0.03,        // water lasts ~1 / (Ke * dt) steps, long enough to reach the valleys
    minSlope: 0.05,
    maxDepth: 0.01,  // water depth at which erosion reaches full strength
    Kt: 0.05,
    talus: 0.025,    // height units; heights are in [0,1], so 0.025 is about 66 deg at 18x height scale
    moistureTau: 2.0, // time constant for moisture to follow water depth (see export.wgsl)
    baseline: 1.0,    // weight of the height/noise moisture baseline; 0 = water only
  };

  stateTex!: GPUTexture[];   // b: height, d: water depth, s: sediment
  fluxTex!: GPUTexture[];    // flux to 4 directions: left, right, top, bottom
  velocityTex!: GPUTexture;  // velocity (u, v)
  wetTex!: GPUTexture;       // EMA of water depth, persists across frames
  stateCur = 0;
  fluxCur = 0;

  heightfieldUniforms!: GPUBuffer;
  heightfieldPipeline!: GPUComputePipeline;
  heightfieldBindGroup!: GPUBindGroup;

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

  exportPipeline!: GPUComputePipeline;
  exportBindGroups!: GPUBindGroup[];

  constructor(height: GPUTexture, moisture: GPUTexture) {
    this.height = height;
    this.moisture = moisture;

    // &steps=N overrides stepsPerFrame; a bench run uses it to keep the GPU busy.
    const steps = new URLSearchParams(location.search).get('steps');
    if (steps !== null) this.sim.stepsPerFrame = Number(steps);
    this.probe = MassProbe.fromUrl(GRID);

    // --- Textures ---
    // Simulation state, ping-ponged between [0] and [1]; stateCur/fluxCur mark
    // the side holding the latest data.
    const stateAndFluxTexDesc: GPUTextureDescriptor = {
      size: [GRID, GRID],
      format: 'rgba32float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    };
    this.stateTex = [0, 1].map(() => device.createTexture(stateAndFluxTexDesc));
    this.fluxTex = [0, 1].map(() => device.createTexture(stateAndFluxTexDesc));

    this.velocityTex = device.createTexture({
      size: [GRID, GRID],
      format: 'rg32float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });

    this.wetTex = device.createTexture({
      size: [GRID, GRID],
      format: 'r32float',
      usage: GPUTextureUsage.STORAGE_BINDING,
    });

    // --- Buffers ---
    // Params: seed, padded to 16 bytes.
    this.heightfieldUniforms = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // SimParams in erosion.wgsl and export.wgsl: 17 f32 = 68 bytes, rounded up
    // to 80 so there is room for 3 more. Grow in 16-byte steps after that.
    this.simUniforms = device.createBuffer({
      size: 80,
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
    // Writes side 0 of state/flux and clears wetTex; reset() sets
    // stateCur/fluxCur back to 0 to match.
    this.heightfieldBindGroup = device.createBindGroup({
      layout: this.heightfieldPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.heightfieldUniforms } },
        { binding: 1, resource: this.stateTex[this.stateCur].createView() },
        { binding: 2, resource: this.fluxTex[this.fluxCur].createView() },
        { binding: 3, resource: this.velocityTex.createView() },
        { binding: 4, resource: this.wetTex.createView() },
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
        { binding: 2, resource: this.wetTex.createView() },
        { binding: 3, resource: {buffer: this.simUniforms } },
        { binding: 4, resource: moisture.createView() },
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

  // `seed` is skipped when the caller already has its own seed control. Such a
  // caller passes a folder of its own, so the controls go straight into it.
  addGui(gui: GUI, seed = true) {
    if (seed) gui.add(this, 'seed').onChange(() => (this.dirty = true));

    const simFolder = seed ? gui.addFolder('Simulation') : gui;
    // One button that reads as the action it will take.
    const toggle = simFolder.add({ toggle: () => {
      this.erode = !this.erode;
      toggle.name(this.erode ? 'pause erosion' : 'start erosion');
    } }, 'toggle').name(this.erode ? 'pause erosion' : 'start erosion');
    simFolder.add(this, 'erosionSteps').name('steps eroded').listen().disable();

    // Labels say what a control does; the symbol from the papers and the
    // shader stays in brackets so the two can be matched up.
    simFolder.add(this.sim, 'stepsPerFrame', 0, 20).step(1).name('steps per frame');
    simFolder.add(this.sim, 'dt', 0.001, 0.2).name('time step (dt)');
    simFolder.add(this.sim, 'rainRate', 0, 0.005).name('rain rate');
    simFolder.add(this.sim, 'Ke', 0, 0.2).name('evaporation (Ke)');
    simFolder.add(this.sim, 'pipeCoeff', 0.1, 20).name('flow speed');

    const hydraulic = simFolder.addFolder('Hydraulic erosion').close();
    hydraulic.add(this.sim, 'Kc', 0, 0.005).name('capacity (Kc)');
    hydraulic.add(this.sim, 'Ks', 0, 1.0).name('pick-up rate (Ks)');
    hydraulic.add(this.sim, 'Kd', 0, 1.0).name('drop rate (Kd)');
    hydraulic.add(this.sim, 'minSlope', 0.01, 0.1).name('min slope');
    hydraulic.add(this.sim, 'maxDepth', 0.001, 0.05).name('full-strength depth');

    const thermal = simFolder.addFolder('Thermal erosion').close();
    thermal.add(this.sim, 'Kt', 0, 1).name('slide rate (Kt)');
    thermal.add(this.sim, 'talus', 0.001, 0.05).name('max step (talus)');

    const moisture = simFolder.addFolder('Moisture').close();
    moisture.add(this.sim, 'moistureScale', 1, 50).name('water weight');
    moisture.add(this.sim, 'moistureTau', 0.1, 20).name('lag (sim s)');
    moisture.add(this.sim, 'baseline', 0.0, 1.0).name('terrain baseline');
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

  // Write height and moisture into the shared textures and update wetTex.
  // Reads state only, no flip.
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
    this.erosionSteps = 0;
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
      this.seed,
      this.sim.baseline,
    ]));

    for (let i = 0; i < this.sim.stepsPerFrame; i++) {
      // Only the first step of the frame is timed, so later steps do not overwrite it.
      const slot = (k: number) => (i === 0 ? k : undefined);
      this.step(encoder, this.rainPipeline, this.rainBindGroups[this.stateCur], 'state', slot(0));
      this.step(encoder, this.fluxPipeline, this.fluxBindGroups[this.stateCur * 2 + this.fluxCur], 'flux', slot(1));
      this.step(encoder, this.waterPipeline, this.waterBindGroups[this.stateCur * 2 + this.fluxCur], 'state', slot(2));
      // Erosion works on geological time and the ecosystem on seasons, so the
      // terrain is shaped first and then frozen; the water passes carry on.
      if (this.erode) this.step(encoder, this.erodeDepositPipeline, this.erodeDepositBindGroups[this.stateCur], 'state', slot(3));
      this.step(encoder, this.transportPipeline, this.transportBindGroups[this.stateCur], 'state', slot(4));
      if (this.erode) this.step(encoder, this.thermalPipeline, this.thermalBindGroups[this.stateCur], 'state', slot(5));
      if (this.erode) this.erosionSteps++;
    }
    this.export(encoder);
    this.probe?.frame(encoder, this.stateTex[this.stateCur]);

    if (this.sim.stepsPerFrame > 0) this.bench?.endFrame(encoder);
  }
}
