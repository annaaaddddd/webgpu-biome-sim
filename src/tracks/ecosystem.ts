import type { GUI } from 'lil-gui';
import { vec3 } from 'wgpu-matrix';
import type { Track } from '../core/track';
import { device, context, canvasFormat } from '../core/device';
import { camera } from '../core/camera';
import {
  GRID,
  WORLD_SIZE,
  HEIGHT_SCALE,
  MAX_SPECIES,
  TERRAIN_FORMAT,
  STATE_FORMAT,
  PLANT_INSTANCE_BYTES,
  DRAW_ARGS_BYTES,
  FLOWER_BUDGET,
  FLOWER_VERTS,
  SKY_COLOR,
  WATER_LEVEL,
  MSAA_SAMPLES,
} from '../core/layout';
import { createBuffer, createGridTexture, createBindGroup, dispatchGrid, RenderTargets } from '../core/gpu';
import { terrainIndices } from './meshes';
import { Flowers } from './flowers';
import commonWgsl from '../shaders/common.wgsl?raw';
import terrainStubWgsl from '../shaders/terrain_stub.wgsl?raw';
import ecosystemWgsl from '../shaders/ecosystem.wgsl?raw';
import compactWgsl from '../shaders/compact.wgsl?raw';
import terrainWgsl from '../shaders/terrain.wgsl?raw';
import plantsWgsl from '../shaders/plants.wgsl?raw';

const VIEWS = ['shaded', 'height', 'moisture', 'slope', 'species', 'viability'];

// per species: height min/max, moisture min/max, max slope, ramp softness
const SPECIES = [
  [0.06, 0.4, 0.5, 1.0, 0.45, 0.1],
  [0.05, 0.4, 0.0, 0.45, 0.5, 0.1],
  [0.35, 0.7, 0.3, 1.0, 0.6, 0.08],
  [0.6, 0.9, 0.0, 0.7, 0.7, 0.08],
];

const prelude = `
const GRID = ${GRID}u;
const WORLD_SIZE = ${WORLD_SIZE}f;
const HEIGHT_SCALE = ${HEIGHT_SCALE}f;
const WATER_LEVEL = ${WATER_LEVEL}f;
const SKY = vec3f(${SKY_COLOR.join(', ')});
const MAX_SPECIES = ${MAX_SPECIES}u;
${Object.entries(FLOWER_BUDGET)
  .map(([k, v]) => `const ${k} = ${v}u;`)
  .join('\n')}
`;

const B = GPUBufferUsage;
const T = GPUTextureUsage;

function shader(code: string) {
  return device.createShaderModule({ code: prelude + commonWgsl + code });
}

function computePipeline(code: string) {
  return device.createComputePipeline({ layout: 'auto', compute: { module: shader(code) } });
}

function renderPipeline(code: string, buffers: GPUVertexBufferLayout[] = []) {
  const module = shader(code);
  return device.createRenderPipeline({
    layout: 'auto',
    vertex: { module, buffers },
    fragment: { module, targets: [{ format: canvasFormat }] },
    primitive: { cullMode: 'none' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    multisample: { count: MSAA_SAMPLES },
  });
}

export class EcosystemTrack implements Track {
  settings = {
    seed: 1,
    speciesCount: MAX_SPECIES,
    spawnRate: 0.5,
    mortality: 0.04,
    lifespan: 80,
    play: false,
    stepsPerSecond: 10,
    view: 'shaded',
    showPlants: true,
  };

  private terrainDirty = true;
  private stateDirty = true;
  private stepIndex = 0;
  private stepBudget = 0;
  private targets = new RenderTargets();
  private skyClear = { r: SKY_COLOR[0] ** (1 / 2.2), g: SKY_COLOR[1] ** (1 / 2.2), b: SKY_COLOR[2] ** (1 / 2.2), a: 1 };

  private height = createGridTexture(GRID, TERRAIN_FORMAT, T.TEXTURE_BINDING | T.STORAGE_BINDING);
  private moisture = createGridTexture(GRID, TERRAIN_FORMAT, T.TEXTURE_BINDING | T.STORAGE_BINDING);
  private state = createGridTexture(GRID, STATE_FORMAT, T.TEXTURE_BINDING | T.COPY_DST);
  private stateNext = createGridTexture(GRID, STATE_FORMAT, T.STORAGE_BINDING | T.COPY_SRC);

  private stubParams = createBuffer(B.UNIFORM | B.COPY_DST, 16);
  private simData = new ArrayBuffer(32 + 32 * MAX_SPECIES);
  private simParams = createBuffer(B.UNIFORM | B.COPY_DST, this.simData.byteLength);
  private drawParams = createBuffer(B.UNIFORM | B.COPY_DST, 16);
  private instances = createBuffer(B.STORAGE, GRID * GRID * PLANT_INSTANCE_BYTES);
  private drawArgs = createBuffer(B.STORAGE | B.INDIRECT | B.COPY_DST, DRAW_ARGS_BYTES);

  private terrainIndexCount = (GRID - 1) * (GRID - 1) * 6;
  private terrainIndexBuffer = createBuffer(B.INDEX, terrainIndices(GRID));
  
  // Use per-species flower shape (created before bind groups use buffer)
  private flowers = new Flowers();

  private stubPipeline = computePipeline(terrainStubWgsl);
  private stepPipeline = computePipeline(ecosystemWgsl);
  private compactPipeline = computePipeline(compactWgsl);
  private terrainPipeline = renderPipeline(terrainWgsl);
  private plantsPipeline = renderPipeline(plantsWgsl); // geometry is from vertex_index

  private stubBindGroup = createBindGroup(this.stubPipeline, [
    this.height.createView(),
    this.moisture.createView(),
    { buffer: this.stubParams },
  ]);
  private stepBindGroup = createBindGroup(this.stepPipeline, [
    this.height.createView(),
    this.moisture.createView(),
    this.state.createView(),
    this.stateNext.createView(),
    { buffer: this.simParams },
  ]);
  private compactBindGroup = createBindGroup(this.compactPipeline, [
    this.height.createView(),
    this.state.createView(),
    { buffer: this.instances },
    { buffer: this.drawArgs },
  ]);
  private terrainBindGroup = createBindGroup(this.terrainPipeline, [
    { buffer: camera.uniformBuffer },
    this.height.createView(),
    this.moisture.createView(),
    this.state.createView(),
    { buffer: this.drawParams },
    { buffer: this.flowers.buffer }
  ]);
  private plantsBindGroup = createBindGroup(
    this.plantsPipeline, 
    [
      { buffer: camera.uniformBuffer }, 
      { buffer: this.instances },
      { buffer: this.flowers.buffer }
    ]
  );

  init(gui: GUI) {
    camera.target = vec3.create(0, 4, 0);
    camera.distance = 105;
    camera.pitch = 0.5;

    const s = this.settings;
    gui.add(s, 'seed', 0, 999, 1).onChange(() => (this.terrainDirty = true));
    gui.add(s, 'speciesCount', 1, MAX_SPECIES, 1).onChange(() => (this.stateDirty = true));
    gui.add(s, 'spawnRate', 0, 1);
    gui.add(s, 'mortality', 0, 0.3);
    gui.add(s, 'lifespan', 5, 250, 1);
    gui.add(s, 'play');
    gui.add(s, 'stepsPerSecond', 1, 60, 1);
    gui.add(s, 'view', VIEWS);
    gui.add(s, 'showPlants');
    gui.add({ reseed: () => (this.stateDirty = true) }, 'reseed');
    this.flowers.addGui(gui);
  }

  frame(encoder: GPUCommandEncoder, dt: number) {
    this.targets.update();
    if (this.terrainDirty) this.generateTerrain(encoder);
    if (this.stateDirty || this.takeStep(dt)) this.step(encoder);
    this.compact(encoder);
    this.draw(encoder);
  }

  private generateTerrain(encoder: GPUCommandEncoder) {
    device.queue.writeBuffer(this.stubParams, 0, new Float32Array([this.settings.seed]));
    dispatchGrid(encoder, this.stubPipeline, this.stubBindGroup, GRID);
    this.terrainDirty = false;
    this.stateDirty = true;
  }

  private takeStep(dt: number) {
    if (!this.settings.play) return false;
    this.stepBudget = Math.min(this.stepBudget + dt * this.settings.stepsPerSecond, 2);
    if (this.stepBudget < 1) return false;
    this.stepBudget -= 1;
    return true;
  }

  private step(encoder: GPUCommandEncoder) {
    const s = this.settings;
    const u32 = new Uint32Array(this.simData);
    const f32 = new Float32Array(this.simData);
    u32.set([this.stateDirty ? 0 : 1, ++this.stepIndex, s.speciesCount, s.lifespan]);
    f32.set([s.mortality, s.spawnRate], 4);
    SPECIES.forEach((row, i) => f32.set(row, 8 + i * 8));
    device.queue.writeBuffer(this.simParams, 0, this.simData);
    dispatchGrid(encoder, this.stepPipeline, this.stepBindGroup, GRID);
    encoder.copyTextureToTexture({ texture: this.stateNext }, { texture: this.state }, [GRID, GRID]);
    this.stateDirty = false;
  }

  private compact(encoder: GPUCommandEncoder) {
    device.queue.writeBuffer(this.drawArgs, 0, new Uint32Array([FLOWER_VERTS,0, 0, 0]));
    dispatchGrid(encoder, this.compactPipeline, this.compactBindGroup, GRID);
  }

  private draw(encoder: GPUCommandEncoder) {
    device.queue.writeBuffer(this.drawParams, 0, new Uint32Array([VIEWS.indexOf(this.settings.view)]));
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.targets.color,
          resolveTarget: context.getCurrentTexture().createView(),
          clearValue: this.skyClear,
          loadOp: 'clear',
          storeOp: 'discard',
        },
      ],
      depthStencilAttachment: {
        view: this.targets.depth,
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'discard',
      },
    });

    pass.setPipeline(this.terrainPipeline);
    pass.setBindGroup(0, this.terrainBindGroup);
    pass.setIndexBuffer(this.terrainIndexBuffer, 'uint32');
    pass.drawIndexed(this.terrainIndexCount);

    if (this.settings.showPlants) {
      pass.setPipeline(this.plantsPipeline);
      pass.setBindGroup(0, this.plantsBindGroup);
      pass.drawIndirect(this.drawArgs, 0);
    }
    pass.end();
  }
}
