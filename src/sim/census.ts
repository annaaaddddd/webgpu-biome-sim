import { device } from '../core/device';
import { GRID, MAX_SPECIES, EVENT_KINDS } from '../core/layout';
import { shaderModule, createBuffer, createBindGroup, dispatchGrid, Readback } from '../core/gpu';
import { PassTimer } from './erosion_bench';
import speciesWgsl from '../shaders/species.wgsl?raw';
import censusWgsl from '../shaders/census.wgsl?raw';

// Cells per side that one census thread sums, a power of two up to 64. &tile=N overrides it
// for a bench run. Measured: 1 to 4 tie, from 8 up the pass slows down with every doubling.
const TILE = Number(new URLSearchParams(location.search).get('tile') ?? 2);
const CENSUS_FIELDS = 4; // Census in census.wgsl
const CENSUS_BYTES = MAX_SPECIES * CENSUS_FIELDS * 4;
const EVENTS_BYTES = MAX_SPECIES * EVENT_KINDS * 4;

const B = GPUBufferUsage;

export interface SpeciesCount {
  alive: number;
  wilting: number;
  viability: number; // mean over living plants, 0..1
  habitat: number;   // share of the map the species could live on, weighted by viability
  events: number[];  // since the last reseed: seeded, sprouted, died of stress, crowding, old age
}

function parse(data: Uint32Array): SpeciesCount[] {
  return Array.from({ length: MAX_SPECIES }, (_, i) => {
    const [alive, wilting, viability, suitability] = data.subarray(i * CENSUS_FIELDS);
    const events = MAX_SPECIES * CENSUS_FIELDS + i * EVENT_KINDS;
    return {
      alive,
      wilting,
      viability: alive ? viability / 255 / alive : 0,
      habitat: suitability / 255 / (GRID * GRID),
      events: Array.from(data.subarray(events, events + EVENT_KINDS)),
    };
  });
}

// Counts plants and habitat per species on the GPU every frame and reads the totals back.
export class Census {
  // births and deaths per species, written by the step pass
  readonly events = createBuffer(B.STORAGE | B.COPY_SRC | B.COPY_DST, EVENTS_BYTES);
  species: SpeciesCount[] = []; // latest totals, a frame or two old

  private counts = createBuffer(B.STORAGE | B.COPY_SRC | B.COPY_DST, CENSUS_BYTES);
  private readback = new Readback(CENSUS_BYTES + EVENTS_BYTES);
  // ?bench=N times the pass on the GPU, see erosion_bench.ts
  private bench = PassTimer.fromUrl(['census'], {
    stepPasses: 1,
    label: `census ${GRID}x${GRID}, tile ${TILE}`,
    file: `census_timing_tile${TILE}.csv`,
  });
  private pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: shaderModule(`const TILE = ${TILE}u;\n` + speciesWgsl + censusWgsl) },
  });
  private bindGroup: GPUBindGroup;

  constructor(height: GPUTexture, moisture: GPUTexture, state: GPUTexture, simParams: GPUBuffer) {
    this.bindGroup = createBindGroup(this.pipeline, [
      height.createView(),
      moisture.createView(),
      state.createView(),
      { buffer: simParams },
      { buffer: this.counts },
    ]);
  }

  // Call after the step pass. reseeded starts the birth and death counts over.
  frame(encoder: GPUCommandEncoder, reseeded: boolean) {
    // last frame's totals, copied before they are cleared
    this.readback.request(
      (copies, target) => {
        copies.copyBufferToBuffer(this.counts, 0, target, 0, CENSUS_BYTES);
        copies.copyBufferToBuffer(this.events, 0, target, CENSUS_BYTES, EVENTS_BYTES);
      },
      (data) => (this.species = parse(new Uint32Array(data))),
    );
    if (reseeded) device.queue.writeBuffer(this.events, 0, new Uint32Array(EVENTS_BYTES / 4));
    device.queue.writeBuffer(this.counts, 0, new Uint32Array(CENSUS_BYTES / 4));
    this.bench?.beginFrame();
    dispatchGrid(encoder, this.pipeline, this.bindGroup, GRID / TILE, this.bench?.writes(0));
    this.bench?.endFrame(encoder);
  }
}
