// Layouts shared by the three tracks: terrain writes height + moisture, the ecosystem step
// turns them into a species map and a PlantInstance list, flowers draw it.

export const GRID = 512;
export const WORLD_SIZE = 100;
export const HEIGHT_SCALE = 18;
export const WATER_LEVEL = 0.05;
export const MAX_SPECIES = 4;

// every render pipeline must match RenderTargets or the merged pass fails
export const MSAA_SAMPLES = 4;

// height and moisture: one texel per grid cell, values in [0, 1]
export const TERRAIN_FORMAT: GPUTextureFormat = 'r32float';

// r = species id (0 = empty), g = age in steps, b = viability * 255, a = unused
export const STATE_FORMAT: GPUTextureFormat = 'rgba8uint';

// vec3f position, u32 species, f32 rotation, f32 age, f32 viability, f32 pad
export const PLANT_INSTANCE_BYTES = 32;

// indexCount, instanceCount (filled by the compact pass), firstIndex, baseVertex, firstInstance
export const DRAW_ARGS_BYTES = 20;

export const SPECIES_COLORS = [
  [0.95, 0.33, 0.38],
  [0.98, 0.8, 0.25],
  [0.64, 0.45, 0.92],
  [0.97, 0.95, 0.9],
];

export const SKY_COLOR = [0.62, 0.74, 0.9];
