// Copies the sim state into the shared output textures the terrain and
// ecosystem tracks read. One thread per texel, runs every frame.

// same struct as erosion.wgsl
struct SimParams {
  dt: f32,
  rainRate: f32,
  moistureScale: f32,
  pipeCoeff: f32,
  cellSize: f32,
  Kc: f32,
  Ks: f32,
  Kd: f32,
  Ke: f32,
  Kt: f32,
  minSlope: f32,
  maxDepth: f32,
  talus: f32,
  moistureTau: f32,
  frameDt: f32,
}

// stateTex[cur]: r = terrain height (b), g = water depth (d), b = sediment (s).
@group(0) @binding(0) var state: texture_2d<f32>;
@group(0) @binding(1) var heightOut: texture_storage_2d<r32float, write>;
// Read back so moisture can keep a running average instead of the instantaneous depth.
@group(0) @binding(2) var moistureOut: texture_storage_2d<r32float, read_write>;
@group(0) @binding(3) var<uniform> sim: SimParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let dims = textureDimensions(state);
  if (id.x >= dims.x || id.y >= dims.y) { return; }

  let cell = textureLoad(state, id.xy, 0);
  textureStore(heightOut, id.xy, vec4f(cell.r, 0, 0, 0));
  // Moisture is an exponential moving average of water depth: it rises while
  // a cell is wet and decays with time constant moistureTau once the water
  // leaves, so the ecosystem sees "recently wet" rather than a flickering
  // per-frame depth. k is derived from the sim time elapsed this frame, so
  // the decay rate does not change with dt or stepsPerFrame.
  let wet = clamp(cell.g * sim.moistureScale, 0, 1);
  let prev = textureLoad(moistureOut, id.xy).r;
  let k = 1.0 - exp(-sim.frameDt / sim.moistureTau);
  textureStore(moistureOut, id.xy, vec4f(mix(prev, wet, k), 0, 0, 0));
}
