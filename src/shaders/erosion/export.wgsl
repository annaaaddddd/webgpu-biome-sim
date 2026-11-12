// Writes the shared height and moisture textures the terrain and ecosystem
// tracks read: height is copied from the sim state, moisture is a height/noise
// baseline plus an EMA of water depth. One thread per texel, runs every frame.

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
  seed: f32,
  baseline: f32,
}

// stateTex[cur]: r = terrain height (b), g = water depth (d), b = sediment (s).
@group(0) @binding(0) var state: texture_2d<f32>;
@group(0) @binding(1) var heightOut: texture_storage_2d<r32float, write>;

// Read back so moisture can keep a running average instead of the instantaneous depth.
@group(0) @binding(2) var wet: texture_storage_2d<r32float, read_write>;   // EMA state
@group(0) @binding(3) var<uniform> sim: SimParams;
@group(0) @binding(4) var moistureOut: texture_storage_2d<r32float, write>; // what the ecosystem reads

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let dims = textureDimensions(state);
  if (id.x >= dims.x || id.y >= dims.y) { return; }

  let cell = textureLoad(state, id.xy, 0);
  textureStore(heightOut, id.xy, vec4f(cell.r, 0, 0, 0));

  // Moisture = height/noise baseline + an EMA of water depth. k is derived from the
  // sim time elapsed this frame so the decay rate does not depend on dt or stepsPerFrame.
  let k = 1.0 - exp(-sim.frameDt / sim.moistureTau);

  let h = cell.r;
  let wetNow = clamp(cell.g * sim.moistureScale, 0, 1);
  let prev = textureLoad(wet, id.xy).r;
  let ema = mix(prev, wetNow, k);
  textureStore(wet, id.xy, vec4f(ema, 0, 0, 0));

  let uv = (vec2f(id.xy) + 0.5) / vec2f(dims);
  let p = uv * 3.0 + sim.seed * vec2f(17.3, 9.1);
  // Same moisture guess as terrain_stub.wgsl; keep the two in sync.
  let base = 0.6 * (1.0 - h) + 0.5 * fbm(p * 2.3 + vec2f(5.0, -3.0)) - 0.15;
  let m = clamp(sim.baseline * base + ema, 0, 1);
  textureStore(moistureOut, id.xy, vec4f(m, 0, 0, 0));
}
