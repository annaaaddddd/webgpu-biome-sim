// Copies the sim state into the shared output textures the terrain and
// ecosystem tracks read. One thread per texel, runs every frame.

// same struct as erosion.wgsl
struct SimParams {
  dt: f32,
  rainRate: f32,
  moistureScale: f32,
}

// stateTex[cur]: r = terrain height (b), g = water depth (d), b = sediment (s).
@group(0) @binding(0) var state: texture_2d<f32>;
@group(0) @binding(1) var heightOut: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var moistureOut: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var<uniform> sim: SimParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let dims = textureDimensions(state);
  if (id.x >= dims.x || id.y >= dims.y) { return; }

  let cell = textureLoad(state, id.xy, 0);
  textureStore(heightOut, id.xy, vec4f(cell.r, 0, 0, 0));
  // Water depth stands in for moisture for now.
  let moist = clamp(cell.g * sim.moistureScale, 0, 1);
  textureStore(moistureOut, id.xy, vec4f(moist, 0, 0, 0));
}
