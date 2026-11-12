// Generates the starting terrain and clears the sim state. One thread per texel.
// Compiled with the shared prelude (fbm from common.wgsl) and shaped like
// terrain_stub.wgsl, so the same seed gives the same map as the stub terrain.

// Must match the Float32Array in ErosionSim.reset().
struct Params {
  seed: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var stateOut: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var fluxOut: texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var velocityOut: texture_storage_2d<rg32float, write>;
@group(0) @binding(4) var wetOut: texture_storage_2d<r32float, write>; // EMA of water depth, see export.wgsl; a new map starts dry

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let dims = textureDimensions(stateOut);
  if (id.x >= dims.x || id.y >= dims.y) { return; }

  // Same shaping as terrain_stub.wgsl: domain-warped fbm, ridges on the high
  // ground, a contrast curve, and a falloff toward the edges so the map ends
  // in lowland (below WATER_LEVEL in the ecosystem renderer).
  let uv = (vec2f(id.xy) + 0.5) / vec2f(dims);
  let p = uv * 3.0 + params.seed * vec2f(17.3, 9.1);
  let warp = vec2f(fbm(p + vec2f(1.7, 9.2)), fbm(p + vec2f(8.3, 2.8))) - 0.5;
  var h = fbm(p + 1.2 * warp);
  let ridge = 1.0 - abs(2.0 * fbm(p * 1.7 + 4.0) - 1.0);
  h = mix(h, ridge * ridge, smoothstep(0.5, 0.7, h));
  h = smoothstep(0.2, 0.85, h);
  h = pow(h, 1.5);
  let center = smoothstep(1.3, 0.6, length(uv - 0.5) * 2.0);
  h *= mix(0.35, 1.0, center);

  textureStore(stateOut, id.xy, vec4f(h, 0, 0, 0));
  textureStore(fluxOut, id.xy, vec4f(0));
  textureStore(velocityOut, id.xy, vec4f(0));
  textureStore(wetOut, id.xy, vec4f(0));
}
