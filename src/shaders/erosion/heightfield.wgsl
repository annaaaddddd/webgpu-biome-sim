// Generates the starting terrain and clears the sim state. One thread per texel.
// The height shaping matches terrain_stub.wgsl so both tracks start from the
// same kind of map; the fbm here is local until the shared one in common.wgsl
// can be included.

// Must match the Float32Array in ErosionSim.reset().
struct Params {
  seed: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var stateOut: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var fluxOut: texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var velocityOut: texture_storage_2d<rg32float, write>;
@group(0) @binding(4) var moistureOut: texture_storage_2d<r32float, write>;

// Cheap 2D -> [0,1) hash.
fn hash(p: vec2f) -> f32 {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

// Value noise: hash the 4 cell corners, blend with a smoothstep curve.
fn valueNoise(p: vec2f) -> f32 {
  let i = floor(p);
  var f = fract(p);

  let n00 = hash(i);
  let n10 = hash(i + vec2f(1.0, 0.0));
  let n01 = hash(i + vec2f(0.0, 1.0));
  let n11 = hash(i + vec2f(1.0, 1.0));

  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(n00, n10, f.x), mix(n01, n11, f.x), f.y);
}

// Six octaves: each one doubles frequency and halves amplitude.
// Divided by the total amplitude so the result stays in [0, 1].
fn fbm(p: vec2f) -> f32 {
  var value = 0.0;
  var totalAmp = 0.0;
  var amplitude = 1.0;
  var frequency = 1.0;
  for (var i = 0; i < 6; i++) {
    value += amplitude * valueNoise(p * frequency);
    totalAmp += amplitude;
    frequency *= 2;
    amplitude *= 0.5;
  }
  return value / totalAmp;
}

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
  // Moisture accumulates over time; a new map starts dry.
  textureStore(moistureOut, id.xy, vec4f(0));
}
