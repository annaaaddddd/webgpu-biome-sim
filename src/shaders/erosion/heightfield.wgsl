// Fills heightTex with fBm value noise. One thread per texel.

// Must match the Float32Array in ErosionTrack.generateHeightfield().
struct Params {
  frequency: f32,
  octaves: f32,
  seed: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var stateOut: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var fluxOut: texture_storage_2d<rgba32float, write>;

// Cheap 2D -> [0,1) hash. Seed shifts the input so each seed gives a new map.
fn hash(p: vec2f) -> f32 {
  return fract(sin(dot(p + vec2f(params.seed), vec2(127.1, 311.7))) * 43758.5453);
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

// Sum of octaves: each one doubles frequency and halves amplitude.
// Divided by the total amplitude so the result stays in [0, 1].
fn fbm(p: vec2f) -> f32 {
  var value = 0.0;
  var totalAmp = 0.0;
  var amplitude = 1.0;
  var frequency = params.frequency;
  for (var i = 0; i < i32(params.octaves); i++) {
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

  // Texel -> uv in [0,1), then sample fBm and store in the red channel.
  let uv = vec2<f32>(id.xy) / vec2f(dims);
  let h = fbm(uv);
  textureStore(stateOut, id.xy, vec4f(h, 0, 0, 0));
  textureStore(fluxOut, id.xy, vec4f(0));
}
