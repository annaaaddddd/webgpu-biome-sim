// Stand-in for the erosion track: noise height plus a moisture guess, both in [0, 1].

struct Params {
  seed: f32,
}

@group(0) @binding(0) var heightOut: texture_storage_2d<r32float, write>;
@group(0) @binding(1) var moistureOut: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let uv = (vec2f(gid.xy) + 0.5) / f32(GRID);
  let p = uv * 3.0 + params.seed * vec2f(17.3, 9.1);
  let warp = vec2f(fbm(p + vec2f(1.7, 9.2)), fbm(p + vec2f(8.3, 2.8))) - 0.5;
  var h = fbm(p + 1.2 * warp);
  let ridge = 1.0 - abs(2.0 * fbm(p * 1.7 + 4.0) - 1.0);
  h = mix(h, ridge * ridge, smoothstep(0.5, 0.7, h));
  h = smoothstep(0.2, 0.85, h);
  h = pow(h, 1.5);
  let center = smoothstep(1.3, 0.6, length(uv - 0.5) * 2.0);
  h *= mix(0.35, 1.0, center);
  let m = 0.6 * (1.0 - h) + 0.5 * fbm(p * 2.3 + vec2f(5.0, -3.0)) - 0.15;
  textureStore(heightOut, vec2i(gid.xy), vec4f(h, 0.0, 0.0, 0.0));
  textureStore(moistureOut, vec2i(gid.xy), vec4f(clamp(m, 0.0, 1.0), 0.0, 0.0, 0.0));
}
