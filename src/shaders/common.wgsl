// Shared by every shader. GRID, WORLD_SIZE, HEIGHT_SCALE, SKY and SPECIES_COLORS are
// injected from core/layout.ts before this file.

struct Camera {
  viewProj: mat4x4f,
  pos: vec3f,
}

struct PlantInstance {
  pos: vec3f,
  species: u32,
  rotation: f32,
  age: f32,
  viability: f32,
  _pad: f32,
}

const SUN_DIR = vec3f(0.645, 0.591, 0.484);

fn pcg3d(vIn: vec3u) -> vec3u {
  var v = vIn * 1664525u + 1013904223u;
  v.x += v.y * v.z;
  v.y += v.z * v.x;
  v.z += v.x * v.y;
  v ^= v >> vec3u(16u);
  v.x += v.y * v.z;
  v.y += v.z * v.x;
  v.z += v.x * v.y;
  return v;
}

fn rand3(p: vec3u) -> vec3f {
  return vec3f(pcg3d(p)) * (1.0 / 4294967296.0);
}

fn hash21(p: vec2f) -> f32 {
  return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453);
}

fn valueNoise(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let a = mix(hash21(i), hash21(i + vec2f(1.0, 0.0)), u.x);
  let b = mix(hash21(i + vec2f(0.0, 1.0)), hash21(i + vec2f(1.0, 1.0)), u.x);
  return mix(a, b, u.y);
}

fn fbm(pIn: vec2f) -> f32 {
  var p = pIn;
  var sum = 0.0;
  var amp = 0.5;
  let rot = mat2x2f(0.8, 0.6, -0.6, 0.8);
  for (var i = 0; i < 6; i++) {
    sum += amp * valueNoise(p);
    p = rot * p * 2.03 + 11.7;
    amp *= 0.5;
  }
  return sum;
}

fn texel(uv: vec2f) -> vec2i {
  return clamp(vec2i(uv * f32(GRID)), vec2i(0), vec2i(i32(GRID) - 1));
}

fn loadHeight(tex: texture_2d<f32>, p: vec2i) -> f32 {
  return textureLoad(tex, clamp(p, vec2i(0), vec2i(i32(GRID) - 1)), 0).r;
}

fn sampleHeight(tex: texture_2d<f32>, uv: vec2f) -> f32 {
  let p = uv * f32(GRID) - 0.5;
  let i = vec2i(floor(p));
  let f = fract(p);
  let bottom = mix(loadHeight(tex, i), loadHeight(tex, i + vec2i(1, 0)), f.x);
  let top = mix(loadHeight(tex, i + vec2i(0, 1)), loadHeight(tex, i + vec2i(1, 1)), f.x);
  return mix(bottom, top, f.y);
}

// dh/dx, dh/dz in world units, central differences
fn heightGradient(tex: texture_2d<f32>, p: vec2i) -> vec2f {
  let dx = loadHeight(tex, p + vec2i(1, 0)) - loadHeight(tex, p - vec2i(1, 0));
  let dz = loadHeight(tex, p + vec2i(0, 1)) - loadHeight(tex, p - vec2i(0, 1));
  let cell = WORLD_SIZE / f32(GRID);
  return vec2f(dx, dz) * HEIGHT_SCALE / (2.0 * cell);
}

fn gradientNormal(g: vec2f) -> vec3f {
  return normalize(vec3f(-g.x, 1.0, -g.y));
}

// 0 flat, 0.5 at 45 degrees, approaches 1 on cliffs
fn gradientSlope(g: vec2f) -> f32 {
  let t = length(g);
  return t / (1.0 + t);
}

fn speciesColor(id: u32) -> vec3f {
  var colors = SPECIES_COLORS;
  return colors[min(id - 1u, 3u)];
}

fn shade(albedo: vec3f, n: vec3f, world: vec3f, camPos: vec3f) -> vec3f {
  let diffuse = max(dot(n, SUN_DIR), 0.0) * vec3f(1.0, 0.95, 0.85);
  let ambient = mix(vec3f(0.16, 0.15, 0.12), SKY * 0.5, n.y * 0.5 + 0.5);
  let color = albedo * (diffuse + ambient);
  let fog = 1.0 - exp(-max(distance(world, camPos) - 60.0, 0.0) * 0.002);
  return mix(color, SKY, fog);
}

fn gamma(c: vec3f) -> vec3f {
  return pow(max(c, vec3f(0.0)), vec3f(1.0 / 2.2));
}
