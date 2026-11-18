// Shared by every shader. GRID, WORLD_SIZE, HEIGHT_SCALE, SKY are
// injected from core/layout.ts before this file.

struct Camera {
  viewProj: mat4x4f,
  pos: vec3f,
  pixelScale: f32 // canvas.height / (2tan(fov / 2)), pixels per world unit
}

struct PlantInstance {
  pos: vec3f,
  species: u32,
  rotation: f32,
  size: f32, // growth, viability, wilt shrink cull fade affects draw scale
  open: f32, // 0 -> 1: budding -> fully opened
  wilt: f32 // 0 -> 1: healthy -> dying
}

// One wind for flowers, clouds and rain, see sim/wind.ts.
struct Wind {
  dir: vec2f,     // unit, world xz, the way it blows
  offset: vec2f,  // how far the air has moved, so gusts travel downwind
  strength: f32,  // 0 calm, 1 normal
  time: f32,
}

// plant lifecycle (ecosystem controls when plant dies, compact controls look)
// state.a has per-plant seed, then the wilt is a countdown in steps
// 0: not wilting
// timing is then a fraction of the plant's lifespan:
// jittered per plant, hashed cell with seed, allows for more sprouting/blooming/wilting randomization
const MAX_WILT_STEPS = 15u; // 4 bit countdown

struct Lifecycle {
  life: f32, // lifespan in steps
  growSteps: f32, // sprout to full size
  bloomStart: f32, // age when petals begin to open
  bloomSteps: f32, // bud -> fully open
  wiltSteps: u32 // length of wilt (3 to MAX_WILT_STEPS)
}

fn plantSeed(a: u32) -> u32 {
  return a & 15u;
}

fn wiltLeft(a: u32) -> u32 {
  return a >> 4u;
}

fn packPlantA(seed: u32, countdown: u32) -> u32 {
  return (seed & 15u) | (min(countdown, MAX_WILT_STEPS) << 4u);
}

fn lifecycle(cell: vec2u, seed: u32, lifespan: u32) -> Lifecycle {
  let j = rand3(vec3u(cell, seed + 0x5eedu));
  let k = rand3(vec3u(cell.yx, seed + 0xb100u));

  var lc: Lifecycle;

  lc.life = min(f32(lifespan) * mix(0.7, 1.3, j.x), 255.0);
  lc.growSteps = max(2.0, lc.life * mix(0.04, 0.1, j.y));
  lc.bloomStart = lc.growSteps * mix(0.4, 0.9, k.x);
  lc.bloomSteps = max(2.0, lc.life * mix(0.04, 0.12, k.y));
  lc.wiltSteps = u32(
    clamp(
      lc.life * mix(0.08, 0.2, j.z),
      3.0,
      f32(MAX_WILT_STEPS))
  );
  
  return lc;
}

const SUN_DIR = vec3f(0.645, 0.591, 0.484);

// per-plant species parameters (look for tracks/flowers.ts for packing)
const F_STEM = 1u;
const F_LEAVES = 2u;
const F_PETALS = 4u;
const F_CENTER = 8u;

struct FlowerParams {
  flags: u32, 
  petalCount: u32,
  leafCount: u32,
  _pad0: u32,
  stemHeight: f32,
  stemRadius: f32,
  stemBend: f32,
  leafLength: f32,
  leafWidth: f32,
  leafDroop: f32,
  petalLength: f32,
  petalWidth: f32,
  petalCup: f32,
  centerRadius: f32,
  boundCenterY: f32, // create a bounding sphere in the flower space (unit plant size)
  boundRadius: f32,
  petalColor: vec4f,
  centerColor: vec4f,
  stemColor: vec4f
} // 112 bytes total

struct CloudParams {
  offset: vec2f, // how far the layer has drifted with the wind, in world units
  cover: f32,    // 0..1, higher = more sky covered
  scale: f32,    // noise frequency; smaller = bigger clouds
  alpha: f32,    // opacity of the full stack
  height: f32,   // y of the bottom layer
  size: f32,     // width of the layer, in world units
  layers: f32,   // layer count, f32 so the struct is all floats
  spacing: f32,  // vertical gap between layers
}

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

// How much cloud is above a world xz position, 0..1. The cloud layer, and
// anything that depends on where the clouds are, should all call this.
// bias raises the cutoff (thinner clouds); each layer of the stack passes its own.
fn cloudDensity(world: vec2f, c: CloudParams, bias: f32) -> f32 {
  // The whole pattern slides with the wind. A second, slower noise bends the
  // lookup so the shapes also change as they move.
  let p = (world - c.offset) * c.scale;
  let warp = fbm(p * 0.5 + c.offset.yx * c.scale * 0.4);
  let n = fbm(p + warp * 1.5);

  // fbm sits mostly in 0.25..0.75, so cover moves the cutoff across that range.
  let cutoff = mix(0.8, 0.2, c.cover) + bias;
  let density = smoothstep(cutoff, cutoff + 0.09, n);

  // Fade out toward the edge of the layer so it has no visible border.
  let edge = 1.0 - smoothstep(0.6, 1.0, length(world) / (c.size * 0.5));
  return density * edge;
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

const SUN_COLOR = vec3f(1.0, 0.95, 0.85);

// ambient lighting (ground and sky)
fn ambientLight(n: vec3f) -> vec3f {
  return mix(vec3f(0.16, 0.15, 0.12), SKY * 0.5, n.y * 0.5 + 0.5);
}

fn applyFog(color: vec3f, world: vec3f, camPos: vec3f) -> vec3f {
  let fog = 1.0 - exp(-max(distance(world, camPos) - 60.0, 0.0) * 0.002);
  return mix(color, SKY, fog);
}

fn shade(albedo: vec3f, n: vec3f, world: vec3f, camPos: vec3f) -> vec3f {
  let diffuse = max(dot(n, SUN_DIR), 0.0) * SUN_COLOR;
  return applyFog(albedo * (diffuse + ambientLight(n)), world, camPos);
}

fn gamma(c: vec3f) -> vec3f {
  return pow(max(c, vec3f(0.0)), vec3f(1.0 / 2.2));
}
