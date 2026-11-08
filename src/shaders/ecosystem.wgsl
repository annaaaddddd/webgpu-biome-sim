// Ecosystem step after Cordonnier et al. 2017, reduced to live or die: each species has a
// viability over height, slope and moisture, empty cells get seeded from plants nearby,
// occupied cells age and die faster the less viable they are. mode 0 seeds the map from scratch.

struct Species {
  height: vec2f,
  moisture: vec2f,
  slopeMax: f32,
  softness: f32,
  _pad: vec2f,
}

struct Sim {
  mode: u32,
  frame: u32,
  speciesCount: u32,
  lifespan: u32,
  mortality: f32,
  spawnRate: f32,
  _pad: vec2f,
  species: array<Species, 4>,
}

struct Pick {
  id: u32,
  viability: f32,
}

@group(0) @binding(0) var heightTex: texture_2d<f32>;
@group(0) @binding(1) var moistureTex: texture_2d<f32>;
@group(0) @binding(2) var stateIn: texture_2d<u32>;
@group(0) @binding(3) var stateOut: texture_storage_2d<rgba8uint, write>;
@group(0) @binding(4) var<uniform> sim: Sim;

const CROWDED_NEIGHBORS = 2u;

fn plateau(x: f32, range: vec2f, soft: f32) -> f32 {
  return smoothstep(range.x - soft, range.x, x) * (1.0 - smoothstep(range.y, range.y + soft, x));
}

fn viability(id: u32, h: f32, slope: f32, m: f32) -> f32 {
  let s = sim.species[id];
  let flatEnough = 1.0 - smoothstep(s.slopeMax * 0.6, s.slopeMax, slope);
  let dryLand = smoothstep(WATER_LEVEL, WATER_LEVEL + 0.03, h);
  return plateau(h, s.height, s.softness) * plateau(m, s.moisture, s.softness) * flatEnough * dryLand;
}

fn bestSpecies(h: f32, slope: f32, m: f32) -> Pick {
  var best = Pick(0u, 0.0);
  for (var id = 0u; id < sim.speciesCount; id++) {
    let v = viability(id, h, slope, m);
    if (v > best.viability) {
      best = Pick(id, v);
    }
  }
  return best;
}

fn plant(id: u32, age: u32, v: f32) -> vec4u {
  return vec4u(id + 1u, age, u32(v * 255.0), 0u);
}

// Species in the i-th cell of the (2r+1)^2 square around p, skipping p itself.
fn neighborSpecies(p: vec2i, r: i32, i: u32) -> u32 {
  let side = 2 * r + 1;
  let count = u32(side * side - 1);
  let k = i32(min(i, count - 1u));
  let kk = select(k, k + 1, k >= (side * side) / 2);
  let q = clamp(p + vec2i(kk % side, kk / side) - r, vec2i(0), vec2i(i32(GRID) - 1));
  return textureLoad(stateIn, q, 0).r;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let p = vec2i(gid.xy);
  let h = loadHeight(heightTex, p);
  let m = textureLoad(moistureTex, p, 0).r;
  let slope = gradientSlope(heightGradient(heightTex, p));
  let rnd = rand3(vec3u(gid.xy, sim.frame));
  var cell = vec4u(0u);

  if (sim.mode == 0u) {
    let best = bestSpecies(h, slope, m);
    if (rnd.x < best.viability * best.viability * sim.spawnRate * 0.3) {
      cell = plant(best.id, u32(mix(0.3, 0.7, rnd.y) * f32(sim.lifespan)), best.viability);
    }
    textureStore(stateOut, p, cell);
    return;
  }

  cell = textureLoad(stateIn, p, 0);
  var occupied = 0u;
  for (var i = 0u; i < 8u; i++) {
    occupied += u32(neighborSpecies(p, 1, i) != 0u);
  }

  if (cell.r != 0u) {
    let v = viability(cell.r - 1u, h, slope, m);
    let death = sim.mortality * (1.0 - v) + select(0.0, 0.25, occupied >= CROWDED_NEIGHBORS);
    if (cell.g + 1u > sim.lifespan || rnd.x < death) {
      cell = vec4u(0u);
    } else {
      cell = plant(cell.r - 1u, cell.g + 1u, v);
    }
  } else {
    let parent = neighborSpecies(p, 2, u32(rnd.z * 24.0));
    if (parent != 0u && occupied == 0u) {
      let v = viability(parent - 1u, h, slope, m);
      if (rnd.y < sim.spawnRate * v * v) {
        cell = plant(parent - 1u, 0u, v);
      }
    } else if (rnd.y < 0.0005) {
      let best = bestSpecies(h, slope, m);
      if (best.viability > 0.5) {
        cell = plant(best.id, 0u, best.viability);
      }
    }
  }
  textureStore(stateOut, p, cell);
}
