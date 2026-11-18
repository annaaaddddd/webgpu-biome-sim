// Ecosystem step after Cordonnier et al. 2017, reduced to live or die: empty cells get seeded
// from healthy plants nearby, occupied cells age and die faster the less viable they are, and a
// dying plant wilts for its own wilt time before its cell frees up (lifecycle in common.wgsl).
// mode 0 seeds the map from scratch. Needs species.wgsl.

@group(0) @binding(0) var heightTex: texture_2d<f32>;
@group(0) @binding(1) var moistureTex: texture_2d<f32>;
@group(0) @binding(2) var stateIn: texture_2d<u32>;
@group(0) @binding(3) var stateOut: texture_storage_2d<rgba8uint, write>;
@group(0) @binding(4) var<uniform> sim: Sim;
@group(0) @binding(5) var<storage, read_write> events: array<atomic<u32>, MAX_SPECIES * EVENT_KINDS>;

const CROWDED_NEIGHBORS = 2u;

// event kinds, in the order sim/census.ts reads them
const SEEDED = 0u;
const SPROUTED = 1u;
const STRESSED = 2u;
const CROWDED = 3u;
const AGED = 4u;
const NO_DEATH = EVENT_KINDS;

fn record(id: u32, kind: u32) {
  atomicAdd(&events[id * EVENT_KINDS + kind], 1u);
}

fn plant(id: u32, age: u32, v: f32, a: u32) -> vec4u {
  return vec4u(id + 1u, min(age, 255u), u32(v * 255.0), a);
}

// Healthy species in the i-th cell of the (2r+1)^2 square around p, skipping p itself.
// A wilting plant counts as empty: it neither seeds nor crowds.
fn neighborSpecies(p: vec2i, r: i32, i: u32) -> u32 {
  let side = 2 * r + 1;
  let count = u32(side * side - 1);
  let k = i32(min(i, count - 1u));
  let kk = select(k, k + 1, k >= (side * side) / 2);
  let q = clamp(p + vec2i(kk % side, kk / side) - r, vec2i(0), vec2i(i32(GRID) - 1));
  let cell = textureLoad(stateIn, q, 0);
  return select(0u, cell.r, wiltLeft(cell.a) == 0u);
}

fn deathCause(aged: bool, v: f32, crowded: bool, roll: f32) -> u32 {
  let stress = sim.mortality * (1.0 - v);
  if (aged) {
    return AGED;
  }
  if (roll < stress) {
    return STRESSED;
  }
  if (crowded && roll < stress + 0.25) {
    return CROWDED;
  }
  return NO_DEATH;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let p = vec2i(gid.xy);
  let h = loadHeight(heightTex, p);
  let m = textureLoad(moistureTex, p, 0).r;
  let slope = gradientSlope(heightGradient(heightTex, p));
  let rnd = rand3(vec3u(gid.xy, sim.frame));
  let newSeed = u32(
    rand3(vec3u(gid.yx, sim.frame + 7919u)).x * 16.0
  ) & 15u; // for plants born this step

  var cell = vec4u(0u);

  if (sim.mode == 0u) {
    let best = bestSpecies(h, slope, m);
    if (rnd.x < best.viability * best.viability * sim.spawnRate * 0.3) {
      cell = plant(
        best.id,
        u32(mix(0.3, 0.7, rnd.y) * f32(sim.lifespan)),
        best.viability,
        packPlantA(newSeed, 0u)
      );
      record(best.id, SPROUTED);
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
    let id = cell.r - 1u;
    let v = viability(id, h, slope, m);
    let seed = plantSeed(cell.a);
    let lc = lifecycle(gid.xy, seed, sim.lifespan);
    let age = cell.g + 1u;

    var left = wiltLeft(cell.a);

    if (left > 0u) {
      // already wilting: count down, remove when it runs out
      left -= 1u;
    } else {
      let aged = f32(age) >= lc.life - f32(lc.wiltSteps);
      let cause = deathCause(aged, v, occupied >= CROWDED_NEIGHBORS, rnd.x);

      if (cause != NO_DEATH) {
        left = lc.wiltSteps; // start wilting instead of vanishing
        record(id, cause);
      }
    }

    let wiltedOut = wiltLeft(cell.a) > 0u && left == 0u;

    if (wiltedOut) {
      cell = vec4u(0u);
    } else {
      cell = plant(id, age, v, packPlantA(seed, left));
    }
  } else {
    let parent = neighborSpecies(p, 2, u32(rnd.z * 24.0));

    if (parent != 0u && occupied == 0u) {
      let v = viability(parent - 1u, h, slope, m);

      if (rnd.y < sim.spawnRate * v * v) {
        cell = plant(parent - 1u, 0u, v, packPlantA(newSeed, 0u));
        record(parent - 1u, SEEDED);
      }
    } else if (rnd.y < 0.0005) {
      let best = bestSpecies(h, slope, m);

      if (best.viability > 0.5) {
        cell = plant(best.id, 0u, best.viability, packPlantA(newSeed, 0u));
        record(best.id, SPROUTED);
      }
    }
  }

  textureStore(stateOut, p, cell);
}
