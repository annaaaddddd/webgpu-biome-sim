// Species niches and the viability they give a cell, after Cordonnier et al. 2017: each factor
// has a plateau response and the weakest one limits the plant. Goes in front of any shader that
// binds `sim: Sim`.

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
  species: array<Species, MAX_SPECIES>,
}

struct Pick {
  id: u32,
  viability: f32,
}

fn plateau(x: f32, range: vec2f, soft: f32) -> f32 {
  return smoothstep(range.x - soft, range.x, x) * (1.0 - smoothstep(range.y, range.y + soft, x));
}

fn viability(id: u32, h: f32, slope: f32, m: f32) -> f32 {
  let s = sim.species[id];
  let climate = min(plateau(h, s.height, s.softness), plateau(m, s.moisture, s.softness));
  let flatEnough = 1.0 - smoothstep(s.slopeMax * 0.6, s.slopeMax, slope);
  let dryLand = smoothstep(WATER_LEVEL, WATER_LEVEL + 0.03, h);
  return min(climate, min(flatEnough, dryLand));
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
