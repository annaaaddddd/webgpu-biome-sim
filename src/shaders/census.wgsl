// Per-species totals over the whole grid. Each thread sums one TILE x TILE block of cells and
// then adds only the totals that are not zero. Skipping the zeros is what makes it cheap; large
// tiles are slower because too few threads share the work.
// Needs species.wgsl and TILE (see sim/census.ts).

struct Census {
  alive: atomic<u32>,
  wilting: atomic<u32>,
  viability: atomic<u32>,   // summed over living plants, x255
  suitability: atomic<u32>, // summed over every cell, x255
}

@group(0) @binding(0) var heightTex: texture_2d<f32>;
@group(0) @binding(1) var moistureTex: texture_2d<f32>;
@group(0) @binding(2) var stateTex: texture_2d<u32>;
@group(0) @binding(3) var<uniform> sim: Sim;
@group(0) @binding(4) var<storage, read_write> census: array<Census, MAX_SPECIES>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  var alive: array<u32, MAX_SPECIES>;
  var wilting: array<u32, MAX_SPECIES>;
  var viable: array<u32, MAX_SPECIES>;
  var suitable: array<u32, MAX_SPECIES>;

  for (var i = 0u; i < TILE * TILE; i++) {
    let p = vec2i(gid.xy * TILE + vec2u(i % TILE, i / TILE));
    let h = loadHeight(heightTex, p);
    let m = textureLoad(moistureTex, p, 0).r;
    let slope = gradientSlope(heightGradient(heightTex, p));
    let cell = textureLoad(stateTex, p, 0);

    for (var id = 0u; id < sim.speciesCount; id++) {
      let v = u32(viability(id, h, slope, m) * 255.0);
      suitable[id] += v;
      if (cell.r != id + 1u) {
        continue;
      }
      if (wiltLeft(cell.a) == 0u) {
        alive[id]++;
        viable[id] += v;
      } else {
        wilting[id]++;
      }
    }
  }

  for (var id = 0u; id < sim.speciesCount; id++) {
    if (alive[id] != 0u) { atomicAdd(&census[id].alive, alive[id]); }
    if (wilting[id] != 0u) { atomicAdd(&census[id].wilting, wilting[id]); }
    if (viable[id] != 0u) { atomicAdd(&census[id].viability, viable[id]); }
    if (suitable[id] != 0u) { atomicAdd(&census[id].suitability, suitable[id]); }
  }
}
