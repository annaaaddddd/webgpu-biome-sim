// Heightfield draw. View 0 is the shaded biome look, the rest are debug maps.
// Needs species.wgsl.

struct Draw {
  view: u32,
}

struct VSOut {
  @builtin(position) clip: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
}

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var heightTex: texture_2d<f32>;
@group(0) @binding(2) var moistureTex: texture_2d<f32>;
@group(0) @binding(3) var stateTex: texture_2d<u32>;
@group(0) @binding(4) var<uniform> draw: Draw;
@group(0) @binding(5) var<storage, read> flowers: array<FlowerParams, MAX_SPECIES>;
@group(0) @binding(6) var<uniform> sim: Sim;

// debug flower species
fn speciesColor(id: u32) -> vec3f {
  return flowers[min(id - 1u, MAX_SPECIES - 1u)].petalColor.rgb;
}

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VSOut {
  let g = vec2i(i32(i % GRID), i32(i / GRID));
  let uv = (vec2f(g) + 0.5) / f32(GRID);
  let xz = (uv - 0.5) * WORLD_SIZE;
  var out: VSOut;
  out.world = vec3f(xz.x, loadHeight(heightTex, g) * HEIGHT_SCALE, xz.y);
  out.normal = gradientNormal(heightGradient(heightTex, g));
  out.uv = uv;
  out.clip = cam.viewProj * vec4f(out.world, 1.0);
  return out;
}

fn biomeColor(h: f32, m: f32, slope: f32, world: vec3f) -> vec3f {
  let sand = vec3f(0.78, 0.72, 0.52);
  let grass = vec3f(0.26, 0.5, 0.17);
  let dryGrass = vec3f(0.64, 0.6, 0.3);
  let rock = vec3f(0.44, 0.42, 0.4);
  let snow = vec3f(0.93, 0.94, 0.96);
  let patchiness = 0.85 + 0.3 * valueNoise(world.xz * 0.35);
  var c = mix(dryGrass, grass, m) * patchiness;
  c = mix(sand, c, smoothstep(0.03, 0.12, h));
  c = mix(c, rock, smoothstep(0.5, 0.75, slope));
  c = mix(c, snow, smoothstep(0.72, 0.85, h) * (1.0 - smoothstep(0.55, 0.85, slope)));
  return c;
}

fn waterColor(world: vec3f) -> vec3f {
  let view = normalize(cam.pos - world);
  let glint = pow(max(dot(reflect(-SUN_DIR, vec3f(0.0, 1.0, 0.0)), view), 0.0), 80.0);
  let water = mix(vec3f(0.08, 0.3, 0.45), SKY * 0.8, 0.35) + glint * 0.6;
  return shade(water, vec3f(0.0, 1.0, 0.0), world, cam.pos);
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  let p = texel(in.uv);
  let h = loadHeight(heightTex, p);
  let m = textureLoad(moistureTex, p, 0).r;
  let slope = gradientSlope(heightGradient(heightTex, p));
  let cell = textureLoad(stateTex, p, 0);
  let n = normalize(in.normal);

  if (draw.view == 0u) {
    // the surface's own height, so the waterline cuts across cells instead of following their edges
    let surface = in.world.y / HEIGHT_SCALE;
    if (surface <= WATER_LEVEL) {
      return vec4f(gamma(waterColor(in.world)), 1.0);
    }
    return vec4f(gamma(shade(biomeColor(surface, m, slope, in.world), n, in.world, cam.pos)), 1.0);
  }

  var color: vec3f;
  switch draw.view {
    case 1u: {
      let low = mix(vec3f(0.08, 0.15, 0.4), vec3f(0.3, 0.6, 0.25), smoothstep(0.0, 0.5, h));
      color = mix(low, vec3f(1.0), smoothstep(0.5, 1.0, h));
    }
    case 2u: {
      color = mix(vec3f(0.85, 0.72, 0.45), vec3f(0.1, 0.35, 0.9), m);
    }
    case 3u: {
      color = vec3f(slope);
    }
    case 4u: {
      color = select(vec3f(0.12), speciesColor(cell.r), cell.r != 0u);
    }
    case 5u: {
      let health = mix(vec3f(0.7, 0.15, 0.1), vec3f(0.2, 0.85, 0.3), f32(cell.b) / 255.0);
      color = select(vec3f(0.08), health, cell.r != 0u);
    }
    case 6u: {
      // where each species could live, occupied or not
      let best = bestSpecies(h, slope, m);
      color = speciesColor(best.id + 1u) * best.viability;
    }
    default: {
      // young to old, brown while wilting
      let age = mix(vec3f(0.5, 0.9, 0.4), vec3f(0.95, 0.85, 0.3), f32(cell.g) / f32(sim.lifespan));
      let stage = mix(age, vec3f(0.45, 0.25, 0.1), f32(cell.a) / f32(WILT_STEPS));
      color = select(vec3f(0.08), stage, cell.r != 0u);
    }
  }
  let light = 0.55 + 0.45 * max(dot(n, SUN_DIR), 0.0);
  return vec4f(gamma(color * light), 1.0);
}
