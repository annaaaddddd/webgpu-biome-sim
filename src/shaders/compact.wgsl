// Packs every occupied cell into the PlantInstance list. The indirect draw's instanceCount
// doubles as the atomic counter.

struct DrawArgs {
  vertexCount: u32,
  instanceCount: atomic<u32>,
  firstVertex: u32,
  firstInstance: u32
}

@group(0) @binding(0) var heightTex: texture_2d<f32>;
@group(0) @binding(1) var stateTex: texture_2d<u32>;
@group(0) @binding(2) var<storage, read_write> instances: array<PlantInstance>;
@group(0) @binding(3) var<storage, read_write> draw: DrawArgs;
@group(0) @binding(4) var<uniform> cam: Camera;
@group(0) @binding(5) var<storage, read> flowers: array<FlowerParams, MAX_SPECIES>;
@group(0) @binding(6) var<uniform> anim: Anim;

// stepPhase: fraction of the way to the next ecosystem step (0..1), so growth and wilting
// move every frame instead of jumping once per step
// lifespan: same value the step uses
struct Anim {
  stepPhase: f32,
  lifespan: u32
}

// Bounding radius (pixels)
// Skip flower if culled, "fade" flower if at culling boundary by shrinking so culling perspective is less drastic
const CULL_PIXELS = 1.0;
const FADE_PIXELS = 4.0;

// If the culling sphere is outside of the six frustum planes, return true.
// Use gribb-hartmann viewproj planes
// in webgpu, clip z is 0..w, so near plane is r2
fn outsideFrustum(c: vec3f, r: f32) -> bool {
  let m = cam.viewProj;
  let r0 = vec4f(m[0][0], m[1][0], m[2][0], m[3][0]);
  let r1 = vec4f(m[0][1], m[1][1], m[2][1], m[3][1]);
  let r2 = vec4f(m[0][2], m[1][2], m[2][2], m[3][2]);
  let r3 = vec4f(m[0][3], m[1][3], m[2][3], m[3][3]);

  var planes = array<vec4f, 6>(
    r3 + r0,
    r3 - r0,
    r3 + r1,
    r3 - r1,
    r2,
    r3 - r2
  );

  for (var i = 0; i < 6; i++) {
    let p = planes[i];
    if (dot(p.xyz, c) + p.w < -r * length(p.xyz)) {
      return true;
    }
  }
  
  return false;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let cell = textureLoad(stateTex, vec2i(gid.xy), 0);
  if (cell.r == 0u) {
    return;
  }
  let rnd = rand3(vec3u(gid.xy, 1234u));
  let uv = (vec2f(gid.xy) + 0.05 + 0.9 * rnd.xy) / f32(GRID);
  let ground = sampleHeight(heightTex, uv);
  // the ground has sunk under water since this plant was placed: not drawn
  if (ground <= WATER_LEVEL) {
    return;
  }
  let pos = vec3f((uv.x - 0.5) * WORLD_SIZE, ground * HEIGHT_SCALE, (uv.y - 0.5) * WORLD_SIZE);

  // Lifecycle
  let lc = lifecycle(gid.xy, plantSeed(cell.a), anim.lifespan);
  let age = f32(cell.g) + anim.stepPhase;
  let growth = smoothstep(0.0, lc.growSteps, age);
  let open = smoothstep(lc.bloomStart, lc.bloomStart + lc.bloomSteps, age);

  var wilt = 0.0;

  let left = wiltLeft(cell.a);

  if (left > 0u) {
    let w = f32(lc.wiltSteps);
    wilt = clamp((w - f32(left) + anim.stepPhase) / w, 0.0, 1.0);
  }

  let viability = f32(cell.b) / 255.0;

  // shrinks away over the last part of the wilt, so it is gone by the time the cell clears
  let size = growth * (0.6 + 0.6 * viability) * (1.0 - smoothstep(0.6, 1.0, wilt));

  // Cull before slot because culled plants will never be drawn
  let fp = flowers[min(cell.r - 1u, MAX_SPECIES - 1u)];
  let radius = fp.boundRadius * size;
  let center = pos + vec3f(0.0, fp.boundCenterY * size, 0.0);
  if (radius <= 0.0 || outsideFrustum(center, radius)) {
    return;
  }

  let pixels = radius * cam.pixelScale / max(distance(center, cam.pos), 1e-3);

  if (pixels < CULL_PIXELS) {
    return;
  }

  let fade = smoothstep(CULL_PIXELS, FADE_PIXELS, pixels);

  let slot = atomicAdd(&draw.instanceCount, 1u);

  instances[slot] = PlantInstance(pos, cell.r, rnd.z * 6.2831853, size * fade, open, wilt);
}
