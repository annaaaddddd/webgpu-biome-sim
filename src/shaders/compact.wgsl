// Packs every occupied cell into the PlantInstance list. The indirect draw's instanceCount
// doubles as the atomic counter.

struct DrawArgs {
  indexCount: u32,
  instanceCount: atomic<u32>,
  firstIndex: u32,
  baseVertex: u32,
  firstInstance: u32,
}

@group(0) @binding(0) var heightTex: texture_2d<f32>;
@group(0) @binding(1) var stateTex: texture_2d<u32>;
@group(0) @binding(2) var<storage, read_write> instances: array<PlantInstance>;
@group(0) @binding(3) var<storage, read_write> draw: DrawArgs;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let cell = textureLoad(stateTex, vec2i(gid.xy), 0);
  if (cell.r == 0u) {
    return;
  }
  let rnd = rand3(vec3u(gid.xy, 1234u));
  let uv = (vec2f(gid.xy) + 0.05 + 0.9 * rnd.xy) / f32(GRID);
  let pos = vec3f((uv.x - 0.5) * WORLD_SIZE, sampleHeight(heightTex, uv) * HEIGHT_SCALE, (uv.y - 0.5) * WORLD_SIZE);
  let slot = atomicAdd(&draw.instanceCount, 1u);
  instances[slot] = PlantInstance(pos, cell.r, rnd.z * 6.2831853, f32(cell.g), f32(cell.b) / 255.0, 0.0);
}
