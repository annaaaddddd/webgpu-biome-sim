// Draws the heightfield as a GRID x GRID mesh. No vertex buffer: the vertex
// index picks the grid point and the height comes from heightTex.

struct VSOut {
  @builtin(position) position: vec4f,
  @location(0) height: f32,
  @location(1) worldPos: vec3f,
  @location(2) uv: vec2f,
}
// Matches Camera.uniformBuffer layout in core/camera.ts.
struct Camera { viewProj: mat4x4f, pos: vec3f }
// view: 0 shaded, 1 height grayscale, 2 moisture only.
struct TerrainUniforms { heightScale: f32, view: f32 }

// Grid resolution and world extent, set from TS via pipeline constants.
override GRID: u32 = 512;
override WORLD_SIZE: f32 = 100.0;

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var heightTex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> terrainUniforms: TerrainUniforms;
@group(0) @binding(3) var moistureTex: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
  // Row-major grid: vi = z * GRID + x.
  let x = vi % GRID;
  let z = vi / GRID;
  let h = textureLoad(heightTex, vec2u(x, z), 0).r;

  // Texel -> uv in [0, 1] -> world xz in [-WORLD_SIZE/2, WORLD_SIZE/2].
  let uv = vec2f(f32(x), f32(z)) / f32(GRID - 1);
  let xz = (uv - 0.5) * WORLD_SIZE;
  let pos = vec3f(xz.x, h * terrainUniforms.heightScale, xz.y);

  var out: VSOut;
  out.position = cam.viewProj * vec4f(pos, 1);
  out.height = h;
  out.worldPos = pos;
  out.uv = uv;
  return out;
}

// Shaded view: green at low height, white at high, tinted blue where wet,
// lit by a fixed directional light. The normal comes from screen-space
// derivatives of the world position (flat-shaded per triangle).
// Debug views skip the lighting so the data reads directly as color.
@fragment
fn fs_main(in: VSOut) -> @location(0) vec4f {
  let m = textureLoad(moistureTex, vec2u(in.uv * f32(GRID - 1)), 0).r;
  let view = u32(terrainUniforms.view);

  if (view == 1u) {
    return vec4f(vec3f(in.height), 1.0);
  }
  if (view == 2u) {
    return vec4f(mix(vec3f(0.05), vec3f(0.2, 0.5, 1.0), m), 1.0);
  }

  // dpdy first: WebGPU's framebuffer y points down, so this order gives an
  // upward-facing normal.
  let n = normalize(cross(dpdy(in.worldPos), dpdx(in.worldPos)));
  let light = max(dot(n, normalize(vec3f(1.0, 1.0, 0.5))), 0.0) * 0.8 + 0.2;
  var albedo = mix(vec3f(0.2, 0.5, 0.2), vec3f(1.0), in.height);
  albedo = mix(albedo, vec3f(0.1, 0.3, 0.8), m);
  return vec4f(albedo * light, 1.0);
}
