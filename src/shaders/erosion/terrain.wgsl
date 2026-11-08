// Draws the heightfield as an N x N grid mesh. No vertex buffer: the vertex
// index picks the grid point and the height comes from heightTex.

struct VSOut {
  @builtin(position) position: vec4f,
  @location(0) height: f32,
  @location(1) worldPos: vec3f,
}
// Matches Camera.uniformBuffer layout in core/camera.ts.
struct Camera { viewProj: mat4x4f, pos: vec3f }
struct TerrainUniforms { heightScale: f32 }

// Grid size, set from TS via pipeline constants.
override N: u32 = 256;

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var heightTex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> terrainUniforms: TerrainUniforms;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
  // Row-major grid: vi = z * N + x.
  let x = vi % N;
  let z = vi / N;
  let h = textureLoad(heightTex, vec2u(x, z), 0).r;

  // Center the grid on the origin, one world unit per texel.
  let pos = vec3f(f32(x) - f32(N)/2, h * terrainUniforms.heightScale, f32(z) - f32(N)/2);

  var out: VSOut;
  out.position = cam.viewProj * vec4f(pos, 1);
  out.height = h;
  out.worldPos = pos;
  return out;
}

// Placeholder shading: green at low height, white at high.
@fragment
fn fs_main(in: VSOut) -> @location(0) vec4f {
  return vec4f(mix(vec3f(0.2, 0.5, 0.2), vec3f(1), in.height), 1);
}
