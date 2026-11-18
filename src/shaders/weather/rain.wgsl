struct RainParams {
  time: f32,
  speed: f32,
  length: f32,
  width: f32,
  alpha: f32,
  top: f32
}

struct VSOut {
  @builtin(position) clip: vec4f,
  @location(0) t: f32,
}

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var heightTex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> rain: RainParams;

@vertex
fn vs_main(@builtin(instance_index) i: u32, @builtin(vertex_index) v: u32) -> VSOut {
  let r = rand3(vec3u(i, 7u, 13u));
  let xz = (r.xy - 0.5) * WORLD_SIZE;
  let uv = r.xy;
  let ground = sampleHeight(heightTex, uv) * HEIGHT_SCALE;
  let span = rain.top - ground;

  // Each drop loops top -> ground; r.z is its phase so they are spread out.
  let fall = fract(r.z + rain.time * rain.speed / span);
  let y = rain.top - fall * span;

  // Quad: 2 triangles, corner from the vertex index. x across, y along the fall.
  let corner = vec2f(f32(v & 1u), f32(v >= 2u && v <= 4u));   // (0,0) (1,0) (0,1) (1,1) (0,1) (1,0)
  let right = normalize(cross(vec3f(0, 1, 0), cam.pos - vec3f(xz.x, y, xz.y)));
  let pos = vec3f(xz.x, y, xz.y) + right * (corner.x - 0.5) * rain.width + vec3f(0, corner.y * rain.length, 0);

  var out: VSOut;
  out.clip = cam.viewProj * vec4f(pos, 1);
  out.t = corner.y;          // 0 at the drop's head, 1 at the tail
  return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4f {
  return vec4f(0.85, 0.9, 1.0, rain.alpha * (1 - in.t));
}
