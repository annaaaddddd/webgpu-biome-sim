// Instanced placeholder flowers positioned by the compact pass. The real flower mesh
// replaces the vertex buffer, the per-instance data stays the same.

struct VSIn {
  @location(0) pos: vec3f,
  @location(1) normal: vec3f,
  @location(2) part: f32,
}

struct VSOut {
  @builtin(position) clip: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
  @location(2) color: vec3f,
}

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<storage, read> instances: array<PlantInstance>;

const PLANT_SIZE = 1.0;
const STEM_COLOR = vec3f(0.25, 0.45, 0.18);

@vertex
fn vs(in: VSIn, @builtin(instance_index) ii: u32) -> VSOut {
  let inst = instances[ii];
  let size = PLANT_SIZE * smoothstep(0.0, 4.0, inst.age) * (0.6 + 0.6 * inst.viability);
  let c = cos(inst.rotation);
  let s = sin(inst.rotation);
  let rot = mat3x3f(vec3f(c, 0.0, s), vec3f(0.0, 1.0, 0.0), vec3f(-s, 0.0, c));
  var out: VSOut;
  out.world = inst.pos + rot * in.pos * size;
  out.normal = rot * in.normal;
  out.color = select(speciesColor(inst.species), STEM_COLOR, in.part < 0.5);
  out.clip = cam.viewProj * vec4f(out.world, 1.0);
  return out;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  return vec4f(gamma(shade(in.color, normalize(in.normal), in.world, cam.pos)), 1.0);
}
