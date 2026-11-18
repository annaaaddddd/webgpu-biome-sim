// CloudParams and cloudDensity() are in common.wgsl.

struct VSOut {
  @builtin(position) clip: vec4f,
  @location(0) world: vec2f,
  @location(1) t: f32, // 0 at the bottom layer, 1 at the top
}

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<uniform> clouds: CloudParams;

// A stack of flat quads above the terrain, one per instance, centered on the map.
@vertex
fn vs_main(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> VSOut {
  // vertex_index 0..5 -> two triangles of a unit quad.
  let corner = vec2f(f32(v & 1u), f32(v >= 2u && v <= 4u));
  let xz = (corner - 0.5) * clouds.size;

  let below = cam.pos.y < clouds.height;
  let layer = select(f32(i), clouds.layers - 1.0 - f32(i), below);
  let levelh = clouds.height + layer * clouds.spacing;

  var out: VSOut;
  out.clip = cam.viewProj * vec4f(xz.x, levelh, xz.y, 1);
  out.world = xz;
  out.t = layer / max((clouds.layers - 1), 1.0);
  return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4f {
  // Higher layers get a higher cutoff, so the stack is wide at the base and
  // narrow at the top.
  let density = cloudDensity(in.world, clouds, in.t * 0.06);

  // Grey underside, white top.
  let color = mix(vec3f(0.78, 0.81, 0.86), vec3f(1.0), in.t);
  // Chosen so N stacked layers at full density composite to clouds.alpha.
  let layerAlpha = 1.0 - pow(1.0 - clouds.alpha, 1.0 / clouds.layers);
  return vec4f(color, density * layerAlpha);
}
