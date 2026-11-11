// Flowers are parametrically generated.  This means one instance per cell, and no vertex buffer.
// The vertex_index is actually split into quad patches (stem, leaves, petal, flower center).
// A patch is a (u, v) grid in [0, 1] where u goes from the base to the tip, and is placed by a surface function.
// Parts can be toggled on and off, as well as leaf and petal count.  These parts are not rasterized.

struct VSOut {
  @builtin(position) clip: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
  @location(2) color: vec3f,
}

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<storage, read> instances: array<PlantInstance>;
@group(0) @binding(2) var<storage, read> flowers: array<FlowerParams, MAX_SPECIES>;

const PI = 3.14159265;
const TAU = 6.2831853;
const GOLDEN_ANGLE = 2.3999632;

// quads per part
const STEM_Q = STEM_U * STEM_V;
const LEAF_Q = LEAF_U * LEAF_V;
const PETAL_Q = PETAL_U * PETAL_V;

const PART_STEM = 0u;
const PART_LEAF = 1u;
const PART_PETAL = 2u;
const PART_CENTER = 3u;

struct PartVertex {
  kind: u32,
  index: u32, // index which leaf or petal
  uv: vec2f
}

// manage vertex_index

// Finding corner (k) of quad (q) in the nu x nv grid
fn patchUV(q: u32, k: u32, nu: u32, nv: u32) -> vec2f {
  var corners = array<vec2u, 6>(
    vec2u(0u, 0u),
    vec2u(1u, 0u),
    vec2u(0u, 1u),
    vec2u(0u, 1u),
    vec2u(1u, 0u),
    vec2u(1u, 1u)
  );

  let c = corners[k];
  return vec2f(
    f32(q % nu + c.x) / f32(nu),
    f32(q / nu + c.y) / f32(nv)
  );
}

fn decode(vi: u32) -> PartVertex {
  var q = vi / 6u;
  let k = vi % 6u;
  if (q < STEM_Q) {
    return PartVertex(PART_STEM, 0u, patchUV(q, k, STEM_U, STEM_V));
  }

  q -= STEM_Q;
  if (q < LEAF_Q * MAX_LEAVES) {
    return PartVertex(PART_LEAF, q / LEAF_Q, patchUV(q % LEAF_Q, k, LEAF_U, LEAF_V));
  }

  q -= LEAF_Q * MAX_LEAVES;
  if (q < PETAL_Q * MAX_PETALS) {
    return PartVertex(PART_PETAL, q / PETAL_Q, patchUV(q % PETAL_Q, k, PETAL_U, PETAL_V));
  }

  q -= PETAL_Q * MAX_PETALS;
  return PartVertex(PART_CENTER, 0u, patchUV(q, k, CTR_U, CTR_V));
}

fn partEnabled(p: FlowerParams, pv: PartVertex) -> bool {
  if (pv.kind == PART_STEM) {
    return (p.flags & F_STEM) != 0u;
  }
  if (pv.kind == PART_LEAF) {
    return (p.flags & F_LEAVES) != 0u && pv.index < p.leafCount;
  }
  if (pv.kind == PART_PETAL) {
    return (p.flags & F_PETALS) != 0u && pv.index < p.petalCount;
  }

  return (p.flags & F_CENTER) != 0u;
}

fn partColor(p: FlowerParams, kind: u32) -> vec3f {
  if (kind == PART_PETAL) {
    return p.petalColor.rgb;
  }
  if (kind == PART_CENTER) {
    return p.centerColor.rgb;
  }
  return p.stemColor.rgb;
}

// SURFACE FUNCTIONS: In "flower space", y is up and base is at the flower's origin.

fn rotY(a: f32) -> mat3x3f {
  let c = cos(a);
  let s = sin(a);
  return mat3x3f(
    vec3f(c, 0.0, s), 
    vec3f(0.0, 1.0, 0.0),
    vec3f (-s, 0.0, c)
  );
}

// Find the centerline of the stem (t in [0, 1]).  Then the "head" of the stem is at spine(p, 1).
fn spine(p: FlowerParams, t: f32) -> vec3f {
  return vec3f(p. stemBend * t * t, 
  t * p.stemHeight, 0.0);
}

// stem tapering (u moves along the stem while v moves around it)
fn stemPos(p: FlowerParams, uv: vec2f) -> vec3f {
  let a = uv.y * TAU;
  let r = p.stemRadius * (1.0 - 0.4 * uv.x);
  return spine(p, uv.x) + r * vec3f(cos(a), 0.0, sin(a));
}

// Leaves will "spiral" up the stem at golden angle (u measures base -> tip while v measures across)
fn leafPos(p: FlowerParams, i: u32, uv: vec2f) -> vec3f {
  let t = mix(0.15, 0.6, (f32(i) + 0.5) / f32(max(p.leafCount, 1u)));
  let along = uv.x * p.leafLength;
  let halfW = 0.5 * p.leafWidth * sin(PI * uv.x);
  let across = (2.0 * uv.y - 1.0) * halfW;
  let fold = abs(2.0 * uv.y - 1.0) * halfW * 0.3;
  let local = vec3f(
    p.stemRadius + along,
    0.5 * along - p.leafDroop * uv.x * uv.x + fold,
    across
  );

  return spine(p, t) + rotY(f32(i) * GOLDEN_ANGLE) * local;
}

// Petals crown the head
// need to fix rotation to follow stem bend
fn petalPos(p: FlowerParams, i: u32, uv: vec2f, open: f32) -> vec3f {
  let tilt = mix(1.35, mix(0.05, 1.1, p.petalCup), open);
  let dir = vec3f(cos(tilt), sin(tilt), 0.0);
  let up = vec3f(-sin(tilt), cos(tilt), 0.0);
  let halfW = 0.5 * p.petalWidth * sin(PI * pow(uv.x, 0.6));

  let x = 2.0 * uv.y - 1.0;
  let local = vec3f(0.8 * p.centerRadius, 0.0, 0.0)
    + dir * (uv.x * p.petalLength)
    + vec3f(0.0, 0.0, x * halfW)
    + up * (x * x * halfW * 0.4);
  
  let yaw = f32(i) * TAU / f32(max(p.petalCount, 1u));

  return spine(p, 1.0) + rotY(yaw) * local;
}

// Flat-ish dome on top of stem
// currently doesn't follow stem bend
// both center and petals are currently oriented globally
fn centerPos(p: FlowerParams, uv: vec2f) -> vec3f {
  let th = uv.x * 0.5 * PI;
  let ph = uv.y * TAU;
  let local = p.centerRadius * vec3f(
    sin(th) * cos(ph),
    0.6 * cos(th),
    sin(th) * sin(ph)
  );

  return spine(p, 1.0) + local;
}

fn partPos(pv: PartVertex, p: FlowerParams, uv: vec2f, open: f32) -> vec3f {
  if (pv.kind == PART_STEM) {
    return stemPos(p, uv);
  }
  if (pv.kind == PART_LEAF) {
    return leafPos(p, pv.index, uv);
  }
  if (pv.kind == PART_PETAL) {
    return petalPos(p, pv.index, uv, open);
  }
  return centerPos(p, uv);
}

// Compute entry points

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
  let inst = instances[ii];
  let fp = flowers[min(inst.species - 1u, MAX_SPECIES - 1u)];
  let pv = decode(vi);

  var out: VSOut;
  if (!partEnabled(fp, pv)) {
    out.clip = vec4f(2.0, 2.0, 2.0, 1.0); // vertex of part goes here to zero
    return out;
  }

  let open = smoothstep(2.0, 10.0, inst.age);
  let size = smoothstep(0.0, 4.0, inst.age) * (0.6 + 0.6 * inst.viability);

  // compute normal from the surface differences; new parts need position dunction
  let e = 2e-3;
  let uvn = clamp(pv.uv, vec2f(0.02), vec2f(0.98));
  let p0 = partPos(pv, fp, uvn, open);
  let du = partPos(pv, fp, uvn + vec2f(e, 0.0), open) - p0;
  let dv = partPos(pv, fp, uvn + vec2f(0.0, e), open) - p0;

  let c = cross(dv, du);
  let n = select(vec3f(0.0, 1.0, 0.0), normalize(c), dot(c, c) > 1e-24);

  let rot = rotY(inst.rotation);
  out.world = inst.pos + rot * partPos(pv, fp, pv.uv, open) * size;
  out.normal = rot * n;
  out.color = partColor(fp, pv.kind);
  out.clip = cam.viewProj * vec4f(out.world, 1.0);
  return out;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  // light the side of the petal/leaf that faces the camera
  var n = normalize(in.normal);
  if (dot(n, cam.pos - in.world) < 0.0) {
    n = -n;
  }
  return vec4f(gamma(shade(in.color, n, in.world, cam.pos)), 1.0);
}
