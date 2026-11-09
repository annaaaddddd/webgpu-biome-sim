struct SimParams {
  dt: f32,
  rainRate: f32,
  moistureScale: f32,
  
  pipeCoeff: f32,
  cellSize: f32,

  Kc: f32,
  Ks: f32,
  Kd: f32,
  Ke: f32,
  Kt: f32,
  minSlope: f32,
  maxDepth: f32,
  talus: f32,       // max height difference (in height units) between neighbors before material slides
  moistureTau: f32, // seconds of sim time for moisture to follow water depth (export pass only)
}

@group(0) @binding(0) var<uniform> sim: SimParams;
@group(0) @binding(1) var stateIn: texture_2d<f32>;
@group(0) @binding(2) var stateOut: texture_storage_2d<rgba32float, write>;

@group(0) @binding(3) var fluxIn: texture_2d<f32>;
@group(0) @binding(4) var fluxOut: texture_storage_2d<rgba32float, write>;

@group(0) @binding(5) var velocityOut: texture_storage_2d<rg32float, write>;
@group(0) @binding(6) var velocityIn: texture_2d<f32>;

@compute @workgroup_size(8, 8)
fn rain(@builtin(global_invocation_id) id: vec3u) {
  let dims = textureDimensions(stateIn);
  if (id.x >= dims.x || id.y >= dims.y) { return; }


  var cell = textureLoad(stateIn, id.xy, 0);
  cell.g += sim.rainRate * sim.dt;
  textureStore(stateOut, id.xy, cell);
}

@compute @workgroup_size(8, 8)
fn flux(@builtin(global_invocation_id) id: vec3u) {
  let dims = vec2i(textureDimensions(stateIn));
  let p = vec2i(id.xy);
  if (any(p >= dims)) { return; }

  let me = textureLoad(stateIn, p, 0);           // me.r = b, me.g = d
  let h = me.r + me.g;           // water level
  var f = textureLoad(fluxIn, p, 0);             // old 4 flux

  // left flux
  let ql = vec2i(id.xy) + vec2i(-1, 0);
  if (all(ql >= vec2i(0)) && all(ql < dims)) {
    let nb = textureLoad(stateIn, ql, 0);
    let dh = h - (nb.r + nb.g);
    f.r = max(0, f.r + sim.dt * sim.pipeCoeff * dh);
  }
  else {
    f.r = 0;
  }

  // right flux
  let qr = vec2i(id.xy) + vec2i(1, 0);
  if (all(qr >= vec2i(0)) && all(qr < dims)) {
    let nb = textureLoad(stateIn, qr, 0);
    let dh = h - (nb.r + nb.g);
    f.g = max(0, f.g + sim.dt * sim.pipeCoeff * dh);
  }
  else {
    f.g = 0;
  }

  // top flux
  let qt = vec2i(id.xy) + vec2i(0, -1);
  if (all(qt >= vec2i(0)) && all(qt < dims)) {
    let nb = textureLoad(stateIn, qt, 0);
    let dh = h - (nb.r + nb.g);
    f.b = max(0, f.b + sim.dt * sim.pipeCoeff * dh);
  }
  else {
    f.b = 0;
  }

  // bottom flux
  let qb = vec2i(id.xy) + vec2i(0, 1);
  if (all(qb >= vec2i(0)) && all(qb < dims)) {
    let nb = textureLoad(stateIn, qb, 0);
    let dh = h - (nb.r + nb.g);
    f.a = max(0, f.a + sim.dt * sim.pipeCoeff * dh);
  }
  else {
    f.a = 0;
  }

  let total = f.r + f.g + f.b + f.a;
  if (total > 0){
    let K = min(1, me.g * sim.cellSize * sim.cellSize / (total * sim.dt));
    f *= K;
  }

  textureStore(fluxOut, p, f);
}

@compute @workgroup_size(8, 8)
fn water(@builtin(global_invocation_id) id: vec3u) {
  let dims = vec2i(textureDimensions(stateIn));
  let p = vec2i(id.xy);
  if (any(p >= dims)) { return; }

  let me = textureLoad(stateIn, p, 0);
  let f = textureLoad(fluxIn, p, 0);

  // Inflow from each side = that neighbor's outflow in the opposite direction.
  // No neighbor past the boundary -> no inflow.
  var in_L = 0.0;  // -x neighbor's +x outflow (.g)
  var in_R = 0.0;  // +x neighbor's -x outflow (.r)
  var in_D = 0.0;  // -y neighbor's +y outflow (.a)
  var in_U = 0.0;  // +y neighbor's -y outflow (.b)
  if (p.x > 0)          { in_L = textureLoad(fluxIn, p + vec2i(-1, 0), 0).g; }
  if (p.x < dims.x - 1) { in_R = textureLoad(fluxIn, p + vec2i( 1, 0), 0).r; }
  if (p.y > 0)          { in_D = textureLoad(fluxIn, p + vec2i(0, -1), 0).a; }
  if (p.y < dims.y - 1) { in_U = textureLoad(fluxIn, p + vec2i(0,  1), 0).b; }

  let inSum  = in_L + in_R + in_D + in_U;
  let outSum = f.r + f.g + f.b + f.a;
  let dV = sim.dt * (inSum - outSum);
  let d2 = max(0, me.g + dV / (sim.cellSize * sim.cellSize));

  let dBar = (me.g + d2) / 2;
  var u = 0.0;
  var v = 0.0;
  if (dBar > 1e-3) {
    u = (in_L - f.r + f.g - in_R) / 2 / (sim.cellSize * dBar);
    v = (in_D - f.b + f.a - in_U) / 2 / (sim.cellSize * dBar);
  }

  // CFL clamp: water may cross at most one cell per step. Shallow cells
  // otherwise produce huge velocities (u ~ 1/dBar).
  let maxSpeed = sim.cellSize / sim.dt;
  let speed = length(vec2f(u, v));
  if (speed > maxSpeed) {
    u *= maxSpeed / speed;
    v *= maxSpeed / speed;
  }

  textureStore(stateOut, p, vec4f(me.r, d2, me.b, 0));
  textureStore(velocityOut, p, vec4f(u, v, 0, 0));
}

@compute @workgroup_size(8, 8)
fn erodeDeposit(@builtin(global_invocation_id) id: vec3u) {
  let dims = vec2i(textureDimensions(stateIn));
  let p = vec2i(id.xy);
  if (any(p >= dims)) { return; }

  let me = textureLoad(stateIn, p, 0);
  let vel = textureLoad(velocityIn, p, 0).xy;
  let speed = length(vel);

  // Neighbor terrain heights for the slope. Past the boundary there is no
  // neighbor, so use our own height: that side contributes zero slope
  // (compare water(), where "no neighbor" means zero inflow instead).
  var b_L = me.r;
  var b_R = me.r;
  var b_D = me.r;
  var b_U = me.r;
  if (p.x > 0)          { b_L = textureLoad(stateIn, p + vec2i(-1, 0), 0).r; }
  if (p.x < dims.x - 1) { b_R = textureLoad(stateIn, p + vec2i( 1, 0), 0).r; }
  if (p.y > 0)          { b_D = textureLoad(stateIn, p + vec2i(0, -1), 0).r; }
  if (p.y < dims.y - 1) { b_U = textureLoad(stateIn, p + vec2i(0,  1), 0).r; }

  let grad = vec2f(b_R - b_L, b_U - b_D) / (2 * sim.cellSize);
  var sinAlpha = length(grad) / sqrt(1 + dot(grad, grad));
  sinAlpha = max(sinAlpha, sim.minSlope);

  // Shallow water carries little: fade capacity in over the first `maxDepth`
  // of water so a thin film of rain can't carve pillars (Jako & Toth 2011).
  let depthFactor = clamp(me.g / sim.maxDepth, 0.0, 1.0);
  let c = sim.Kc * sinAlpha * speed * depthFactor;

  var b = me.r;
  var s = me.b;
  var delta = 0.0;
  if (c > s) {
    delta = sim.Ks * (c - s) * sim.dt;
    delta = min(delta, b);
    b -= delta;
    s += delta;
  }
  else {
    delta = sim.Kd * (s - c) * sim.dt;
    b += delta;
    s -= delta;
  }
  b = max(b, 0);

  textureStore(stateOut, p, vec4f(b, me.g, s, 0));
}

@compute @workgroup_size(8, 8)
fn transport(@builtin(global_invocation_id) id: vec3u) {
  let dims = vec2i(textureDimensions(stateIn));
  let p = vec2i(id.xy);
  if (any(p >= dims)) { return; }

  let me = textureLoad(stateIn, p, 0);
  let vel = textureLoad(velocityIn, p, 0).xy;

  var src = vec2f(p) - vel * sim.dt / sim.cellSize;
  src = clamp(src, vec2f(0), vec2f(dims - 1));

  let i0 = vec2i(floor(src));
  let i1 = min(i0 + 1, dims - 1);
  let f = fract(src);
  let s00 = textureLoad(stateIn, i0, 0).b;
  let s10 = textureLoad(stateIn, vec2i(i1.x, i0.y), 0).b;
  let s01 = textureLoad(stateIn, vec2i(i0.x, i1.y), 0).b;
  let s11 = textureLoad(stateIn, i1, 0).b;
  let s = mix(mix(s00, s10, f.x), mix(s01, s11, f.x), f.y);

  let d = me.g * (1 - sim.Ke * sim.dt);

  textureStore(stateOut, p, vec4f(me.r, d, s, 0));
}

@compute @workgroup_size(8, 8)
fn thermal(@builtin(global_invocation_id) id: vec3u) {
  let dims = vec2i(textureDimensions(stateIn));
  let p = vec2i(id.xy);
  if (any(p >= dims)) { return; }

  let me = textureLoad(stateIn, p, 0);
  var b = me.r;
  let talus = sim.talus;

  // Symmetric talus transfer: for each neighbor pair, the higher cell loses
  // half the excess and the lower cell gains it. The neighbor computes the
  // same pair with the opposite sign, so material is conserved in one pass.
  // Past the boundary there is no neighbor, so nothing moves that way.
  var dTotal = 0.0;
  if (p.x > 0) {
    let diff = b - textureLoad(stateIn, p + vec2i(-1, 0), 0).r;
    let excess = abs(diff) - talus;
    if (excess > 0.0) { dTotal -= sign(diff) * sim.Kt * excess * sim.dt * 0.5; }
  }
  if (p.x < dims.x - 1) {
    let diff = b - textureLoad(stateIn, p + vec2i(1, 0), 0).r;
    let excess = abs(diff) - talus;
    if (excess > 0.0) { dTotal -= sign(diff) * sim.Kt * excess * sim.dt * 0.5; }
  }
  if (p.y > 0) {
    let diff = b - textureLoad(stateIn, p + vec2i(0, -1), 0).r;
    let excess = abs(diff) - talus;
    if (excess > 0.0) { dTotal -= sign(diff) * sim.Kt * excess * sim.dt * 0.5; }
  }
  if (p.y < dims.y - 1) {
    let diff = b - textureLoad(stateIn, p + vec2i(0, 1), 0).r;
    let excess = abs(diff) - talus;
    if (excess > 0.0) { dTotal -= sign(diff) * sim.Kt * excess * sim.dt * 0.5; }
  }

  b = max(b + dTotal, 0.0);
  textureStore(stateOut, p, vec4f(b, me.g, me.b, 0));
}
