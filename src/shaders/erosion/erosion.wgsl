struct SimParams {
  dt: f32,
  rainRate: f32,
  moistureScale: f32,
  
  pipeCoeff: f32,
  cellSize: f32,
}

@group(0) @binding(0) var<uniform> sim: SimParams;
@group(0) @binding(1) var stateIn: texture_2d<f32>;
@group(0) @binding(2) var stateOut: texture_storage_2d<rgba32float, write>;

@group(0) @binding(3) var fluxIn: texture_2d<f32>;
@group(0) @binding(4) var fluxOut: texture_storage_2d<rgba32float, write>;



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
