struct SimParams {
  dt: f32,
  rainRate: f32,
  moistureScale: f32,
}

@group(0) @binding(0) var<uniform> sim: SimParams;
@group(0) @binding(1) var stateIn: texture_2d<f32>;
@group(0) @binding(2) var stateOut: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8)
fn rain(@builtin(global_invocation_id) id: vec3u) {
  let dims = textureDimensions(stateIn);
  if (id.x >= dims.x || id.y >= dims.y) { return; }


  var cell = textureLoad(stateIn, id.xy, 0);
  cell.g += sim.rainRate * sim.dt;
  textureStore(stateOut, id.xy, cell);
}
