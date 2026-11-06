import type { GUI } from 'lil-gui';

// One per M1 track. init() sets up buffers/pipelines and GUI controls;
// frame() records the track's passes into the encoder each frame.
export interface Track {
  init(gui: GUI): Promise<void> | void;
  frame(encoder: GPUCommandEncoder, dt: number): void;
}
