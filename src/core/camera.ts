import { mat4, vec3 } from 'wgpu-matrix';
import { device, canvas } from './device';

// Shared orbit camera, created once in main.ts after initWebGPU().
// Drag to rotate, scroll to zoom.
// Uniform layout: viewProj (mat4x4f) + position (vec3f) + pad = 80 bytes.
export let camera: Camera;

export function initCamera() {
  camera = new Camera();
}

export class Camera {
  target = vec3.create(0, 0, 0);
  distance = 10;
  yaw = 0.6;
  pitch = 0.5;
  fov = Math.PI / 4;

  uniformBuffer: GPUBuffer;
  private data = new Float32Array(20);

  constructor() {
    this.uniformBuffer = device.createBuffer({
      size: this.data.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    let dragging = false;
    canvas.addEventListener('mousedown', () => (dragging = true));
    window.addEventListener('mouseup', () => (dragging = false));
    window.addEventListener('mousemove', (e) => {
      if (!dragging) return;
      this.yaw -= e.movementX * 0.005;
      this.pitch = Math.min(1.5, Math.max(-1.5, this.pitch + e.movementY * 0.005));
    });
    canvas.addEventListener('wheel', (e) => {
      this.distance *= 1 + e.deltaY * 0.001;
    });
  }

  get position() {
    const cp = Math.cos(this.pitch);
    return vec3.add(
      this.target,
      vec3.create(this.distance * cp * Math.sin(this.yaw), this.distance * Math.sin(this.pitch), this.distance * cp * Math.cos(this.yaw)),
    );
  }

  update() {
    const pos = this.position;
    const view = mat4.lookAt(pos, this.target, vec3.create(0, 1, 0));
    const proj = mat4.perspective(this.fov, canvas.width / canvas.height, 0.1, 1000);
    this.data.set(mat4.multiply(proj, view), 0);
    this.data.set(pos, 16);
    device.queue.writeBuffer(this.uniformBuffer, 0, this.data);
  }
}
