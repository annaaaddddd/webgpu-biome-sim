import { mat4, vec3 } from 'wgpu-matrix';
import { device, canvas } from './device';

// Shared camera, created once in main.ts after initWebGPU().
// Orbit: drag to rotate, scroll to zoom.
// Walking: the eye sits at target, drag to look, WASD to move, Shift to run.
// Uniform layout: viewProj (mat4x4f) + position (vec3f) + pad = 80 bytes.
export let camera: Camera;

const WALK_SPEED = 5; // world units per second
const RUN_FACTOR = 3;

export function initCamera() {
  camera = new Camera();
}

export class Camera {
  target = vec3.create(0, 0, 0);
  distance = 10;
  yaw = 0.6;
  pitch = 0.5;
  fov = Math.PI / 4;
  walking = false;

  uniformBuffer: GPUBuffer;
  private data = new Float32Array(20);
  private keys = new Set<string>();

  constructor() {
    this.uniformBuffer = device.createBuffer({
      size: this.data.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
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
    window.addEventListener('keydown', (e) => this.keys.add(e.code));
    window.addEventListener('keyup', (e) => this.keys.delete(e.code));
    window.addEventListener('blur', () => this.keys.clear());
  }

  // Unit vector from the target toward the orbiting eye. The view looks the other way.
  private get back() {
    const cp = Math.cos(this.pitch);
    return vec3.create(cp * Math.sin(this.yaw), Math.sin(this.pitch), cp * Math.cos(this.yaw));
  }

  get position() {
    return this.walking ? this.target : vec3.addScaled(this.target, this.back, this.distance);
  }

  // Moves the target on the ground plane. Whoever owns the terrain sets target y.
  private walk(dt: number) {
    const axis = (plus: string, minus: string) => +this.keys.has(plus) - +this.keys.has(minus);
    const forward = axis('KeyW', 'KeyS');
    const right = axis('KeyD', 'KeyA');
    const step = WALK_SPEED * (this.keys.has('ShiftLeft') ? RUN_FACTOR : 1) * dt;
    const s = Math.sin(this.yaw);
    const c = Math.cos(this.yaw);
    this.target[0] += (c * right - s * forward) * step;
    this.target[2] -= (s * right + c * forward) * step;
  }

  update(dt: number) {
    if (this.walking) this.walk(dt);
    const pos = this.position;
    const view = mat4.lookAt(pos, vec3.subtract(pos, this.back), vec3.create(0, 1, 0));
    const proj = mat4.perspective(this.fov, canvas.width / canvas.height, 0.1, 1000);
    this.data.set(mat4.multiply(proj, view), 0);
    this.data.set(pos, 16);
    this.data[19] = canvas.height / (2 * Math.tan(this.fov / 2)) // camera.pixelScale
    device.queue.writeBuffer(this.uniformBuffer, 0, this.data);
  }
}
