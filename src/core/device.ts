export let device: GPUDevice;
export let canvas: HTMLCanvasElement;
export let context: GPUCanvasContext;
export let canvasFormat: GPUTextureFormat;

export async function initWebGPU(canvasId = 'mainCanvas') {
  if (!navigator.gpu) {
    throw new Error('WebGPU is not supported in this browser');
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error('No WebGPU adapter found');
  }

  // tracks time individual passes on the GPU
  const requiredFeatures: GPUFeatureName[] = adapter.features.has('timestamp-query') ? ['timestamp-query'] : [];
  device = await adapter.requestDevice({ requiredFeatures });
  device.addEventListener('uncapturederror', (e) => console.error(e.error.message));
  device.lost.then((info) => {
    console.error(`WebGPU device lost: ${info.message}`);
  });

  canvas = document.getElementById(canvasId) as HTMLCanvasElement;
  context = canvas.getContext('webgpu') as GPUCanvasContext;
  canvasFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format: canvasFormat });

  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);
}

export function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(canvas.clientWidth * dpr);
  canvas.height = Math.floor(canvas.clientHeight * dpr);
}
