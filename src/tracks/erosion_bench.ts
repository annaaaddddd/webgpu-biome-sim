import { device } from '../core/device';

// GPU timing for compute passes, driven by URL params:
//   ?bench=N      record N samples, then print mean/median/p95 per pass to the
//                 console and download the raw samples as CSV
//   &warmup=M     skip the first M readings (default 60)
// Needs the 'timestamp-query' feature, requested in device.ts when available.
// Without Chrome's WebGPU developer features flag the timestamps are coarsely
// quantized and the numbers are not meaningful.
//
// Usage per frame:
//   bench.beginFrame();                               // maps last frame's readback
//   encoder.beginComputePass({ timestampWrites: bench.writes(slot) });
//   bench.endFrame(encoder);                          // resolves this frame's queries
// A timer that has finished becomes a no-op, so callers do not need to check.

export interface BenchOptions {
  samples: number;
  warmup: number;
  stepPasses: number; // the first N passes make up one sim step; the rest are per-frame
  label: string;      // printed with the table, e.g. "erosion 512x512, 10 steps/frame"
  file: string;       // download name for the CSV
}

export class PassTimer {
  private querySet: GPUQuerySet;
  private resolveBuf: GPUBuffer;
  private readBuf: GPUBuffer;
  private copied = false;  // resolve + copy was encoded this frame
  private reading = false; // readBuf is mapped or about to be
  private samples: number[][] = [];
  private warmup: number;
  private done = false;
  readonly names: readonly string[];
  readonly opts: BenchOptions;

  // Returns undefined when the URL has no ?bench=N.
  static fromUrl(names: readonly string[], opts: Omit<BenchOptions, 'samples' | 'warmup'>) {
    const query = new URLSearchParams(location.search);
    const samples = Number(query.get('bench') ?? 0);
    if (samples <= 0) return undefined;
    const warmup = Number(query.get('warmup') ?? 60);
    return new PassTimer(names, { ...opts, samples, warmup });
  }

  constructor(names: readonly string[], opts: BenchOptions) {
    if (!device.features.has('timestamp-query')) throw new Error('bench mode needs the timestamp-query feature');
    this.names = names;
    this.opts = opts;
    const count = names.length * 2;
    this.querySet = device.createQuerySet({ type: 'timestamp', count });
    this.resolveBuf = device.createBuffer({ size: count * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    this.readBuf = device.createBuffer({ size: count * 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this.warmup = opts.warmup;
  }

  // Timestamp writes for pass `slot` (index into names); undefined when not timing.
  writes(slot?: number): GPUComputePassTimestampWrites | undefined {
    if (this.done || slot === undefined) return undefined;
    return { querySet: this.querySet, beginningOfPassWriteIndex: slot * 2, endOfPassWriteIndex: slot * 2 + 1 };
  }

  // Call at the start of a frame. Maps the previous frame's readback, so the
  // buffer is never both a copy target and mapped within one submit.
  beginFrame() {
    if (this.done || !this.copied || this.reading) return;
    this.copied = false;
    this.reading = true;
    this.readBuf.mapAsync(GPUMapMode.READ).then(() => {
      const t = new BigInt64Array(this.readBuf.getMappedRange());
      const ms = this.names.map((_, i) => Number(t[i * 2 + 1] - t[i * 2]) / 1e6);
      this.readBuf.unmap();
      this.reading = false;

      if (this.warmup > 0) this.warmup--;
      else this.samples.push(ms);
      if (this.samples.length >= this.opts.samples) this.finish();
    });
  }

  // Call after the timed passes have been encoded.
  endFrame(encoder: GPUCommandEncoder) {
    if (this.done || this.reading) return;
    const count = this.names.length * 2;
    encoder.resolveQuerySet(this.querySet, 0, count, this.resolveBuf, 0);
    encoder.copyBufferToBuffer(this.resolveBuf, 0, this.readBuf, 0, count * 8);
    this.copied = true;
  }

  private finish() {
    const stat = (xs: number[]) => {
      const s = [...xs].sort((a, b) => a - b);
      const q = (p: number) => s[Math.min(s.length - 1, Math.floor(p * s.length))];
      return { mean: xs.reduce((a, b) => a + b, 0) / xs.length, median: q(0.5), p95: q(0.95) };
    };
    const rows = this.names.map((pass, i) => ({ pass, ...stat(this.samples.map((f) => f[i])) }));
    const step = this.samples.map((f) => f.slice(0, this.opts.stepPasses).reduce((a, b) => a + b, 0));
    rows.push({ pass: 'step', ...stat(step) });
    console.log(`bench: ${this.opts.label}, ${this.samples.length} samples, ms per pass`);
    console.table(rows);

    const csv = [
      ['sample', ...this.names, 'step'].join(','),
      ...this.samples.map((f, i) => [i, ...f.map((x) => x.toFixed(4)), step[i].toFixed(4)].join(',')),
    ].join('\n');
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
    a.download = this.opts.file;
    a.click();

    this.querySet.destroy();
    this.resolveBuf.destroy();
    this.readBuf.destroy();
    this.samples = [];
    this.done = true;
  }
}
