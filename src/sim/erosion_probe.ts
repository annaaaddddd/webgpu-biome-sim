import { device } from '../core/device';

// Mass bookkeeping for the erosion sim, enabled with ?probe. Once a second it
// reads the whole state texture back and logs the grid totals of bedrock,
// sediment and water. The map edges are open to the sea, so bedrock + sediment
// falls slowly as rivers carry sediment off the map; a rise, a sudden drop or
// a NaN points at a pass that does not conserve material.
//
// Usage per frame, after the sim steps have been encoded:
//   probe.frame(encoder, stateTex);
export class MassProbe {
  private readBuf: GPUBuffer;
  private reading = false;
  private copied = false;
  private frames = 0;
  private start = performance.now();
  private base?: { land: number; sed: number };
  readonly size: number;
  readonly every: number;

  static fromUrl(size: number) {
    const query = new URLSearchParams(location.search);
    return query.has('probe') ? new MassProbe(size, 60) : undefined;
  }

  constructor(size: number, every: number) {
    this.size = size;
    this.every = every;
    // rgba32float rows are 16 * size bytes, already a multiple of 256 for size >= 16.
    this.readBuf = device.createBuffer({ size: size * size * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  }

  frame(encoder: GPUCommandEncoder, state: GPUTexture) {
    if (this.copied && !this.reading) this.read();
    if (++this.frames % this.every !== 0 || this.reading || this.copied) return;
    encoder.copyTextureToBuffer({ texture: state }, { buffer: this.readBuf, bytesPerRow: this.size * 16 }, [this.size, this.size]);
    this.copied = true;
  }

  private read() {
    this.copied = false;
    this.reading = true;
    this.readBuf.mapAsync(GPUMapMode.READ).then(() => {
      const f = new Float32Array(this.readBuf.getMappedRange());
      let land = 0, water = 0, sed = 0, maxWater = 0;
      for (let i = 0; i < f.length; i += 4) {
        land += f[i];
        water += f[i + 1];
        sed += f[i + 2];
        if (f[i + 1] > maxWater) maxWater = f[i + 1];
      }
      this.readBuf.unmap();
      this.reading = false;

      const n = this.size * this.size;
      this.base ??= { land, sed };
      const t = ((performance.now() - this.start) / 1000).toFixed(0);
      const total = land + sed;
      const drift = total - (this.base.land + this.base.sed);
      console.log(
        `probe t=${t}s  land ${(land / n).toFixed(5)}  sed ${(sed / n).toFixed(5)}  land+sed ${(total / n).toFixed(5)}` +
        `  drift ${(drift / n).toExponential(2)}  water ${(water / n).toFixed(5)}  maxWater ${maxWater.toFixed(3)}`,
      );
    });
  }
}
