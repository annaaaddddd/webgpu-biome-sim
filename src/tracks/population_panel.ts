import type { SpeciesCount } from '../sim/census';

const COLUMNS = ['', 'plants', 'viability', 'habitat', 'seeded', 'sprouted', 'stress', 'crowded', 'aged'];
const HISTORY = 240; // samples in the chart
const SAMPLE_SECONDS = 0.25;

const css = (color: number[]) => `rgb(${color.map((v) => Math.round(v * 255)).join(' ')})`;

// Bottom-left readout of the GPU census: one row per species and plants over time.
// Hidden until `visible` is set.
export class PopulationPanel {
  private root = document.createElement('div');
  private cells: HTMLTableCellElement[][];
  private chart = document.createElement('canvas');
  private ledger = document.createElement('div');
  private history: number[][] = []; // plants per species, one entry per sample
  private sinceSample = 0;

  constructor(speciesCount: number) {
    const table = document.createElement('table');
    const head = table.insertRow();
    COLUMNS.forEach((name) => (head.insertCell().textContent = name));
    this.cells = Array.from({ length: speciesCount }, () => {
      const row = table.insertRow();
      return COLUMNS.map(() => row.insertCell());
    });
    this.chart.width = 400;
    this.chart.height = 80;

    this.root.id = 'population';
    this.root.hidden = true;
    this.root.append(table, this.chart, this.ledger);
    document.body.append(this.root);
  }

  get visible() {
    return !this.root.hidden;
  }

  set visible(on: boolean) {
    this.root.hidden = !on;
  }

  update(dt: number, species: SpeciesCount[], colors: number[][]) {
    this.sinceSample += dt;
    if (this.sinceSample < SAMPLE_SECONDS || species.length === 0) return;
    this.sinceSample = 0;

    species.forEach((s, i) => {
      const text = [
        '■',
        s.alive.toLocaleString(),
        s.viability.toFixed(2),
        `${(s.habitat * 100).toFixed(1)}%`,
        ...s.events.map((n) => n.toLocaleString()),
      ];
      text.forEach((value, column) => (this.cells[i][column].textContent = value));
      this.cells[i][0].style.color = css(colors[i]);
    });

    // every plant alive was born once and has not died: births - deaths must equal the count
    const drift = species.reduce((sum, s) => {
      const [seeded, sprouted, ...deaths] = s.events;
      return sum + Math.abs(seeded + sprouted - deaths.reduce((a, b) => a + b, 0) - s.alive);
    }, 0);
    this.ledger.textContent = `births - deaths = plants: ${drift === 0 ? 'ok' : `off by ${drift}`}`;

    this.history.push(species.map((s) => s.alive));
    if (this.history.length > HISTORY) this.history.shift();
    this.draw(colors);
  }

  private draw(colors: number[][]) {
    const { width, height } = this.chart;
    const ctx = this.chart.getContext('2d')!;
    ctx.clearRect(0, 0, width, height);
    const peak = Math.max(1, ...this.history.flat());
    colors.forEach((color, i) => {
      ctx.strokeStyle = css(color);
      ctx.beginPath();
      this.history.forEach((sample, t) =>
        ctx.lineTo((t / (HISTORY - 1)) * width, height - 1 - (sample[i] / peak) * (height - 2)),
      );
      ctx.stroke();
    });
  }
}
