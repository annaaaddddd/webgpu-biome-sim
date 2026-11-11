import type { GUI } from 'lil-gui';
import { device } from '../core/device';
import { createBuffer } from '../core/gpu';
import { MAX_SPECIES, FLOWER_BUDGET, FLOWER_PARAMS_BYTES, SPECIES_COLORS } from '../core/layout';

// each species has a user editable flower shape
export interface FlowerSpecies {
    stem: boolean;
    leaves: boolean;
    petals: boolean;
    center: boolean;
    petalCount: number;
    leafCount: number;
    stemHeight: number;
    stemRadius: number;
    stemBend: number;
    leafLength: number;
    leafWidth: number;
    leafDroop: number;
    petalLength: number;
    petalWidth: number;
    petalCup: number;
    centerRadius: number;
    petalColor: number[];
    centerColor: number[];
    stemColor: number[];
}

// unit height flowers (plants.wgsl responsible for scaling)
function defaults(petalColor: number[]): FlowerSpecies {
    return {
        stem: true,
        leaves: true,
        petals: true,
        center: true,
        petalCount: 6,
        leafCount: 2,
        stemHeight: 1,
        stemRadius: 0.03,
        stemBend: 0.1,
        leafLength: 0.3,
        leafWidth: 0.12,
        leafDroop: 0.1,
        petalLength: 0.25,
        petalWidth: 0.15,
        petalCup: 0.4,
        centerRadius: 0.06,
        petalColor: [...petalColor],
        centerColor: [0.90, 0.75, 0.20],
        stemColor: [0.25, 0.45, 0.18]
    };
}

export class Flowers {
    species: FlowerSpecies[] = SPECIES_COLORS.slice(0, MAX_SPECIES).map(defaults);
    buffer = createBuffer(GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, FLOWER_PARAMS_BYTES * MAX_SPECIES);
    private data = new ArrayBuffer(FLOWER_PARAMS_BYTES * MAX_SPECIES);

    constructor() {
        this.upload();
    }

    addGui(gui: GUI) {
        const root = gui.addFolder('Flowers');
        this.species.forEach((f, i) => {
            const s = root.addFolder(`Species ${i + 1}`).close();
            s.add(f, 'stem');
            s.add(f, 'leaves');
            s.add(f, 'petals');
            s.add(f, 'center');
            s.add(f, 'petalCount', 1, FLOWER_BUDGET.MAX_PETALS, 1);
            s.add(f, 'leafCount', 0, FLOWER_BUDGET.MAX_LEAVES, 1);
            s.add(f, 'stemHeight', 0.2, 2);
            s.add(f, 'stemRadius', 0.005, 0.1);
            s.add(f, 'stemBend', -0.5, 0.5);
            s.add(f, 'leafLength', 0.05, 0.6);
            s.add(f, 'leafWidth', 0.02, 0.3);
            s.add(f, 'leafDroop', 0, 0.5);
            s.add(f, 'petalLength', 0.05, 0.6);
            s.add(f, 'petalWidth', 0.02, 0.4);
            s.add(f, 'petalCup', 0, 1);
            s.add(f, 'centerRadius', 0, 0.2);
            s.addColor(f, 'petalColor');
            s.addColor(f, 'centerColor');
            s.addColor(f, 'stemColor');
        });

        root.onChange(() => this.upload());
    }

    // field order and offsets (recall FlowerParams struct in common)
    upload() {
        const u32 = new Uint32Array(this.data);
        const f32 = new Float32Array(this.data);
        this.species.forEach((f, i) => {
            const o = (i * FLOWER_PARAMS_BYTES) / 4;
            u32[o] = +f.stem | (+f.leaves << 1) | (+f.petals << 2) | (+f.center << 3);
            u32[o + 1] = Math.min(Math.round(f.petalCount), FLOWER_BUDGET.MAX_PETALS);
            u32[o + 2] = Math.min(Math.round(f.leafCount), FLOWER_BUDGET.MAX_LEAVES);

            f32.set(
                [
                    f.stemHeight,
                    f.stemRadius,
                    f.stemBend,
                    f.leafLength,
                    f.leafWidth,
                    f.leafDroop,
                    f.petalLength,
                    f.petalWidth,
                    f.petalCup,
                    f.centerRadius
                ],
                o + 4
            );

            f32.set(
                [
                    ...f.petalColor,
                    1,
                    ...f.centerColor,
                    1,
                    ...f.stemColor,
                    1
                ],
                o + 16
            );
        });

        device.queue.writeBuffer(this.buffer, 0, this.data);
    }
}