export function terrainIndices(n: number) {
  const indices = new Uint32Array((n - 1) * (n - 1) * 6);
  let k = 0;
  for (let z = 0; z < n - 1; z++) {
    for (let x = 0; x < n - 1; x++) {
      const i = z * n + x;
      indices.set([i, i + n, i + 1, i + 1, i + n, i + n + 1], k);
      k += 6;
    }
  }
  return indices;
}

// Stand-in flower: a thin stem topped by a cup. Interleaved position, normal, part
// (0 = stem, 1 = head), unit height.
export function plantProxyMesh() {
  const vertices: number[] = [];
  const indices: number[] = [];

  const push = (pos: number[], normal: number[], part: number) => {
    vertices.push(...pos, ...normal, part);
    return vertices.length / 7 - 1;
  };
  const ring = (y: number, radius: number, count: number, normalY: number, part: number) =>
    Array.from({ length: count }, (_, i) => {
      const a = (i / count) * Math.PI * 2;
      const c = Math.cos(a);
      const s = Math.sin(a);
      const k = Math.hypot(1, normalY);
      return push([c * radius, y, s * radius], [c / k, normalY / k, s / k], part);
    });

  const base = ring(0, 0.05, 6, 0.1, 0);
  const top = ring(1, 0.025, 6, 0.1, 0);
  for (let i = 0; i < 6; i++) {
    const j = (i + 1) % 6;
    indices.push(base[i], top[i], top[j], base[i], top[j], base[j]);
  }

  const center = push([0, 0.95, 0], [0, 1, 0], 1);
  const petals = ring(1.2, 0.22, 8, 0.7, 1);
  for (let i = 0; i < 8; i++) {
    indices.push(center, petals[i], petals[(i + 1) % 8]);
  }

  const padded = new Uint16Array(Math.ceil(indices.length / 2) * 2);
  padded.set(indices);
  return { vertices: new Float32Array(vertices), indices: padded, indexCount: indices.length };
}
