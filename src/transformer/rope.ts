// Rotary Positional Embeddings (RoPE) - Optimized
// =====================================

import { Tensor } from "../tensor/tensor";

export interface RoPEFreqs {
  cos: number[][]; // [maxSeqLen, dim / 2]
  sin: number[][]; // [maxSeqLen, dim / 2]
}

export function precomputeRoPE(dim: number, maxSeqLen: number = 2048, base: number = 10000.0): RoPEFreqs {
  if (dim % 2 !== 0) {
    throw new Error(`RoPE head dimension must be even, got ${dim}`);
  }

  const halfDim: number = dim / 2;
  const thetas: number[] = [];
  for (let i: number = 0; i < halfDim; i++) {
    // theta_i = base^(-2i / dim)
    const exponent: number = -(2.0 * i) / dim;
    thetas.push(Math.pow(base, exponent));
  }

  const cosTable: number[][] = [];
  const sinTable: number[][] = [];

  for (let pos: number = 0; pos < maxSeqLen; pos++) {
    const cosRow: number[] = [];
    const sinRow: number[] = [];
    for (let i: number = 0; i < halfDim; i++) {
      const angle: number = pos * thetas[i];
      cosRow.push(Math.cos(angle));
      sinRow.push(Math.sin(angle));
    }
    cosTable.push(cosRow);
    sinTable.push(sinRow);
  }

  return { cos: cosTable, sin: sinTable };
}

// Original applyRoPE for backward compatibility (creates a copy)
export function applyRoPE(x: Tensor, startPos: number = 0, freqs?: RoPEFreqs, base: number = 10000.0): Tensor {
  // Input x: [seqLen, numHeads, headDim] or [numHeads, seqLen, headDim] or [seqLen, headDim]
  const ndim: number = x.ndim();
  const headDim: number = x.shape[ndim - 1];

  if (headDim % 2 !== 0) {
    throw new Error(`Head dimension for RoPE must be even, got ${headDim}`);
  }

  const halfDim: number = headDim / 2;
  let seqLen: number;

  if (ndim === 2) {
    seqLen = x.shape[0];
  } else if (ndim === 3) {
    seqLen = x.shape[0]; // [seqLen, numHeads, headDim]
  } else if (ndim === 4) {
    seqLen = x.shape[1]; // [batch, seqLen, numHeads, headDim]
  } else {
    throw new Error(`Unsupported tensor dimension for RoPE: ${ndim}D`);
  }

  const ropeFreqs: RoPEFreqs = freqs || precomputeRoPE(headDim, startPos + seqLen + 10, base);
  const out: Tensor = x.clone();

  // Rotate each 2D pair (x[2i], x[2i+1])
  if (ndim === 2) {
    // x: [seqLen, headDim]
    for (let s: number = 0; s < seqLen; s++) {
      const pos: number = startPos + s;
      const cosRow: number[] = ropeFreqs.cos[pos];
      const sinRow: number[] = ropeFreqs.sin[pos];

      for (let i: number = 0; i < halfDim; i++) {
        const x0: number = x.get(s, 2 * i);
        const x1: number = x.get(s, 2 * i + 1);
        const c: number = cosRow[i];
        const sn: number = sinRow[i];

        out.set(x0 * c - x1 * sn, s, 2 * i);
        out.set(x0 * sn + x1 * c, s, 2 * i + 1);
      }
    }
  } else if (ndim === 3) {
    // x: [seqLen, numHeads, headDim]
    const numHeads: number = x.shape[1];
    for (let s: number = 0; s < seqLen; s++) {
      const pos: number = startPos + s;
      const cosRow: number[] = ropeFreqs.cos[pos];
      const sinRow: number[] = ropeFreqs.sin[pos];

      for (let h: number = 0; h < numHeads; h++) {
        for (let i: number = 0; i < halfDim; i++) {
          const x0: number = x.get(s, h, 2 * i);
          const x1: number = x.get(s, h, 2 * i + 1);
          const c: number = cosRow[i];
          const sn: number = sinRow[i];

          out.set(x0 * c - x1 * sn, s, h, 2 * i);
          out.set(x0 * sn + x1 * c, s, h, 2 * i + 1);
        }
      }
    }
  }

  return out;
}

// Optimized in-place RoPE application - modifies tensor directly, no allocation
export function applyRoPEInPlace(tensor: Tensor, startPos: number, freqs?: RoPEFreqs): void {
  if (!freqs) return;

  const ndim = tensor.ndim();
  const headDim = tensor.shape[ndim - 1];

  if (headDim % 2 !== 0) {
    throw new Error(`RoPE requires even headDim, got ${headDim}`);
  }

  const halfDim = headDim / 2;
  let seqLen: number;
  let numHeads = 1;

  if (ndim === 2) {
    seqLen = tensor.shape[0];
  } else if (ndim === 3) {
    seqLen = tensor.shape[0];
    numHeads = tensor.shape[1];
  } else {
    throw new Error(`Unsupported tensor dimension for in-place RoPE: ${ndim}D`);
  }

  const data = tensor.data;
  const strides = tensor.strides;
  const offset = tensor.offset;

  for (let s = 0; s < seqLen; s++) {
    const pos = startPos + s;
    const cos = freqs.cos[pos];
    const sin = freqs.sin[pos];

    if (!cos || !sin) continue;

    const rowOffset = offset + s * strides[0];

    for (let h = 0; h < numHeads; h++) {
      const headOffset = rowOffset + h * strides[1];

      for (let d = 0; d < halfDim; d++) {
        const idx1 = headOffset + (2 * d) * strides[2];
        const idx2 = headOffset + (2 * d + 1) * strides[2];

        const x1 = data[idx1];
        const x2 = data[idx2];

        data[idx1] = x1 * cos[d] - x2 * sin[d];
        data[idx2] = x1 * sin[d] + x2 * cos[d];
      }
    }
  }
}
