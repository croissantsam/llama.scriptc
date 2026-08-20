// Root Mean Square Normalization (RMSNorm)
// ==========================================
// RMS(x) = sqrt(mean(x^2) + eps)
// y = (x / RMS(x)) * weight

import { Tensor } from "../tensor/tensor";

export class RMSNorm {
  dim: number;
  eps: number;
  weight: Tensor; // learnable scale parameter [dim]

  constructor(dim: number, eps: number = 1e-6, weight?: Tensor) {
    this.dim = dim;
    this.eps = eps;

    if (weight) {
      if (weight.shape.length !== 1 || weight.shape[0] !== dim) {
        throw new Error(`RMSNorm weight shape mismatch: expected [${dim}], got [${weight.shape.join(", ")}]`);
      }
      this.weight = weight;
    } else {
      this.weight = Tensor.ones([dim]);
    }
  }

  forward(x: Tensor): Tensor {
    const lastDim: number = x.shape[x.ndim() - 1];
    if (lastDim !== this.dim) {
      throw new Error(`RMSNorm dimension mismatch: expected ${this.dim}, got ${lastDim}`);
    }

    const out: Tensor = Tensor.zeros(x.shape, x.dtype);
    const numRows: number = x.size / this.dim;

    for (let r: number = 0; r < numRows; r++) {
      const rowOffset: number = r * this.dim;

      // 1. Calculate mean(x^2)
      let sumSq: number = 0;
      for (let i: number = 0; i < this.dim; i++) {
        const val: number = x.getFlat(rowOffset + i);
        sumSq += val * val;
      }
      const meanSq: number = sumSq / this.dim;

      // 2. RMS = sqrt(meanSq + eps)
      const rms: number = Math.sqrt(meanSq + this.eps);
      const invRms: number = 1.0 / rms;

      // 3. Normalize and scale by weight
      for (let i: number = 0; i < this.dim; i++) {
        const val: number = x.getFlat(rowOffset + i);
        const gamma: number = this.weight.get(i);
        out.setFlat(rowOffset + i, val * invRms * gamma);
      }
    }

    return out;
  }
}
