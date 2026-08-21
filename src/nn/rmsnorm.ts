// Root Mean Square Normalization (RMSNorm) - Optimized
// ==========================================
// RMS(x) = sqrt(mean(x^2) + eps)
// y = (x / RMS(x)) * weight

import { Tensor } from "../tensor/tensor";

export class RMSNorm {
  dim: number;
  eps: number;
  weight: Tensor; // learnable scale parameter [dim]

  // Pre-allocated output buffer
  private _outBuffer?: Tensor;

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

    // Reuse output buffer if possible
    let out: Tensor;
    if (this._outBuffer && shapesEqual(this._outBuffer.shape, x.shape)) {
      out = this._outBuffer;
    } else {
      out = Tensor.zeros(x.shape, x.dtype);
      this._outBuffer = out;
    }

    const numRows: number = x.size / this.dim;
    const xData = x.data;
    const outData = out.data;
    const wData = this.weight.data;

    const xStrides = x.strides;
    const outStrides = out.strides;
    const xOffset = x.offset;
    const outOffset = out.offset;

    if (x.isContiguous() && out.isContiguous()) {
      // Fast path for contiguous tensors
      for (let r = 0; r < numRows; r++) {
        const rowOffset = xOffset + r * this.dim;
        const outRowOffset = outOffset + r * this.dim;

        // 1. Calculate mean(x^2)
        let sumSq = 0.0;
        for (let i = 0; i < this.dim; i++) {
          const val = xData[rowOffset + i];
          sumSq += val * val;
        }
        const meanSq = sumSq / this.dim;

        // 2. RMS = sqrt(meanSq + eps)
        const invRms = 1.0 / Math.sqrt(meanSq + this.eps);

        // 3. Normalize and scale by weight
        for (let i = 0; i < this.dim; i++) {
          const val = xData[rowOffset + i];
          outData[outRowOffset + i] = val * invRms * wData[i];
        }
      }
    } else {
      // General path with strides
      for (let r = 0; r < numRows; r++) {
        // Calculate flat offset for this row
        let rowOffset = xOffset;
        let outRowOffset = outOffset;
        let rem = r;
        
        // Compute multi-dimensional offset
        for (let d = 0; d < x.ndim() - 1; d++) {
          const stride = xStrides[d];
          const size = x.shape[d];
          const coord = Math.floor(rem / (x.size / (size * stride)));
          rowOffset += coord * stride;
          outRowOffset += coord * outStrides[d];
          rem %= (x.size / (size * stride));
        }
        rowOffset += rem * xStrides[x.ndim() - 1];
        outRowOffset += rem * outStrides[out.ndim() - 1];

        // 1. Calculate mean(x^2)
        let sumSq = 0.0;
        for (let i = 0; i < this.dim; i++) {
          const val = xData[rowOffset + i * xStrides[x.ndim() - 1]];
          sumSq += val * val;
        }
        const meanSq = sumSq / this.dim;

        // 2. RMS = sqrt(meanSq + eps)
        const invRms = 1.0 / Math.sqrt(meanSq + this.eps);

        // 3. Normalize and scale by weight
        for (let i = 0; i < this.dim; i++) {
          const val = xData[rowOffset + i * xStrides[x.ndim() - 1]];
          outData[outRowOffset + i * outStrides[out.ndim() - 1]] = val * invRms * wData[i];
        }
      }
    }

    return out;
  }
}

// Helper function to compare shapes
function shapesEqual(a: number[], b: number[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}
