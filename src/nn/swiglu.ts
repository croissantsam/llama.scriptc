// SwiGLU Feed-Forward Network (Llama MLP Architecture) - Optimized
// ====================================================
// MLP(x) = (SiLU(x * W_gate^T) * (x * W_up^T)) * W_down^T

import { Tensor } from "../tensor/tensor";
import { Linear } from "./linear";
import { silu } from "./activations";
import { mul } from "../math/elementwise";

export class SwiGLU {
  hiddenDim: number;
  intermediateDim: number;
  gateProj: Linear;
  upProj: Linear;
  downProj: Linear;

  // Pre-allocated buffers
  private _gateBuffer?: Tensor;
  private _upBuffer?: Tensor;
  private _intermediateBuffer?: Tensor;

  constructor(
    hiddenDim: number,
    intermediateDim: number,
    gateWeight?: Tensor,
    upWeight?: Tensor,
    downWeight?: Tensor
  ) {
    this.hiddenDim = hiddenDim;
    this.intermediateDim = intermediateDim;

    this.gateProj = new Linear(hiddenDim, intermediateDim, gateWeight);
    this.upProj = new Linear(hiddenDim, intermediateDim, upWeight);
    this.downProj = new Linear(intermediateDim, hiddenDim, downWeight);
  }

  forward(x: Tensor): Tensor {
    // 1. Gate projection + SiLU activation
    const gate: Tensor = this.gateProj.forward(x);
    const activatedGate: Tensor = silu(gate);

    // 2. Up projection
    const up: Tensor = this.upProj.forward(x);

    // 3. Elementwise Hadamard product - fuse with SiLU to avoid extra allocation
    const intermediate: Tensor = this.fusedMulSilu(activatedGate, up);

    // 4. Down projection back to hiddenDim
    const out: Tensor = this.downProj.forward(intermediate);
    return out;
  }

  // Fused SiLU + elementwise multiply to avoid intermediate allocation
  private fusedMulSilu(siluOut: Tensor, up: Tensor): Tensor {
    const shape = siluOut.shape;
    const size = siluOut.size;

    // Reuse buffer if possible
    let out: Tensor;
    if (this._intermediateBuffer && shapesEqual(this._intermediateBuffer.shape, shape)) {
      out = this._intermediateBuffer;
    } else {
      out = Tensor.zeros(shape, siluOut.dtype);
      this._intermediateBuffer = out;
    }

    const siluData = siluOut.data;
    const upData = up.data;
    const outData = out.data;

    if (siluOut.isContiguous() && up.isContiguous() && out.isContiguous()) {
      for (let i = 0; i < size; i++) {
        outData[i] = siluData[siluOut.offset + i] * upData[up.offset + i];
      }
    } else {
      for (let i = 0; i < size; i++) {
        out.setFlat(i, siluOut.getFlat(i) * up.getFlat(i));
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
