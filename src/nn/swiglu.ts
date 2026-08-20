// SwiGLU Feed-Forward Network (Llama MLP Architecture)
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

    // 3. Elementwise Hadamard product
    const intermediate: Tensor = mul(activatedGate, up);

    // 4. Down projection back to hiddenDim
    const out: Tensor = this.downProj.forward(intermediate);
    return out;
  }
}
