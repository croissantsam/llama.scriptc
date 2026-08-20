// Neural Network Activation Functions (SiLU / Swish, Sigmoid)
// ==============================================================

import { Tensor } from "../tensor/tensor";

export function sigmoidScalar(x: number): number {
  if (x >= 0) {
    const z: number = Math.exp(-x);
    return 1.0 / (1.0 + z);
  } else {
    const z: number = Math.exp(x);
    return z / (1.0 + z);
  }
}

export function siluScalar(x: number): number {
  return x * sigmoidScalar(x);
}

export function sigmoid(x: Tensor): Tensor {
  const out: Tensor = Tensor.zeros(x.shape, x.dtype);
  for (let i: number = 0; i < x.size; i++) {
    out.setFlat(i, sigmoidScalar(x.getFlat(i)));
  }
  return out;
}

export function silu(x: Tensor): Tensor {
  const out: Tensor = Tensor.zeros(x.shape, x.dtype);
  for (let i: number = 0; i < x.size; i++) {
    out.setFlat(i, siluScalar(x.getFlat(i)));
  }
  return out;
}
