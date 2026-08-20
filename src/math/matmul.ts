// Matrix Multiplication (2D and Batched 3D)
// ===========================================

import { Tensor } from "../tensor/tensor";

export function matmul(a: Tensor, b: Tensor): Tensor {
  // Case 1: 2D @ 2D: (M, K) @ (K, N) -> (M, N)
  if (a.ndim() === 2 && b.ndim() === 2) {
    return matmul2D(a, b);
  }

  // Case 2: 1D @ 2D: (K) @ (K, N) -> (N)
  if (a.ndim() === 1 && b.ndim() === 2) {
    const a2D: Tensor = a.view([1, a.shape[0]]);
    const res2D: Tensor = matmul2D(a2D, b);
    return res2D.view([res2D.shape[1]]);
  }

  // Case 3: 2D @ 1D: (M, K) @ (K) -> (M)
  if (a.ndim() === 2 && b.ndim() === 1) {
    const b2D: Tensor = b.view([b.shape[0], 1]);
    const res2D: Tensor = matmul2D(a, b2D);
    return res2D.view([res2D.shape[0]]);
  }

  // Case 4: 3D @ 3D (Batched Matmul): (B, M, K) @ (B, K, N) -> (B, M, N)
  if (a.ndim() === 3 && b.ndim() === 3) {
    return matmul3DBatched(a, b);
  }

  throw new Error(`Unsupported matmul dimensions: ${a.ndim()}D with shape [${a.shape.join(", ")}] and ${b.ndim()}D with shape [${b.shape.join(", ")}]`);
}

function matmul2D(a: Tensor, b: Tensor): Tensor {
  const M: number = a.shape[0];
  const K1: number = a.shape[1];
  const K2: number = b.shape[0];
  const N: number = b.shape[1];

  if (K1 !== K2) {
    throw new Error(`matmul dimension mismatch: inner dimensions (${K1} and ${K2}) must match. Shapes: [${a.shape.join(", ")}] @ [${b.shape.join(", ")}]`);
  }

  const out: Tensor = Tensor.zeros([M, N], a.dtype);

  // Standard triple loop matmul
  for (let i: number = 0; i < M; i++) {
    for (let k: number = 0; k < K1; k++) {
      const aVal: number = a.get(i, k);
      for (let j: number = 0; j < N; j++) {
        const current: number = out.get(i, j);
        out.set(current + aVal * b.get(k, j), i, j);
      }
    }
  }
  return out;
}

function matmul3DBatched(a: Tensor, b: Tensor): Tensor {
  const B1: number = a.shape[0];
  const B2: number = b.shape[0];
  const M: number = a.shape[1];
  const K1: number = a.shape[2];
  const K2: number = b.shape[1];
  const N: number = b.shape[2];

  if (B1 !== B2) {
    throw new Error(`Batched matmul batch size mismatch: ${B1} vs ${B2}`);
  }
  if (K1 !== K2) {
    throw new Error(`Batched matmul inner dimension mismatch: ${K1} vs ${K2}`);
  }

  const out: Tensor = Tensor.zeros([B1, M, N], a.dtype);

  for (let batch: number = 0; batch < B1; batch++) {
    for (let i: number = 0; i < M; i++) {
      for (let k: number = 0; k < K1; k++) {
        const aVal: number = a.get(batch, i, k);
        for (let j: number = 0; j < N; j++) {
          const current: number = out.get(batch, i, j);
          out.set(current + aVal * b.get(batch, k, j), batch, i, j);
        }
      }
    }
  }
  return out;
}
