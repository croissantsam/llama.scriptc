// Matrix Multiplication (2D and Batched 3D) - Optimized
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
  const K: number = a.shape[1];
  const N: number = b.shape[1];

  if (K !== b.shape[0]) {
    throw new Error(`matmul dimension mismatch: inner dimensions (${K} and ${b.shape[0]}) must match. Shapes: [${a.shape.join(", ")}] @ [${b.shape.join(", ")}]`);
  }

  const out: Tensor = Tensor.zeros([M, N], a.dtype);

  // Get raw data arrays for direct access (avoiding get/set overhead)
  const aData = a.data;
  const bData = b.data;
  const outData = out.data;

  // Get strides for non-contiguous tensors
  const aStrides = a.strides;
  const bStrides = b.strides;
  const outStrides = out.strides;

  const aOffset = a.offset;
  const bOffset = b.offset;
  const outOffset = out.offset;

  // Optimized loop ordering: i, j, k for better cache locality
  // This accesses a row-wise, b column-wise (transposed), out row-wise
  if (a.isContiguous() && b.isContiguous() && out.isContiguous()) {
    // Fast path for contiguous tensors - use flat array access
    for (let i = 0; i < M; i++) {
      const aRowOffset = aOffset + i * aStrides[0];
      const outRowOffset = outOffset + i * outStrides[0];
      for (let j = 0; j < N; j++) {
        let sum = 0.0;
        const bColOffset = bOffset + j; // b is [K, N], so stride[1] = 1 for contiguous
        for (let k = 0; k < K; k++) {
          sum += aData[aRowOffset + k] * bData[bOffset + k * bStrides[0] + j];
        }
        outData[outRowOffset + j] = sum;
      }
    }
  } else {
    // General path with strides
    for (let i = 0; i < M; i++) {
      for (let j = 0; j < N; j++) {
        let sum = 0.0;
        for (let k = 0; k < K; k++) {
          const aIdx = aOffset + i * aStrides[0] + k * aStrides[1];
          const bIdx = bOffset + k * bStrides[0] + j * bStrides[1];
          sum += aData[aIdx] * bData[bIdx];
        }
        outData[outOffset + i * outStrides[0] + j * outStrides[1]] = sum;
      }
    }
  }
  return out;
}

function matmul3DBatched(a: Tensor, b: Tensor): Tensor {
  const B: number = a.shape[0];
  const M: number = a.shape[1];
  const K: number = a.shape[2];
  const N: number = b.shape[2];

  if (B !== b.shape[0]) {
    throw new Error(`Batched matmul batch size mismatch: ${B} vs ${b.shape[0]}`);
  }
  if (K !== b.shape[1]) {
    throw new Error(`Batched matmul inner dimension mismatch: ${K} vs ${b.shape[1]}`);
  }

  const out: Tensor = Tensor.zeros([B, M, N], a.dtype);

  const aData = a.data;
  const bData = b.data;
  const outData = out.data;

  const aStrides = a.strides;
  const bStrides = b.strides;
  const outStrides = out.strides;

  const aOffset = a.offset;
  const bOffset = b.offset;
  const outOffset = out.offset;

  if (a.isContiguous() && b.isContiguous() && out.isContiguous()) {
    // Fast path for contiguous tensors
    for (let batch = 0; batch < B; batch++) {
      const aBatchOffset = aOffset + batch * aStrides[0];
      const bBatchOffset = bOffset + batch * bStrides[0];
      const outBatchOffset = outOffset + batch * outStrides[0];

      for (let i = 0; i < M; i++) {
        const aRowOffset = aBatchOffset + i * aStrides[1];
        const outRowOffset = outBatchOffset + i * outStrides[1];
        for (let j = 0; j < N; j++) {
          let sum = 0.0;
          for (let k = 0; k < K; k++) {
            sum += aData[aRowOffset + k * aStrides[2]] * bData[bBatchOffset + k * bStrides[1] + j * bStrides[2]];
          }
          outData[outRowOffset + j * outStrides[2]] = sum;
        }
      }
    }
  } else {
    // General path with strides
    for (let batch = 0; batch < B; batch++) {
      for (let i = 0; i < M; i++) {
        for (let j = 0; j < N; j++) {
          let sum = 0.0;
          for (let k = 0; k < K; k++) {
            const aIdx = aOffset + batch * aStrides[0] + i * aStrides[1] + k * aStrides[2];
            const bIdx = bOffset + batch * bStrides[0] + k * bStrides[1] + j * bStrides[2];
            sum += aData[aIdx] * bData[bIdx];
          }
          outData[outOffset + batch * outStrides[0] + i * outStrides[1] + j * outStrides[2]] = sum;
        }
      }
    }
  }
  return out;
}
