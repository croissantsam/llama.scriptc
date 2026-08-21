// Numerically Stable Softmax Implementation - Optimized
// ==========================================

import { Tensor } from "../tensor/tensor";

export function softmax(x: Tensor, axis: number = -1): Tensor {
  const normAxis: number = axis < 0 ? axis + x.ndim() : axis;
  if (normAxis < 0 || normAxis >= x.ndim()) {
    throw new Error(`Invalid axis ${axis} for tensor with ${x.ndim()} dimensions`);
  }

  // 1D Softmax
  if (x.ndim() === 1) {
    return softmax1D(x);
  }

  // 2D Softmax (e.g. [Batch, Vocab] or [Batch, SeqLen])
  if (x.ndim() === 2 && normAxis === 1) {
    return softmax2DLastAxis(x);
  }

  // 3D Softmax (e.g. [Batch, SeqLen, SeqLen] in attention)
  if (x.ndim() === 3 && normAxis === 2) {
    return softmax3DLastAxis(x);
  }

  // Generic fallback: flatten all leading dimensions into a 2D matrix, apply softmax along last dimension, and reshape back
  if (normAxis === x.ndim() - 1) {
    const lastDim: number = x.shape[x.ndim() - 1];
    const leadingDim: number = x.size / lastDim;
    const flat2D: Tensor = x.clone().view([leadingDim, lastDim]);
    const out2D: Tensor = softmax2DLastAxis(flat2D);
    return out2D.view(x.shape);
  }

  throw new Error(`Softmax along axis ${axis} is currently supported for the last axis only`);
}

function softmax1D(x: Tensor): Tensor {
  const n: number = x.size;
  if (n === 0) return Tensor.zeros([0], x.dtype);

  const out: Tensor = Tensor.zeros([n], x.dtype);
  const xData = x.data;
  const outData = out.data;
  const xOffset = x.offset;
  const outOffset = out.offset;

  // Step 1: Find max value for numerical stability
  let maxVal: number = -Infinity;
  if (x.isContiguous()) {
    for (let i = 0; i < n; i++) {
      const val = xData[xOffset + i];
      if (val > maxVal) maxVal = val;
    }
  } else {
    for (let i = 0; i < n; i++) {
      const val = x.get(i);
      if (val > maxVal) maxVal = val;
    }
  }

  // Step 2: Compute exp(x_i - maxVal) and accumulate sum
  let expSum: number = 0;
  if (x.isContiguous()) {
    for (let i = 0; i < n; i++) {
      const expVal = Math.exp(xData[xOffset + i] - maxVal);
      outData[outOffset + i] = expVal;
      expSum += expVal;
    }
  } else {
    for (let i = 0; i < n; i++) {
      const expVal = Math.exp(x.get(i) - maxVal);
      outData[outOffset + i] = expVal;
      expSum += expVal;
    }
  }

  // Step 3: Normalize in-place
  const invExpSum = 1.0 / expSum;
  for (let i = 0; i < n; i++) {
    outData[outOffset + i] = outData[outOffset + i] * invExpSum;
  }
  return out;
}

function softmax2DLastAxis(x: Tensor): Tensor {
  const rows: number = x.shape[0];
  const cols: number = x.shape[1];
  const out: Tensor = Tensor.zeros([rows, cols], x.dtype);

  const xData = x.data;
  const outData = out.data;
  const xStrides = x.strides;
  const outStrides = out.strides;
  const xOffset = x.offset;
  const outOffset = out.offset;

  if (x.isContiguous() && out.isContiguous()) {
    // Fast path for contiguous tensors
    for (let r = 0; r < rows; r++) {
      const rowOffset = xOffset + r * cols;
      const outRowOffset = outOffset + r * cols;

      // 1. Max for row
      let maxVal = -Infinity;
      for (let c = 0; c < cols; c++) {
        const val = xData[rowOffset + c];
        if (val > maxVal) maxVal = val;
      }

      // 2. Exp & Sum for row
      let expSum = 0.0;
      for (let c = 0; c < cols; c++) {
        const e = Math.exp(xData[rowOffset + c] - maxVal);
        outData[outRowOffset + c] = e;
        expSum += e;
      }

      // 3. Normalize row in-place
      const invExpSum = 1.0 / expSum;
      for (let c = 0; c < cols; c++) {
        outData[outRowOffset + c] = outData[outRowOffset + c] * invExpSum;
      }
    }
  } else {
    // General path with strides
    for (let r = 0; r < rows; r++) {
      // 1. Max for row
      let maxVal = -Infinity;
      for (let c = 0; c < cols; c++) {
        const val = x.get(r, c);
        if (val > maxVal) maxVal = val;
      }

      // 2. Exp & Sum for row
      let expSum = 0.0;
      for (let c = 0; c < cols; c++) {
        const e = Math.exp(x.get(r, c) - maxVal);
        out.set(e, r, c);
        expSum += e;
      }

      // 3. Normalize row
      const invExpSum = 1.0 / expSum;
      for (let c = 0; c < cols; c++) {
        const val = out.get(r, c);
        out.set(val * invExpSum, r, c);
      }
    }
  }
  return out;
}

function softmax3DLastAxis(x: Tensor): Tensor {
  const d0: number = x.shape[0];
  const d1: number = x.shape[1];
  const d2: number = x.shape[2];
  const out: Tensor = Tensor.zeros([d0, d1, d2], x.dtype);

  const xData = x.data;
  const outData = out.data;
  const xStrides = x.strides;
  const outStrides = out.strides;
  const xOffset = x.offset;
  const outOffset = out.offset;

  if (x.isContiguous() && out.isContiguous()) {
    // Fast path for contiguous tensors
    for (let i = 0; i < d0; i++) {
      for (let j = 0; j < d1; j++) {
        const rowOffset = xOffset + i * xStrides[0] + j * xStrides[1];
        const outRowOffset = outOffset + i * outStrides[0] + j * outStrides[1];

        // 1. Max
        let maxVal = -Infinity;
        for (let k = 0; k < d2; k++) {
          const val = xData[rowOffset + k];
          if (val > maxVal) maxVal = val;
        }

        // 2. Exp & Sum
        let expSum = 0.0;
        for (let k = 0; k < d2; k++) {
          const e = Math.exp(xData[rowOffset + k] - maxVal);
          outData[outRowOffset + k] = e;
          expSum += e;
        }

        // 3. Normalize in-place
        const invExpSum = 1.0 / expSum;
        for (let k = 0; k < d2; k++) {
          outData[outRowOffset + k] = outData[outRowOffset + k] * invExpSum;
        }
      }
    }
  } else {
    // General path with strides
    for (let i = 0; i < d0; i++) {
      for (let j = 0; j < d1; j++) {
        // 1. Max
        let maxVal = -Infinity;
        for (let k = 0; k < d2; k++) {
          const val = x.get(i, j, k);
          if (val > maxVal) maxVal = val;
        }

        // 2. Exp & Sum
        let expSum = 0.0;
        for (let k = 0; k < d2; k++) {
          const e = Math.exp(x.get(i, j, k) - maxVal);
          out.set(e, i, j, k);
          expSum += e;
        }

        // 3. Normalize
        const invExpSum = 1.0 / expSum;
        for (let k = 0; k < d2; k++) {
          const val = out.get(i, j, k);
          out.set(val * invExpSum, i, j, k);
        }
      }
    }
  }
  return out;
}
