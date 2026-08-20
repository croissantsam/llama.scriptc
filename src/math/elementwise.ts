// Elementwise mathematical operations on Tensors
// ===============================================

import { Tensor } from "../tensor/tensor";
import { broadcastShapes, shapesEqual } from "../tensor/shape";

// --- Binary operations ---

export function add(a: Tensor, b: Tensor | number): Tensor {
  if (typeof b === "number") {
    return addScalar(a, b);
  }
  return binaryOp(a, b, (x: number, y: number): number => x + y);
}

export function sub(a: Tensor, b: Tensor | number): Tensor {
  if (typeof b === "number") {
    return addScalar(a, -b);
  }
  return binaryOp(a, b, (x: number, y: number): number => x - y);
}

export function mul(a: Tensor, b: Tensor | number): Tensor {
  if (typeof b === "number") {
    return scale(a, b);
  }
  return binaryOp(a, b, (x: number, y: number): number => x * y);
}

export function div(a: Tensor, b: Tensor | number): Tensor {
  if (typeof b === "number") {
    if (b === 0) throw new Error("Division by scalar zero");
    return scale(a, 1.0 / b);
  }
  return binaryOp(a, b, (x: number, y: number): number => {
    if (y === 0) throw new Error("Division by zero in elementwise div");
    return x / y;
  });
}

export function scale(a: Tensor, scalar: number): Tensor {
  const out: Tensor = Tensor.zeros(a.shape, a.dtype);
  for (let i: number = 0; i < a.size; i++) {
    out.setFlat(i, a.getFlat(i) * scalar);
  }
  return out;
}

export function addScalar(a: Tensor, scalar: number): Tensor {
  const out: Tensor = Tensor.zeros(a.shape, a.dtype);
  for (let i: number = 0; i < a.size; i++) {
    out.setFlat(i, a.getFlat(i) + scalar);
  }
  return out;
}

// --- Unary operations ---

export function exp(a: Tensor): Tensor {
  const out: Tensor = Tensor.zeros(a.shape, a.dtype);
  for (let i: number = 0; i < a.size; i++) {
    out.setFlat(i, Math.exp(a.getFlat(i)));
  }
  return out;
}

export function sqrt(a: Tensor): Tensor {
  const out: Tensor = Tensor.zeros(a.shape, a.dtype);
  for (let i: number = 0; i < a.size; i++) {
    const val: number = a.getFlat(i);
    if (val < 0) {
      throw new Error(`Cannot compute sqrt of negative number ${val} at index ${i}`);
    }
    out.setFlat(i, Math.sqrt(val));
  }
  return out;
}

export function rsqrt(a: Tensor, eps: number = 0): Tensor {
  const out: Tensor = Tensor.zeros(a.shape, a.dtype);
  for (let i: number = 0; i < a.size; i++) {
    const val: number = a.getFlat(i) + eps;
    if (val <= 0) {
      throw new Error(`Cannot compute rsqrt of non-positive number ${val} at index ${i}`);
    }
    out.setFlat(i, 1.0 / Math.sqrt(val));
  }
  return out;
}

export function neg(a: Tensor): Tensor {
  return scale(a, -1.0);
}

export function abs(a: Tensor): Tensor {
  const out: Tensor = Tensor.zeros(a.shape, a.dtype);
  for (let i: number = 0; i < a.size; i++) {
    out.setFlat(i, Math.abs(a.getFlat(i)));
  }
  return out;
}

export function clamp(a: Tensor, minVal: number, maxVal: number): Tensor {
  const out: Tensor = Tensor.zeros(a.shape, a.dtype);
  for (let i: number = 0; i < a.size; i++) {
    const val: number = a.getFlat(i);
    const clamped: number = Math.max(minVal, Math.min(maxVal, val));
    out.setFlat(i, clamped);
  }
  return out;
}

// --- Helper: Generic binary op with broadcasting ---

function binaryOp(a: Tensor, b: Tensor, op: (x: number, y: number) => number): Tensor {
  // Fast path: same shapes
  if (shapesEqual(a.shape, b.shape)) {
    const out: Tensor = Tensor.zeros(a.shape, a.dtype);
    for (let i: number = 0; i < a.size; i++) {
      out.setFlat(i, op(a.getFlat(i), b.getFlat(i)));
    }
    return out;
  }

  // Broadcasting path
  const outShape: number[] = broadcastShapes(a.shape, b.shape);
  const out: Tensor = Tensor.zeros(outShape, a.dtype);

  // Iterate over all coordinates of the broadcasted shape
  const ndim: number = outShape.length;
  const currentCoord: number[] = [];
  for (let i: number = 0; i < ndim; i++) {
    currentCoord.push(0);
  }

  for (let flat: number = 0; flat < out.size; flat++) {
    // Map coordinate to tensor A and tensor B
    const coordA: number[] = mapToOriginalCoord(currentCoord, outShape, a.shape);
    const coordB: number[] = mapToOriginalCoord(currentCoord, outShape, b.shape);

    const valA: number = a.get(...coordA);
    const valB: number = b.get(...coordB);
    out.setFlat(flat, op(valA, valB));

    // Increment coordinates (row-major)
    for (let d: number = ndim - 1; d >= 0; d--) {
      currentCoord[d] = currentCoord[d] + 1;
      if (currentCoord[d] < outShape[d] || d === 0) {
        break;
      }
      currentCoord[d] = 0;
    }
  }

  return out;
}

function mapToOriginalCoord(outCoord: number[], outShape: number[], origShape: number[]): number[] {
  const origDim: number = origShape.length;
  const outDim: number = outShape.length;
  const offset: number = outDim - origDim;
  const coord: number[] = [];

  for (let i: number = 0; i < origDim; i++) {
    const outIdx: number = outCoord[offset + i];
    const origSize: number = origShape[i];
    coord.push(origSize === 1 ? 0 : outIdx);
  }
  return coord;
}
