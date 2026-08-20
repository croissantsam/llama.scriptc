// Tensor reductions (sum, mean, max, argmax)
// ===========================================

import { Tensor } from "../tensor/tensor";

// --- Full tensor reductions (scalar results) ---

export function sumAll(a: Tensor): number {
  let total: number = 0;
  for (let i: number = 0; i < a.size; i++) {
    total += a.getFlat(i);
  }
  return total;
}

export function meanAll(a: Tensor): number {
  if (a.size === 0) {
    throw new Error("Cannot compute mean of an empty tensor");
  }
  return sumAll(a) / a.size;
}

export function maxAll(a: Tensor): number {
  if (a.size === 0) {
    throw new Error("Cannot compute max of an empty tensor");
  }
  let maximum: number = -Infinity;
  for (let i: number = 0; i < a.size; i++) {
    const val: number = a.getFlat(i);
    if (val > maximum) {
      maximum = val;
    }
  }
  return maximum;
}

export function minAll(a: Tensor): number {
  if (a.size === 0) {
    throw new Error("Cannot compute min of an empty tensor");
  }
  let minimum: number = Infinity;
  for (let i: number = 0; i < a.size; i++) {
    const val: number = a.getFlat(i);
    if (val < minimum) {
      minimum = val;
    }
  }
  return minimum;
}

export function argmaxAll(a: Tensor): number {
  if (a.size === 0) {
    throw new Error("Cannot compute argmax of an empty tensor");
  }
  let maxIdx: number = 0;
  let maxVal: number = a.getFlat(0);
  for (let i: number = 1; i < a.size; i++) {
    const val: number = a.getFlat(i);
    if (val > maxVal) {
      maxVal = val;
      maxIdx = i;
    }
  }
  return maxIdx;
}

// --- Axis-wise reductions (producing reduced tensors) ---

export function sum(a: Tensor, axis?: number, keepDims: boolean = false): Tensor {
  if (axis === undefined) {
    return Tensor.fromArray([sumAll(a)], keepDims ? padOnes(a.ndim()) : [1]);
  }
  return reduceAxis(a, axis, keepDims, 0, (acc: number, val: number): number => acc + val);
}

export function mean(a: Tensor, axis?: number, keepDims: boolean = false): Tensor {
  if (axis === undefined) {
    return Tensor.fromArray([meanAll(a)], keepDims ? padOnes(a.ndim()) : [1]);
  }
  const s: Tensor = sum(a, axis, keepDims);
  const normalizedAxis: number = normalizeAxis(axis, a.ndim());
  const dimSize: number = a.shape[normalizedAxis];
  for (let i: number = 0; i < s.size; i++) {
    s.setFlat(i, s.getFlat(i) / dimSize);
  }
  return s;
}

export function max(a: Tensor, axis?: number, keepDims: boolean = false): Tensor {
  if (axis === undefined) {
    return Tensor.fromArray([maxAll(a)], keepDims ? padOnes(a.ndim()) : [1]);
  }
  return reduceAxis(a, axis, keepDims, -Infinity, (acc: number, val: number): number => Math.max(acc, val));
}

export function argmax(a: Tensor, axis?: number): number[] {
  if (axis === undefined) {
    return [argmaxAll(a)];
  }
  const normalizedAxis: number = normalizeAxis(axis, a.ndim());
  if (a.ndim() === 1) {
    return [argmaxAll(a)];
  }
  if (a.ndim() === 2 && normalizedAxis === 1) {
    // 2D row-wise argmax (common in LLM sampling over vocab)
    const rows: number = a.shape[0];
    const cols: number = a.shape[1];
    const result: number[] = [];
    for (let r: number = 0; r < rows; r++) {
      let maxIdx: number = 0;
      let maxVal: number = a.get(r, 0);
      for (let c: number = 1; c < cols; c++) {
        const val: number = a.get(r, c);
        if (val > maxVal) {
          maxVal = val;
          maxIdx = c;
        }
      }
      result.push(maxIdx);
    }
    return result;
  }
  throw new Error(`argmax along axis ${axis} for ${a.ndim()}D tensor is not implemented yet`);
}

// --- Helpers ---

function normalizeAxis(axis: number, ndim: number): number {
  if (axis < -ndim || axis >= ndim) {
    throw new Error(`Axis ${axis} is out of bounds for tensor of dimension ${ndim}`);
  }
  return axis < 0 ? axis + ndim : axis;
}

function padOnes(count: number): number[] {
  const ones: number[] = [];
  for (let i: number = 0; i < count; i++) {
    ones.push(1);
  }
  return ones;
}

function reduceAxis(
  a: Tensor,
  axis: number,
  keepDims: boolean,
  initialVal: number,
  reducer: (acc: number, val: number) => number
): Tensor {
  const normAxis: number = normalizeAxis(axis, a.ndim());
  const outShape: number[] = [];
  for (let i: number = 0; i < a.shape.length; i++) {
    if (i === normAxis) {
      if (keepDims) outShape.push(1);
    } else {
      outShape.push(a.shape[i]);
    }
  }

  // Handle reducing a 1D tensor to 0D (or 1-element)
  if (outShape.length === 0) {
    outShape.push(1);
  }

  const out: Tensor = Tensor.fill(outShape, initialVal, a.dtype);

  // 2D Fast Path
  if (a.ndim() === 2) {
    const rows: number = a.shape[0];
    const cols: number = a.shape[1];
    if (normAxis === 1) {
      // Reduce along columns (e.g. sum each row)
      for (let r: number = 0; r < rows; r++) {
        let acc: number = initialVal;
        for (let c: number = 0; c < cols; c++) {
          acc = reducer(acc, a.get(r, c));
        }
        out.setFlat(r, acc);
      }
      return out;
    } else if (normAxis === 0) {
      // Reduce along rows (e.g. sum each col)
      for (let c: number = 0; c < cols; c++) {
        let acc: number = initialVal;
        for (let r: number = 0; r < rows; r++) {
          acc = reducer(acc, a.get(r, c));
        }
        out.setFlat(c, acc);
      }
      return out;
    }
  }

  // 1D Fast Path
  if (a.ndim() === 1) {
    let acc: number = initialVal;
    for (let i: number = 0; i < a.size; i++) {
      acc = reducer(acc, a.get(i));
    }
    out.setFlat(0, acc);
    return out;
  }

  // Generic N-D reduction
  const inCoords: number[] = [];
  for (let i: number = 0; i < a.ndim(); i++) {
    inCoords.push(0);
  }

  for (let flat: number = 0; flat < a.size; flat++) {
    const val: number = a.get(...inCoords);
    const outCoords: number[] = [];
    for (let i: number = 0; i < inCoords.length; i++) {
      if (i === normAxis) {
        if (keepDims) outCoords.push(0);
      } else {
        outCoords.push(inCoords[i]);
      }
    }
    const current: number = out.get(...outCoords);
    out.set(reducer(current, val), ...outCoords);

    // Advance inCoords
    for (let d: number = a.ndim() - 1; d >= 0; d--) {
      inCoords[d] = inCoords[d] + 1;
      if (inCoords[d] < a.shape[d] || d === 0) {
        break;
      }
      inCoords[d] = 0;
    }
  }

  return out;
}
