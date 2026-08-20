// Shape and Stride manipulation for multi-dimensional Tensors
// ============================================================

export function shapeToSize(shape: number[]): number {
  if (shape.length === 0) return 0;
  let size: number = 1;
  for (let i: number = 0; i < shape.length; i++) {
    const dim: number = shape[i];
    if (dim < 0) {
      throw new Error(`Invalid negative dimension ${dim} at axis ${i}`);
    }
    size *= dim;
  }
  return size;
}

export function computeStrides(shape: number[]): number[] {
  const ndim: number = shape.length;
  if (ndim === 0) return [];

  const strides: number[] = [];
  for (let i: number = 0; i < ndim; i++) {
    strides.push(0);
  }

  let currentStride: number = 1;
  for (let i: number = ndim - 1; i >= 0; i--) {
    strides[i] = currentStride;
    currentStride *= shape[i];
  }
  return strides;
}

export function validateShape(shape: number[]): void {
  for (let i: number = 0; i < shape.length; i++) {
    if (shape[i] < 0 || Math.floor(shape[i]) !== shape[i]) {
      throw new Error(`Dimension at axis ${i} must be a non-negative integer, got ${shape[i]}`);
    }
  }
}

export function shapesEqual(a: number[], b: number[]): boolean {
  if (a.length !== b.length) return false;
  for (let i: number = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

export function getFlatOffset(indices: number[], strides: number[], offset: number = 0): number {
  let flat: number = offset;
  for (let i: number = 0; i < indices.length; i++) {
    flat += indices[i] * strides[i];
  }
  return flat;
}

export function areBroadcastable(a: number[], b: number[]): boolean {
  const maxLen: number = Math.max(a.length, b.length);
  const padA: number[] = padShape(a, maxLen);
  const padB: number[] = padShape(b, maxLen);

  for (let i: number = 0; i < maxLen; i++) {
    const da: number = padA[i];
    const db: number = padB[i];
    if (da !== db && da !== 1 && db !== 1) {
      return false;
    }
  }
  return true;
}

export function broadcastShapes(a: number[], b: number[]): number[] {
  const maxLen: number = Math.max(a.length, b.length);
  const padA: number[] = padShape(a, maxLen);
  const padB: number[] = padShape(b, maxLen);
  const outShape: number[] = [];

  for (let i: number = 0; i < maxLen; i++) {
    const da: number = padA[i];
    const db: number = padB[i];
    if (da === db) {
      outShape.push(da);
    } else if (da === 1) {
      outShape.push(db);
    } else if (db === 1) {
      outShape.push(da);
    } else {
      throw new Error(`Incompatible shapes for broadcasting: [${a.join(", ")}] and [${b.join(", ")}]`);
    }
  }
  return outShape;
}

function padShape(shape: number[], targetLength: number): number[] {
  const padCount: number = targetLength - shape.length;
  const result: number[] = [];
  for (let i: number = 0; i < padCount; i++) {
    result.push(1);
  }
  for (let i: number = 0; i < shape.length; i++) {
    result.push(shape[i]);
  }
  return result;
}
