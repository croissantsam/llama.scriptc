// Core Tensor implementation for ScriptC
// =======================================

import { DType } from "./dtype";
import { computeStrides, shapeToSize, validateShape, shapesEqual, getFlatOffset } from "./shape";

export class Tensor {
  data: number[];
  shape: number[];
  strides: number[];
  offset: number;
  size: number;
  dtype: DType;
  isView: boolean;

  constructor(
    data: number[],
    shape: number[],
    strides?: number[],
    offset: number = 0,
    dtype: DType = DType.Float64,
    isView: boolean = false
  ) {
    validateShape(shape);
    this.shape = shape.slice();
    this.size = shapeToSize(shape);
    this.strides = strides ? strides.slice() : computeStrides(shape);
    this.offset = offset;
    this.dtype = dtype;
    this.isView = isView;
    this.data = data;
  }

  // --- Factory constructors ---

  static zeros(shape: number[], dtype: DType = DType.Float64): Tensor {
    const size: number = shapeToSize(shape);
    const data: number[] = [];
    for (let i: number = 0; i < size; i++) {
      data.push(0.0);
    }
    return new Tensor(data, shape, undefined, 0, dtype, false);
  }

  static ones(shape: number[], dtype: DType = DType.Float64): Tensor {
    const size: number = shapeToSize(shape);
    const data: number[] = [];
    for (let i: number = 0; i < size; i++) {
      data.push(1.0);
    }
    return new Tensor(data, shape, undefined, 0, dtype, false);
  }

  static fill(shape: number[], val: number, dtype: DType = DType.Float64): Tensor {
    const size: number = shapeToSize(shape);
    const data: number[] = [];
    for (let i: number = 0; i < size; i++) {
      data.push(val);
    }
    return new Tensor(data, shape, undefined, 0, dtype, false);
  }

  static fromArray(data: number[], shape: number[], dtype: DType = DType.Float64): Tensor {
    const expectedSize: number = shapeToSize(shape);
    if (data.length !== expectedSize) {
      throw new Error(`Data length (${data.length}) does not match shape total size (${expectedSize})`);
    }
    return new Tensor(data.slice(), shape, undefined, 0, dtype, false);
  }

  static from2D(matrix: number[][], dtype: DType = DType.Float64): Tensor {
    const rows: number = matrix.length;
    if (rows === 0) return Tensor.zeros([0, 0], dtype);
    const cols: number = matrix[0].length;
    const data: number[] = [];
    for (let r: number = 0; r < rows; r++) {
      if (matrix[r].length !== cols) {
        throw new Error(`Jagged matrix at row ${r}: expected ${cols}, got ${matrix[r].length}`);
      }
      for (let c: number = 0; c < cols; c++) {
        data.push(matrix[r][c]);
      }
    }
    return new Tensor(data, [rows, cols], undefined, 0, dtype, false);
  }

  // --- Properties ---

  ndim(): number {
    return this.shape.length;
  }

  isContiguous(): boolean {
    const expectedStrides: number[] = computeStrides(this.shape);
    return shapesEqual(this.strides, expectedStrides);
  }

  // --- Element Access ---

  get(...indices: number[]): number {
    this.checkBounds(indices);
    const flat: number = getFlatOffset(indices, this.strides, this.offset);
    return this.data[flat];
  }

  set(value: number, ...indices: number[]): void {
    this.checkBounds(indices);
    const flat: number = getFlatOffset(indices, this.strides, this.offset);
    this.data[flat] = value;
  }

  getFlat(index: number): number {
    if (index < 0 || index >= this.size) {
      throw new Error(`Flat index out of bounds: ${index} (size: ${this.size})`);
    }
    if (this.isContiguous()) {
      return this.data[this.offset + index];
    }
    // Compute multi-dim coordinates for non-contiguous views
    const coords: number[] = this.flatToCoords(index);
    return this.get(...coords);
  }

  setFlat(index: number, value: number): void {
    if (index < 0 || index >= this.size) {
      throw new Error(`Flat index out of bounds: ${index} (size: ${this.size})`);
    }
    if (this.isContiguous()) {
      this.data[this.offset + index] = value;
    } else {
      const coords: number[] = this.flatToCoords(index);
      this.set(value, ...coords);
    }
  }

  private checkBounds(indices: number[]): void {
    if (indices.length !== this.shape.length) {
      throw new Error(`Index dimension mismatch: expected ${this.shape.length}, got ${indices.length}`);
    }
    for (let i: number = 0; i < indices.length; i++) {
      const idx: number = indices[i];
      if (idx < 0 || idx >= this.shape[i]) {
        throw new Error(`Index ${idx} out of bounds for axis ${i} with size ${this.shape[i]}`);
      }
    }
  }

  private flatToCoords(flatIndex: number): number[] {
    const coords: number[] = [];
    let rem: number = flatIndex;
    for (let i: number = 0; i < this.shape.length; i++) {
      let stride: number = 1;
      for (let j: number = i + 1; j < this.shape.length; j++) {
        stride *= this.shape[j];
      }
      const coord: number = Math.floor(rem / stride);
      coords.push(coord);
      rem %= stride;
    }
    return coords;
  }

  // --- Views and Reshaping ---

  view(newShape: number[], offset: number = 0): Tensor {
    const newSize: number = shapeToSize(newShape);
    if (newSize !== this.size) {
      throw new Error(`Cannot view shape [${this.shape.join(", ")}] (size ${this.size}) as [${newShape.join(", ")}] (size ${newSize})`);
    }
    if (!this.isContiguous()) {
      throw new Error("Cannot create non-contiguous view directly; call clone() or contiguous() first");
    }
    return new Tensor(
      this.data,
      newShape,
      computeStrides(newShape),
      this.offset + offset,
      this.dtype,
      true
    );
  }

  reshape(newShape: number[]): Tensor {
    const newSize: number = shapeToSize(newShape);
    if (newSize !== this.size) {
      throw new Error(`Cannot reshape [${this.shape.join(", ")}] (size ${this.size}) into [${newShape.join(", ")}] (size ${newSize})`);
    }
    if (this.isContiguous()) {
      return new Tensor(this.data, newShape, computeStrides(newShape), this.offset, this.dtype, this.isView);
    }
    // If not contiguous, copy data to a new contiguous tensor
    const copyData: number[] = this.toArray();
    return new Tensor(copyData, newShape, computeStrides(newShape), 0, this.dtype, false);
  }

  transpose(): Tensor {
    if (this.ndim() !== 2) {
      throw new Error(`Transpose is only supported for 2D tensors, got ${this.ndim()}D`);
    }
    const newShape: number[] = [this.shape[1], this.shape[0]];
    const newStrides: number[] = [this.strides[1], this.strides[0]];
    return new Tensor(this.data, newShape, newStrides, this.offset, this.dtype, true);
  }

  clone(): Tensor {
    const dataCopy: number[] = this.toArray();
    return new Tensor(dataCopy, this.shape.slice(), computeStrides(this.shape), 0, this.dtype, false);
  }

  contiguous(): Tensor {
    if (this.isContiguous() && this.offset === 0 && !this.isView) {
      return this;
    }
    return this.clone();
  }

  toArray(): number[] {
    const out: number[] = [];
    for (let i: number = 0; i < this.size; i++) {
      out.push(this.getFlat(i));
    }
    return out;
  }

  to2DArray(): number[][] {
    if (this.ndim() !== 2) {
      throw new Error(`to2DArray requires a 2D tensor, got ${this.ndim()}D`);
    }
    const rows: number = this.shape[0];
    const cols: number = this.shape[1];
    const matrix: number[][] = [];
    for (let r: number = 0; r < rows; r++) {
      const row: number[] = [];
      for (let c: number = 0; c < cols; c++) {
        row.push(this.get(r, c));
      }
      matrix.push(row);
    }
    return matrix;
  }

  toString(): string {
    return `Tensor(shape=[${this.shape.join(", ")}], dtype=${this.dtype}, size=${this.size})`;
  }
}
