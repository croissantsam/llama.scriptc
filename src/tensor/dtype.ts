// Data types supported by the Tensor system
// ==========================================

export enum DType {
  Float64 = "float64",
  Float32 = "float32",
  Int32 = "int32"
}

export function bytesPerElement(dtype: DType): number {
  switch (dtype) {
    case DType.Float64:
      return 8;
    case DType.Float32:
      return 4;
    case DType.Int32:
      return 4;
    default:
      return 8;
  }
}
