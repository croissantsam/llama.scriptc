// Tensor Pool for Reducing Allocations
// =====================================
// Reuses tensor buffers to minimize garbage collection pressure

import { Tensor } from "./tensor";
import { DType } from "./dtype";
import { computeStrides, shapeToSize } from "./shape";

interface PoolKey {
  shape: number[];
  dtype: DType;
}

function poolKeyToString(key: PoolKey): string {
  return `${key.dtype}:${key.shape.join(",")}`;
}

export class TensorPool {
  private pools: Map<string, Tensor[]> = new Map();
  private maxPoolSize: number = 100; // Maximum tensors per shape

  // Get a tensor from the pool or create a new one
  get(shape: number[], dtype: DType = DType.Float64): Tensor {
    const key = poolKeyToString({ shape, dtype });
    const pool = this.pools.get(key);

    if (pool && pool.length > 0) {
      const tensor = pool.pop()!;
      // Reset the tensor data to zeros
      const data = tensor.data;
      for (let i = 0; i < data.length; i++) {
        data[i] = 0;
      }
      return tensor;
    }

    // Create new tensor
    const size = shapeToSize(shape);
    const data: number[] = new Array(size).fill(0);
    return new Tensor(data, shape, computeStrides(shape), 0, dtype, false);
  }

  // Return a tensor to the pool
  release(tensor: Tensor): void {
    if (!tensor || tensor.isView) {
      // Don't pool views or null tensors
      return;
    }

    const key = poolKeyToString({ shape: tensor.shape, dtype: tensor.dtype });
    let pool = this.pools.get(key);

    if (!pool) {
      pool = [];
      this.pools.set(key, pool);
    }

    if (pool.length < this.maxPoolSize) {
      // Reset tensor state
      tensor.offset = 0;
      tensor.strides = computeStrides(tensor.shape);
      tensor.isView = false;
      pool.push(tensor);
    }
    // If pool is full, let the tensor be garbage collected
  }

  // Clear all pools
  clear(): void {
    this.pools.clear();
  }

  // Get pool statistics
  getStats(): { totalTensors: number; shapes: number } {
    let totalTensors = 0;
    for (const pool of this.pools.values()) {
      totalTensors += pool.length;
    }
    return { totalTensors, shapes: this.pools.size };
  }
}

// Global tensor pool instance
export const globalTensorPool = new TensorPool();

// Convenience functions
export function pooledZeros(shape: number[], dtype: DType = DType.Float64): Tensor {
  return globalTensorPool.get(shape, dtype);
}

export function releaseToPool(tensor: Tensor): void {
  globalTensorPool.release(tensor);
}