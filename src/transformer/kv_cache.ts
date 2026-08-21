// Key-Value Cache for Autoregressive Transformer Inference - Optimized
// ==========================================================

import { Tensor } from "../tensor/tensor";
import { computeStrides } from "../tensor/shape";

export class LayerKVCache {
  maxSeqLen: number;
  numKVHeads: number;
  headDim: number;

  k: Tensor; // [maxSeqLen, numKVHeads, headDim]
  v: Tensor; // [maxSeqLen, numKVHeads, headDim]
  curLen: number;

  constructor(maxSeqLen: number, numKVHeads: number, headDim: number) {
    this.maxSeqLen = maxSeqLen;
    this.numKVHeads = numKVHeads;
    this.headDim = headDim;
    this.curLen = 0;

    this.k = Tensor.zeros([maxSeqLen, numKVHeads, headDim]);
    this.v = Tensor.zeros([maxSeqLen, numKVHeads, headDim]);
  }

  update(startPos: number, kChunk: Tensor, vChunk: Tensor): void {
    // kChunk, vChunk shape: [chunkSeqLen, numKVHeads, headDim]
    const chunkLen: number = kChunk.shape[0];
    if (startPos + chunkLen > this.maxSeqLen) {
      throw new Error(`KV Cache overflow: startPos (${startPos}) + chunkLen (${chunkLen}) exceeds maxSeqLen (${this.maxSeqLen})`);
    }

    // Fast path: if both tensors are contiguous, use direct memory copy
    if (kChunk.isContiguous() && vChunk.isContiguous() && this.k.isContiguous() && this.v.isContiguous()) {
      const kData = this.k.data;
      const vData = this.v.data;
      const kChunkData = kChunk.data;
      const vChunkData = vChunk.data;
      
      const kStride = this.k.strides[0];
      const vStride = this.v.strides[0];
      const chunkStride = kChunk.strides[0];
      
      const kOffset = this.k.offset + startPos * kStride;
      const vOffset = this.v.offset + startPos * vStride;
      const chunkOffset = kChunk.offset;
      
      const elementsPerToken = this.numKVHeads * this.headDim;
      
      for (let s = 0; s < chunkLen; s++) {
        const srcOffset = chunkOffset + s * chunkStride;
        const dstKOffset = kOffset + s * kStride;
        const dstVOffset = vOffset + s * vStride;
        
        // Copy entire token at once
        for (let i = 0; i < elementsPerToken; i++) {
          kData[dstKOffset + i] = kChunkData[srcOffset + i];
          vData[dstVOffset + i] = vChunkData[srcOffset + i];
        }
      }
    } else {
      // General path with strides
      for (let s: number = 0; s < chunkLen; s++) {
        const pos: number = startPos + s;
        for (let h: number = 0; h < this.numKVHeads; h++) {
          for (let d: number = 0; d < this.headDim; d++) {
            this.k.set(kChunk.get(s, h, d), pos, h, d);
            this.v.set(vChunk.get(s, h, d), pos, h, d);
          }
        }
      }
    }

    this.curLen = Math.max(this.curLen, startPos + chunkLen);
  }

  // Return a VIEW of the keys up to totalLen - no copy!
  getKeysView(totalLen: number): Tensor {
    if (totalLen > this.maxSeqLen) {
      throw new Error(`Requested length ${totalLen} exceeds maxSeqLen ${this.maxSeqLen}`);
    }
    // Create a view into the existing buffer
    return new Tensor(
      this.k.data,
      [totalLen, this.numKVHeads, this.headDim],
      computeStrides([totalLen, this.numKVHeads, this.headDim]),
      this.k.offset,
      this.k.dtype,
      true
    );
  }

  // Return a VIEW of the values up to totalLen - no copy!
  getValuesView(totalLen: number): Tensor {
    if (totalLen > this.maxSeqLen) {
      throw new Error(`Requested length ${totalLen} exceeds maxSeqLen ${this.maxSeqLen}`);
    }
    // Create a view into the existing buffer
    return new Tensor(
      this.v.data,
      [totalLen, this.numKVHeads, this.headDim],
      computeStrides([totalLen, this.numKVHeads, this.headDim]),
      this.v.offset,
      this.v.dtype,
      true
    );
  }

  // Legacy methods for backward compatibility (still create copies)
  getKeys(totalLen: number): Tensor {
    return this.getKeysView(totalLen).clone();
  }

  getValues(totalLen: number): Tensor {
    return this.getValuesView(totalLen).clone();
  }

  reset(): void {
    this.curLen = 0;
    // Don't reallocate - just zero the existing buffers
    const kData = this.k.data;
    const vData = this.v.data;
    for (let i = 0; i < kData.length; i++) {
      kData[i] = 0;
      vData[i] = 0;
    }
  }
}

export class KVCache {
  numLayers: number;
  layers: LayerKVCache[];

  constructor(numLayers: number, maxSeqLen: number, numKVHeads: number, headDim: number) {
    this.numLayers = numLayers;
    this.layers = [];
    for (let i: number = 0; i < numLayers; i++) {
      this.layers.push(new LayerKVCache(maxSeqLen, numKVHeads, headDim));
    }
  }

  getLayer(layerIdx: number): LayerKVCache {
    if (layerIdx < 0 || layerIdx >= this.numLayers) {
      throw new Error(`Invalid layer index ${layerIdx} for KVCache with ${this.numLayers} layers`);
    }
    return this.layers[layerIdx];
  }

  reset(): void {
    for (let i: number = 0; i < this.numLayers; i++) {
      this.layers[i].reset();
    }
  }
}
