// Key-Value Cache for Autoregressive Transformer Inference
// ==========================================================

import { Tensor } from "../tensor/tensor";

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

    for (let s: number = 0; s < chunkLen; s++) {
      const pos: number = startPos + s;
      for (let h: number = 0; h < this.numKVHeads; h++) {
        for (let d: number = 0; d < this.headDim; d++) {
          this.k.set(kChunk.get(s, h, d), pos, h, d);
          this.v.set(vChunk.get(s, h, d), pos, h, d);
        }
      }
    }

    this.curLen = Math.max(this.curLen, startPos + chunkLen);
  }

  getKeys(totalLen: number): Tensor {
    // Return keys from position 0 to totalLen - 1: [totalLen, numKVHeads, headDim]
    const out: Tensor = Tensor.zeros([totalLen, this.numKVHeads, this.headDim]);
    for (let s: number = 0; s < totalLen; s++) {
      for (let h: number = 0; h < this.numKVHeads; h++) {
        for (let d: number = 0; d < this.headDim; d++) {
          out.set(this.k.get(s, h, d), s, h, d);
        }
      }
    }
    return out;
  }

  getValues(totalLen: number): Tensor {
    // Return values from position 0 to totalLen - 1: [totalLen, numKVHeads, headDim]
    const out: Tensor = Tensor.zeros([totalLen, this.numKVHeads, this.headDim]);
    for (let s: number = 0; s < totalLen; s++) {
      for (let h: number = 0; h < this.numKVHeads; h++) {
        for (let d: number = 0; d < this.headDim; d++) {
          out.set(this.v.get(s, h, d), s, h, d);
        }
      }
    }
    return out;
  }

  reset(): void {
    this.curLen = 0;
    this.k = Tensor.zeros([this.maxSeqLen, this.numKVHeads, this.headDim]);
    this.v = Tensor.zeros([this.maxSeqLen, this.numKVHeads, this.headDim]);
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
