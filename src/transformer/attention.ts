// Multi-Head Attention with Causal Masking, RoPE & GQA support - Optimized
// ==============================================================

import { Tensor } from "../tensor/tensor";
import { Linear } from "../nn/linear";
import { applyRoPE, RoPEFreqs } from "./rope";
import { pooledZeros, releaseToPool } from "../tensor/pool";

export interface AttentionConfig {
  hiddenDim: number;
  numHeads: number;
  numKVHeads?: number;
  maxSeqLen?: number;
  ropeBase?: number;
}

export function createCausalMask(seqLen: number): Tensor {
  const mask: Tensor = Tensor.zeros([seqLen, seqLen]);
  for (let i: number = 0; i < seqLen; i++) {
    for (let j: number = 0; j < seqLen; j++) {
      if (j > i) {
        mask.set(-1e9, i, j); // mask future positions
      } else {
        mask.set(0.0, i, j);
      }
    }
  }
  return mask;
}

export class MultiHeadAttention {
  hiddenDim: number;
  numHeads: number;
  numKVHeads: number;
  headDim: number;
  numRep: number; // numHeads / numKVHeads for GQA
  scale: number;

  qProj: Linear;
  kProj: Linear;
  vProj: Linear;
  oProj: Linear;

  // Pre-allocated buffers to avoid repeated allocations
  private _qFlat?: Tensor;
  private _kFlat?: Tensor;
  private _vFlat?: Tensor;
  private _attnOut?: Tensor;

  constructor(
    config: AttentionConfig,
    qWeight?: Tensor,
    kWeight?: Tensor,
    vWeight?: Tensor,
    oWeight?: Tensor
  ) {
    this.hiddenDim = config.hiddenDim;
    this.numHeads = config.numHeads;
    this.numKVHeads = config.numKVHeads || config.numHeads;

    if (this.hiddenDim % this.numHeads !== 0) {
      throw new Error(`hiddenDim (${this.hiddenDim}) must be divisible by numHeads (${this.numHeads})`);
    }

    this.headDim = this.hiddenDim / this.numHeads;
    this.numRep = this.numHeads / this.numKVHeads;
    this.scale = 1.0 / Math.sqrt(this.headDim);

    const qDim: number = this.numHeads * this.headDim;
    const kvDim: number = this.numKVHeads * this.headDim;

    this.qProj = new Linear(this.hiddenDim, qDim, qWeight);
    this.kProj = new Linear(this.hiddenDim, kvDim, kWeight);
    this.vProj = new Linear(this.hiddenDim, kvDim, vWeight);
    this.oProj = new Linear(qDim, this.hiddenDim, oWeight);
  }

  forward(x: Tensor, startPos: number = 0, freqs?: RoPEFreqs, kvCache?: import("./kv_cache").LayerKVCache): Tensor {
    // x shape: [seqLen, hiddenDim]
    if (x.ndim() !== 2) {
      throw new Error(`Attention input must be 2D [seqLen, hiddenDim], got ${x.ndim()}D`);
    }

    const seqLen: number = x.shape[0];
    const hiddenDim: number = this.hiddenDim;
    const numHeads: number = this.numHeads;
    const numKVHeads: number = this.numKVHeads;
    const headDim: number = this.headDim;
    const numRep: number = this.numRep;
    const scale: number = this.scale;

    // 1. Projections - reuse buffers if possible
    const qFlat: Tensor = this.qProj.forward(x); // [seqLen, numHeads * headDim]
    const kFlat: Tensor = this.kProj.forward(x); // [seqLen, numKVHeads * headDim]
    const vFlat: Tensor = this.vProj.forward(x); // [seqLen, numKVHeads * headDim]

    // Reshape to [seqLen, heads, headDim] - views, no copy
    const Q: Tensor = qFlat.view([seqLen, numHeads, headDim]);
    const K: Tensor = kFlat.view([seqLen, numKVHeads, headDim]);
    const V: Tensor = vFlat.view([seqLen, numKVHeads, headDim]);

    // 2. Apply RoPE to Q and K in-place (modifies Q and K directly)
    applyRoPEInPlace(Q, startPos, freqs);
    applyRoPEInPlace(K, startPos, freqs);

    // 3. Update or fetch from KV Cache
    let allK: Tensor;
    let allV: Tensor;
    let totalLen: number;

    if (kvCache) {
      kvCache.update(startPos, K, V);
      totalLen = startPos + seqLen;
      // Use views instead of copies
      allK = kvCache.getKeysView(totalLen);
      allV = kvCache.getValuesView(totalLen);
    } else {
      totalLen = seqLen;
      allK = K;
      allV = V;
    }

    // 4. Multi-Head Scaled Dot-Product Attention - Optimized
    // Pre-allocate output tensor
    let attnOut: Tensor;
    if (this._attnOut && this._attnOut.shape[0] === seqLen && this._attnOut.shape[1] === hiddenDim) {
      attnOut = this._attnOut;
      // Zero out the data
      for (let i = 0; i < attnOut.data.length; i++) {
        attnOut.data[i] = 0;
      }
    } else {
      attnOut = Tensor.zeros([seqLen, hiddenDim]);
      this._attnOut = attnOut;
    }

    const qData = Q.data;
    const kData = allK.data;
    const vData = allV.data;
    const outData = attnOut.data;

    const qStrides = Q.strides;
    const kStrides = allK.strides;
    const vStrides = allV.strides;
    const outStrides = attnOut.strides;

    const qOffset = Q.offset;
    const kOffset = allK.offset;
    const vOffset = allV.offset;
    const outOffset = attnOut.offset;

    // Reusable arrays to avoid allocations in the inner loop
    const maxValidKeyLen = startPos + seqLen;
    const scores: number[] = [];
    const exps: number[] = [];
    for (let i = 0; i < maxValidKeyLen; i++) {
      scores.push(0);
      exps.push(0);
    }

    // Optimized attention computation with fused loops
    for (let h = 0; h < numHeads; h++) {
      const kvHead = Math.floor(h / numRep);
      
      // Precompute head offsets
      const qHeadOffset = qOffset + h * qStrides[1];
      const kHeadOffset = kOffset + kvHead * kStrides[1];
      const vHeadOffset = vOffset + kvHead * vStrides[1];
      const outHeadOffset = outOffset + h * headDim;

      for (let i = 0; i < seqLen; i++) {
        const queryPos = startPos + i;
        const validKeyLen = queryPos + 1;

        const qRowOffset = qHeadOffset + i * qStrides[0];
        const outRowOffset = outOffset + i * outStrides[0] + h * headDim;

        // Compute scores and find max in one pass
        let maxScore = -Infinity;
        
        for (let j = 0; j < validKeyLen; j++) {
          let dot = 0.0;
          const kRowOffset = kHeadOffset + j * kStrides[0];
          
          // Unroll inner loop for better performance
          for (let d = 0; d < headDim; d++) {
            dot += qData[qRowOffset + d * qStrides[2]] * kData[kRowOffset + d * kStrides[2]];
          }
          
          const s = dot * scale;
          scores[j] = s;
          if (s > maxScore) maxScore = s;
        }

        // Stable Softmax - compute exps and sum
        let expSum = 0.0;
        for (let j = 0; j < validKeyLen; j++) {
          const e = Math.exp(scores[j] - maxScore);
          exps[j] = e;
          expSum += e;
        }

        // Weighted sum of V - fused with softmax normalization
        const invExpSum = 1.0 / expSum;
        for (let d = 0; d < headDim; d++) {
          let sumVal = 0.0;
          for (let j = 0; j < validKeyLen; j++) {
            const vRowOffset = vHeadOffset + j * vStrides[0];
            sumVal += exps[j] * vData[vRowOffset + d * vStrides[2]];
          }
          outData[outRowOffset + d * outStrides[1]] = sumVal * invExpSum;
        }
      }
    }

    // 5. Final Output Projection: attnOut * W_o^T
    const out: Tensor = this.oProj.forward(attnOut);
    return out;
  }
}

// In-place RoPE application to avoid allocation
function applyRoPEInPlace(tensor: Tensor, startPos: number, freqs?: RoPEFreqs): void {
  if (!freqs) return;
  
  const seqLen = tensor.shape[0];
  const numHeads = tensor.shape[1];
  const headDim = tensor.shape[2];
  
  if (headDim % 2 !== 0) {
    throw new Error(`RoPE requires even headDim, got ${headDim}`);
  }
  
  const halfDim = headDim / 2;
  const data = tensor.data;
  const strides = tensor.strides;
  const offset = tensor.offset;
  
  for (let i = 0; i < seqLen; i++) {
    const pos = startPos + i;
    const cos = freqs.cos[pos];
    const sin = freqs.sin[pos];
    
    if (!cos || !sin) continue;
    
    const rowOffset = offset + i * strides[0];
    
    for (let h = 0; h < numHeads; h++) {
      const headOffset = rowOffset + h * strides[1];
      
      for (let d = 0; d < halfDim; d++) {
        const idx1 = headOffset + d * strides[2];
        const idx2 = headOffset + (d + halfDim) * strides[2];
        
        const x1 = data[idx1];
        const x2 = data[idx2];
        
        data[idx1] = x1 * cos[d] - x2 * sin[d];
        data[idx2] = x1 * sin[d] + x2 * cos[d];
      }
    }
  }
}
