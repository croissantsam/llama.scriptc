// Multi-Head Attention with Causal Masking, RoPE & GQA support
// ==============================================================

import { Tensor } from "../tensor/tensor";
import { Linear } from "../nn/linear";
import { softmax } from "../math/softmax";
import { applyRoPE, RoPEFreqs } from "./rope";

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

    // 1. Projections
    const qFlat: Tensor = this.qProj.forward(x); // [seqLen, numHeads * headDim]
    const kFlat: Tensor = this.kProj.forward(x); // [seqLen, numKVHeads * headDim]
    const vFlat: Tensor = this.vProj.forward(x); // [seqLen, numKVHeads * headDim]

    // Reshape to [seqLen, heads, headDim]
    const Q: Tensor = qFlat.view([seqLen, this.numHeads, this.headDim]);
    const K: Tensor = kFlat.view([seqLen, this.numKVHeads, this.headDim]);
    const V: Tensor = vFlat.view([seqLen, this.numKVHeads, this.headDim]);

    // 2. Apply RoPE to Q and K
    const qRotated: Tensor = applyRoPE(Q, startPos, freqs);
    const kRotated: Tensor = applyRoPE(K, startPos, freqs);

    // 3. Update or fetch from KV Cache
    let allK: Tensor;
    let allV: Tensor;
    let totalLen: number;

    if (kvCache) {
      kvCache.update(startPos, kRotated, V);
      totalLen = startPos + seqLen;
      allK = kvCache.getKeys(totalLen);
      allV = kvCache.getValues(totalLen);
    } else {
      totalLen = seqLen;
      allK = kRotated;
      allV = V;
    }

    // 4. Multi-Head Scaled Dot-Product Attention
    const attnOut: Tensor = Tensor.zeros([seqLen, this.hiddenDim]);

    for (let h: number = 0; h < this.numHeads; h++) {
      const kvHead: number = Math.floor(h / this.numRep);

      for (let i: number = 0; i < seqLen; i++) {
        const queryPos: number = startPos + i;
        const validKeyLen: number = queryPos + 1; // can attend up to its own position

        // Compute scores for this query token across valid keys
        const scores: number[] = [];
        let maxScore: number = -Infinity;

        for (let j: number = 0; j < validKeyLen; j++) {
          let dot: number = 0.0;
          for (let d: number = 0; d < this.headDim; d++) {
            dot += qRotated.get(i, h, d) * allK.get(j, kvHead, d);
          }
          const s: number = dot * this.scale;
          scores.push(s);
          if (s > maxScore) maxScore = s;
        }

        // Stable Softmax
        const exps: number[] = [];
        let expSum: number = 0.0;
        for (let j: number = 0; j < validKeyLen; j++) {
          const e: number = Math.exp(scores[j] - maxScore);
          exps.push(e);
          expSum += e;
        }

        // Weighted sum of V
        for (let d: number = 0; d < this.headDim; d++) {
          let sumVal: number = 0.0;
          for (let j: number = 0; j < validKeyLen; j++) {
            sumVal += (exps[j] / expSum) * allV.get(j, kvHead, d);
          }
          attnOut.set(sumVal, i, h * this.headDim + d);
        }
      }
    }

    // 5. Final Output Projection: attnOut * W_o^T
    const out: Tensor = this.oProj.forward(attnOut);
    return out;
  }
}
