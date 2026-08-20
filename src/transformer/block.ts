// Transformer Block (Pre-RMSNorm + Causal Attention + SwiGLU MLP + Residuals)
// ===========================================================================

import { Tensor } from "../tensor/tensor";
import { RMSNorm } from "../nn/rmsnorm";
import { SwiGLU } from "../nn/swiglu";
import { MultiHeadAttention, AttentionConfig } from "./attention";
import { RoPEFreqs } from "./rope";
import { add } from "../math/elementwise";

export interface TransformerBlockConfig {
  hiddenDim: number;
  numHeads: number;
  numKVHeads?: number;
  intermediateDim: number;
  normEps?: number;
}

export class TransformerBlock {
  hiddenDim: number;
  attentionNorm: RMSNorm;
  attention: MultiHeadAttention;
  ffnNorm: RMSNorm;
  feedForward: SwiGLU;

  constructor(config: TransformerBlockConfig) {
    this.hiddenDim = config.hiddenDim;
    const eps: number = config.normEps || 1e-6;

    // 1. Pre-Attention Normalization
    this.attentionNorm = new RMSNorm(config.hiddenDim, eps);

    // 2. Self Attention
    const attnConfig: AttentionConfig = {
      hiddenDim: config.hiddenDim,
      numHeads: config.numHeads,
      numKVHeads: config.numKVHeads || config.numHeads
    };
    this.attention = new MultiHeadAttention(attnConfig);

    // 3. Pre-FFN Normalization
    this.ffnNorm = new RMSNorm(config.hiddenDim, eps);

    // 4. SwiGLU Feed-Forward Network
    this.feedForward = new SwiGLU(config.hiddenDim, config.intermediateDim);
  }

  forward(x: Tensor, startPos: number = 0, freqs?: RoPEFreqs, kvCache?: import("./kv_cache").LayerKVCache): Tensor {
    // x shape: [seqLen, hiddenDim]

    // Step 1: Pre-RMSNorm -> Attention -> Residual Connection
    const normAttnInput: Tensor = this.attentionNorm.forward(x);
    const attnOut: Tensor = this.attention.forward(normAttnInput, startPos, freqs, kvCache);
    const h: Tensor = add(x, attnOut);

    // Step 2: Pre-RMSNorm -> SwiGLU MLP -> Residual Connection
    const normFFNInput: Tensor = this.ffnNorm.forward(h);
    const ffnOut: Tensor = this.feedForward.forward(normFFNInput);
    const out: Tensor = add(h, ffnOut);

    return out;
  }
}
