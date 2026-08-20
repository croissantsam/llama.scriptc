// Complete Transformer Architecture for Text Generation
// ========================================================
// tokens -> tokEmbeddings -> N x TransformerBlock -> RMSNorm -> lmHead (logits)

import { Tensor } from "../tensor/tensor";
import { Embedding } from "../nn/embedding";
import { Linear } from "../nn/linear";
import { RMSNorm } from "../nn/rmsnorm";
import { TransformerBlock, TransformerBlockConfig } from "../transformer/block";
import { precomputeRoPE, RoPEFreqs } from "../transformer/rope";
import { KVCache } from "../transformer/kv_cache";
import { ModelConfig } from "./config";

export class TransformerModel {
  config: ModelConfig;
  tokEmbeddings: Embedding;
  layers: TransformerBlock[];
  norm: RMSNorm;
  lmHead: Linear;
  ropeFreqs: RoPEFreqs;

  constructor(config: ModelConfig) {
    this.config = config;
    const eps: number = config.normEps || 1e-6;
    const maxSeqLen: number = config.maxSeqLen || 512;
    const numKVHeads: number = config.numKVHeads || config.numHeads;
    const ropeBase: number = config.ropeBase || 10000.0;
    const headDim: number = config.hiddenDim / config.numHeads;

    // 1. Token Embeddings
    this.tokEmbeddings = new Embedding(config.vocabSize, config.hiddenDim);

    // 2. Transformer Blocks
    this.layers = [];
    const blockConfig: TransformerBlockConfig = {
      hiddenDim: config.hiddenDim,
      numHeads: config.numHeads,
      numKVHeads,
      intermediateDim: config.intermediateDim,
      normEps: eps
    };

    for (let i: number = 0; i < config.numLayers; i++) {
      this.layers.push(new TransformerBlock(blockConfig));
    }

    // 3. Final Pre-Logits Normalization
    this.norm = new RMSNorm(config.hiddenDim, eps);

    // 4. Output Projection (LM Head): [hiddenDim] -> [vocabSize]
    this.lmHead = new Linear(config.hiddenDim, config.vocabSize);

    // 5. Precomputed RoPE Frequencies
    this.ropeFreqs = precomputeRoPE(headDim, maxSeqLen, ropeBase);
  }

  forward(tokens: number[] | Tensor, startPos: number = 0, kvCache?: KVCache, lastTokenOnly: boolean = false): Tensor {
    // 1. Embed input tokens: [seqLen, hiddenDim]
    let h: Tensor = this.tokEmbeddings.forward(tokens);

    // 2. Pass through each Transformer Layer
    for (let l: number = 0; l < this.config.numLayers; l++) {
      const layerCache: import("../transformer/kv_cache").LayerKVCache | undefined = kvCache ? kvCache.getLayer(l) : undefined;
      h = this.layers[l].forward(h, startPos, this.ropeFreqs, layerCache);
    }

    // If only the last token logits are needed (e.g. prompt prefill), slice before expensive LM head projection
    if (lastTokenOnly && h.shape[0] > 1) {
      const lastRow: Tensor = Tensor.zeros([1, this.config.hiddenDim]);
      const lastIdx: number = h.shape[0] - 1;
      for (let d: number = 0; d < this.config.hiddenDim; d++) {
        lastRow.set(h.get(lastIdx, d), 0, d);
      }
      h = lastRow;
    }

    // 3. Final RMSNorm
    const normalized: Tensor = this.norm.forward(h);

    // 4. Project to vocabulary logits: [seqLen or 1, vocabSize]
    const logits: Tensor = this.lmHead.forward(normalized);
    return logits;
  }

  createKVCache(): KVCache {
    const maxSeqLen: number = this.config.maxSeqLen || 512;
    const numKVHeads: number = this.config.numKVHeads || this.config.numHeads;
    const headDim: number = this.config.hiddenDim / this.config.numHeads;
    return new KVCache(this.config.numLayers, maxSeqLen, numKVHeads, headDim);
  }
}
