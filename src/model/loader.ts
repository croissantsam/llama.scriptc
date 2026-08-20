// Model Serialization and Deserialization (JSON format)
// ========================================================

import * as fs from "fs";
import { TransformerModel } from "./transformer";
import { ModelConfig } from "./config";
import { Tensor } from "../tensor/tensor";
import { GGUFReader } from "./gguf";

export interface SerializedModel {
  config: ModelConfig;
  weights: { [name: string]: number[] };
}

export function saveModel(model: TransformerModel, filePath: string): void {
  const serialized: SerializedModel = {
    config: model.config,
    weights: {
      "tok_embeddings": model.tokEmbeddings.weight.toArray(),
      "norm": model.norm.weight.toArray(),
      "lm_head": model.lmHead.weight.toArray()
    }
  };

  // Add weights for each layer
  for (let l: number = 0; l < model.config.numLayers; l++) {
    const layer = model.layers[l];
    serialized.weights[`layer_${l}.attn_norm`] = layer.attentionNorm.weight.toArray();
    serialized.weights[`layer_${l}.q_proj`] = layer.attention.qProj.weight.toArray();
    serialized.weights[`layer_${l}.k_proj`] = layer.attention.kProj.weight.toArray();
    serialized.weights[`layer_${l}.v_proj`] = layer.attention.vProj.weight.toArray();
    serialized.weights[`layer_${l}.o_proj`] = layer.attention.oProj.weight.toArray();
    serialized.weights[`layer_${l}.ffn_norm`] = layer.ffnNorm.weight.toArray();
    serialized.weights[`layer_${l}.gate_proj`] = layer.feedForward.gateProj.weight.toArray();
    serialized.weights[`layer_${l}.up_proj`] = layer.feedForward.upProj.weight.toArray();
    serialized.weights[`layer_${l}.down_proj`] = layer.feedForward.downProj.weight.toArray();
  }

  const jsonStr: string = JSON.stringify(serialized);
  fs.writeFileSync(filePath, jsonStr, "utf-8");
}

export function loadModel(filePath: string): TransformerModel {
  const content: string = fs.readFileSync(filePath, "utf-8");
  const data: SerializedModel = JSON.parse(content) as SerializedModel;

  const model: TransformerModel = new TransformerModel(data.config);

  // Load embeddings, norm, lm_head
  if (data.weights["tok_embeddings"]) {
    model.tokEmbeddings.weight = Tensor.fromArray(data.weights["tok_embeddings"], [data.config.vocabSize, data.config.hiddenDim]);
  }
  if (data.weights["norm"]) {
    model.norm.weight = Tensor.fromArray(data.weights["norm"], [data.config.hiddenDim]);
  }
  if (data.weights["lm_head"]) {
    model.lmHead.weight = Tensor.fromArray(data.weights["lm_head"], [data.config.vocabSize, data.config.hiddenDim]);
  }

  // Load layers
  const numKVHeads: number = data.config.numKVHeads || data.config.numHeads;
  const headDim: number = data.config.hiddenDim / data.config.numHeads;

  for (let l: number = 0; l < data.config.numLayers; l++) {
    const layer = model.layers[l];
    if (data.weights[`layer_${l}.attn_norm`]) {
      layer.attentionNorm.weight = Tensor.fromArray(data.weights[`layer_${l}.attn_norm`], [data.config.hiddenDim]);
    }
    if (data.weights[`layer_${l}.q_proj`]) {
      layer.attention.qProj.weight = Tensor.fromArray(data.weights[`layer_${l}.q_proj`], [data.config.numHeads * headDim, data.config.hiddenDim]);
    }
    if (data.weights[`layer_${l}.k_proj`]) {
      layer.attention.kProj.weight = Tensor.fromArray(data.weights[`layer_${l}.k_proj`], [numKVHeads * headDim, data.config.hiddenDim]);
    }
    if (data.weights[`layer_${l}.v_proj`]) {
      layer.attention.vProj.weight = Tensor.fromArray(data.weights[`layer_${l}.v_proj`], [numKVHeads * headDim, data.config.hiddenDim]);
    }
    if (data.weights[`layer_${l}.o_proj`]) {
      layer.attention.oProj.weight = Tensor.fromArray(data.weights[`layer_${l}.o_proj`], [data.config.hiddenDim, data.config.numHeads * headDim]);
    }
    if (data.weights[`layer_${l}.ffn_norm`]) {
      layer.ffnNorm.weight = Tensor.fromArray(data.weights[`layer_${l}.ffn_norm`], [data.config.hiddenDim]);
    }
    if (data.weights[`layer_${l}.gate_proj`]) {
      layer.feedForward.gateProj.weight = Tensor.fromArray(data.weights[`layer_${l}.gate_proj`], [data.config.intermediateDim, data.config.hiddenDim]);
    }
    if (data.weights[`layer_${l}.up_proj`]) {
      layer.feedForward.upProj.weight = Tensor.fromArray(data.weights[`layer_${l}.up_proj`], [data.config.intermediateDim, data.config.hiddenDim]);
    }
    if (data.weights[`layer_${l}.down_proj`]) {
      layer.feedForward.downProj.weight = Tensor.fromArray(data.weights[`layer_${l}.down_proj`], [data.config.hiddenDim, data.config.intermediateDim]);
    }
  }

  return model;
}

export function loadGGUFModel(filePath: string, maxLayers?: number): TransformerModel {
  const reader: GGUFReader = new GGUFReader(filePath);
  const fullConfig: ModelConfig = reader.getModelConfig();

  const numLayers: number = maxLayers !== undefined ? Math.min(maxLayers, fullConfig.numLayers) : fullConfig.numLayers;
  const config: ModelConfig = {
    vocabSize: fullConfig.vocabSize,
    hiddenDim: fullConfig.hiddenDim,
    numLayers: numLayers,
    numHeads: fullConfig.numHeads,
    numKVHeads: fullConfig.numKVHeads,
    intermediateDim: fullConfig.intermediateDim,
    maxSeqLen: Math.min(2048, fullConfig.maxSeqLen || 2048),
    normEps: fullConfig.normEps,
    ropeBase: fullConfig.ropeBase
  };

  const model: TransformerModel = new TransformerModel(config);

  console.log(`[Loader] Loading embeddings (${config.vocabSize} x ${config.hiddenDim})...`);
  if (reader.tensorMap.has("token_embd.weight")) {
    model.tokEmbeddings.weight = reader.loadTensor("token_embd.weight");
  }

  console.log(`[Loader] Loading final norm and LM head...`);
  if (reader.tensorMap.has("output_norm.weight")) {
    model.norm.weight = reader.loadTensor("output_norm.weight");
  }
  if (reader.tensorMap.has("output.weight")) {
    model.lmHead.weight = reader.loadTensor("output.weight");
  } else if (reader.tensorMap.has("token_embd.weight")) {
    // Tied weights if output.weight is not present
    model.lmHead.weight = model.tokEmbeddings.weight;
  }

  console.log(`[Loader] Loading ${numLayers} transformer layers...`);
  for (let l: number = 0; l < numLayers; l++) {
    const layer = model.layers[l];

    // Attention norm
    if (reader.tensorMap.has(`blk.${l}.attn_norm.weight`)) {
      layer.attentionNorm.weight = reader.loadTensor(`blk.${l}.attn_norm.weight`);
    }

    // Attention Q, K, V, O projections
    if (reader.tensorMap.has(`blk.${l}.attn_q.weight`)) {
      layer.attention.qProj.weight = reader.loadTensor(`blk.${l}.attn_q.weight`);
    }
    if (reader.tensorMap.has(`blk.${l}.attn_q.bias`)) {
      layer.attention.qProj.bias = reader.loadTensor(`blk.${l}.attn_q.bias`);
    }

    if (reader.tensorMap.has(`blk.${l}.attn_k.weight`)) {
      layer.attention.kProj.weight = reader.loadTensor(`blk.${l}.attn_k.weight`);
    }
    if (reader.tensorMap.has(`blk.${l}.attn_k.bias`)) {
      layer.attention.kProj.bias = reader.loadTensor(`blk.${l}.attn_k.bias`);
    }

    if (reader.tensorMap.has(`blk.${l}.attn_v.weight`)) {
      layer.attention.vProj.weight = reader.loadTensor(`blk.${l}.attn_v.weight`);
    }
    if (reader.tensorMap.has(`blk.${l}.attn_v.bias`)) {
      layer.attention.vProj.bias = reader.loadTensor(`blk.${l}.attn_v.bias`);
    }

    if (reader.tensorMap.has(`blk.${l}.attn_output.weight`)) {
      layer.attention.oProj.weight = reader.loadTensor(`blk.${l}.attn_output.weight`);
    }

    // FFN norm
    if (reader.tensorMap.has(`blk.${l}.ffn_norm.weight`)) {
      layer.ffnNorm.weight = reader.loadTensor(`blk.${l}.ffn_norm.weight`);
    }

    // SwiGLU MLP
    if (reader.tensorMap.has(`blk.${l}.ffn_gate.weight`)) {
      layer.feedForward.gateProj.weight = reader.loadTensor(`blk.${l}.ffn_gate.weight`);
    }
    if (reader.tensorMap.has(`blk.${l}.ffn_up.weight`)) {
      layer.feedForward.upProj.weight = reader.loadTensor(`blk.${l}.ffn_up.weight`);
    }
    if (reader.tensorMap.has(`blk.${l}.ffn_down.weight`)) {
      layer.feedForward.downProj.weight = reader.loadTensor(`blk.${l}.ffn_down.weight`);
    }
  }

  reader.close();
  console.log(`[Loader] Model loaded successfully.`);
  return model;
}
