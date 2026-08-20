// Transformer Model Configuration
// ================================

export interface ModelConfig {
  vocabSize: number;
  hiddenDim: number;
  numLayers: number;
  numHeads: number;
  numKVHeads?: number;
  intermediateDim: number;
  maxSeqLen?: number;
  normEps?: number;
  ropeBase?: number;
}

export function createDefaultConfig(vocabSize: number = 1000, hiddenDim: number = 128): ModelConfig {
  return {
    vocabSize,
    hiddenDim,
    numLayers: 2,
    numHeads: 4,
    numKVHeads: 4,
    intermediateDim: 512,
    maxSeqLen: 512,
    normEps: 1e-6,
    ropeBase: 10000.0
  };
}
