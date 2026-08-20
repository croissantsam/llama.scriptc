// Optimized Sampling Strategies for Large LLM Vocabularies
// ==========================================================

import { Tensor } from "../tensor/tensor";

export interface SamplerOptions {
  temperature?: number;         // 0.0 = greedy, >0.0 = sampling (default: 0.0)
  topK?: number;                // keep top K logits (e.g. 40, default: 0 = disabled)
  topP?: number;                // nucleus sampling cumulative probability (e.g. 0.9, default: 1.0 = disabled)
  repetitionPenalty?: number;   // repetition penalty multiplier (default: 1.0 = disabled)
}

export class Sampler {
  sample(logits: Tensor, generatedTokens: number[] = [], options: SamplerOptions = {}): number {
    const vocabSize: number = logits.size;
    const temp: number = options.temperature !== undefined ? options.temperature : 0.0;
    const topK: number = options.topK !== undefined ? options.topK : 0;
    const topP: number = options.topP !== undefined ? options.topP : 1.0;
    const repPenalty: number = options.repetitionPenalty !== undefined ? options.repetitionPenalty : 1.0;

    // 1. Direct access to raw logits
    const rawData: number[] = logits.data;

    // 2. Greedy Sampling (temperature == 0.0)
    if (temp <= 0.0) {
      let maxIdx: number = 0;
      let maxVal: number = -Infinity;

      // Handle repetition penalty on seen tokens efficiently
      const seenSet: Set<number> = new Set<number>(generatedTokens);

      for (let i: number = 0; i < vocabSize; i++) {
        let val: number = rawData[i];
        if (repPenalty > 1.0 && seenSet.has(i)) {
          val = val < 0 ? val * repPenalty : val / repPenalty;
        }
        if (val > maxVal) {
          maxVal = val;
          maxIdx = i;
        }
      }
      return maxIdx;
    }

    // 3. Temperature & Top-K candidate extraction (O(N * K) instead of O(N log N) 150k sort)
    const effectiveK: number = (topK > 0 && topK < vocabSize) ? topK : Math.min(50, vocabSize);

    // Keep top K (indices and scaled values)
    const candIndices: number[] = [];
    const candValues: number[] = [];
    const seenSet: Set<number> = new Set<number>(generatedTokens);

    for (let i: number = 0; i < vocabSize; i++) {
      let val: number = rawData[i];
      if (repPenalty > 1.0 && seenSet.has(i)) {
        val = val < 0 ? val * repPenalty : val / repPenalty;
      }
      val = val / temp;

      if (candValues.length < effectiveK) {
        candIndices.push(i);
        candValues.push(val);
      } else {
        // Find minimum among current candidates
        let minIdx: number = 0;
        let minVal: number = candValues[0];
        for (let k: number = 1; k < effectiveK; k++) {
          if (candValues[k] < minVal) {
            minVal = candValues[k];
            minIdx = k;
          }
        }
        if (val > minVal) {
          candIndices[minIdx] = i;
          candValues[minIdx] = val;
        }
      }
    }

    const numCands: number = candValues.length;

    // 4. Softmax over the K candidates only
    let maxVal: number = candValues[0];
    for (let k: number = 1; k < numCands; k++) {
      if (candValues[k] > maxVal) maxVal = candValues[k];
    }

    const exps: number[] = [];
    let expSum: number = 0.0;
    for (let k: number = 0; k < numCands; k++) {
      const e: number = Math.exp(candValues[k] - maxVal);
      exps.push(e);
      expSum = expSum + e;
    }

    const probs: number[] = [];
    for (let k: number = 0; k < numCands; k++) {
      probs.push(expSum > 0 ? exps[k] / expSum : 1.0 / numCands);
    }

    // 5. Categorical sample over K candidates
    const rand: number = Math.random();
    let cumProb: number = 0.0;
    for (let k: number = 0; k < numCands; k++) {
      cumProb = cumProb + probs[k];
      if (rand <= cumProb) {
        return candIndices[k];
      }
    }

    return candIndices[numCands - 1];
  }
}
