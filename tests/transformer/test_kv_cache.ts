// Tests for KV Cache Autoregressive Equivalence
// ===============================================

import { Tensor } from "../../src/tensor/tensor";
import { MultiHeadAttention } from "../../src/transformer/attention";
import { LayerKVCache, KVCache } from "../../src/transformer/kv_cache";
import { assert, assertEqual, assertArrayEqual, assertClose, runTests, TestCase } from "../../src/testing/assert";

const tests: TestCase[] = [
  {
    name: "LayerKVCache update and retrieval",
    fn: () => {
      const maxSeqLen: number = 8;
      const numKVHeads: number = 2;
      const headDim: number = 4;
      const cache: LayerKVCache = new LayerKVCache(maxSeqLen, numKVHeads, headDim);

      const k1: Tensor = Tensor.ones([2, numKVHeads, headDim]);
      const v1: Tensor = Tensor.fill([2, numKVHeads, headDim], 5.0);
      cache.update(0, k1, v1);
      assertEqual(cache.curLen, 2);

      const kRetrieved: Tensor = cache.getKeys(2);
      assertArrayEqual(kRetrieved.shape, [2, numKVHeads, headDim]);
      assertEqual(kRetrieved.get(0, 0, 0), 1.0);
      assertEqual(kRetrieved.get(1, 1, 3), 1.0);

      const vRetrieved: Tensor = cache.getValues(2);
      assertEqual(vRetrieved.get(0, 0, 0), 5.0);
      assertEqual(vRetrieved.get(1, 1, 3), 5.0);
    }
  },
  {
    name: "Autoregressive Equivalence: Step-by-step KV Cache vs Full Recompute",
    fn: () => {
      const hiddenDim: number = 8;
      const numHeads: number = 2;
      const mha: MultiHeadAttention = new MultiHeadAttention({
        hiddenDim,
        numHeads
      });

      // 3 prompt tokens
      const prompt: Tensor = Tensor.from2D([
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
        [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7]
      ]);

      // 1 newly generated token
      const nextTok: Tensor = Tensor.from2D([
        [3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7]
      ]);

      // Method A: Full recomputation with all 4 tokens together [4, 8]
      const fullSeq: Tensor = Tensor.from2D([
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
        [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7],
        [3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7]
      ]);
      const fullOut: Tensor = mha.forward(fullSeq, 0); // shape [4, 8]

      // Method B: Autoregressive with KV Cache
      const cache: LayerKVCache = new LayerKVCache(16, numHeads, hiddenDim / numHeads);

      // 1. Prefill prompt (positions 0, 1, 2)
      const promptOut: Tensor = mha.forward(prompt, 0, undefined, cache);

      // 2. Decode next token (position 3, seqLen=1)
      const nextOut: Tensor = mha.forward(nextTok, 3, undefined, cache); // shape [1, 8]

      // Verify that nextOut [1, 8] is IDENTICAL to fullOut at row 3 [8]
      assertArrayEqual(nextOut.shape, [1, hiddenDim]);
      for (let d: number = 0; d < hiddenDim; d++) {
        assertClose(nextOut.get(0, d), fullOut.get(3, d), 1e-6);
      }
    }
  }
];

export function runKVCacheTests(): boolean {
  return runTests("KV Cache Autoregressive Equivalence", tests);
}

runKVCacheTests();
