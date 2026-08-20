// Tests for Causal Multi-Head Attention
// =======================================

import { Tensor } from "../../src/tensor/tensor";
import { createCausalMask, MultiHeadAttention } from "../../src/transformer/attention";
import { assert, assertEqual, assertArrayEqual, assertClose, runTests, TestCase } from "../../src/testing/assert";

const tests: TestCase[] = [
  {
    name: "Causal Mask creation",
    fn: () => {
      const mask: Tensor = createCausalMask(3);
      assertArrayEqual(mask.shape, [3, 3]);

      // Lower triangle (j <= i) must be 0
      assertEqual(mask.get(0, 0), 0);
      assertEqual(mask.get(1, 0), 0);
      assertEqual(mask.get(1, 1), 0);
      assertEqual(mask.get(2, 0), 0);
      assertEqual(mask.get(2, 1), 0);
      assertEqual(mask.get(2, 2), 0);

      // Upper triangle (j > i) must be masked (-1e9)
      assertEqual(mask.get(0, 1), -1e9);
      assertEqual(mask.get(0, 2), -1e9);
      assertEqual(mask.get(1, 2), -1e9);
    }
  },
  {
    name: "Multi-Head Attention forward pass dimensions",
    fn: () => {
      const hiddenDim: number = 8;
      const numHeads: number = 2; // headDim = 4
      const mha: MultiHeadAttention = new MultiHeadAttention({
        hiddenDim,
        numHeads
      });

      const seqLen: number = 4;
      const x: Tensor = Tensor.ones([seqLen, hiddenDim]);
      const out: Tensor = mha.forward(x);

      assertArrayEqual(out.shape, [seqLen, hiddenDim]);
    }
  },
  {
    name: "Causality validation: future tokens do NOT affect past outputs",
    fn: () => {
      const hiddenDim: number = 4;
      const numHeads: number = 2;
      const mha: MultiHeadAttention = new MultiHeadAttention({
        hiddenDim,
        numHeads
      });

      // Sequence 1: 2 tokens
      const x1: Tensor = Tensor.from2D([
        [1.0, 2.0, 3.0, 4.0],
        [0.5, 1.5, 2.5, 3.5]
      ]);
      const out1: Tensor = mha.forward(x1);

      // Sequence 2: Same first 2 tokens + 1 extra token appended
      const x2: Tensor = Tensor.from2D([
        [1.0, 2.0, 3.0, 4.0],
        [0.5, 1.5, 2.5, 3.5],
        [99.0, -88.0, 77.0, -66.0] // Future token
      ]);
      const out2: Tensor = mha.forward(x2);

      // Output at position 0 must be EXACTLY identical between seq1 and seq2
      for (let d: number = 0; d < hiddenDim; d++) {
        assertClose(out1.get(0, d), out2.get(0, d), 1e-6);
        assertClose(out1.get(1, d), out2.get(1, d), 1e-6);
      }
    }
  },
  {
    name: "Grouped-Query Attention (GQA) forward pass",
    fn: () => {
      const hiddenDim: number = 8;
      const numHeads: number = 4;   // 4 query heads
      const numKVHeads: number = 2; // 2 KV heads (each shared across 2 Q heads)
      const gqa: MultiHeadAttention = new MultiHeadAttention({
        hiddenDim,
        numHeads,
        numKVHeads
      });

      const x: Tensor = Tensor.ones([3, hiddenDim]);
      const out: Tensor = gqa.forward(x);
      assertArrayEqual(out.shape, [3, hiddenDim]);
    }
  }
];

export function runAttentionTests(): boolean {
  return runTests("Causal Multi-Head Attention", tests);
}

runAttentionTests();
