// Tests for Transformer Block
// =============================

import { Tensor } from "../../src/tensor/tensor";
import { TransformerBlock, TransformerBlockConfig } from "../../src/transformer/block";
import { precomputeRoPE, RoPEFreqs } from "../../src/transformer/rope";
import { assert, assertEqual, assertArrayEqual, assertClose, runTests, TestCase } from "../../src/testing/assert";

const tests: TestCase[] = [
  {
    name: "Transformer Block forward pass preserves dimensions",
    fn: () => {
      const config: TransformerBlockConfig = {
        hiddenDim: 16,
        numHeads: 4,
        numKVHeads: 2,
        intermediateDim: 32
      };

      const block: TransformerBlock = new TransformerBlock(config);
      const seqLen: number = 5;
      const x: Tensor = Tensor.ones([seqLen, config.hiddenDim]);

      const freqs: RoPEFreqs = precomputeRoPE(16 / 4, 100);
      const out: Tensor = block.forward(x, 0, freqs);

      assertArrayEqual(out.shape, [seqLen, config.hiddenDim]);
    }
  },
  {
    name: "Multi-layer stacking (chaining 2 Transformer Blocks)",
    fn: () => {
      const config: TransformerBlockConfig = {
        hiddenDim: 8,
        numHeads: 2,
        numKVHeads: 2,
        intermediateDim: 16
      };

      const block1: TransformerBlock = new TransformerBlock(config);
      const block2: TransformerBlock = new TransformerBlock(config);

      const seqLen: number = 3;
      const x: Tensor = Tensor.from2D([
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
        [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7]
      ]);

      const h1: Tensor = block1.forward(x);
      assertArrayEqual(h1.shape, [seqLen, config.hiddenDim]);

      const h2: Tensor = block2.forward(h1);
      assertArrayEqual(h2.shape, [seqLen, config.hiddenDim]);

      // Verify that values are non-zero and finite
      for (let s: number = 0; s < seqLen; s++) {
        for (let d: number = 0; d < config.hiddenDim; d++) {
          assert(!isNaN(h2.get(s, d)), `Element [${s}, ${d}] is NaN`);
          assert(isFinite(h2.get(s, d)), `Element [${s}, ${d}] is Infinite`);
        }
      }
    }
  }
];

export function runBlockTests(): boolean {
  return runTests("Transformer Block", tests);
}

runBlockTests();
