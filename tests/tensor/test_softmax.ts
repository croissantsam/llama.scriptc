// Tests for Softmax Implementation
// =================================

import { Tensor } from "../../src/tensor/tensor";
import { softmax } from "../../src/math/softmax";
import { sumAll } from "../../src/math/reduction";
import { assert, assertEqual, assertArrayEqual, assertClose, assertThrows, runTests, TestCase } from "../../src/testing/assert";

const tests: TestCase[] = [
  {
    name: "1D Softmax basic distribution and sum to 1.0",
    fn: () => {
      const logits: Tensor = Tensor.fromArray([2.0, 1.0, 0.1], [3]);
      const probs: Tensor = softmax(logits);

      assertArrayEqual(probs.shape, [3]);
      assertClose(sumAll(probs), 1.0);

      // Verify individual values:
      // max = 2.0
      // exp(0) = 1, exp(-1) = 0.367879, exp(-1.9) = 0.149568
      // sum = 1.517448
      assertClose(probs.get(0), 1.0 / 1.517448, 1e-4);
      assertClose(probs.get(1), 0.367879 / 1.517448, 1e-4);
      assertClose(probs.get(2), 0.149568 / 1.517448, 1e-4);
    }
  },
  {
    name: "Numerical stability with extreme large values (no overflow/NaN)",
    fn: () => {
      // If not stable, Math.exp(1000) = Infinity -> NaN
      const largeLogits: Tensor = Tensor.fromArray([1000.0, 1001.0, 1002.0], [3]);
      const probs: Tensor = softmax(largeLogits);

      assertClose(sumAll(probs), 1.0);
      assert(!isNaN(probs.get(0)), "Should not be NaN");
      assert(!isNaN(probs.get(1)), "Should not be NaN");
      assert(!isNaN(probs.get(2)), "Should not be NaN");

      // Relative differences between [0, 1, 2] and [1000, 1001, 1002] are identical!
      const normalLogits: Tensor = Tensor.fromArray([0.0, 1.0, 2.0], [3]);
      const normalProbs: Tensor = softmax(normalLogits);
      assertClose(probs.get(0), normalProbs.get(0), 1e-6);
      assertClose(probs.get(1), normalProbs.get(1), 1e-6);
      assertClose(probs.get(2), normalProbs.get(2), 1e-6);
    }
  },
  {
    name: "Uniform distribution from equal logits",
    fn: () => {
      const logits: Tensor = Tensor.fromArray([5.0, 5.0, 5.0, 5.0], [4]);
      const probs: Tensor = softmax(logits);
      assertClose(probs.get(0), 0.25);
      assertClose(probs.get(1), 0.25);
      assertClose(probs.get(2), 0.25);
      assertClose(probs.get(3), 0.25);
    }
  },
  {
    name: "2D Softmax row-wise across vocab",
    fn: () => {
      const batchLogits: Tensor = Tensor.from2D([
        [1.0, 2.0, 3.0],
        [10.0, 10.0, 10.0]
      ]);
      const probs: Tensor = softmax(batchLogits, 1);
      assertArrayEqual(probs.shape, [2, 3]);

      // Row 0 sum = 1.0
      const row0Sum: number = probs.get(0, 0) + probs.get(0, 1) + probs.get(0, 2);
      assertClose(row0Sum, 1.0);

      // Row 1 sum = 1.0 (uniform)
      assertClose(probs.get(1, 0), 1.0 / 3.0);
      assertClose(probs.get(1, 1), 1.0 / 3.0);
      assertClose(probs.get(1, 2), 1.0 / 3.0);
    }
  },
  {
    name: "3D Softmax for Attention Weights",
    fn: () => {
      // Shape [1, 2, 2]: 1 batch, 2 query tokens, 2 key positions
      const attnScores: Tensor = Tensor.fromArray([
        1.0, 2.0,
        3.0, 1.0
      ], [1, 2, 2]);

      const attnWeights: Tensor = softmax(attnScores, 2);
      assertArrayEqual(attnWeights.shape, [1, 2, 2]);

      const sumQuery0: number = attnWeights.get(0, 0, 0) + attnWeights.get(0, 0, 1);
      const sumQuery1: number = attnWeights.get(0, 1, 0) + attnWeights.get(0, 1, 1);
      assertClose(sumQuery0, 1.0);
      assertClose(sumQuery1, 1.0);
    }
  }
];

export function runSoftmaxTests(): boolean {
  return runTests("Softmax Normalization", tests);
}

runSoftmaxTests();
