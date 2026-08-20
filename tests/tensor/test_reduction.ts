// Tests for Tensor Reductions
// =============================

import { Tensor } from "../../src/tensor/tensor";
import { sumAll, meanAll, maxAll, minAll, argmaxAll, sum, mean, max, argmax } from "../../src/math/reduction";
import { assert, assertEqual, assertArrayEqual, assertClose, assertThrows, runTests, TestCase } from "../../src/testing/assert";

const tests: TestCase[] = [
  {
    name: "Full tensor reductions (sumAll, meanAll, maxAll, minAll, argmaxAll)",
    fn: () => {
      const a: Tensor = Tensor.fromArray([1, 5, 2, 8, 4], [5]);
      assertEqual(sumAll(a), 20);
      assertEqual(meanAll(a), 4);
      assertEqual(maxAll(a), 8);
      assertEqual(minAll(a), 1);
      assertEqual(argmaxAll(a), 3);
    }
  },
  {
    name: "2D sum along axis 0 (columns) and axis 1 (rows)",
    fn: () => {
      const a: Tensor = Tensor.from2D([[1, 2, 3], [4, 5, 6]]); // shape [2, 3]

      // Sum rows (axis 1) -> shape [2] -> [6, 15]
      const sumRows: Tensor = sum(a, 1);
      assertArrayEqual(sumRows.shape, [2]);
      assertEqual(sumRows.get(0), 6);
      assertEqual(sumRows.get(1), 15);

      // Sum cols (axis 0) -> shape [3] -> [5, 7, 9]
      const sumCols: Tensor = sum(a, 0);
      assertArrayEqual(sumCols.shape, [3]);
      assertEqual(sumCols.get(0), 5);
      assertEqual(sumCols.get(1), 7);
      assertEqual(sumCols.get(2), 9);
    }
  },
  {
    name: "2D sum with keepDims",
    fn: () => {
      const a: Tensor = Tensor.from2D([[1, 2], [3, 4]]);
      const sumK: Tensor = sum(a, 1, true);
      assertArrayEqual(sumK.shape, [2, 1]);
      assertEqual(sumK.get(0, 0), 3);
      assertEqual(sumK.get(1, 0), 7);
    }
  },
  {
    name: "2D mean along axis",
    fn: () => {
      const a: Tensor = Tensor.from2D([[2, 4], [6, 8]]);
      const mRows: Tensor = mean(a, 1);
      assertArrayEqual(mRows.shape, [2]);
      assertEqual(mRows.get(0), 3);
      assertEqual(mRows.get(1), 7);

      const mCols: Tensor = mean(a, 0);
      assertArrayEqual(mCols.shape, [2]);
      assertEqual(mCols.get(0), 4);
      assertEqual(mCols.get(1), 6);
    }
  },
  {
    name: "2D max along axis",
    fn: () => {
      const a: Tensor = Tensor.from2D([[10, 2], [3, 40]]);
      const maxRows: Tensor = max(a, 1);
      assertEqual(maxRows.get(0), 10);
      assertEqual(maxRows.get(1), 40);

      const maxCols: Tensor = max(a, 0);
      assertEqual(maxCols.get(0), 10);
      assertEqual(maxCols.get(1), 40);
    }
  },
  {
    name: "argmax 1D and 2D row-wise",
    fn: () => {
      const logits1D: Tensor = Tensor.fromArray([0.1, 0.9, 0.4, 0.2], [4]);
      assertArrayEqual(argmax(logits1D), [1]);

      const logits2D: Tensor = Tensor.from2D([
        [0.1, 0.8, 0.1], // max idx 1
        [0.7, 0.2, 0.1], // max idx 0
        [0.0, 0.3, 0.9]  // max idx 2
      ]);
      const picked: number[] = argmax(logits2D, 1);
      assertArrayEqual(picked, [1, 0, 2]);
    }
  }
];

export function runReductionTests(): boolean {
  return runTests("Tensor Reduction Operations", tests);
}

runReductionTests();
