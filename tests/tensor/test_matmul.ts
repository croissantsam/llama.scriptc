// Tests for Matrix Multiplication
// =================================

import { Tensor } from "../../src/tensor/tensor";
import { matmul } from "../../src/math/matmul";
import { assert, assertEqual, assertArrayEqual, assertClose, assertThrows, runTests, TestCase } from "../../src/testing/assert";

const tests: TestCase[] = [
  {
    name: "Canonical 2x2 Matmul example from instruction",
    fn: () => {
      // A = [[1, 2], [3, 4]]
      // B = [[5, 6], [7, 8]]
      // C = A @ B = [[19, 22], [43, 50]]
      const A: Tensor = Tensor.from2D([[1, 2], [3, 4]]);
      const B: Tensor = Tensor.from2D([[5, 6], [7, 8]]);
      const C: Tensor = matmul(A, B);

      assertArrayEqual(C.shape, [2, 2]);
      assertEqual(C.get(0, 0), 19);
      assertEqual(C.get(0, 1), 22);
      assertEqual(C.get(1, 0), 43);
      assertEqual(C.get(1, 1), 50);
    }
  },
  {
    name: "Rectangular 2D Matmul (2x3 @ 3x4 -> 2x4)",
    fn: () => {
      const A: Tensor = Tensor.from2D([
        [1, 2, 3],
        [4, 5, 6]
      ]);
      const B: Tensor = Tensor.from2D([
        [7, 8, 9, 10],
        [11, 12, 13, 14],
        [15, 16, 17, 18]
      ]);
      const C: Tensor = matmul(A, B);

      assertArrayEqual(C.shape, [2, 4]);
      // Row 0: [1*7+2*11+3*15, 1*8+2*12+3*16, 1*9+2*13+3*17, 1*10+2*14+3*18] = [74, 80, 86, 92]
      assertEqual(C.get(0, 0), 74);
      assertEqual(C.get(0, 1), 80);
      assertEqual(C.get(0, 2), 86);
      assertEqual(C.get(0, 3), 92);
      // Row 1: [4*7+5*11+6*15, 4*8+5*12+6*16, 4*9+5*13+6*17, 4*10+5*14+6*18] = [173, 188, 203, 218]
      assertEqual(C.get(1, 0), 173);
      assertEqual(C.get(1, 1), 188);
      assertEqual(C.get(1, 2), 203);
      assertEqual(C.get(1, 3), 218);
    }
  },
  {
    name: "Identity matrix multiplication",
    fn: () => {
      const A: Tensor = Tensor.from2D([[3, 7], [1, 9]]);
      const I: Tensor = Tensor.from2D([[1, 0], [0, 1]]);
      const AI: Tensor = matmul(A, I);
      assertArrayEqual(AI.to2DArray()[0], [3, 7]);
      assertArrayEqual(AI.to2DArray()[1], [1, 9]);
    }
  },
  {
    name: "1D @ 2D (vector @ matrix)",
    fn: () => {
      const v: Tensor = Tensor.fromArray([1, 2, 3], [3]); // (3)
      const M: Tensor = Tensor.from2D([[1, 2], [3, 4], [5, 6]]); // (3, 2)
      const res: Tensor = matmul(v, M); // (2)

      assertArrayEqual(res.shape, [2]);
      assertEqual(res.get(0), 1 * 1 + 2 * 3 + 3 * 5); // 22
      assertEqual(res.get(1), 1 * 2 + 2 * 4 + 3 * 6); // 28
    }
  },
  {
    name: "2D @ 1D (matrix @ vector)",
    fn: () => {
      const M: Tensor = Tensor.from2D([[1, 2], [3, 4], [5, 6]]); // (3, 2)
      const v: Tensor = Tensor.fromArray([10, 20], [2]); // (2)
      const res: Tensor = matmul(M, v); // (3)

      assertArrayEqual(res.shape, [3]);
      assertEqual(res.get(0), 1 * 10 + 2 * 20); // 50
      assertEqual(res.get(1), 3 * 10 + 4 * 20); // 110
      assertEqual(res.get(2), 5 * 10 + 6 * 20); // 170
    }
  },
  {
    name: "3D Batched Matmul (2, 2, 2) @ (2, 2, 2)",
    fn: () => {
      // Batch 0: [[1, 2], [3, 4]] @ [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
      // Batch 1: [[1, 0], [0, 1]] @ [[2, 3], [4, 5]] = [[2, 3], [4, 5]]
      const A: Tensor = Tensor.fromArray([
        1, 2, 3, 4,
        1, 0, 0, 1
      ], [2, 2, 2]);

      const B: Tensor = Tensor.fromArray([
        5, 6, 7, 8,
        2, 3, 4, 5
      ], [2, 2, 2]);

      const C: Tensor = matmul(A, B);
      assertArrayEqual(C.shape, [2, 2, 2]);
      assertEqual(C.get(0, 0, 0), 19);
      assertEqual(C.get(0, 0, 1), 22);
      assertEqual(C.get(0, 1, 0), 43);
      assertEqual(C.get(0, 1, 1), 50);

      assertEqual(C.get(1, 0, 0), 2);
      assertEqual(C.get(1, 0, 1), 3);
      assertEqual(C.get(1, 1, 0), 4);
      assertEqual(C.get(1, 1, 1), 5);
    }
  },
  {
    name: "Dimension mismatch error handling",
    fn: () => {
      const A: Tensor = Tensor.zeros([2, 3]);
      const B: Tensor = Tensor.zeros([4, 2]);
      assertThrows(() => matmul(A, B), "dimension mismatch");
    }
  }
];

export function runMatmulTests(): boolean {
  return runTests("Matrix Multiplication (Matmul)", tests);
}

runMatmulTests();
