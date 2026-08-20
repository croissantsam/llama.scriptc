// Tests for Tensor Core Class
// =============================

import { Tensor } from "../../src/tensor/tensor";
import { assert, assertEqual, assertArrayEqual, assertThrows, runTests, TestCase } from "../../src/testing/assert";

const tests: TestCase[] = [
  {
    name: "Tensor.zeros and ones creation",
    fn: () => {
      const z: Tensor = Tensor.zeros([2, 3]);
      assertEqual(z.ndim(), 2);
      assertEqual(z.size, 6);
      assertArrayEqual(z.shape, [2, 3]);
      assertArrayEqual(z.strides, [3, 1]);
      assertEqual(z.get(0, 0), 0);
      assertEqual(z.get(1, 2), 0);

      const o: Tensor = Tensor.ones([3]);
      assertEqual(o.ndim(), 1);
      assertEqual(o.size, 3);
      assertEqual(o.get(0), 1);
      assertEqual(o.get(2), 1);
    }
  },
  {
    name: "Tensor.fill and fromArray",
    fn: () => {
      const f: Tensor = Tensor.fill([2, 2], 42);
      assertEqual(f.get(0, 0), 42);
      assertEqual(f.get(1, 1), 42);

      const a: Tensor = Tensor.fromArray([1, 2, 3, 4], [2, 2]);
      assertEqual(a.get(0, 0), 1);
      assertEqual(a.get(0, 1), 2);
      assertEqual(a.get(1, 0), 3);
      assertEqual(a.get(1, 1), 4);
    }
  },
  {
    name: "Tensor.from2D and to2DArray",
    fn: () => {
      const mat: number[][] = [[1, 2], [3, 4], [5, 6]];
      const t: Tensor = Tensor.from2D(mat);
      assertArrayEqual(t.shape, [3, 2]);
      assertEqual(t.get(0, 0), 1);
      assertEqual(t.get(2, 1), 6);

      const back: number[][] = t.to2DArray();
      assertEqual(back.length, 3);
      assertEqual(back[0].length, 2);
      assertEqual(back[0][0], 1);
      assertEqual(back[2][1], 6);
    }
  },
  {
    name: "Tensor get/set mutability",
    fn: () => {
      const t: Tensor = Tensor.zeros([3, 3]);
      t.set(99, 1, 2);
      assertEqual(t.get(1, 2), 99);
      assertEqual(t.get(0, 0), 0);
    }
  },
  {
    name: "Tensor bounds checking",
    fn: () => {
      const t: Tensor = Tensor.zeros([2, 3]);
      assertThrows(() => t.get(2, 0), "out of bounds");
      assertThrows(() => t.get(0, 3), "out of bounds");
      assertThrows(() => t.get(-1, 0), "out of bounds");
      assertThrows(() => t.get(0), "dimension mismatch");
      assertThrows(() => t.get(0, 0, 0), "dimension mismatch");
    }
  },
  {
    name: "Tensor reshape",
    fn: () => {
      const t: Tensor = Tensor.fromArray([1, 2, 3, 4, 5, 6], [2, 3]);
      const r: Tensor = t.reshape([3, 2]);
      assertArrayEqual(r.shape, [3, 2]);
      assertEqual(r.get(0, 0), 1);
      assertEqual(r.get(0, 1), 2);
      assertEqual(r.get(1, 0), 3);
      assertEqual(r.get(2, 1), 6);

      const flat: Tensor = t.reshape([6]);
      assertArrayEqual(flat.shape, [6]);
      assertEqual(flat.get(5), 6);

      assertThrows(() => t.reshape([2, 2]), "Cannot reshape");
    }
  },
  {
    name: "Tensor view sharing buffer",
    fn: () => {
      const t: Tensor = Tensor.fromArray([1, 2, 3, 4], [2, 2]);
      const v: Tensor = t.view([4]);
      assertEqual(v.isView, true);
      assertEqual(v.get(2), 3);

      // Mutating view should mutate original
      v.set(100, 2);
      assertEqual(t.get(1, 0), 100);
    }
  },
  {
    name: "Tensor transpose 2D",
    fn: () => {
      const t: Tensor = Tensor.from2D([[1, 2, 3], [4, 5, 6]]);
      assertArrayEqual(t.shape, [2, 3]);
      const tr: Tensor = t.transpose();
      assertArrayEqual(tr.shape, [3, 2]);
      assertEqual(tr.get(0, 0), 1);
      assertEqual(tr.get(0, 1), 4);
      assertEqual(tr.get(1, 0), 2);
      assertEqual(tr.get(1, 1), 5);
      assertEqual(tr.get(2, 0), 3);
      assertEqual(tr.get(2, 1), 6);
    }
  },
  {
    name: "Tensor clone deep copy",
    fn: () => {
      const t: Tensor = Tensor.fromArray([1, 2, 3], [3]);
      const c: Tensor = t.clone();
      c.set(999, 0);
      assertEqual(c.get(0), 999);
      assertEqual(t.get(0), 1); // Original unchanged
    }
  }
];

export function runTensorTests(): boolean {
  return runTests("Tensor Core Class", tests);
}

runTensorTests();
