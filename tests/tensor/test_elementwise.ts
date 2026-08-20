// Tests for Elementwise Operations
// ==================================

import { Tensor } from "../../src/tensor/tensor";
import { add, sub, mul, div, scale, exp, sqrt, rsqrt, neg, abs, clamp } from "../../src/math/elementwise";
import { assert, assertEqual, assertArrayEqual, assertClose, assertThrows, runTests, TestCase } from "../../src/testing/assert";

const tests: TestCase[] = [
  {
    name: "add/sub/mul/div with scalars",
    fn: () => {
      const a: Tensor = Tensor.fromArray([1, 2, 3, 4], [2, 2]);
      const added: Tensor = add(a, 10);
      assertArrayEqual(added.toArray(), [11, 12, 13, 14]);

      const subbed: Tensor = sub(a, 1);
      assertArrayEqual(subbed.toArray(), [0, 1, 2, 3]);

      const scaled: Tensor = mul(a, 2.5);
      assertArrayEqual(scaled.toArray(), [2.5, 5, 7.5, 10]);

      const divided: Tensor = div(a, 2);
      assertArrayEqual(divided.toArray(), [0.5, 1, 1.5, 2]);
    }
  },
  {
    name: "add/sub/mul/div same shape tensors",
    fn: () => {
      const a: Tensor = Tensor.fromArray([10, 20, 30, 40], [2, 2]);
      const b: Tensor = Tensor.fromArray([1, 2, 3, 4], [2, 2]);

      assertArrayEqual(add(a, b).toArray(), [11, 22, 33, 44]);
      assertArrayEqual(sub(a, b).toArray(), [9, 18, 27, 36]);
      assertArrayEqual(mul(a, b).toArray(), [10, 40, 90, 160]);
      assertArrayEqual(div(a, b).toArray(), [10, 10, 10, 10]);
    }
  },
  {
    name: "binary ops with broadcasting (2D + 1D)",
    fn: () => {
      const a: Tensor = Tensor.from2D([[1, 2, 3], [4, 5, 6]]); // shape [2, 3]
      const b: Tensor = Tensor.fromArray([10, 20, 30], [3]);    // shape [3] -> broadcasts to [2, 3]

      const res: Tensor = add(a, b);
      assertArrayEqual(res.shape, [2, 3]);
      assertEqual(res.get(0, 0), 11);
      assertEqual(res.get(0, 1), 22);
      assertEqual(res.get(0, 2), 33);
      assertEqual(res.get(1, 0), 14);
      assertEqual(res.get(1, 1), 25);
      assertEqual(res.get(1, 2), 36);
    }
  },
  {
    name: "exp function",
    fn: () => {
      const a: Tensor = Tensor.fromArray([0, 1, 2], [3]);
      const e: Tensor = exp(a);
      assertClose(e.get(0), 1.0);
      assertClose(e.get(1), Math.E);
      assertClose(e.get(2), Math.E * Math.E);
    }
  },
  {
    name: "sqrt and rsqrt functions",
    fn: () => {
      const a: Tensor = Tensor.fromArray([4, 16, 64], [3]);
      const s: Tensor = sqrt(a);
      assertClose(s.get(0), 2.0);
      assertClose(s.get(1), 4.0);
      assertClose(s.get(2), 8.0);

      const r: Tensor = rsqrt(a);
      assertClose(r.get(0), 0.5);
      assertClose(r.get(1), 0.25);
      assertClose(r.get(2), 0.125);
    }
  },
  {
    name: "rsqrt with epsilon for numerical stability",
    fn: () => {
      const a: Tensor = Tensor.fromArray([0, 3], [2]);
      const r: Tensor = rsqrt(a, 1e-6);
      assertClose(r.get(0), 1.0 / Math.sqrt(1e-6));
      assertClose(r.get(1), 1.0 / Math.sqrt(3 + 1e-6));
    }
  },
  {
    name: "neg, abs, and clamp",
    fn: () => {
      const a: Tensor = Tensor.fromArray([-5, 0, 10], [3]);
      assertArrayEqual(neg(a).toArray(), [5, -0, -10]);
      assertArrayEqual(abs(a).toArray(), [5, 0, 10]);

      const c: Tensor = clamp(a, -2, 7);
      assertArrayEqual(c.toArray(), [-2, 0, 7]);
    }
  },
  {
    name: "error handling on invalid operations",
    fn: () => {
      const a: Tensor = Tensor.fromArray([1, 2], [2]);
      const b: Tensor = Tensor.fromArray([1, 2, 3], [3]);
      assertThrows(() => add(a, b), "Incompatible shapes");
      assertThrows(() => div(a, 0), "Division by scalar zero");
      assertThrows(() => sqrt(Tensor.fromArray([-1], [1])), "negative number");
    }
  }
];

export function runElementwiseTests(): boolean {
  return runTests("Elementwise Math Operations", tests);
}

runElementwiseTests();
