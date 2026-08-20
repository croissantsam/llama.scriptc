// Tests for Rotary Positional Embeddings (RoPE)
// =============================================

import { Tensor } from "../../src/tensor/tensor";
import { precomputeRoPE, applyRoPE } from "../../src/transformer/rope";
import { assert, assertEqual, assertArrayEqual, assertClose, runTests, TestCase } from "../../src/testing/assert";

const tests: TestCase[] = [
  {
    name: "RoPE at position 0 is Identity transform",
    fn: () => {
      const dim: number = 4;
      const x: Tensor = Tensor.from2D([
        [1.0, 2.0, 3.0, 4.0] // position 0
      ]);
      const rotated: Tensor = applyRoPE(x, 0);

      // At position 0, cos=1, sin=0, so x should be untouched
      assertClose(rotated.get(0, 0), 1.0);
      assertClose(rotated.get(0, 1), 2.0);
      assertClose(rotated.get(0, 2), 3.0);
      assertClose(rotated.get(0, 3), 4.0);
    }
  },
  {
    name: "RoPE preserves L2 norm (unitary transformation)",
    fn: () => {
      const dim: number = 4;
      const x: Tensor = Tensor.from2D([
        [2.0, 3.0, 4.0, 5.0]
      ]);
      const normBefore: number = Math.sqrt(2*2 + 3*3 + 4*4 + 5*5);

      // Rotate at position 10
      const rotated: Tensor = applyRoPE(x, 10);
      const r0: number = rotated.get(0, 0);
      const r1: number = rotated.get(0, 1);
      const r2: number = rotated.get(0, 2);
      const r3: number = rotated.get(0, 3);
      const normAfter: number = Math.sqrt(r0*r0 + r1*r1 + r2*r2 + r3*r3);

      assertClose(normAfter, normBefore, 1e-6);
    }
  },
  {
    name: "RoPE 3D tensor across heads and sequence positions",
    fn: () => {
      // shape: [seqLen=3, numHeads=2, headDim=4]
      const seqLen: number = 3;
      const numHeads: number = 2;
      const headDim: number = 4;

      const x: Tensor = Tensor.ones([seqLen, numHeads, headDim]);
      const rotated: Tensor = applyRoPE(x, 0);

      assertArrayEqual(rotated.shape, [seqLen, numHeads, headDim]);

      // At position 0: elements are unchanged (all 1.0)
      for (let h: number = 0; h < numHeads; h++) {
        for (let d: number = 0; d < headDim; d++) {
          assertClose(rotated.get(0, h, d), 1.0);
        }
      }

      // At position 1: elements are rotated
      const r0: number = rotated.get(1, 0, 0);
      const r1: number = rotated.get(1, 0, 1);
      // Norm of pair (r0, r1) should equal norm of (1, 1) = sqrt(2)
      assertClose(Math.sqrt(r0 * r0 + r1 * r1), Math.sqrt(2.0), 1e-6);
    }
  }
];

export function runRoPETests(): boolean {
  return runTests("Rotary Positional Embeddings (RoPE)", tests);
}

runRoPETests();
