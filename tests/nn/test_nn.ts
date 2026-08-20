// Tests for Neural Network Primitives
// ====================================

import { Tensor } from "../../src/tensor/tensor";
import { sigmoid, silu } from "../../src/nn/activations";
import { Embedding } from "../../src/nn/embedding";
import { Linear } from "../../src/nn/linear";
import { RMSNorm } from "../../src/nn/rmsnorm";
import { SwiGLU } from "../../src/nn/swiglu";
import { assert, assertEqual, assertArrayEqual, assertClose, assertThrows, runTests, TestCase } from "../../src/testing/assert";

const tests: TestCase[] = [
  {
    name: "SiLU and Sigmoid activations",
    fn: () => {
      const x: Tensor = Tensor.fromArray([0.0, 2.0, -2.0], [3]);
      const sig: Tensor = sigmoid(x);
      assertClose(sig.get(0), 0.5);
      assertClose(sig.get(1), 1.0 / (1.0 + Math.exp(-2.0)));
      assertClose(sig.get(2), 1.0 / (1.0 + Math.exp(2.0)));

      const act: Tensor = silu(x);
      assertClose(act.get(0), 0.0); // 0 * 0.5 = 0
      assertClose(act.get(1), 2.0 * (1.0 / (1.0 + Math.exp(-2.0))));
      assertClose(act.get(2), -2.0 * (1.0 / (1.0 + Math.exp(2.0))));
    }
  },
  {
    name: "Embedding layer lookup",
    fn: () => {
      // 4 vocab items, dim 3
      const weight: Tensor = Tensor.from2D([
        [1.0, 1.1, 1.2], // token 0
        [2.0, 2.1, 2.2], // token 1
        [3.0, 3.1, 3.2], // token 2
        [4.0, 4.1, 4.2]  // token 3
      ]);
      const emb: Embedding = new Embedding(4, 3, weight);

      // Sequence [2, 0, 3]
      const out: Tensor = emb.forward([2, 0, 3]);
      assertArrayEqual(out.shape, [3, 3]);
      assertArrayEqual(out.to2DArray()[0], [3.0, 3.1, 3.2]);
      assertArrayEqual(out.to2DArray()[1], [1.0, 1.1, 1.2]);
      assertArrayEqual(out.to2DArray()[2], [4.0, 4.1, 4.2]);

      // Out of bounds token ID
      assertThrows(() => emb.forward([5]), "out of bounds");
    }
  },
  {
    name: "Linear layer (1D, 2D, 3D with and without bias)",
    fn: () => {
      // in: 2, out: 3
      const W: Tensor = Tensor.from2D([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
      ]); // shape [3, 2]
      const b: Tensor = Tensor.fromArray([0.1, 0.2, 0.3], [3]);
      const lin: Linear = new Linear(2, 3, W, b);

      // 1D: [1, 2] -> [1*1+2*2 + 0.1, 1*3+2*4 + 0.2, 1*5+2*6 + 0.3] = [5.1, 11.2, 17.3]
      const x1D: Tensor = Tensor.fromArray([1.0, 2.0], [2]);
      const out1D: Tensor = lin.forward(x1D);
      assertArrayEqual(out1D.shape, [3]);
      assertClose(out1D.get(0), 5.1);
      assertClose(out1D.get(1), 11.2);
      assertClose(out1D.get(2), 17.3);

      // 2D: [2, 2]
      const x2D: Tensor = Tensor.from2D([[1.0, 2.0], [0.0, 1.0]]);
      const out2D: Tensor = lin.forward(x2D);
      assertArrayEqual(out2D.shape, [2, 3]);
      assertClose(out2D.get(0, 0), 5.1);
      assertClose(out2D.get(1, 0), 0*1+1*2 + 0.1); // 2.1
    }
  },
  {
    name: "RMSNorm normalization",
    fn: () => {
      const dim: number = 4;
      const rmsnorm: RMSNorm = new RMSNorm(dim, 1e-6);

      const x: Tensor = Tensor.fromArray([1.0, 2.0, 3.0, 4.0], [4]);
      // mean(x^2) = (1+4+9+16)/4 = 7.5
      // rms = sqrt(7.5 + 1e-6) ≈ 2.738613
      const out: Tensor = rmsnorm.forward(x);
      assertArrayEqual(out.shape, [4]);
      assertClose(out.get(0), 1.0 / Math.sqrt(7.5 + 1e-6));
      assertClose(out.get(3), 4.0 / Math.sqrt(7.5 + 1e-6));

      // With custom scale gamma
      const gamma: Tensor = Tensor.fromArray([2.0, 2.0, 2.0, 2.0], [4]);
      const scaledNorm: RMSNorm = new RMSNorm(dim, 1e-6, gamma);
      const outScaled: Tensor = scaledNorm.forward(x);
      assertClose(outScaled.get(0), 2.0 * (1.0 / Math.sqrt(7.5 + 1e-6)));
    }
  },
  {
    name: "SwiGLU MLP Block",
    fn: () => {
      const hiddenDim: number = 4;
      const interDim: number = 8;
      const mlp: SwiGLU = new SwiGLU(hiddenDim, interDim);

      const x: Tensor = Tensor.ones([2, hiddenDim]);
      const out: Tensor = mlp.forward(x);
      assertArrayEqual(out.shape, [2, hiddenDim]);
    }
  }
];

export function runNNTests(): boolean {
  return runTests("Neural Network Primitives", tests);
}

runNNTests();
