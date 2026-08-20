// Tests for Sampler Strategies
// =============================

import { Tensor } from "../../src/tensor/tensor";
import { Sampler } from "../../src/sampling/sampler";
import { assert, assertEqual, runTests, TestCase } from "../../src/testing/assert";

const tests: TestCase[] = [
  {
    name: "Greedy sampling always picks maximum logit (temperature = 0)",
    fn: () => {
      const sampler: Sampler = new Sampler();
      const logits: Tensor = Tensor.fromArray([1.0, 5.0, 2.0, 0.5], [4]);
      const token: number = sampler.sample(logits, [], { temperature: 0.0 });
      assertEqual(token, 1); // max at index 1
    }
  },
  {
    name: "Repetition penalty reduces probability of seen tokens",
    fn: () => {
      const sampler: Sampler = new Sampler();
      // Token 1 has highest logit (5.0), Token 0 has 4.5
      const logits: Tensor = Tensor.fromArray([4.5, 5.0, 1.0], [3]);

      // If token 1 was already generated and rep penalty is 2.0:
      // token 1 logit becomes 5.0 / 2.0 = 2.5 < 4.5
      // Greedy should now pick token 0!
      const token: number = sampler.sample(logits, [1], { temperature: 0.0, repetitionPenalty: 2.0 });
      assertEqual(token, 0);
    }
  },
  {
    name: "Top-K restricts candidate tokens",
    fn: () => {
      const sampler: Sampler = new Sampler();
      // Only token 0 (100.0) and token 1 (99.0) are in Top-2
      const logits: Tensor = Tensor.fromArray([100.0, 99.0, 1.0, 0.0], [4]);

      // Sample 20 times with Top-K = 2
      for (let i: number = 0; i < 20; i++) {
        const token: number = sampler.sample(logits, [], { temperature: 1.0, topK: 2 });
        assert(token === 0 || token === 1, `Sampled token ${token} should be in Top-2`);
      }
    }
  }
];

export function runSamplerTests(): boolean {
  return runTests("Sampling Strategies", tests);
}

runSamplerTests();
