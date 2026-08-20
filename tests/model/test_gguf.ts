// Test GGUF Parser on Qwen2.5-0.5B-Instruct Q8_0
// ===============================================

import { GGUFReader } from "../../src/model/gguf";
import { ModelConfig } from "../../src/model/config";
import { Tensor } from "../../src/tensor/tensor";
import { assert, assertEqual, runTests, TestCase } from "../../src/testing/assert";

const tests: TestCase[] = [
  {
    name: "GGUF Header and Metadata Parsing",
    fn: () => {
      const ggufPath = "models/qwen2.5-0.5b-instruct-q8_0.gguf";
      const reader = new GGUFReader(ggufPath);

      assertEqual(reader.version, 3);
      assertEqual(reader.tensorCount, 291);
      assert(reader.metadataCount > 0, "Metadata count should be > 0");

      const config: ModelConfig = reader.getModelConfig();
      console.log(`\n      Architecture: ${reader.stringMeta.get("general.architecture")}`);
      console.log(`      Hidden Dim  : ${config.hiddenDim}`);
      console.log(`      Num Layers  : ${config.numLayers}`);
      console.log(`      Num Heads   : ${config.numHeads} (KV Heads: ${config.numKVHeads})`);
      console.log(`      Intermediate: ${config.intermediateDim}`);
      console.log(`      Vocab Size  : ${config.vocabSize}`);
      console.log(`      RoPE Base   : ${config.ropeBase}`);

      assertEqual(config.hiddenDim, 896);
      assertEqual(config.numLayers, 24);
      assertEqual(config.numHeads, 14);
      assertEqual(config.numKVHeads || 0, 2);
      assertEqual(config.intermediateDim, 4864);

      reader.close();
    }
  },
  {
    name: "GGUF Q8_0 Tensor Loading and Dequantization",
    fn: () => {
      const ggufPath = "models/qwen2.5-0.5b-instruct-q8_0.gguf";
      const reader = new GGUFReader(ggufPath);

      // Load layer 0 attention norm (F32)
      const normTensor: Tensor = reader.loadTensor("blk.0.attn_norm.weight");
      assertEqual(normTensor.shape[0], 896);
      assert(!isNaN(normTensor.get(0)), "Norm weight should not be NaN");

      // Load layer 0 Q projection (Q8_0)
      const qTensor: Tensor = reader.loadTensor("blk.0.attn_q.weight");
      console.log(`\n      blk.0.attn_q.weight shape: [${qTensor.shape.join(", ")}]`);
      assert(!isNaN(qTensor.get(0, 0)), "Q weight should not be NaN");

      reader.close();
    }
  }
];

export function runGGUFTests(): boolean {
  return runTests("GGUF v3 Parser & Q8_0 Dequantizer", tests);
}

runGGUFTests();
