// Tests for Full Transformer Model and Loader
// ===============================================

import * as fs from "fs";
import * as path from "path";
import { Tensor } from "../../src/tensor/tensor";
import { TransformerModel } from "../../src/model/transformer";
import { ModelConfig, createDefaultConfig } from "../../src/model/config";
import { saveModel, loadModel } from "../../src/model/loader";
import { KVCache } from "../../src/transformer/kv_cache";
import { assert, assertEqual, assertArrayEqual, assertClose, runTests, TestCase } from "../../src/testing/assert";

const tests: TestCase[] = [
  {
    name: "Transformer Model initialization and forward pass",
    fn: () => {
      const config: ModelConfig = {
        vocabSize: 50,
        hiddenDim: 16,
        numLayers: 2,
        numHeads: 4,
        numKVHeads: 2,
        intermediateDim: 32,
        maxSeqLen: 64
      };

      const model: TransformerModel = new TransformerModel(config);
      const tokens: number[] = [1, 5, 12]; // sequence of 3 tokens
      const logits: Tensor = model.forward(tokens);

      assertArrayEqual(logits.shape, [3, config.vocabSize]);
      for (let s: number = 0; s < 3; s++) {
        for (let v: number = 0; v < config.vocabSize; v++) {
          assert(!isNaN(logits.get(s, v)), `Logit at [${s}, ${v}] is NaN`);
        }
      }
    }
  },
  {
    name: "Transformer Model with KV Cache forward decoding",
    fn: () => {
      const config: ModelConfig = {
        vocabSize: 20,
        hiddenDim: 8,
        numLayers: 2,
        numHeads: 2,
        intermediateDim: 16,
        maxSeqLen: 32
      };

      const model: TransformerModel = new TransformerModel(config);
      const kvCache: KVCache = model.createKVCache();

      // 1. Prefill with 2 tokens: [0, 1]
      const promptLogits: Tensor = model.forward([5, 8], 0, kvCache);
      assertArrayEqual(promptLogits.shape, [2, config.vocabSize]);

      // 2. Decode next token: [12] at position 2
      const decodeLogits: Tensor = model.forward([12], 2, kvCache);
      assertArrayEqual(decodeLogits.shape, [1, config.vocabSize]);
    }
  },
  {
    name: "Model save to JSON and load back",
    fn: () => {
      const config: ModelConfig = {
        vocabSize: 10,
        hiddenDim: 4,
        numLayers: 1,
        numHeads: 2,
        intermediateDim: 8,
        maxSeqLen: 16
      };

      const model: TransformerModel = new TransformerModel(config);
      // Set distinct weight value to verify preservation
      model.tokEmbeddings.weight.set(3.14159, 2, 1);

      const tmpFile: string = path.join(path.dirname(process.argv[1]), "test_model.json");
      saveModel(model, tmpFile);

      assert(fs.existsSync(tmpFile), "Model file should exist");

      const loaded: TransformerModel = loadModel(tmpFile);
      assertEqual(loaded.config.vocabSize, 10);
      assertEqual(loaded.config.hiddenDim, 4);
      assertClose(loaded.tokEmbeddings.weight.get(2, 1), 3.14159);

      // Clean up
      fs.unlinkSync(tmpFile);
    }
  }
];

export function runModelTests(): boolean {
  return runTests("Transformer Model & Loader", tests);
}

runModelTests();
