// Tests for End-to-End Generation Pipeline
// ==========================================

import { Tokenizer } from "../../src/tokenizer/tokenizer";
import { TransformerModel } from "../../src/model/transformer";
import { ModelConfig } from "../../src/model/config";
import { Generator, GenerationResult } from "../../src/runtime/generator";
import { assert, assertEqual, runTests, TestCase } from "../../src/testing/assert";

const tests: TestCase[] = [
  {
    name: "End-to-End Generation produces tokens and text",
    fn: () => {
      const vocabWords: string[] = ["Hello", "world", "ScriptC", "is", "fast"];
      const tokenizer: Tokenizer = new Tokenizer(vocabWords);

      const config: ModelConfig = {
        vocabSize: tokenizer.vocabSize,
        hiddenDim: 16,
        numLayers: 2,
        numHeads: 4,
        numKVHeads: 2,
        intermediateDim: 32,
        maxSeqLen: 64
      };

      const model: TransformerModel = new TransformerModel(config);
      const generator: Generator = new Generator(model, tokenizer);

      const streamTokens: string[] = [];
      const result: GenerationResult = generator.generate(
        "Hello",
        6,
        { temperature: 0.0 }, // greedy
        (tok: string) => {
          streamTokens.push(tok);
        }
      );

      assertEqual(result.generatedTokens.length, 6);
      assertEqual(streamTokens.length, 6);
      assert(result.text.length > 0, "Result text should not be empty");
    }
  }
];

export function runGeneratorTests(): boolean {
  return runTests("End-to-End Generation Pipeline", tests);
}

runGeneratorTests();
