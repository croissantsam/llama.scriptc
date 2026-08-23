// Tests for Tokenizer
// ====================

import * as fs from "fs";
import { Tokenizer } from "../../src/tokenizer/tokenizer";
import { GGUFTokenizer } from "../../src/tokenizer/gguf_tokenizer";
import { assert, assertEqual, assertArrayEqual, runTests, TestCase } from "../../src/testing/assert";

const tests: TestCase[] = [
  {
    name: "Tokenizer special tokens",
    fn: () => {
      const tok: Tokenizer = new Tokenizer();
      assertEqual(tok.bosTokenId, 1);
      assertEqual(tok.eosTokenId, 2);
      assertEqual(tok.unkTokenId, 0);
      assertEqual(tok.padTokenId, 3);
      assertEqual(tok.idToToken(1), "<s>");
      assertEqual(tok.idToToken(2), "</s>");
    }
  },
  {
    name: "Tokenizer encode and decode roundtrip",
    fn: () => {
      const tok: Tokenizer = new Tokenizer();
      const text: string = "Hello, world! 123";

      const encoded: number[] = tok.encode(text, false, false);
      const decoded: string = tok.decode(encoded);

      assertEqual(decoded, text);
    }
  },
  {
    name: "Tokenizer with custom subwords vocabulary",
    fn: () => {
      const customWords: string[] = ["Hello", "world", "LLM", "ScriptC"];
      const tok: Tokenizer = new Tokenizer(customWords);

      // "Hello world" should encode into 3 tokens: ["Hello", " ", "world"]
      const encoded: number[] = tok.encode("Hello world", false, false);
      assertEqual(encoded.length, 3);
      assertEqual(tok.idToToken(encoded[0]), "Hello");
      assertEqual(tok.idToToken(encoded[1]), " ");
      assertEqual(tok.idToToken(encoded[2]), "world");

      const decoded: string = tok.decode(encoded);
      assertEqual(decoded, "Hello world");
    }
  },
  {
    name: "Tokenizer handles BOS and EOS flags",
    fn: () => {
      const tok: Tokenizer = new Tokenizer();
      const encoded: number[] = tok.encode("Hi", true, true);

      assertEqual(encoded[0], tok.bosTokenId);
      assertEqual(encoded[encoded.length - 1], tok.eosTokenId);

      // With skipSpecialTokens = true (default)
      assertEqual(tok.decode(encoded), "Hi");
      // With skipSpecialTokens = false
      assertEqual(tok.decode(encoded, false), "<s>Hi</s>");
    }
  },
  {
    name: "GGUF Qwen tokenizer matches reference chat prompt ids",
    fn: () => {
      const modelPath = "models/qwen2.5-0.5b-instruct-q8_0.gguf";
      if (!fs.existsSync(modelPath)) {
        return;
      }

      const tok: GGUFTokenizer = new GGUFTokenizer(modelPath);
      const prompt = "<|im_start|>user\nHello, my name is<|im_end|>\n<|im_start|>assistant\n";
      const encoded: number[] = tok.encode(prompt, tok.addBosByDefault, false);

      assertArrayEqual(encoded, [
        151644,
        872,
        198,
        9707,
        11,
        847,
        829,
        374,
        151645,
        198,
        151644,
        77091,
        198
      ]);
      assertEqual(tok.decode(encoded, true), "user\nHello, my name is\nassistant\n");
    }
  }
];

export function runTokenizerTests(): boolean {
  return runTests("Tokenizer", tests);
}

runTokenizerTests();
