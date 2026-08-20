// Tests for Tokenizer
// ====================

import { Tokenizer } from "../../src/tokenizer/tokenizer";
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
  }
];

export function runTokenizerTests(): boolean {
  return runTests("Tokenizer", tests);
}

runTokenizerTests();
