// Standalone LLM Text Generation Demo in ScriptC
// ===============================================

import { Tokenizer } from "../src/tokenizer/tokenizer";
import { TransformerModel } from "../src/model/transformer";
import { ModelConfig } from "../src/model/config";
import { Generator, GenerationResult } from "../src/runtime/generator";

console.log("=========================================================");
console.log("    🚀 LLAMA.SCRIPTC — NATIVE TRANSFORMER LLM ENGINE     ");
console.log("=========================================================\n");

// 1. Initialize Tokenizer with standard vocabulary
const customVocab: string[] = [
  "The", " quick", " brown", " fox", " jumps", " over", " the", " lazy", " dog",
  "Bonjour", " je", " m'appelle", " ScriptC", " un", " moteur", " LLM", " ultra", " rapide",
  "Intelligence", " Artificielle", " Transformer", " Attention", " Llama"
];
const tokenizer: Tokenizer = new Tokenizer(customVocab);

console.log(`[1/4] Tokenizer initialized with ${tokenizer.vocabSize} vocabulary tokens.`);

// 2. Configure Miniature Llama Transformer Model (conforme à l'instruction Phase 10)
const config: ModelConfig = {
  vocabSize: tokenizer.vocabSize,
  hiddenDim: 128,
  numLayers: 2,
  numHeads: 4,
  numKVHeads: 4,
  intermediateDim: 512,
  maxSeqLen: 256,
  normEps: 1e-6,
  ropeBase: 10000.0
};

console.log("[2/4] Model Architecture:");
console.log(`      - Vocab Size       : ${config.vocabSize}`);
console.log(`      - Hidden Dimension : ${config.hiddenDim}`);
console.log(`      - Number of Layers : ${config.numLayers}`);
console.log(`      - Attention Heads  : ${config.numHeads} (KV Heads: ${config.numKVHeads})`);
console.log(`      - Intermediate Dim : ${config.intermediateDim}`);
console.log(`      - Context Window   : ${config.maxSeqLen}`);

const model: TransformerModel = new TransformerModel(config);
console.log("[3/4] Transformer model weights initialized.\n");

// 3. Create Generation Pipeline
const generator: Generator = new Generator(model, tokenizer);

const prompt: string = process.argv.length > 2 ? process.argv[2] : "Bonjour, je m'appelle";
const maxTokens: number = 20;

console.log(`[4/4] Starting generation for prompt: "${prompt}"`);
console.log("---------------------------------------------------------");
process.stdout.write(prompt);

const startTime: number = Date.now();

const result: GenerationResult = generator.generate(
  prompt,
  maxTokens,
  {
    temperature: 0.7,
    topK: 40,
    topP: 0.9,
    repetitionPenalty: 1.1
  },
  (tok: string) => {
    process.stdout.write(tok);
  }
);

const elapsedMs: number = Date.now() - startTime;
const genCount: number = result.generatedTokens.length;
const tokensPerSec: number = elapsedMs > 0 ? (genCount / (elapsedMs / 1000.0)) : 0;

console.log("\n---------------------------------------------------------");
console.log(`\n✨ Generation finished in ${elapsedMs}ms`);
console.log(`   - Prompt tokens    : ${result.promptTokens.length}`);
console.log(`   - Generated tokens : ${genCount}`);
console.log(`   - Total tokens     : ${result.totalTokens}`);
console.log(`   - Speed            : ${tokensPerSec.toFixed(1)} tokens/sec (${(elapsedMs / Math.max(1, genCount)).toFixed(2)} ms/token)`);
console.log("\n=========================================================");
