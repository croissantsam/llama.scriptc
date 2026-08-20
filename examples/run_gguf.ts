// Execute Qwen2.5-0.5B-Instruct Q8_0 GGUF in ScriptC
// ====================================================

import { loadGGUFModel } from "../src/model/loader";
import { GGUFTokenizer } from "../src/tokenizer/gguf_tokenizer";
import { Generator, GenerationResult } from "../src/runtime/generator";
import { TransformerModel } from "../src/model/transformer";

console.log("=========================================================");
console.log("     🚀 QWEN2.5-0.5B INSTRUCT (Q8_0 GGUF) IN SCRIPTC     ");
console.log("=========================================================\n");

const modelPath = "models/qwen2.5-0.5b-instruct-q8_0.gguf";

// 1. Load Tokenizer
console.log("[1/3] Loading Tokenizer from GGUF...");
const tokenizer = new GGUFTokenizer(modelPath);
console.log(`      Vocabulary size: ${tokenizer.vocabSize} tokens`);
console.log(`      BOS Token: ${tokenizer.bosTokenId}, EOS Token: ${tokenizer.eosTokenId}\n`);

// 2. Load Model Weights (24 Layers, 0.5B parameters)
// By default we can load 4 layers for instant response or full 24 layers if requested
const numLayersArg = process.argv.length > 2 ? parseInt(process.argv[2], 10) : 24;
console.log(`[2/3] Loading ${numLayersArg} layers of Qwen2.5-0.5B Model...`);
const loadStart = Date.now();
const model: TransformerModel = loadGGUFModel(modelPath, numLayersArg);
console.log(`      Model loaded in ${Date.now() - loadStart}ms\n`);

// 3. Setup Generator
const generator: Generator = new Generator(model, tokenizer);

const promptText = process.argv.length > 3 ? process.argv[3] : "Hello, my name is";
const formattedPrompt = `<|im_start|>user\n${promptText}<|im_end|>\n<|im_start|>assistant\n`;

console.log("[3/3] Generating response for prompt:");
console.log(`"${promptText}"\n`);
console.log("------------------ Response ------------------");

const genStart = Date.now();
const result: GenerationResult = generator.generate(
  formattedPrompt,
  10, // max new tokens
  {
    temperature: 0.0, // greedy for most coherent output
    repetitionPenalty: 1.1
  },
  (tok: string) => {
    process.stdout.write(tok);
  }
);

const genTime = Date.now() - genStart;
const tps = genTime > 0 ? (result.generatedTokens.length / (genTime / 1000.0)) : 0;

console.log("\n----------------------------------------------");
console.log(`\n✨ Generation completed:`);
console.log(`   - Prompt tokens    : ${result.promptTokens.length}`);
console.log(`   - Generated tokens : ${result.generatedTokens.length}`);
console.log(`   - Total latency    : ${genTime}ms`);
console.log(`   - Throughput       : ${tps.toFixed(1)} tokens/sec`);
console.log("\n=========================================================");
