// End-to-End LLM Generation Pipeline
// ===================================

import { TransformerModel } from "../model/transformer";
import { Tokenizer } from "../tokenizer/tokenizer";
import { Sampler, SamplerOptions } from "../sampling/sampler";
import { KVCache } from "../transformer/kv_cache";
import { Tensor } from "../tensor/tensor";

export interface GenerationResult {
  text: string;
  promptTokens: number[];
  generatedTokens: number[];
  totalTokens: number;
}

export class Generator {
  model: TransformerModel;
  tokenizer: Tokenizer;
  sampler: Sampler;

  constructor(model: TransformerModel, tokenizer: Tokenizer, sampler?: Sampler) {
    this.model = model;
    this.tokenizer = tokenizer;
    this.sampler = sampler || new Sampler();
  }

  generate(
    prompt: string,
    maxNewTokens: number = 50,
    options: SamplerOptions = {},
    onToken?: (tokenText: string) => void
  ): GenerationResult {
    // 1. Encode prompt
    const promptTokens: number[] = this.tokenizer.encode(prompt, this.tokenizer.addBosByDefault, false);
    if (promptTokens.length === 0) {
      throw new Error("Prompt produced 0 tokens");
    }

    const generatedTokens: number[] = [];
    const allTokens: number[] = promptTokens.slice();
    let emittedLen = this.tokenizer.decode(promptTokens, true).length;

    // 2. Initialize KV Cache
    const kvCache: KVCache = this.model.createKVCache();

    // 3. Prefill Phase (process entire prompt in one forward pass, compute logits for last token only)
    const prefillLogits: Tensor = this.model.forward(promptTokens, 0, kvCache, true);

    // Extract logits of the last prompt token (shape is now [1, vocabSize] or [vocabSize])
    const lastLogits: Tensor = Tensor.zeros([this.model.config.vocabSize]);
    const rowIdx: number = prefillLogits.shape.length > 1 ? prefillLogits.shape[0] - 1 : 0;
    for (let v: number = 0; v < this.model.config.vocabSize; v++) {
      lastLogits.set(prefillLogits.get(rowIdx, v), v);
    }

    // 4. Sample the first generated token
    let nextTok: number = this.sampler.sample(lastLogits, allTokens, options);

    // 5. Autoregressive Decode Loop (one token at a time)
    while (generatedTokens.length < maxNewTokens) {
      if (nextTok === this.tokenizer.eosTokenId) {
        break;
      }

      generatedTokens.push(nextTok);
      allTokens.push(nextTok);

      if (onToken) {
        // Stream properly byte-decoded text (handles multi-byte UTF-8 tokens
        // split across tokens by only emitting complete characters).
        const decodedSoFar: string = this.tokenizer.decode(allTokens, true);
        if (decodedSoFar.length > emittedLen) {
          onToken(decodedSoFar.substring(emittedLen));
          emittedLen = decodedSoFar.length;
        }
      }

      const curPos: number = promptTokens.length + generatedTokens.length - 1;
      const decodeLogits: Tensor = this.model.forward([nextTok], curPos, kvCache);

      // Extract 1D logits
      const nextLogits1D: Tensor = Tensor.zeros([this.model.config.vocabSize]);
      for (let v: number = 0; v < this.model.config.vocabSize; v++) {
        nextLogits1D.set(decodeLogits.get(0, v), v);
      }

      nextTok = this.sampler.sample(nextLogits1D, allTokens, options);
    }

    const fullText: string = this.tokenizer.decode(allTokens, true);

    return {
      text: fullText,
      promptTokens,
      generatedTokens,
      totalTokens: allTokens.length
    };
  }
}
