// Embedding Layer (Token ID to Vector representation)
// ====================================================

import { Tensor } from "../tensor/tensor";

export class Embedding {
  numEmbeddings: number; // vocab size
  embeddingDim: number;  // hidden dimension
  weight: Tensor;        // shape [numEmbeddings, embeddingDim]

  constructor(numEmbeddings: number, embeddingDim: number, weight?: Tensor) {
    this.numEmbeddings = numEmbeddings;
    this.embeddingDim = embeddingDim;

    if (weight) {
      if (weight.shape[0] !== numEmbeddings || weight.shape[1] !== embeddingDim) {
        throw new Error(`Embedding weight shape mismatch: expected [${numEmbeddings}, ${embeddingDim}], got [${weight.shape.join(", ")}]`);
      }
      this.weight = weight;
    } else {
      this.weight = Tensor.zeros([numEmbeddings, embeddingDim]);
    }
  }

  forward(input: number[] | Tensor): Tensor {
    // If input is an array of token IDs [token_0, token_1, ..., token_N]
    if (Array.isArray(input)) {
      const seqLen: number = input.length;
      const out: Tensor = Tensor.zeros([seqLen, this.embeddingDim]);

      for (let s: number = 0; s < seqLen; s++) {
        const tokenId: number = input[s];
        this.checkTokenId(tokenId);
        for (let d: number = 0; d < this.embeddingDim; d++) {
          out.set(this.weight.get(tokenId, d), s, d);
        }
      }
      return out;
    }

    // If input is a 1D Tensor of token IDs [seqLen]
    if (input.ndim() === 1) {
      const seqLen: number = input.size;
      const out: Tensor = Tensor.zeros([seqLen, this.embeddingDim]);

      for (let s: number = 0; s < seqLen; s++) {
        const tokenId: number = Math.floor(input.get(s));
        this.checkTokenId(tokenId);
        for (let d: number = 0; d < this.embeddingDim; d++) {
          out.set(this.weight.get(tokenId, d), s, d);
        }
      }
      return out;
    }

    // If input is a 2D Tensor of token IDs [batchSize, seqLen]
    if (input.ndim() === 2) {
      const batchSize: number = input.shape[0];
      const seqLen: number = input.shape[1];
      const out: Tensor = Tensor.zeros([batchSize, seqLen, this.embeddingDim]);

      for (let b: number = 0; b < batchSize; b++) {
        for (let s: number = 0; s < seqLen; s++) {
          const tokenId: number = Math.floor(input.get(b, s));
          this.checkTokenId(tokenId);
          for (let d: number = 0; d < this.embeddingDim; d++) {
            out.set(this.weight.get(tokenId, d), b, s, d);
          }
        }
      }
      return out;
    }

    throw new Error(`Unsupported input dimension for Embedding: ${input.ndim()}D`);
  }

  private checkTokenId(tokenId: number): void {
    if (tokenId < 0 || tokenId >= this.numEmbeddings) {
      throw new Error(`Token ID ${tokenId} is out of bounds for vocab size ${this.numEmbeddings}`);
    }
  }
}
