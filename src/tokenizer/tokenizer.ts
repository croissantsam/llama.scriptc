// Tokenizer implementation (Character/Byte/BPE Vocab)
// ===================================================

export interface TokenizerVocab {
  [token: string]: number;
}

export class Tokenizer {
  tokenToIdMap: Map<string, number>;
  idToTokenMap: string[];
  bosTokenId: number;
  eosTokenId: number;
  unkTokenId: number;
  padTokenId: number;

  constructor(customVocab?: string[]) {
    this.tokenToIdMap = new Map<string, number>();
    this.idToTokenMap = [];

    // 1. Special tokens (0..3)
    this.unkTokenId = this.addToken("<unk>");
    this.bosTokenId = this.addToken("<s>");
    this.eosTokenId = this.addToken("</s>");
    this.padTokenId = this.addToken("<pad>");

    // 2. ASCII and common characters (4..131)
    for (let c: number = 32; c < 127; c++) {
      this.addToken(String.fromCharCode(c));
    }
    // Newline, tab, carriage return
    this.addToken("\n");
    this.addToken("\t");
    this.addToken("\r");

    // 3. Custom vocabulary / words if provided
    if (customVocab) {
      for (let i: number = 0; i < customVocab.length; i++) {
        const word: string = customVocab[i];
        if (!this.tokenToIdMap.has(word)) {
          this.addToken(word);
        }
      }
    }
  }

  get vocabSize(): number {
    return this.idToTokenMap.length;
  }

  addToken(token: string): number {
    if (this.tokenToIdMap.has(token)) {
      return this.tokenToIdMap.get(token)!;
    }
    const id: number = this.idToTokenMap.length;
    this.tokenToIdMap.set(token, id);
    this.idToTokenMap.push(token);
    return id;
  }

  tokenToId(token: string): number {
    const id: number | undefined = this.tokenToIdMap.get(token);
    return id !== undefined ? id : this.unkTokenId;
  }

  idToToken(id: number): string {
    if (id < 0 || id >= this.idToTokenMap.length) {
      return "<unk>";
    }
    return this.idToTokenMap[id];
  }

  encode(text: string, addBos: boolean = true, addEos: boolean = false): number[] {
    const tokens: number[] = [];

    if (addBos) {
      tokens.push(this.bosTokenId);
    }

    // Longest-matching tokenization (Greedy Forward Parsing)
    let i: number = 0;
    while (i < text.length) {
      let matched: boolean = false;
      // Try to find the longest token starting at i (up to length 32)
      const maxMatchLen: number = Math.min(32, text.length - i);
      for (let len: number = maxMatchLen; len >= 1; len--) {
        const sub: string = text.substring(i, i + len);
        if (this.tokenToIdMap.has(sub)) {
          tokens.push(this.tokenToIdMap.get(sub)!);
          i += len;
          matched = true;
          break;
        }
      }

      if (!matched) {
        // Fallback: character or unk
        tokens.push(this.unkTokenId);
        i++;
      }
    }

    if (addEos) {
      tokens.push(this.eosTokenId);
    }

    return tokens;
  }

  decode(tokenIds: number[], skipSpecialTokens: boolean = true): string {
    let text: string = "";
    for (let i: number = 0; i < tokenIds.length; i++) {
      const id: number = tokenIds[i];
      if (skipSpecialTokens) {
        if (id === this.bosTokenId || id === this.eosTokenId || id === this.padTokenId) {
          continue;
        }
      }
      text += this.idToToken(id);
    }
    return text;
  }
}
