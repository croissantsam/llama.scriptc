// GGUF Tokenizer for Qwen2 and Llama Models with Byte-level UTF-8 Decoding
// =========================================================================

import * as fs from "fs";
import { GGUF_MAGIC } from "../model/gguf";
import { Tokenizer } from "./tokenizer";

export class GGUFTokenizer extends Tokenizer {
  tokens: string[] = [];
  mergeRanks: Map<string, number> = new Map<string, number>();
  specialTokenIds: Set<number> = new Set<number>();
  specialTokens: string[] = [];
  specialIds: number[] = [];
  // Inverse of the GPT-2 bytes_to_unicode map: codepoint -> original byte
  private static cpToByte: Map<number, number> = GGUFTokenizer.buildCpToByteMap();
  private static byteToCp: number[] = GGUFTokenizer.buildByteToCpMap();

  private static buildCpToByteMap(): Map<number, number> {
    const bs: number[] = [];
    for (let b = 33; b < 127; b++) bs.push(b);
    for (let b = 161; b < 173; b++) bs.push(b);
    for (let b = 174; b < 256; b++) bs.push(b);
    const cs = bs.slice();
    let n = 0;
    for (let b = 0; b < 256; b++) {
      if (!bs.includes(b)) {
        bs.push(b);
        cs.push(256 + n);
        n++;
      }
    }
    const m = new Map<number, number>();
    for (let i = 0; i < bs.length; i++) {
      m.set(cs[i], bs[i]);
    }
    return m;
  }

  private static buildByteToCpMap(): number[] {
    const byteToCp: number[] = [];
    for (let i = 0; i < 256; i++) {
      byteToCp.push(0);
    }
    const cpToByte = GGUFTokenizer.buildCpToByteMap();
    for (let cp = 0; cp < 512; cp++) {
      const byte = cpToByte.get(cp);
      if (byte !== undefined) {
        byteToCp[byte] = cp;
      }
    }
    return byteToCp;
  }

  constructor(filePath: string) {
    super();
    this.addBosByDefault = false;
    this.loadFromGGUF(filePath);
  }

  get vocabSize(): number {
    return this.tokens.length > 0 ? this.tokens.length : this.idToTokenMap.length;
  }

  loadFromGGUF(filePath: string): void {
    const fd = fs.openSync(filePath, "r");
    const stat = fs.statSync(filePath);
    const readSize = Math.min(32 * 1024 * 1024, stat.size);
    const buf = Buffer.alloc(readSize);
    fs.readSync(fd, buf, 0, readSize, 0);

    let pos = 0;
    const magic = buf.readUInt32LE(pos); pos += 4;
    if (magic !== GGUF_MAGIC) throw new Error("Invalid GGUF");
    const version = buf.readUInt32LE(pos); pos += 4;
    const tensorCount = buf.readUInt32LE(pos); pos += 8;
    const metadataCount = buf.readUInt32LE(pos); pos += 8;

    for (let i = 0; i < metadataCount && pos < readSize; i++) {
      const keyLen = buf.readUInt32LE(pos); pos += 8;
      const key = buf.toString("utf8", pos, pos + keyLen); pos += keyLen;
      const valType = buf.readUInt32LE(pos); pos += 4;

      if (key === "tokenizer.ggml.tokens" && valType === 9) { // ARRAY
        const itemType = buf.readUInt32LE(pos); pos += 4;
        const count = buf.readUInt32LE(pos); pos += 8;

        for (let t = 0; t < count && pos < readSize; t++) {
          const strLen = buf.readUInt32LE(pos); pos += 8;
          const tokenStr = buf.toString("utf8", pos, pos + strLen); pos += strLen;
          this.tokens.push(tokenStr);
          this.idToTokenMap.push(tokenStr);
          this.tokenToIdMap.set(tokenStr, t);
        }
      } else if (key === "tokenizer.ggml.merges" && valType === 9) {
        const itemType = buf.readUInt32LE(pos); pos += 4;
        const count = buf.readUInt32LE(pos); pos += 8;

        for (let m = 0; m < count && pos < readSize; m++) {
          const strLen = buf.readUInt32LE(pos); pos += 8;
          const merge = buf.toString("utf8", pos, pos + strLen); pos += strLen;
          this.mergeRanks.set(merge, m);
        }
      } else if (key === "tokenizer.ggml.bos_token_id" && (valType === 4 || valType === 5)) {
        this.bosTokenId = buf.readUInt32LE(pos); pos += 4;
      } else if (key === "tokenizer.ggml.eos_token_id" && (valType === 4 || valType === 5)) {
        this.eosTokenId = buf.readUInt32LE(pos); pos += 4;
      } else {
        // Skip metadata
        if (valType === 0 || valType === 1 || valType === 7) pos += 1;
        else if (valType === 2 || valType === 3) pos += 2;
        else if (valType === 4 || valType === 5 || valType === 6) pos += 4;
        else if (valType === 10 || valType === 11 || valType === 12) pos += 8;
        else if (valType === 8) {
          const sLen = buf.readUInt32LE(pos); pos += 8 + sLen;
        } else if (valType === 9) {
          const iType = buf.readUInt32LE(pos); pos += 4;
          const aCount = buf.readUInt32LE(pos); pos += 8;
          if (iType === 8) {
            for (let k = 0; k < aCount && pos < readSize; k++) {
              const sLen = buf.readUInt32LE(pos); pos += 8 + sLen;
            }
          } else if (iType === 0 || iType === 1 || iType === 7) pos += aCount * 1;
          else if (iType === 2 || iType === 3) pos += aCount * 2;
          else if (iType === 4 || iType === 5 || iType === 6) pos += aCount * 4;
          else if (iType === 10 || iType === 11 || iType === 12) pos += aCount * 8;
        }
      }
    }

    fs.closeSync(fd);

    for (let i = 0; i < this.tokens.length; i++) {
      const tok = this.tokens[i];
      if (this.looksLikeSpecialToken(tok)) {
        this.specialTokenIds.add(i);
        this.specialTokens.push(tok);
        this.specialIds.push(i);
      }
    }
  }

  idToToken(id: number): string {
    if (id < 0 || id >= this.tokens.length) {
      return "<unk>";
    }
    return this.tokens[id];
  }

  encode(text: string, addBos: boolean = false, addEos: boolean = false): number[] {
    const ids: number[] = [];
    if (addBos) {
      ids.push(this.bosTokenId);
    }

    let pos = 0;
    while (pos < text.length) {
      let matchedSpecial = false;
      let bestSpecial = "";
      let bestId = -1;

      for (let i = 0; i < this.specialTokens.length; i++) {
        const tok = this.specialTokens[i];
        if (tok.length > bestSpecial.length && this.startsWithAt(text, tok, pos)) {
          bestSpecial = tok;
          bestId = this.specialIds[i];
          matchedSpecial = true;
        }
      }

      if (matchedSpecial) {
        ids.push(bestId);
        pos += bestSpecial.length;
        continue;
      }

      let nextSpecialPos = text.length;
      for (let i = 0; i < this.specialTokens.length; i++) {
        const tok = this.specialTokens[i];
        const found = text.indexOf(tok, pos + 1);
        if (found >= 0 && found < nextSpecialPos) {
          nextSpecialPos = found;
        }
      }

      const segment = text.substring(pos, nextSpecialPos);
      const pieces = this.bytePairEncode(this.bytesToUnicode(segment));
      for (let i = 0; i < pieces.length; i++) {
        const id = this.tokenToIdMap.get(pieces[i]);
        ids.push(id !== undefined ? id : this.unkTokenId);
      }
      pos = nextSpecialPos;
    }

    if (addEos) {
      ids.push(this.eosTokenId);
    }
    return ids;
  }

  private looksLikeSpecialToken(tok: string): boolean {
    return tok.length >= 4
      && tok.charCodeAt(0) === 60
      && tok.charCodeAt(1) === 124
      && tok.charCodeAt(tok.length - 2) === 124
      && tok.charCodeAt(tok.length - 1) === 62;
  }

  private startsWithAt(text: string, prefix: string, pos: number): boolean {
    if (pos + prefix.length > text.length) {
      return false;
    }
    for (let i = 0; i < prefix.length; i++) {
      if (text.charCodeAt(pos + i) !== prefix.charCodeAt(i)) {
        return false;
      }
    }
    return true;
  }

  private bytesToUnicode(text: string): string {
    const bytes = Buffer.from(text, "utf8");
    let out = "";
    for (let i = 0; i < bytes.length; i++) {
      out += String.fromCharCode(GGUFTokenizer.byteToCp[bytes[i]]);
    }
    return out;
  }

  private bytePairEncode(piece: string): string[] {
    if (piece.length === 0) {
      return [];
    }

    const word: string[] = [];
    for (let i = 0; i < piece.length; i++) {
      word.push(piece.charAt(i));
    }

    while (word.length > 1) {
      let bestRank = Infinity;
      let bestIndex = -1;

      for (let i = 0; i < word.length - 1; i++) {
        const rank = this.mergeRanks.get(word[i] + " " + word[i + 1]);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestIndex = i;
        }
      }

      if (bestIndex < 0) {
        break;
      }

      word[bestIndex] = word[bestIndex] + word[bestIndex + 1];
      word.splice(bestIndex + 1, 1);
    }

    return word;
  }

  decode(tokenIds: number[], skipSpecialTokens: boolean = true): string {
    const rawBytes: number[] = [];

    for (let i = 0; i < tokenIds.length; i++) {
      const id = tokenIds[i];
      if (skipSpecialTokens) {
        if (id === this.bosTokenId || id === this.eosTokenId || this.specialTokenIds.has(id)) {
          continue;
        }
      }
      const tokStr = this.idToToken(id);

      for (let c = 0; c < tokStr.length; c++) {
        const code = tokStr.charCodeAt(c);
        // Invert the GPT-2 bytes_to_unicode mapping to recover the raw byte
        const b = GGUFTokenizer.cpToByte.get(code);
        rawBytes.push(b !== undefined ? b : code & 0xff);
      }
    }

    const outBuf = Buffer.from(rawBytes);
    return outBuf.toString("utf8");
  }
}
