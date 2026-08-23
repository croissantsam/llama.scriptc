// GGUF Tokenizer for Qwen2 and Llama Models with Byte-level UTF-8 Decoding
// =========================================================================

import * as fs from "fs";
import { GGUF_MAGIC } from "../model/gguf";
import { Tokenizer } from "./tokenizer";

export class GGUFTokenizer extends Tokenizer {
  tokens: string[] = [];
  // Inverse of the GPT-2 bytes_to_unicode map: codepoint -> original byte
  private static cpToByte: Map<number, number> = GGUFTokenizer.buildCpToByteMap();

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

  constructor(filePath: string) {
    super();
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
  }

  idToToken(id: number): string {
    if (id < 0 || id >= this.tokens.length) {
      return "<unk>";
    }
    return this.tokens[id];
  }

  decode(tokenIds: number[], skipSpecialTokens: boolean = true): string {
    const rawBytes: number[] = [];

    for (let i = 0; i < tokenIds.length; i++) {
      const id = tokenIds[i];
      if (skipSpecialTokens) {
        if (id === this.bosTokenId || id === this.eosTokenId || id === 151644 || id === 151645) {
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
