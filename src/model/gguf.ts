// GGUF v3 Binary Parser and Q8_0 Dequantizer for ScriptC
// ========================================================

import * as fs from "fs";
import { Tensor } from "../tensor/tensor";
import { ModelConfig } from "./config";

export const GGUF_MAGIC: number = 0x46554747; // "GGUF" in little-endian

export interface GGUFTensorInfo {
  name: string;
  shape: number[];
  type: number;
  offset: number;
}

export function f16ToF32(h: number): number {
  const s: number = (h >> 15) & 1;
  const e: number = (h >> 10) & 0x1f;
  const f: number = h & 0x3ff;
  if (e === 0) {
    if (f === 0) return s ? -0.0 : 0.0;
    return (s ? -1.0 : 1.0) * Math.pow(2, -14) * (f / 1024.0);
  } else if (e === 31) {
    return f ? NaN : (s ? -Infinity : Infinity);
  }
  return (s ? -1.0 : 1.0) * Math.pow(2, e - 15) * (1.0 + f / 1024.0);
}

export class GGUFReader {
  fd: number;
  buffer: Buffer;
  pos: number;
  fileSize: number;

  version: number = 0;
  tensorCount: number = 0;
  metadataCount: number = 0;
  
  stringMeta: Map<string, string> = new Map<string, string>();
  numMeta: Map<string, number> = new Map<string, number>();
  boolMeta: Map<string, boolean> = new Map<string, boolean>();

  tensorList: GGUFTensorInfo[] = [];
  tensorMap: Map<string, GGUFTensorInfo> = new Map<string, GGUFTensorInfo>();
  
  tensorDataOffset: number = 0;
  alignment: number = 32;

  constructor(filePath: string) {
    this.fd = fs.openSync(filePath, "r");
    const stat = fs.statSync(filePath);
    this.fileSize = stat.size;
    // Read first 16 MB for headers and metadata
    const headerBufSize = Math.min(16 * 1024 * 1024, this.fileSize);
    this.buffer = Buffer.alloc(headerBufSize);
    fs.readSync(this.fd, this.buffer, 0, headerBufSize, 0);
    this.pos = 0;

    this.parseHeader();
    this.parseMetadata();
    this.parseTensorInfos();
  }

  readUint8(): number {
    const val = this.buffer.readUInt8(this.pos);
    this.pos += 1;
    return val;
  }

  readInt8(): number {
    const val = this.buffer.readInt8(this.pos);
    this.pos += 1;
    return val;
  }

  readUint16(): number {
    const val = this.buffer.readUInt16LE(this.pos);
    this.pos += 2;
    return val;
  }

  readInt16(): number {
    const val = this.buffer.readInt16LE(this.pos);
    this.pos += 2;
    return val;
  }

  readUint32(): number {
    const val = this.buffer.readUInt32LE(this.pos);
    this.pos += 4;
    return val;
  }

  readInt32(): number {
    const val = this.buffer.readInt32LE(this.pos);
    this.pos += 4;
    return val;
  }

  readFloat32(): number {
    const val = this.buffer.readFloatLE(this.pos);
    this.pos += 4;
    return val;
  }

  readFloat64(): number {
    const val = this.buffer.readDoubleLE(this.pos);
    this.pos += 8;
    return val;
  }

  readUint64(): number {
    const low = this.buffer.readUInt32LE(this.pos);
    const high = this.buffer.readUInt32LE(this.pos + 4);
    this.pos += 8;
    return low + high * 4294967296;
  }

  readString(): string {
    const len = this.readUint64();
    if (len === 0) return "";
    const str = this.buffer.toString("utf8", this.pos, this.pos + len);
    this.pos += len;
    return str;
  }

  skipValue(type: number): void {
    if (type === 0 || type === 1 || type === 7) { // 8-bit
      this.pos += 1;
    } else if (type === 2 || type === 3) { // 16-bit
      this.pos += 2;
    } else if (type === 4 || type === 5 || type === 6) { // 32-bit
      this.pos += 4;
    } else if (type === 10 || type === 11 || type === 12) { // 64-bit
      this.pos += 8;
    } else if (type === 8) { // string
      const len = this.readUint64();
      this.pos += len;
    } else if (type === 9) { // array
      const itemType = this.readUint32();
      const count = this.readUint64();
      for (let i = 0; i < count; i++) {
        this.skipValue(itemType);
      }
    }
  }

  parseHeader(): void {
    const magic = this.readUint32();
    if (magic !== GGUF_MAGIC) {
      throw new Error(`Invalid GGUF magic: 0x${magic.toString(16)}, expected 0x${GGUF_MAGIC.toString(16)}`);
    }
    this.version = this.readUint32();
    this.tensorCount = this.readUint64();
    this.metadataCount = this.readUint64();
  }

  parseMetadata(): void {
    for (let i = 0; i < this.metadataCount; i++) {
      const key = this.readString();
      const valType = this.readUint32();

      if (valType === 0 || valType === 1 || valType === 2 || valType === 3 || valType === 4 || valType === 5) {
        if (valType === 0) this.numMeta.set(key, this.readUint8());
        else if (valType === 1) this.numMeta.set(key, this.readInt8());
        else if (valType === 2) this.numMeta.set(key, this.readUint16());
        else if (valType === 3) this.numMeta.set(key, this.readInt16());
        else if (valType === 4) this.numMeta.set(key, this.readUint32());
        else if (valType === 5) this.numMeta.set(key, this.readInt32());
      } else if (valType === 6) {
        this.numMeta.set(key, this.readFloat32());
      } else if (valType === 7) {
        this.boolMeta.set(key, this.readUint8() !== 0);
      } else if (valType === 8) {
        this.stringMeta.set(key, this.readString());
      } else if (valType === 10 || valType === 11) {
        this.numMeta.set(key, this.readUint64());
      } else if (valType === 12) {
        this.numMeta.set(key, this.readFloat64());
      } else if (valType === 9) {
        // array
        const itemType = this.readUint32();
        const count = this.readUint64();
        // Skip array contents to avoid allocating 150k strings in metadata
        for (let j = 0; j < count; j++) {
          this.skipValue(itemType);
        }
      } else {
        this.skipValue(valType);
      }
    }

    if (this.numMeta.has("general.alignment")) {
      this.alignment = this.numMeta.get("general.alignment")!;
    }
  }

  parseTensorInfos(): void {
    for (let i = 0; i < this.tensorCount; i++) {
      const name = this.readString();
      const nDims = this.readUint32();
      const shape: number[] = [];
      for (let d = 0; d < nDims; d++) {
        shape.push(this.readUint64());
      }
      // GGUF stores dimensions reversed (column-major / innermost first)
      shape.reverse();

      const type = this.readUint32();
      const offset = this.readUint64();

      const info: GGUFTensorInfo = {
        name,
        shape,
        type,
        offset
      };

      this.tensorList.push(info);
      this.tensorMap.set(name, info);
    }

    // Align to general.alignment
    const remainder = this.pos % this.alignment;
    if (remainder !== 0) {
      this.pos += this.alignment - remainder;
    }
    this.tensorDataOffset = this.pos;
  }

  getModelConfig(): ModelConfig {
    const arch: string = this.stringMeta.has("general.architecture") ? this.stringMeta.get("general.architecture")! : "qwen2";
    const hiddenDim: number = this.numMeta.has(`${arch}.embedding_length`) ? this.numMeta.get(`${arch}.embedding_length`)! : 896;
    const numLayers: number = this.numMeta.has(`${arch}.block_count`) ? this.numMeta.get(`${arch}.block_count`)! : 24;
    const numHeads: number = this.numMeta.has(`${arch}.attention.head_count`) ? this.numMeta.get(`${arch}.attention.head_count`)! : 14;
    const numKVHeads: number = this.numMeta.has(`${arch}.attention.head_count_kv`) ? this.numMeta.get(`${arch}.attention.head_count_kv`)! : 2;
    const intermediateDim: number = this.numMeta.has(`${arch}.feed_forward_length`) ? this.numMeta.get(`${arch}.feed_forward_length`)! : 4864;
    const maxSeqLen: number = this.numMeta.has(`${arch}.context_length`) ? this.numMeta.get(`${arch}.context_length`)! : 32768;
    const normEps: number = this.numMeta.has(`${arch}.attention.layer_norm_rms_epsilon`) ? this.numMeta.get(`${arch}.attention.layer_norm_rms_epsilon`)! : 1e-6;
    const ropeBase: number = this.numMeta.has(`${arch}.rope.freq_base`) ? this.numMeta.get(`${arch}.rope.freq_base`)! : 1000000.0;

    let vocabSize: number = 151936;
    if (this.numMeta.has(`${arch}.vocab_size`)) {
      vocabSize = this.numMeta.get(`${arch}.vocab_size`)!;
    }

    return {
      vocabSize,
      hiddenDim,
      numLayers,
      numHeads,
      numKVHeads,
      intermediateDim,
      maxSeqLen,
      normEps,
      ropeBase
    };
  }

  loadTensor(name: string): Tensor {
    if (!this.tensorMap.has(name)) {
      throw new Error(`Tensor not found in GGUF: ${name}`);
    }
    const info = this.tensorMap.get(name)!;

    const filePos = this.tensorDataOffset + info.offset;
    let numElements = 1;
    for (let d = 0; d < info.shape.length; d++) {
      numElements *= info.shape[d];
    }

    if (info.type === 0) { // F32
      const bytesToRead = numElements * 4;
      const buf = Buffer.alloc(bytesToRead);
      fs.readSync(this.fd, buf, 0, bytesToRead, filePos);
      const data: number[] = [];
      for (let i = 0; i < numElements; i++) {
        data.push(buf.readFloatLE(i * 4));
      }
      return Tensor.fromArray(data, info.shape);
    } else if (info.type === 1) { // F16
      const bytesToRead = numElements * 2;
      const buf = Buffer.alloc(bytesToRead);
      fs.readSync(this.fd, buf, 0, bytesToRead, filePos);
      const data: number[] = [];
      for (let i = 0; i < numElements; i++) {
        data.push(f16ToF32(buf.readUInt16LE(i * 2)));
      }
      return Tensor.fromArray(data, info.shape);
    } else if (info.type === 8) { // Q8_0
      // Q8_0: 32 elements per block (34 bytes: 2-byte float16 scale + 32 int8 quants)
      const blockSize = 32;
      const numBlocks = Math.ceil(numElements / blockSize);
      const bytesToRead = numBlocks * 34;
      const buf = Buffer.alloc(bytesToRead);
      fs.readSync(this.fd, buf, 0, bytesToRead, filePos);

      const data: number[] = [];

      for (let b = 0; b < numBlocks; b++) {
        const blockOffset = b * 34;
        const scaleF16 = buf.readUInt16LE(blockOffset);
        const scale = f16ToF32(scaleF16);

        for (let i = 0; i < blockSize && data.length < numElements; i++) {
          const q = buf.readInt8(blockOffset + 2 + i);
          data.push(scale * q);
        }
      }

      return Tensor.fromArray(data, info.shape);
    } else {
      throw new Error(`Unsupported GGML tensor type: ${info.type} for tensor ${name}`);
    }
  }

  close(): void {
    fs.closeSync(this.fd);
  }
}
