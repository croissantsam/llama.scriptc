// Linear Transformation Layer (y = x * W^T + bias) - Optimized
// ==================================================

import { Tensor } from "../tensor/tensor";
import { matmul } from "../math/matmul";
import { add } from "../math/elementwise";

export class Linear {
  inFeatures: number;
  outFeatures: number;
  weight: Tensor;      // shape [outFeatures, inFeatures]
  bias?: Tensor;       // shape [outFeatures] (optional)

  // Pre-allocated output buffer for 2D case
  private _outBuffer?: Tensor;

  constructor(inFeatures: number, outFeatures: number, weight?: Tensor, bias?: Tensor) {
    this.inFeatures = inFeatures;
    this.outFeatures = outFeatures;

    if (weight) {
      if (weight.shape[0] !== outFeatures || weight.shape[1] !== inFeatures) {
        throw new Error(`Linear weight shape mismatch: expected [${outFeatures}, ${inFeatures}], got [${weight.shape.join(", ")}]`);
      }
      this.weight = weight;
    } else {
      this.weight = Tensor.zeros([outFeatures, inFeatures]);
    }

    if (bias) {
      if (bias.shape.length !== 1 || bias.shape[0] !== outFeatures) {
        throw new Error(`Linear bias shape mismatch: expected [${outFeatures}], got [${bias.shape.join(", ")}]`);
      }
      this.bias = bias;
    }
  }

  forward(x: Tensor): Tensor {
    const ndim: number = x.ndim();

    // Case 1: 1D input [inFeatures] -> [outFeatures]
    if (ndim === 1) {
      if (x.shape[0] !== this.inFeatures) {
        throw new Error(`Linear input dimension mismatch: expected ${this.inFeatures}, got ${x.shape[0]}`);
      }
      const weightT: Tensor = this.weight.transpose();
      let out = matmul(x, weightT);
      if (this.bias) {
        out = add(out, this.bias);
      }
      return out;
    }
    // Case 2: 2D input [seqLen, inFeatures] -> [seqLen, outFeatures]
    else if (ndim === 2) {
      return this.forward2D(x);
    }
    // Case 3: 3D input [batch, seqLen, inFeatures] -> [batch, seqLen, outFeatures]
    else if (ndim === 3) {
      return this.forward3D(x);
    }
    else {
      throw new Error(`Unsupported input dimension for Linear: ${x.ndim()}D`);
    }
  }

  private forward2D(x: Tensor): Tensor {
    const seqLen: number = x.shape[0];
    const inFeat: number = this.inFeatures;
    const outFeat: number = this.outFeatures;
    const hasBias: boolean = this.bias !== undefined;

    // Reuse output buffer if possible
    let out: Tensor;
    if (this._outBuffer && this._outBuffer.shape[0] === seqLen && this._outBuffer.shape[1] === outFeat) {
      out = this._outBuffer;
    } else {
      out = Tensor.zeros([seqLen, outFeat]);
      this._outBuffer = out;
    }

    const xData = x.data;
    const wData = this.weight.data;
    const outData = out.data;
    const bData = hasBias ? this.bias!.data : null;

    const xStrides = x.strides;
    const wStrides = this.weight.strides;
    const outStrides = out.strides;

    const xOffset = x.offset;
    const wOffset = this.weight.offset;
    const outOffset = out.offset;

    // Optimized: iterate over output elements, compute dot product
    // This is more cache-friendly than the previous approach
    if (x.isContiguous() && this.weight.isContiguous() && out.isContiguous()) {
      // Fast path for contiguous tensors
      for (let i = 0; i < seqLen; i++) {
        const xRowOffset = xOffset + i * inFeat;
        const outRowOffset = outOffset + i * outFeat;
        
        for (let j = 0; j < outFeat; j++) {
          let sumVal = 0.0;
          const wRowOffset = wOffset + j * inFeat;
          
          // Unroll inner loop for better performance
          for (let k = 0; k < inFeat; k++) {
            sumVal += xData[xRowOffset + k] * wData[wRowOffset + k];
          }
          
          if (hasBias) {
            sumVal += bData![j];
          }
          
          outData[outRowOffset + j] = sumVal;
        }
      }
    } else {
      // General path with strides
      for (let i = 0; i < seqLen; i++) {
        for (let j = 0; j < outFeat; j++) {
          let sumVal = 0.0;
          for (let k = 0; k < inFeat; k++) {
            const xIdx = xOffset + i * xStrides[0] + k * xStrides[1];
            const wIdx = wOffset + j * wStrides[0] + k * wStrides[1];
            sumVal += xData[xIdx] * wData[wIdx];
          }
          if (hasBias) {
            sumVal += bData![j];
          }
          outData[outOffset + i * outStrides[0] + j * outStrides[1]] = sumVal;
        }
      }
    }

    return out;
  }

  private forward3D(x: Tensor): Tensor {
    const batch: number = x.shape[0];
    const seqLen: number = x.shape[1];
    const hasBias: boolean = this.bias !== undefined;

    if (x.shape[2] !== this.inFeatures) {
      throw new Error(`Linear input feature mismatch: expected ${this.inFeatures}, got ${x.shape[2]}`);
    }

    // Reshape to 2D [batch * seqLen, inFeatures], use optimized forward2D, then reshape back
    const flat2D: Tensor = x.clone().view([batch * seqLen, this.inFeatures]);
    const res2D: Tensor = this.forward2D(flat2D);
    let out: Tensor = res2D.view([batch, seqLen, this.outFeatures]);

    // Add optional bias (already fused in forward2D if we use it directly)
    // But since we're using view, we need to handle bias separately for 3D
    if (hasBias && out !== res2D) {
      out = add(out, this.bias!);
    }
    return out;
  }
}
