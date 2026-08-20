// Linear Transformation Layer (y = x * W^T + bias)
// ==================================================

import { Tensor } from "../tensor/tensor";
import { matmul } from "../math/matmul";
import { add } from "../math/elementwise";

export class Linear {
  inFeatures: number;
  outFeatures: number;
  weight: Tensor;      // shape [outFeatures, inFeatures]
  bias?: Tensor;       // shape [outFeatures] (optional)

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
    const weightT: Tensor = this.weight.transpose();

    let out: Tensor;

    // Case 1: 1D input [inFeatures] -> [outFeatures]
    if (ndim === 1) {
      if (x.shape[0] !== this.inFeatures) {
        throw new Error(`Linear input dimension mismatch: expected ${this.inFeatures}, got ${x.shape[0]}`);
      }
      out = matmul(x, weightT);
      if (this.bias) {
        out = add(out, this.bias);
      }
      return out;
    }
    // Case 2: 2D input [seqLen, inFeatures] -> [seqLen, outFeatures]
    else if (ndim === 2) {
      // Input [seqLen, inFeatures] @ W^T [inFeatures, outFeatures] -> [seqLen, outFeatures]
      const seqLen: number = x.shape[0];
      const inFeat: number = this.inFeatures;
      const outFeat: number = this.outFeatures;
      const outData: number[] = [];
      const xData: number[] = x.data;
      const wData: number[] = this.weight.data;
      const hasBias: boolean = this.bias !== undefined;
      const bData: number[] = hasBias ? this.bias!.data : [];

      for (let i: number = 0; i < seqLen; i++) {
        const xOffset: number = i * inFeat;
        for (let j: number = 0; j < outFeat; j++) {
          let sumVal: number = 0.0;
          const wOffset: number = j * inFeat;
          for (let k: number = 0; k < inFeat; k++) {
            sumVal = sumVal + xData[xOffset + k] * wData[wOffset + k];
          }
          if (hasBias) {
            sumVal = sumVal + bData[j];
          }
          outData.push(sumVal);
        }
      }
      return Tensor.fromArray(outData, [seqLen, outFeat]);
    }
    // Case 3: 3D input [batch, seqLen, inFeatures] -> [batch, seqLen, outFeatures]
    else if (ndim === 3) {
      const batch: number = x.shape[0];
      const seqLen: number = x.shape[1];
      if (x.shape[2] !== this.inFeatures) {
        throw new Error(`Linear input feature mismatch: expected ${this.inFeatures}, got ${x.shape[2]}`);
      }
      // Reshape to 2D [batch * seqLen, inFeatures], matmul, then reshape back to [batch, seqLen, outFeatures]
      const flat2D: Tensor = x.clone().view([batch * seqLen, this.inFeatures]);
      const res2D: Tensor = matmul(flat2D, weightT);
      let out: Tensor = res2D.view([batch, seqLen, this.outFeatures]);

      // Add optional bias
      if (this.bias) {
        out = add(out, this.bias);
      }
      return out;
    }
    else {
      throw new Error(`Unsupported input dimension for Linear: ${x.ndim()}D`);
    }
  }
}
