// Master Test Runner for all LLM Tensor & Math Tests
// ====================================================

import { runShapeTests } from "./tensor/test_shape";
import { runTensorTests } from "./tensor/test_tensor";
import { runElementwiseTests } from "./tensor/test_elementwise";
import { runReductionTests } from "./tensor/test_reduction";
import { runMatmulTests } from "./tensor/test_matmul";
import { runSoftmaxTests } from "./tensor/test_softmax";
import { runNNTests } from "./nn/test_nn";
import { runRoPETests } from "./transformer/test_rope";
import { runAttentionTests } from "./transformer/test_attention";
import { runBlockTests } from "./transformer/test_block";
import { runKVCacheTests } from "./transformer/test_kv_cache";
import { runTokenizerTests } from "./tokenizer/test_tokenizer";
import { runModelTests } from "./model/test_model";
import { runSamplerTests } from "./sampling/test_sampler";
import { runGeneratorTests } from "./runtime/test_generator";

console.log("=================================================");
console.log("   LLM INFERENCE ENGINE (ScriptC) TEST SUITE     ");
console.log("=================================================");

const startTime: number = Date.now();

const results: boolean[] = [
  runShapeTests(),
  runTensorTests(),
  runElementwiseTests(),
  runReductionTests(),
  runMatmulTests(),
  runSoftmaxTests(),
  runNNTests(),
  runRoPETests(),
  runAttentionTests(),
  runBlockTests(),
  runKVCacheTests(),
  runTokenizerTests(),
  runModelTests(),
  runSamplerTests(),
  runGeneratorTests()
];

const totalTime: number = Date.now() - startTime;

let allPassed: boolean = true;
for (let i: number = 0; i < results.length; i++) {
  if (!results[i]) {
    allPassed = false;
    break;
  }
}

console.log("\n=================================================");
if (allPassed) {
  console.log(`🎉 ALL TEST SUITES PASSED in ${totalTime}ms!`);
} else {
  console.log("❌ SOME SUITES FAILED!");
  process.exit(1);
}
console.log("=================================================");
