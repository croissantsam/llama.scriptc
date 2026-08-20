// Phase 0: Test ScriptC math capabilities
// ========================================

console.log("--- Basic Math ---");
console.log("Math.sqrt(2):", Math.sqrt(2));
console.log("Math.sqrt(4):", Math.sqrt(4));
console.log("Math.exp(0):", Math.exp(0));
console.log("Math.exp(1):", Math.exp(1));
console.log("Math.log(1):", Math.log(1));
console.log("Math.log(Math.E):", Math.log(Math.E));
console.log("Math.abs(-5):", Math.abs(-5));
console.log("Math.max(3, 7):", Math.max(3, 7));
console.log("Math.min(3, 7):", Math.min(3, 7));
console.log("Math.PI:", Math.PI);
console.log("Math.E:", Math.E);

// --- Trigonometric ---
console.log("\n--- Trig ---");
console.log("Math.sin(0):", Math.sin(0));
console.log("Math.cos(0):", Math.cos(0));
console.log("Math.sin(Math.PI/2):", Math.sin(Math.PI / 2));
console.log("Math.cos(Math.PI):", Math.cos(Math.PI));

// --- Power / Log ---
console.log("\n--- Power / Log ---");
console.log("Math.pow(2, 10):", Math.pow(2, 10));
console.log("2 ** 10:", 2 ** 10);
console.log("Math.log2(1024):", Math.log2(1024));
console.log("Math.log10(1000):", Math.log10(1000));

// --- Softmax components ---
console.log("\n--- Softmax building blocks ---");
const logits: number[] = [2.0, 1.0, 0.1];

// max for numerical stability
let maxVal: number = -Infinity;
for (let i: number = 0; i < logits.length; i++) {
  if (logits[i] > maxVal) maxVal = logits[i];
}
console.log("max:", maxVal);

// exp(x - max)
const exps: number[] = [];
for (let i: number = 0; i < logits.length; i++) {
  exps.push(Math.exp(logits[i] - maxVal));
}
console.log("exp(x-max):", exps);

// sum
let expSum: number = 0;
for (let i: number = 0; i < exps.length; i++) {
  expSum += exps[i];
}
console.log("sum:", expSum);

// softmax
const softmax: number[] = [];
for (let i: number = 0; i < exps.length; i++) {
  softmax.push(exps[i] / expSum);
}
console.log("softmax:", softmax);

// verify sum = 1
let checkSum: number = 0;
for (let i: number = 0; i < softmax.length; i++) {
  checkSum += softmax[i];
}
console.log("softmax sum:", checkSum, "≈ 1?", Math.abs(checkSum - 1.0) < 1e-10);

// --- rsqrt (1/sqrt) ---
console.log("\n--- rsqrt ---");
function rsqrt(x: number): number {
  return 1.0 / Math.sqrt(x);
}
console.log("rsqrt(4):", rsqrt(4));  // 0.5
console.log("rsqrt(64):", rsqrt(64));  // 0.125

// --- RMSNorm components ---
console.log("\n--- RMSNorm building blocks ---");
const x: number[] = [1.0, 2.0, 3.0, 4.0];
const eps: number = 1e-6;

// mean(x^2)
let sumSq: number = 0;
for (let i: number = 0; i < x.length; i++) {
  sumSq += x[i] * x[i];
}
const meanSq: number = sumSq / x.length;
console.log("mean(x^2):", meanSq);

// rms = sqrt(mean(x^2) + eps)
const rms: number = Math.sqrt(meanSq + eps);
console.log("rms:", rms);

// normalize
const normalized: number[] = [];
for (let i: number = 0; i < x.length; i++) {
  normalized.push(x[i] / rms);
}
console.log("normalized:", normalized);

console.log("\n✅ math.ts passed");
