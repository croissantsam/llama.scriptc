// Phase 0: Test ScriptC performance measurement
// ===============================================

// --- Date.now() for timing ---
console.log("--- Performance measurement ---");

function benchMatmulNaive(n: number): number {
  // Create n×n matrices
  const a: number[][] = [];
  const b: number[][] = [];
  for (let i: number = 0; i < n; i++) {
    const rowA: number[] = [];
    const rowB: number[] = [];
    for (let j: number = 0; j < n; j++) {
      rowA.push(i * n + j);
      rowB.push(j * n + i);
    }
    a.push(rowA);
    b.push(rowB);
  }

  // Multiply
  const start: number = Date.now();
  const c: number[][] = [];
  for (let i: number = 0; i < n; i++) {
    const row: number[] = [];
    for (let j: number = 0; j < n; j++) {
      let sum: number = 0;
      for (let k: number = 0; k < n; k++) {
        sum += a[i][k] * b[k][j];
      }
      row.push(sum);
    }
    c.push(row);
  }
  const elapsed: number = Date.now() - start;
  return elapsed;
}

// Warmup
benchMatmulNaive(32);

// Benchmark different sizes
const sizes: number[] = [32, 64, 128, 256];
for (let si: number = 0; si < sizes.length; si++) {
  const n: number = sizes[si];
  const ms: number = benchMatmulNaive(n);
  const flops: number = 2 * n * n * n;
  const gflops: number = ms > 0 ? flops / (ms / 1000) / 1e9 : 0;
  console.log(`matmul ${n}x${n}: ${ms}ms (${gflops.toFixed(2)} GFLOPS)`);
}

// --- Loop performance ---
console.log("\n--- Loop performance ---");
const loopSize: number = 10000000;
const startLoop: number = Date.now();
let acc: number = 0;
for (let i: number = 0; i < loopSize; i++) {
  acc += i;
}
const loopTime: number = Date.now() - startLoop;
console.log(`${loopSize} iterations: ${loopTime}ms, result: ${acc}`);
console.log(`iterations/ms: ${Math.round(loopSize / Math.max(loopTime, 1))}`);

// --- Array fill performance ---
console.log("\n--- Array fill performance ---");
const fillSize: number = 1000000;
const startFill: number = Date.now();
const fillArr: number[] = [];
for (let i: number = 0; i < fillSize; i++) {
  fillArr.push(0);
}
const fillTime: number = Date.now() - startFill;
console.log(`Fill ${fillSize} elements: ${fillTime}ms`);

console.log("\n✅ perf.ts passed");
