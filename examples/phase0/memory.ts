// Phase 0: Test ScriptC memory patterns and contiguous arrays
// ==============================================================

// --- Contiguous number[] allocation ---
console.log("--- Contiguous number[] allocation ---");
const size: number = 100000;
const memoryBlock: number[] = [];
for (let i: number = 0; i < size; i++) {
  memoryBlock.push(0.0);
}
console.log("memoryBlock length:", memoryBlock.length);
console.log("first:", memoryBlock[0], "last:", memoryBlock[size - 1]);

// --- Fast fill & read ---
console.log("\n--- Fill & Read ---");
for (let i: number = 0; i < 100; i++) {
  memoryBlock[i] = i * 1.5;
}
console.log("memoryBlock[0]:", memoryBlock[0], "memoryBlock[50]:", memoryBlock[50], "memoryBlock[99]:", memoryBlock[99]);

// --- Flat buffer to multi-dim indexing (Strides) ---
console.log("\n--- Strided indexing ---");
// 3x4 matrix in flat array: index = row * 4 + col
const rows: number = 3;
const cols: number = 4;
const flatMatrix: number[] = [];
for (let r: number = 0; r < rows; r++) {
  for (let c: number = 0; c < cols; c++) {
    flatMatrix.push(r * 10 + c);
  }
}

function get2D(data: number[], r: number, c: number, numCols: number): number {
  return data[r * numCols + c];
}

console.log("matrix (flat):", flatMatrix);
console.log("matrix[1, 2] expecting 12:", get2D(flatMatrix, 1, 2, cols));
console.log("matrix[2, 3] expecting 23:", get2D(flatMatrix, 2, 3, cols));

// --- Slice / View emulation ---
console.log("\n--- Sub-slice copy ---");
const row1: number[] = flatMatrix.slice(4, 8); // row 1 (elements 4..7)
console.log("row 1 extracted:", row1);

// --- Large scale allocation test ---
console.log("\n--- Large scale allocation (1M floats) ---");
const big1M: number[] = [];
const startAlloc: number = Date.now();
for (let i: number = 0; i < 1000000; i++) {
  big1M.push(i * 0.1);
}
const endAlloc: number = Date.now();
console.log(`1,000,000 floats allocated in ${endAlloc - startAlloc}ms`);

console.log("\n✅ memory.ts passed");
