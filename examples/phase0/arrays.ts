// Phase 0: Test ScriptC array capabilities
// ==========================================

// --- Basic array operations ---
console.log("--- Basic array operations ---");
const arr: number[] = [1, 2, 3, 4, 5];
console.log("arr:", arr);
console.log("arr.length:", arr.length);
console.log("arr[0]:", arr[0]);
console.log("arr[4]:", arr[4]);

// --- Push / pop ---
console.log("\n--- Push / Pop ---");
const dynamic: number[] = [];
dynamic.push(10);
dynamic.push(20);
dynamic.push(30);
console.log("after push:", dynamic);
const popped: number = dynamic.pop();
console.log("popped:", popped, "remaining:", dynamic);

// --- Slice ---
console.log("\n--- Slice ---");
const original: number[] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
console.log("slice(2, 5):", original.slice(2, 5));
console.log("slice(5):", original.slice(5));
console.log("slice(-3):", original.slice(-3));

// --- Map / Filter / Reduce ---
console.log("\n--- Map / Filter / Reduce ---");
const doubled: number[] = arr.map((x: number): number => x * 2);
console.log("map x*2:", doubled);

const evens: number[] = arr.filter((x: number): boolean => x % 2 === 0);
console.log("filter even:", evens);

const total: number = arr.reduce((acc: number, x: number): number => acc + x, 0);
console.log("reduce sum:", total);

// --- indexOf / includes ---
console.log("\n--- indexOf / includes ---");
console.log("indexOf(3):", arr.indexOf(3));
console.log("indexOf(99):", arr.indexOf(99));
console.log("includes(3):", arr.includes(3));
console.log("includes(99):", arr.includes(99));

// --- Nested arrays (2D) ---
console.log("\n--- 2D arrays ---");
const matrix: number[][] = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
console.log("matrix[1][2]:", matrix[1][2]);
console.log("rows:", matrix.length, "cols:", matrix[0].length);

// --- Fill pattern ---
console.log("\n--- Fill pattern ---");
const filled: number[] = [];
for (let i: number = 0; i < 10; i++) {
  filled.push(i * i);
}
console.log("squares:", filled);

// --- Sort ---
console.log("\n--- Sort ---");
const unsorted: number[] = [3, 1, 4, 1, 5, 9, 2, 6];
const sorted: number[] = unsorted.slice().sort((a: number, b: number): number => a - b);
console.log("sorted:", sorted);

// --- Large array ---
console.log("\n--- Large array ---");
const big: number[] = [];
for (let i: number = 0; i < 100000; i++) {
  big.push(i);
}
console.log("big array length:", big.length);
console.log("big[99999]:", big[99999]);

// --- Bounds checking ---
// ScriptC: out-of-bounds = trap, so we test with safe access
console.log("\n--- Bounds safety ---");
const safe: number[] = [10, 20, 30];
if (safe.length > 2) {
  console.log("safe[2]:", safe[2]);
}
console.log("Safe bounds checking works");

console.log("\n✅ arrays.ts passed");
