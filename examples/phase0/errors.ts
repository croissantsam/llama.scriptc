// Phase 0: Test ScriptC error handling
// =====================================

// --- try/catch with user throw ---
console.log("--- try/catch ---");
function divide(a: number, b: number): number {
  if (b === 0) {
    throw new Error("Division by zero");
  }
  return a / b;
}

try {
  console.log("10/3 =", divide(10, 3));
  console.log("10/0 =", divide(10, 0));
} catch (e: unknown) {
  if (e instanceof Error) {
    console.log("Caught error:", e.message);
  }
}

// --- Error types ---
console.log("\n--- Error types ---");
try {
  throw new RangeError("out of range");
} catch (e: unknown) {
  if (e instanceof RangeError) {
    console.log("Caught RangeError:", e.message);
  }
}

try {
  throw new TypeError("wrong type");
} catch (e: unknown) {
  if (e instanceof TypeError) {
    console.log("Caught TypeError:", e.message);
  }
}

// --- Nested try/catch ---
console.log("\n--- Nested try/catch ---");
function risky(level: number): number {
  if (level <= 0) {
    throw new Error(`fail at level ${level}`);
  }
  try {
    return risky(level - 1) + 1;
  } catch (e: unknown) {
    if (e instanceof Error) {
      console.log(`Caught at level ${level}: ${e.message}`);
    }
    return -1;
  }
}
console.log("result:", risky(3));

// --- Validation pattern (for tensor ops) ---
console.log("\n--- Validation pattern ---");
function assertShape(shape: number[]): void {
  for (let i: number = 0; i < shape.length; i++) {
    if (shape[i] <= 0) {
      throw new Error(`Invalid dimension at index ${i}: ${shape[i]}`);
    }
  }
}

try {
  assertShape([2, 3, 4]);
  console.log("Valid shape: [2, 3, 4]");
} catch (e: unknown) {
  if (e instanceof Error) console.log("Error:", e.message);
}

try {
  assertShape([2, 0, 4]);
  console.log("Should not reach here");
} catch (e: unknown) {
  if (e instanceof Error) console.log("Caught:", e.message);
}

try {
  assertShape([2, -1, 4]);
  console.log("Should not reach here");
} catch (e: unknown) {
  if (e instanceof Error) console.log("Caught:", e.message);
}

// --- Dimension mismatch pattern ---
console.log("\n--- Dimension mismatch ---");
function checkMatmulDims(aShape: number[], bShape: number[]): void {
  if (aShape.length !== 2 || bShape.length !== 2) {
    throw new Error("matmul requires 2D tensors");
  }
  if (aShape[1] !== bShape[0]) {
    throw new Error(`matmul dimension mismatch: (${aShape[0]},${aShape[1]}) @ (${bShape[0]},${bShape[1]})`);
  }
}

try {
  checkMatmulDims([2, 3], [3, 4]);
  console.log("Dims OK: (2,3) @ (3,4)");
} catch (e: unknown) {
  if (e instanceof Error) console.log("Error:", e.message);
}

try {
  checkMatmulDims([2, 3], [4, 5]);
  console.log("Should not reach here");
} catch (e: unknown) {
  if (e instanceof Error) console.log("Caught:", e.message);
}

console.log("\n✅ errors.ts passed");
