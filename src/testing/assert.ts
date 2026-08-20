// Test assertions and runner for ScriptC LLM engine
// ==================================================

export function assert(condition: boolean, message: string = "Assertion failed"): void {
  if (!condition) {
    throw new Error(`[ASSERTION FAILED] ${message}`);
  }
}

export function assertEqual(actual: number | string | boolean, expected: number | string | boolean, message: string = ""): void {
  if (actual !== expected) {
    const detail: string = message.length > 0 ? ` (${message})` : "";
    throw new Error(`[ASSERTION FAILED] Expected ${expected}, got ${actual}${detail}`);
  }
}

export function assertArrayEqual(actual: number[], expected: number[], message: string = ""): void {
  if (actual.length !== expected.length) {
    throw new Error(`[ASSERTION FAILED] Array length mismatch: got ${actual.length}, expected ${expected.length}. ${message}`);
  }
  for (let i: number = 0; i < actual.length; i++) {
    if (actual[i] !== expected[i]) {
      throw new Error(`[ASSERTION FAILED] Array element at index ${i}: got ${actual[i]}, expected ${expected[i]}. ${message}`);
    }
  }
}

export function assertClose(actual: number, expected: number, eps: number = 1e-5, message: string = ""): void {
  const diff: number = Math.abs(actual - expected);
  if (diff > eps) {
    const detail: string = message.length > 0 ? ` (${message})` : "";
    throw new Error(`[ASSERTION FAILED] Expected ≈ ${expected}, got ${actual} (diff: ${diff} > eps: ${eps})${detail}`);
  }
}

export function assertThrows(fn: () => void, expectedMessageSubstr: string = ""): void {
  let threw: boolean = false;
  try {
    fn();
  } catch (e: unknown) {
    threw = true;
    if (expectedMessageSubstr.length > 0 && e instanceof Error) {
      if (!e.message.includes(expectedMessageSubstr)) {
        throw new Error(`[ASSERTION FAILED] Threw '${e.message}', expected substring '${expectedMessageSubstr}'`);
      }
    }
  }
  if (!threw) {
    throw new Error("[ASSERTION FAILED] Expected function to throw an error, but it did not.");
  }
}

export interface TestCase {
  name: string;
  fn: () => void;
}

export function runTests(suiteName: string, tests: TestCase[]): boolean {
  console.log(`\n=== Running Test Suite: ${suiteName} ===`);
  let passed: number = 0;
  let failed: number = 0;

  for (let i: number = 0; i < tests.length; i++) {
    const t: TestCase = tests[i];
    try {
      t.fn();
      console.log(`  ✓ ${t.name}`);
      passed++;
    } catch (e: unknown) {
      console.log(`  ✗ ${t.name}`);
      if (e instanceof Error) {
        console.log(`    Error: ${e.message}`);
      }
      failed++;
    }
  }

  console.log(`\nResults: ${passed} passed, ${failed} failed (${tests.length} total)`);
  if (failed > 0) {
    console.log(`❌ Some tests failed in ${suiteName}`);
    return false;
  } else {
    console.log(`✅ All tests passed in ${suiteName}`);
    return true;
  }
}
