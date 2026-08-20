// Phase 0: Test ScriptC type system and numeric capabilities
// ==========================================================

// --- Basic types ---
const n: number = 42;
const s: string = "hello";
const b: boolean = true;
console.log("number:", n, "string:", s, "boolean:", b);

// --- Number precision (IEEE 754 double) ---
console.log("\n--- Number precision ---");
console.log("MAX_SAFE_INTEGER:", Number.MAX_SAFE_INTEGER);
console.log("MIN_SAFE_INTEGER:", Number.MIN_SAFE_INTEGER);
console.log("EPSILON:", Number.EPSILON);
console.log("MAX_VALUE:", Number.MAX_VALUE);
console.log("MIN_VALUE:", Number.MIN_VALUE);

// --- Float precision tests ---
console.log("\n--- Float precision ---");
console.log("0.1 + 0.2 =", 0.1 + 0.2);
console.log("0.1 + 0.2 === 0.3?", 0.1 + 0.2 === 0.3);
console.log("Math.abs(0.1 + 0.2 - 0.3) < 1e-15?", Math.abs(0.1 + 0.2 - 0.3) < 1e-15);

// --- Numeric conversions ---
console.log("\n--- Numeric conversions ---");
console.log("parseInt('42'):", parseInt("42"));
console.log("parseFloat('3.14'):", parseFloat("3.14"));
console.log("Number('123'):", Number("123"));
console.log("Math.floor(3.7):", Math.floor(3.7));
console.log("Math.ceil(3.2):", Math.ceil(3.2));
console.log("Math.round(3.5):", Math.round(3.5));
console.log("Math.trunc(3.7):", Math.trunc(3.7));
console.log("Math.trunc(-3.7):", Math.trunc(-3.7));

// --- Bitwise (integers via f64) ---
console.log("\n--- Bitwise operations ---");
console.log("5 | 3 =", 5 | 3);
console.log("5 & 3 =", 5 & 3);
console.log("5 ^ 3 =", 5 ^ 3);
console.log("~5 =", ~5);
console.log("5 << 1 =", 5 << 1);
console.log("20 >> 2 =", 20 >> 2);

// --- Special values ---
console.log("\n--- Special values ---");
console.log("Infinity:", Infinity);
console.log("-Infinity:", -Infinity);
console.log("NaN:", NaN);
console.log("isNaN(NaN):", isNaN(NaN));
console.log("isFinite(Infinity):", isFinite(Infinity));
console.log("isFinite(42):", isFinite(42));

// --- Template literals and string operations ---
console.log("\n--- String operations ---");
const name: string = "LLM";
console.log(`Hello, ${name}!`);
console.log("length:", name.length);
console.log("charAt(0):", name.charAt(0));
console.log("'abc'.indexOf('b'):", "abc".indexOf("b"));
console.log("'hello world'.split(' '):", "hello world".split(" "));

console.log("\n✅ types.ts passed");
