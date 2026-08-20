// Phase 0: Test ScriptC file I/O
// ===============================

import * as fs from "fs";
import * as path from "path";

// --- Write text file ---
console.log("--- Write text file ---");
const tmpDir: string = path.join(path.dirname(process.argv[1]), "tmp_test");
// Create temp directory
if (!fs.existsSync(tmpDir)) {
  fs.mkdirSync(tmpDir, { recursive: true });
}

const textFile: string = path.join(tmpDir, "test.txt");
fs.writeFileSync(textFile, "Hello from ScriptC!\nLine 2\nLine 3\n");
console.log("Wrote:", textFile);

// --- Read text file ---
console.log("\n--- Read text file ---");
const content: string = fs.readFileSync(textFile, "utf-8");
console.log("Content:", content);
console.log("Lines:", content.split("\n").length);

// --- Write binary data ---
console.log("\n--- Write binary data ---");
const binFile: string = path.join(tmpDir, "test.bin");
const rawBytes: number[] = [0x48, 0x65, 0x6c, 0x6c, 0x6f]; // "Hello"
const buffer: Buffer = Buffer.from(rawBytes);
fs.writeFileSync(binFile, buffer);
console.log("Wrote binary:", buffer.length, "bytes");

// --- Read binary data ---
console.log("\n--- Read binary data ---");
const readBuf: Buffer = fs.readFileSync(binFile);
console.log("Read binary:", readBuf.length, "bytes");
console.log("Byte values:", readBuf[0], readBuf[1], readBuf[2], readBuf[3], readBuf[4]);

// --- JSON model format test ---
console.log("\n--- JSON serialization test ---");
const jsonFile: string = path.join(tmpDir, "model.json");
const modelData: string = JSON.stringify({
  config: { vocab_size: 1000, hidden_size: 128 },
  weights: [0.1, 0.2, 0.3, -0.4]
});
fs.writeFileSync(jsonFile, modelData);
const loadedJson: string = fs.readFileSync(jsonFile, "utf-8");
console.log("JSON loaded length:", loadedJson.length);
fs.unlinkSync(jsonFile);

// --- File stats ---
console.log("\n--- File stats ---");
const stats: fs.Stats = fs.statSync(binFile);
console.log("Size:", stats.size, "bytes");
console.log("Is file:", stats.isFile());

// --- Cleanup ---
fs.unlinkSync(textFile);
fs.unlinkSync(binFile);
fs.rmdirSync(tmpDir);
console.log("\nCleaned up temp files");

console.log("\n✅ files.ts passed");
