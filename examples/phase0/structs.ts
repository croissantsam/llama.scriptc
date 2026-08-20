// Phase 0: Test ScriptC structs, interfaces, classes
// ===================================================

// --- Interface / Record ---
console.log("--- Interface / Record ---");
interface Point {
  x: number;
  y: number;
}

const p: Point = { x: 3, y: 4 };
console.log("point:", p.x, p.y);

function distance(a: Point, b: Point): number {
  const dx: number = a.x - b.x;
  const dy: number = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

const origin: Point = { x: 0, y: 0 };
console.log("distance:", distance(origin, p));

// --- Class ---
console.log("\n--- Class ---");
class Vector {
  data: number[];
  size: number;

  constructor(data: number[]) {
    this.data = data;
    this.size = data.length;
  }

  get(i: number): number {
    return this.data[i];
  }

  set(i: number, value: number): void {
    this.data[i] = value;
  }

  dot(other: Vector): number {
    let sum: number = 0;
    for (let i: number = 0; i < this.size; i++) {
      sum += this.data[i] * other.data[i];
    }
    return sum;
  }

  toString(): string {
    return `Vector([${this.data.join(", ")}])`;
  }
}

const v1: Vector = new Vector([1, 2, 3]);
const v2: Vector = new Vector([4, 5, 6]);
console.log("v1:", v1.toString());
console.log("v2:", v2.toString());
console.log("dot:", v1.dot(v2));

// --- Mutation ---
console.log("\n--- Mutation ---");
v1.set(0, 10);
console.log("after set:", v1.toString());

// --- Class with methods ---
console.log("\n--- Class composition ---");

class TensorShape {
  dims: number[];

  constructor(dims: number[]) {
    this.dims = dims.slice(); // defensive copy
  }

  ndim(): number {
    return this.dims.length;
  }

  totalSize(): number {
    let size: number = 1;
    for (let i: number = 0; i < this.dims.length; i++) {
      size *= this.dims[i];
    }
    return size;
  }

  equals(other: TensorShape): boolean {
    if (this.dims.length !== other.dims.length) return false;
    for (let i: number = 0; i < this.dims.length; i++) {
      if (this.dims[i] !== other.dims[i]) return false;
    }
    return true;
  }

  toString(): string {
    return `Shape(${this.dims.join("x")})`;
  }
}

const s1: TensorShape = new TensorShape([2, 3, 4]);
const s2: TensorShape = new TensorShape([2, 3, 4]);
const s3: TensorShape = new TensorShape([4, 3, 2]);
console.log("s1:", s1.toString(), "ndim:", s1.ndim(), "size:", s1.totalSize());
console.log("s1 === s2?", s1.equals(s2));
console.log("s1 === s3?", s1.equals(s3));

// --- Record shapes are exact structs in ScriptC ---
console.log("\n--- Record shapes ---");
interface Config {
  hiddenSize: number;
  numLayers: number;
  numHeads: number;
}

const config: Config = {
  hiddenSize: 128,
  numLayers: 2,
  numHeads: 4,
};
console.log("config:", config.hiddenSize, config.numLayers, config.numHeads);

console.log("\n✅ structs.ts passed");
