// Tests for Tensor Shape and Strides
// ===================================

import { computeStrides, shapeToSize, validateShape, areBroadcastable, broadcastShapes, shapesEqual } from "../../src/tensor/shape";
import { assert, assertEqual, assertArrayEqual, assertThrows, runTests, TestCase } from "../../src/testing/assert";

const tests: TestCase[] = [
  {
    name: "shapeToSize calculation",
    fn: () => {
      assertEqual(shapeToSize([]), 0);
      assertEqual(shapeToSize([5]), 5);
      assertEqual(shapeToSize([2, 3]), 6);
      assertEqual(shapeToSize([2, 3, 4]), 24);
      assertEqual(shapeToSize([1, 1, 1]), 1);
    }
  },
  {
    name: "computeStrides row-major",
    fn: () => {
      assertArrayEqual(computeStrides([]), []);
      assertArrayEqual(computeStrides([5]), [1]);
      assertArrayEqual(computeStrides([2, 3]), [3, 1]);
      assertArrayEqual(computeStrides([2, 3, 4]), [12, 4, 1]);
      assertArrayEqual(computeStrides([1, 10, 5, 2]), [100, 10, 2, 1]);
    }
  },
  {
    name: "validateShape rejects invalid dims",
    fn: () => {
      validateShape([1, 2, 3]); // should not throw
      assertThrows(() => validateShape([-1, 2]), "non-negative integer");
      assertThrows(() => validateShape([2.5, 3]), "non-negative integer");
    }
  },
  {
    name: "shapesEqual comparison",
    fn: () => {
      assert(shapesEqual([2, 3], [2, 3]));
      assert(!shapesEqual([2, 3], [3, 2]));
      assert(!shapesEqual([2, 3], [2, 3, 1]));
    }
  },
  {
    name: "areBroadcastable checking",
    fn: () => {
      assert(areBroadcastable([2, 3], [2, 3]));
      assert(areBroadcastable([2, 3], [1, 3]));
      assert(areBroadcastable([2, 3], [3]));
      assert(areBroadcastable([2, 1, 4], [3, 4]));
      assert(!areBroadcastable([2, 3], [2, 4]));
      assert(!areBroadcastable([2, 3], [4]));
    }
  },
  {
    name: "broadcastShapes calculation",
    fn: () => {
      assertArrayEqual(broadcastShapes([2, 3], [2, 3]), [2, 3]);
      assertArrayEqual(broadcastShapes([2, 3], [1, 3]), [2, 3]);
      assertArrayEqual(broadcastShapes([2, 1], [2, 4]), [2, 4]);
      assertArrayEqual(broadcastShapes([2, 3], [3]), [2, 3]);
      assertArrayEqual(broadcastShapes([1, 4], [3, 1]), [3, 4]);
      assertThrows(() => broadcastShapes([2, 3], [2, 4]), "Incompatible shapes");
    }
  }
];

export function runShapeTests(): boolean {
  return runTests("Tensor Shape & Strides", tests);
}

// Execute when run directly
runShapeTests();
