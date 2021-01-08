import { createMatrix, createRandomMatrix } from "./matrix";

/* create Matrix */
test("should create a matrix with 1 rows and 2 cols", () => {
  const m = createMatrix(1, 2);
  const data = m.data();
  expect(data).toStrictEqual([[0, 1]]);
});

test("should create a matrix with 2 rows and 3 cols", () => {
  const m = createMatrix(2, 3);
  const data = m.data();
  expect(data).toStrictEqual([
    [0, 1, 2],
    [0, 1, 2],
  ]);
});

test("should create a matrix from an array", () => {
  const m = createMatrix([
    [0, 1, 2],
    [0, 1, 2],
  ]);
  const data = m.data();
  expect(data).toStrictEqual([
    [0, 1, 2],
    [0, 1, 2],
  ]);
});

/* createRandomMatrix */
test("should create a random matrix", () => {
  const m1 = createRandomMatrix(3, 2);
  expect(m1.rows).toBe(3);
  expect(m1.cols).toBe(2);

  const m2 = createRandomMatrix(3, 2);
  expect(m1).not.toStrictEqual(m2);
});

/* isSameSize */
test("should be same size", () => {
  const m1 = createRandomMatrix(3, 2);
  const m2 = createRandomMatrix(3, 2);

  expect(m1.isSameSize(m2)).toBe(true);
});

test("should not be same size", () => {
  const m1 = createRandomMatrix(3, 2);
  const m2 = createRandomMatrix(2, 3);

  expect(m1.isSameSize(m2)).toBe(false);
});

/* map */
test("should double value in a amtrix with map", () => {
  const a = createMatrix([
    [0, 1],
    [2, 4],
  ]);
  const result = a.map((x) => x * 2);
  expect(result.data()).toStrictEqual([
    [0, 2],
    [4, 8],
  ]);
});

/* zip */
test("should zip 2 matrices", () => {
  const a = createMatrix([
    [0, 1],
    [2, 4],
  ]);
  const b = createMatrix([
    [0, 1],
    [2, 4],
  ]);
  const result = a.zip(b, ([a, b]) => a + b);
  expect(result.data()).toStrictEqual([
    [0, 2],
    [4, 8],
  ]);
});

test("should zip 2 matrices and apply a callback", () => {
  const a = createMatrix([
    [0, 1],
    [2, 4],
  ]);
  const b = createMatrix([
    [0, 1],
    [2, 4],
  ]);
  const result = a.zip(b, ([a, b]) => a * b);
  expect(result.data()).toStrictEqual([
    [0, 1],
    [4, 16],
  ]);
});

/* transpose */
test("should transpose a matrix", () => {
  const m = createMatrix([
    [1, 3, 5],
    [2, 4, 6],
  ]);
  const result = m.transpose(m);
  expect(result.data()).toStrictEqual([
    [1, 2],
    [3, 4],
    [5, 6],
  ]);
});

/* add */
test("should add a 10 scalar to a matrix", () => {
  const m = createMatrix([
    [0, 1, 2],
    [0, 1, 2],
  ]);
  const result = m.add(10);
  const data = result.data();
  expect(data).toStrictEqual([
    [10, 11, 12],
    [10, 11, 12],
  ]);
});

test("should add 2 matrices element wise", () => {
  const a = createMatrix([
    [0, 1],
    [2, 4],
  ]);
  const b = createMatrix([
    [0, 1],
    [2, 4],
  ]);
  const result = a.add(b);
  expect(result.data()).toStrictEqual([
    [0, 2],
    [4, 8],
  ]);
});

/* subsstract */
test("should substract a 10 scalar to a matrix", () => {
  const m = createMatrix([
    [0, 1, 2],
    [0, 1, 2],
  ]);
  const result = m.subtract(10);
  const data = result.data();
  expect(data).toStrictEqual([
    [-10, -9, -8],
    [-10, -9, -8],
  ]);
});

test("should substract 2 matrices (2x2) element wise", () => {
  const a = createMatrix([
    [0, 1],
    [2, 4],
  ]);
  const b = createMatrix([
    [0, 1],
    [2, 4],
  ]);
  const result = a.subtract(b);
  expect(result.data()).toStrictEqual([
    [0, 0],
    [0, 0],
  ]);
});

test("should substract 2 matrices (2x3) element wise", () => {
  const a = createMatrix([
    [0, 1, 5],
    [2, 4, 5],
  ]);
  const b = createMatrix([
    [0, 1, 5],
    [2, 4, 5],
  ]);
  const result = a.subtract(b);
  expect(result.data()).toStrictEqual([
    [0, 0, 0],
    [0, 0, 0],
  ]);
});

/* multiply */
test("should multiply a 10 scalar with a matrix", () => {
  const m = createMatrix([
    [0, 1, 2],
    [0, 1, 2],
  ]);
  const result = m.multiply(10);
  const data = result.data();
  expect(data).toStrictEqual([
    [0, 10, 20],
    [0, 10, 20],
  ]);
});

test("should mutiply 2 matrices element wise", () => {
  const a = createMatrix([
    [0, 1],
    [2, 4],
  ]);
  const b = createMatrix([
    [0, 1],
    [2, 4],
  ]);
  const result = a.multiply(b);
  expect(result.data()).toStrictEqual([
    [0, 1],
    [4, 16],
  ]);
});

test("should mutiply 2 matrices element wise (3x2)", () => {
  const a = createMatrix([
    [0, 1, 2],
    [3, 4, 5],
  ]);
  const b = createMatrix([
    [10, 10, 10],
    [10, 10, 10],
  ]);
  const result = a.multiply(b);
  expect(result.data()).toStrictEqual([
    [0, 10, 20],
    [30, 40, 50],
  ]);
});

/* dot */
test("should dot mutiply 2 matrices (2x3) . (3x2)", () => {
  const a = createMatrix([
    [1, 2, 1],
    [0, 1, 1],
  ]);
  const b = createMatrix([
    [2, 5],
    [6, 7],
    [1, 1],
  ]);
  const result = a.dot(b);
  expect(result.data()).toStrictEqual([
    [15, 20],
    [7, 8],
  ]);
});

test("should dot mutiply 2 matrices (1x3) . (3x1)", () => {
  const a = createMatrix([[1, 3, -5]]);
  const b = createMatrix([[4], [-2], [-1]]);
  const result = a.dot(b);
  expect(result.data()).toStrictEqual([[3]]);
});

test("should dot mutiply 2 matrices codetrain", () => {
  const a = createMatrix([
    [6, 7, 0],
    [7, 2, 6],
  ]);
  const b = createMatrix([
    [5, 3],
    [1, 1],
    [5, 1],
  ]);
  const result = a.dot(b);
  expect(result.data()).toStrictEqual([
    [37, 25],
    [67, 29],
  ]);
});

test("should not dot mutiply and throw because a (2x3) and b (2x2)", () => {
  const a = createMatrix([
    [6, 7, 0],
    [7, 2, 6],
  ]);
  const b = createMatrix([
    [5, 3],
    [1, 1],
  ]);
  expect(() => {
    a.dot(b);
  }).toThrow();
});

test("should not dot mutiply and throw because a (2x2) and b (3x2)", () => {
  const a = createMatrix([
    [6, 7],
    [7, 2],
  ]);
  const b = createMatrix([
    [5, 3],
    [1, 1],
    [5, 1],
  ]);
  expect(() => {
    a.dot(b);
  }).toThrow();
});

/* sum */
test("should sum values in a matrix", () => {
  const a = createMatrix([
    [0, 1],
    [2, 4],
  ]);
  const result = a.sum();
  expect(result).toBe(7);
});
