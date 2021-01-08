/* eslint-disable @typescript-eslint/consistent-type-assertions */
import * as R from "ramda";

export type MatrixData = number[][];

export interface IMatrix {
  data: () => MatrixData;
  readonly rows: number;
  readonly cols: number;
  isSameSize: (b: IMatrix) => boolean;
  transpose: () => IMatrix;
  print: () => void;
  // reduce operation
  // reduce: <T>(callback: (acc: T, elem: number) => T, start: T) => T;
  sum: () => number;
  // map operation
  map: (callback: (x: number) => number) => IMatrix;
  zip: (
    b: MatrixData | IMatrix,
    callback: (input: [a: number, b: number]) => number
  ) => IMatrix;
  randomize: () => IMatrix;
  add: (b: number | MatrixData | IMatrix) => IMatrix;
  subtract: (b: number | MatrixData | IMatrix) => IMatrix;
  multiply: (b: number | MatrixData | IMatrix) => IMatrix;
  dot: (b: MatrixData | IMatrix) => IMatrix;
}

export const isMatrixData = Array.isArray;
export const getMatrixData = (data: MatrixData | IMatrix) =>
  isMatrixData(data) ? data : data.data();

export const getMatrix = (data: MatrixData | IMatrix) =>
  isMatrixData(data) ? createMatrix(data) : data;

export const checkSize = (data: MatrixData) =>
  R.all((row: number[]) => row.length === data[0].length)(data);

const _isSameSize = (a: IMatrix, b: IMatrix) =>
  a.cols === b.cols && a.rows === b.rows;
const isSameSize = R.curry(_isSameSize);

export const transpose = (data: MatrixData | IMatrix) =>
  R.pipe(
    <(x: MatrixData) => MatrixData>R.transpose,
    createMatrix
  )(getMatrixData(data));

const _reduce = <T>(
  data: MatrixData | IMatrix,
  callback: (acc: T, elem: number) => T,
  start: T
): T =>
  R.pipe(
    <(data: MatrixData) => number[]>R.flatten,
    R.reduce(callback, start)
  )(getMatrixData(data));
export const reduce = R.curry(_reduce);

const _map = (data: MatrixData | IMatrix, callback: (x: number) => number) =>
  R.pipe(
    R.map((row) => R.map(callback, row)),
    createMatrix // output a new Matrix
  )(getMatrixData(data)); // if not a number[][] then get number[][] from data methods of IMatrix
export const map = R.curry(_map);

const _zip = (
  input_a: MatrixData | IMatrix,
  input_b: MatrixData | IMatrix,
  callback: (input: [a: number, b: number]) => number
) => {
  const a = getMatrixData(input_a);
  const b = getMatrixData(input_b);

  if (!isSameSize(getMatrix(input_a), getMatrix(input_b)))
    throw new Error("input_a and input_b need to be same size");

  return R.pipe(
    R.map<[number[], number[]], number[]>(([a_row, b_row]) =>
      R.map(callback, R.zip(a_row, b_row))
    ),
    createMatrix
  )(R.zip(a, b));
};
export const zip = R.curry(_zip);

const _randomize = (rows: number, cols: number) =>
  createRandomMatrix(rows, cols);
const randomize = R.curry(_randomize);

const _add = (data: MatrixData | IMatrix, b: number | MatrixData | IMatrix) => {
  if (typeof b == "number") {
    return map(data, (a) => a + b);
  }
  return zip(data, b, ([a, b]) => a + b);
};
export const add = R.curry(_add);

const _subtract = (
  data: MatrixData | IMatrix,
  b: number | MatrixData | IMatrix
) => {
  if (typeof b == "number") {
    return map(data, (a) => a - b);
  }
  return zip(data, b, ([a, b]) => a - b);
};
export const subtract = R.curry(_subtract);

const _multiply = (
  data: MatrixData | IMatrix,
  b: number | MatrixData | IMatrix
) => {
  if (typeof b == "number") {
    return map(data, (a) => a * b);
  }
  return zip(data, b, ([a, b]) => a * b);
};
export const multiply = R.curry(_multiply);

// http://matrixmultiplication.xyz/
// a . b = a * bT (T = transposed)
const _dot = (input_a: MatrixData | IMatrix, input_b: MatrixData | IMatrix) => {
  const a = getMatrix(input_a);
  const b = getMatrix(input_b);
  const a_data = getMatrixData(input_a);
  const b_data = getMatrixData(input_b);

  if (a.cols !== b.rows)
    throw new Error(`a columns need to match b rows (${a.cols} != ${b.rows})`);

  const result = createMatrix(a.rows, b.cols);
  let result_data = result.data();
  for (let i = 0; i < result.rows; i++) {
    for (let j = 0; j < result.cols; j++) {
      let sum = 0;
      for (let k = 0; k < a.cols; k++) {
        sum += a_data[i][k] * b_data[k][j];
      }
      result_data[i][j] = sum;
    }
  }

  return createMatrix(result_data);
};
export const dot = R.curry(_dot);

const _sum = (data: MatrixData | IMatrix) =>
  R.pipe(<(data: MatrixData) => number[]>R.flatten, R.sum)(getMatrixData(data));
export const sum = R.curry(_sum);

// Function Factory to create a Matrix
export function createMatrix(
  this: any,
  rows: number | MatrixData,
  cols?: number,
  callback = ([row, col]: [number, number]) => col
): IMatrix {
  // data
  const data: MatrixData = isMatrixData(rows)
    ? R.clone(rows)
    : R.map((row: number) =>
        R.pipe(
          R.map((col: number) => <[number, number]>[row, col]),
          R.map(callback)
        )(R.range(0, <number>cols))
      )(R.range(0, rows));

  if (!checkSize(data))
    throw new Error("The input array does not have uniform size.");

  // methods
  return {
    data: () => R.clone(data),
    get rows() {
      return data.length;
    },
    get cols() {
      return R.head(data)?.length || -1;
    },
    isSameSize: (b: IMatrix) => isSameSize(createMatrix(data), b),
    transpose: () => transpose(data),
    print: () => console.table(data),
    // reduce: reduce(data),
    sum: () => sum(data),
    map: map(data),
    zip: zip(data),
    randomize: () => randomize(data.length, data[0].length),
    add: add(data),
    subtract: subtract(data),
    multiply: multiply(data),
    dot: dot(data),
  };
}
export function createRandomMatrix(rows: number, cols?: number) {
  return createMatrix(rows, cols, () => Math.random() * 2 - 1);
}

export default createMatrix;
