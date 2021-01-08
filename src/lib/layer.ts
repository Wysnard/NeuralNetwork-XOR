import * as R from "ramda";

import createMatrix, {
  IMatrix,
  createRandomMatrix,
  getMatrix,
  dot,
  add,
  map,
  MatrixData,
} from "./matrix";
import { dsigmoid, sigmoid } from "./utils";

export interface ILayer {
  print: () => void;
  predict: (input: IMatrix) => IMatrix;
  optimize: (
    output_error: IMatrix,
    input: IMatrix,
    learning_rate: number
  ) => ILayer;
}

const _predict = (
  inputDimension: number,
  activationFn: (x: number) => number,
  weights: IMatrix,
  bias: IMatrix,
  input: IMatrix
): IMatrix =>
  R.pipe<MatrixData | IMatrix, IMatrix, IMatrix, IMatrix, IMatrix, IMatrix>(
    getMatrix,
    (x) => {
      if (x.rows !== inputDimension)
        throw new Error(
          `Your input needs to match the input dimension (${x.rows} != ${inputDimension})`
        );
      else if (x.cols !== 1)
        throw new Error("Your input needs to have 1 column");
      return x;
    },
    dot(weights),
    add(bias),
    map(R.__, activationFn)
  )(input);
export const predict = R.curry(_predict);

const _optimize = (
  inputDimension: number,
  units: number,
  fn: {
    activationFn: (x: number) => number;
    dActivationFn: (x: number) => number;
  },
  weights: IMatrix,
  bias: IMatrix,
  output_error: IMatrix,
  input: IMatrix,
  learning_rate = 0.1
): ILayer => {
  // apply derivative
  const gradient: IMatrix = _predict(
    inputDimension,
    fn.dActivationFn,
    weights,
    bias,
    input
  )
    .dot(output_error)
    .multiply(learning_rate);

  const weights_deltas = gradient.dot(input.transpose());
  const new_weights = weights.add(weights_deltas);
  const new_bias = bias.add(gradient);

  return createLayer({
    inputDimension,
    units,
    fn,
    weights: new_weights,
    bias: new_bias,
  });
};
const optimize = R.curry(_optimize);

interface ILayerParam {
  inputDimension: number;
  units: number;
  fn?: {
    activationFn: (x: number) => number;
    dActivationFn: (x: number) => number;
  };
  weights?: IMatrix;
  bias?: IMatrix;
}

/**
 * Create a Layer which is used in a Neural Network
 * @param {number} inputDimension Input Matrix dimension
 * @param {number} units Number of nodes in the layer
 * @param {{activationFn: (x: number) => number, dActivationFn: (x: number) => number}} fnActivation Function with its derivative
 * @param {IMatrix} weights Weights Matrix
 * @param {IMatrix} bias Bias Matrix
 * @returns {ILayer}
 */
export function createLayer({
  inputDimension,
  units,
  fn = { activationFn: sigmoid, dActivationFn: dsigmoid },
  weights = createRandomMatrix(units, inputDimension),
  bias = createMatrix(units, 1, () => 1),
}: ILayerParam): ILayer {
  if (inputDimension !== weights.cols)
    throw new Error(
      `inputDimension needs to match weights columns (${inputDimension} != ${weights.cols})`
    );
  if (units !== weights.rows)
    throw new Error(
      `units needs to match weights columns (${units} != ${weights.rows})`
    );
  if (units !== bias.rows)
    throw new Error(
      `units needs to match bias columns (${units} != ${bias.rows})`
    );

  return {
    print: () => {
      console.log("WEIGHT MATRIX");
      console.table(weights.data());
      console.log("BIAS MATRIX");
      console.table(bias.data());
    },
    predict: predict(inputDimension, fn.activationFn, weights, bias),
    optimize: optimize(
      inputDimension,
      units,
      { activationFn: fn.activationFn, dActivationFn: fn.dActivationFn },
      weights,
      bias
    ),
  };
}
