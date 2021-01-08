/* eslint-disable @typescript-eslint/consistent-type-assertions */
import * as R from "ramda";

import { shuffle } from "./utils";
import { ILayer } from "./layer";
import { IMatrix } from "./matrix";

const _predict = (layers: ILayer[], input: IMatrix) =>
  R.reduce((acc, elem) => elem.predict(acc), input, layers);
const predict = R.curry(_predict);

const _layeredPredict = (layers: ILayer[], input: IMatrix) =>
  R.reduce<ILayer, [IMatrix, IMatrix[]]>(
    ([acc, layeredResult], elem) => {
      const result = elem.predict(acc);
      return [result, [...layeredResult, result]];
    },
    [input, [] as IMatrix[]],
    layers
  )[1];
const layeredPredict = R.curry(_layeredPredict);

const _error = (layers: ILayer[], input: IMatrix, target: IMatrix): IMatrix =>
  target.subtract(predict(layers, input));
const error = R.curry(_error);

const _errors = (
  layers: ILayer[],
  inputs: IMatrix[],
  targets: IMatrix[]
): IMatrix[] =>
  R.map<[IMatrix, IMatrix], IMatrix>(
    ([input, target]) => _error(layers, input, target),
    R.zip(inputs, targets)
  );
const errors = R.curry(_errors);

const _optimize = (
  layers: ILayer[],
  input: IMatrix,
  target: IMatrix,
  learning_rate = 0.1
) => {
  const layerResult = layeredPredict(layers, input);
  const layeredInput = R.pipe<IMatrix[], IMatrix[], IMatrix[]>(
    R.dropLast<IMatrix>(1),
    R.concat([input])
  )(layerResult);
  const output_error = error(layers, input, target);
  return R.pipe(
    R.map<[ILayer, IMatrix], ILayer>(([layer, layerInput]) =>
      layer.optimize(output_error, layerInput, learning_rate)
    ),
    createNeuralNetwork
  )(R.zip(layers, layeredInput));
};
export const optimize = R.curry(_optimize);

interface Ptrain {
  inputs: IMatrix[];
  targets: IMatrix[];
  epochs?: number;
  learning_rate?: number;
}

const _train = (
  layers: ILayer[],
  { inputs, targets, epochs = 100, learning_rate = 0.1 }: Ptrain
): INeuralNetwork =>
  R.reduce(
    (nn) =>
      R.reduce<[IMatrix, IMatrix], INeuralNetwork>(
        (nn, [input, target]) => nn.optimize(input, target, learning_rate),
        nn,
        shuffle(R.zip(inputs, targets))
      ), // nn.optimize(input, target),
    createNeuralNetwork(layers),
    R.range(0, epochs)
  );
const train = R.curry(_train);

export interface INeuralNetwork {
  print: () => void;
  error: (input: IMatrix, target: IMatrix) => IMatrix;
  errors: (inputs: IMatrix[], targets: IMatrix[]) => IMatrix[];
  predict: (input: IMatrix) => IMatrix;
  layeredPredict: (input: IMatrix) => IMatrix[];
  optimize: (
    input: IMatrix,
    target: IMatrix,
    learning_rate?: number
  ) => INeuralNetwork;
  train: (parameters: Omit<Ptrain, "layers">) => INeuralNetwork;
}

export function createNeuralNetwork(layers: ILayer[]): INeuralNetwork {
  return {
    print: () => R.forEach((layer) => layer.print(), layers),
    error: error(layers),
    errors: errors(layers),
    predict: predict(layers),
    layeredPredict: layeredPredict(layers),
    optimize: <
      (
        input: IMatrix,
        target: IMatrix,
        learning_rate?: number
      ) => INeuralNetwork
    >optimize(layers),
    train: train(layers),
  };
}
