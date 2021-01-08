import * as R from "ramda";

import { createMatrix } from "./matrix";
import { createLayer } from "./layer";
import { createNeuralNetwork } from "./nn";

/* createNeuralNetwork */
test("should create a Neural Network with a very simple layer and predict", () => {
  const nn = createNeuralNetwork([
    createLayer({
      inputDimension: 1,
      units: 1,
      fn: { activationFn: (x) => x, dActivationFn: (x) => x },
      weights: createMatrix([[1]]),
      bias: createMatrix([[0]]),
    }),
  ]);
  const input = createMatrix([[1]]);
  const result = nn.predict(input);
  expect(result.data()).toStrictEqual([[1]]);
});

test("should create a Neural Network with 2 very simple layers and predict", () => {
  const nn = createNeuralNetwork([
    createLayer({
      inputDimension: 1,
      units: 1,
      fn: { activationFn: (x) => x, dActivationFn: (x) => x },
      weights: createMatrix([[1]]),
      bias: createMatrix([[0]]),
    }),
    createLayer({
      inputDimension: 1,
      units: 1,
      fn: { activationFn: (x) => x, dActivationFn: (x) => x },
      weights: createMatrix([[1]]),
      bias: createMatrix([[0]]),
    }),
  ]);
  const input = createMatrix([[1]]);
  const result = nn.predict(input);
  expect(result.data()).toStrictEqual([[1]]);
});

test("should create a Neural Network with one very simple layers with 2 inputs and predict", () => {
  const weights = createMatrix([[1, 1]]);
  const bias = createMatrix([[1]]);
  const nn = createNeuralNetwork([
    createLayer({
      inputDimension: 2,
      units: 1,
      fn: { activationFn: (x) => x, dActivationFn: (x) => x },
      weights,
      bias,
    }),
  ]);
  const input = createMatrix([[1], [1]]);
  const result = nn.predict(input);
  expect(result.data()).toStrictEqual([[3]]);
});

/* predict */
test("should create a Neural Network and just predict", () => {
  const nn = createNeuralNetwork([
    createLayer({ inputDimension: 1, units: 1 }),
  ]);
  const input = createMatrix([[1]]);
  const _result = nn.predict(input);
});

/* layeredpredict */
test("should create a Neural Network with 2 very simple layers and layered predict", () => {
  const nn = createNeuralNetwork([
    createLayer({
      inputDimension: 1,
      units: 1,
      fn: { activationFn: (x) => x, dActivationFn: (x) => x },
      weights: createMatrix([[1]]),
      bias: createMatrix([[0]]),
    }),
    createLayer({
      inputDimension: 1,
      units: 1,
      fn: { activationFn: (x) => x, dActivationFn: (x) => x },
      weights: createMatrix([[1]]),
      bias: createMatrix([[1]]),
    }),
  ]);
  const input = createMatrix([[1]]);
  const result = nn.layeredPredict(input);
  expect(result.length).toBe(2);
  expect(R.map((layerResult) => layerResult.data(), result)).toStrictEqual([
    [[1]],
    [[2]],
  ]);
});

/* optimize */
test("should create a Neural Network and just optimize", () => {
  const nn = createNeuralNetwork([
    createLayer({ inputDimension: 1, units: 1 }),
  ]);
  const input = createMatrix([[1]]);
  const output = createMatrix([[1]]);
  const initial_error = nn.error(input, output).sum();
  const result = R.reduce(
    (nn) => nn.optimize(input, output),
    nn,
    R.range(0, 100)
  );
  const result_error = result.error(input, output).sum();
  expect(result_error - initial_error).toBeLessThanOrEqual(0);
});

test("should create a Neural Network with a very simple layer and optimize it", () => {
  const weights = createMatrix([[1, 1]]);
  const bias = createMatrix([[0]]);
  const nn = createNeuralNetwork([
    createLayer({
      inputDimension: 2,
      units: 1,
      fn: { activationFn: (x) => x, dActivationFn: (x) => x },
      weights,
      bias,
    }),
  ]);
  const input = createMatrix([[1], [1]]);
  const output = createMatrix([[2]]);
  nn.optimize(input, output);
});

test("should create a Neural Network with 2 layers and optimize it", () => {
  const nn = createNeuralNetwork([
    createLayer({
      inputDimension: 2,
      units: 2,
      fn: { activationFn: (x) => x, dActivationFn: (x) => x },
      weights: createMatrix([
        [1, 1],
        [1, 1],
      ]),
      bias: createMatrix([[0], [0]]),
    }),
    createLayer({
      inputDimension: 2,
      units: 1,
      fn: { activationFn: (x) => x, dActivationFn: (x) => x },
      weights: createMatrix([[1, 1]]),
      bias: createMatrix([[0]]),
    }),
  ]);
  const input = createMatrix([[1], [1]]);
  const output = createMatrix([[1]]);
  //   nn.print();
  nn.optimize(input, output);
  //   nn.print();
});

/* train */
test("should create a Neural Network and just train it", () => {
  const nn = createNeuralNetwork([
    createLayer({ inputDimension: 1, units: 1 }),
  ]);
  const input = createMatrix([[1]]);
  const output = createMatrix([[1]]);
  const initial_error = nn.error(input, output).sum();
  const trained_nn = nn.train({ inputs: [input], targets: [output] });
  const trained_error = trained_nn.error(input, output).sum();
  expect(trained_error - initial_error).toBeLessThanOrEqual(0);
});

test("should reproduce AND operator with Neural Network", () => {
  const nn = createNeuralNetwork([
    createLayer({ inputDimension: 2, units: 1 }),
  ]);
  const inputs = [
    createMatrix([[1], [1]]),
    createMatrix([[0], [1]]),
    createMatrix([[1], [0]]),
    createMatrix([[0], [0]]),
  ];
  const targets = [
    createMatrix([[1]]),
    createMatrix([[0]]),
    createMatrix([[0]]),
    createMatrix([[0]]),
  ];
  const trained_nn = nn.train({ inputs, targets, learning_rate: 1 });

  // Check each input/outputs
  expect(
    trained_nn.predict(createMatrix([[1], [1]])).sum()
  ).toBeGreaterThanOrEqual(0.6);
  expect(
    trained_nn.predict(createMatrix([[0], [1]])).sum()
  ).toBeLessThanOrEqual(0.4);
  expect(
    trained_nn.predict(createMatrix([[1], [0]])).sum()
  ).toBeLessThanOrEqual(0.4);
  expect(
    trained_nn.predict(createMatrix([[0], [0]])).sum()
  ).toBeLessThanOrEqual(0.4);

  // Compare initial errors against the trained nn errors
  const initial_errors = R.reduce(
    (acc, elem) => elem.sum() + acc,
    0,
    nn.errors(inputs, targets)
  );
  const trained_errors = R.reduce(
    (acc, elem) => elem.sum() + acc,
    0,
    trained_nn.errors(inputs, targets)
  );
  expect(Math.abs(initial_errors)).toBeGreaterThanOrEqual(
    Math.abs(trained_errors)
  );
});

test("should reproduce XOR operator with Neural Network", () => {
  const nn = createNeuralNetwork([
    createLayer({ inputDimension: 2, units: 2 }),
    createLayer({ inputDimension: 2, units: 1 }),
  ]);
  const inputs = [
    createMatrix([[1], [1]]),
    createMatrix([[0], [1]]),
    createMatrix([[1], [0]]),
    createMatrix([[0], [0]]),
  ];
  const targets = [
    createMatrix([[0]]),
    createMatrix([[1]]),
    createMatrix([[1]]),
    createMatrix([[0]]),
  ];
  const trained_nn = nn.train({ inputs, targets });

  // // Check each input/outputs
  // expect(
  //   trained_nn.predict(createMatrix([[1], [1]])).sum()
  // ).toBeLessThanOrEqual(0.5);
  // expect(
  //   trained_nn.predict(createMatrix([[1], [0]])).sum()
  // ).toBeGreaterThanOrEqual(0.5);
  // expect(
  //   trained_nn.predict(createMatrix([[0], [1]])).sum()
  // ).toBeGreaterThanOrEqual(0.5);
  // expect(
  //   trained_nn.predict(createMatrix([[0], [0]])).sum()
  // ).toBeLessThanOrEqual(0.5);

  // Compare initial errors against the trained nn errors
  const initial_errors = R.reduce(
    (acc, elem) => elem.sum() + acc,
    0,
    nn.errors(inputs, targets)
  );
  const trained_errors = R.reduce(
    (acc, elem) => elem.sum() + acc,
    0,
    trained_nn.errors(inputs, targets)
  );
  expect(Math.abs(initial_errors)).toBeGreaterThanOrEqual(
    Math.abs(trained_errors)
  );
});
