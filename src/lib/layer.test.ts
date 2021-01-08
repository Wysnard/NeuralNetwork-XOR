import { createMatrix } from "./matrix";
import { createLayer } from "./layer";

/* create Matrix */
test("should create a very simple layer and predict", () => {
  const layer = createLayer({
    inputDimension: 1,
    units: 1,
    fn: { activationFn: (x) => x, dActivationFn: (x) => x },
    weights: createMatrix([[1]]),
    bias: createMatrix([[0]]),
  });
  const input = createMatrix([[1]]);
  const result = layer.predict(input);
  expect(result.data()).toStrictEqual([[1]]);
});

test("should create a 2 -> 1 layer and predict", () => {
  const weights = createMatrix([[1, 1]]);
  const bias = createMatrix([[1]]);
  const layer = createLayer({
    inputDimension: 2,
    units: 1,
    fn: { activationFn: (x) => x, dActivationFn: (x) => x },
    weights,
    bias,
  });
  const input = createMatrix([[1], [1]]);
  const result = layer.predict(input);
  expect(result.data()).toStrictEqual([[3]]);
});

test("should create a 1 -> 4 layer and predict", () => {
  const weights = createMatrix([[1], [1], [0], [0]]);
  const bias = createMatrix([[1], [0], [1], [0]]);
  const layer = createLayer({
    inputDimension: 1,
    units: 4,
    fn: { activationFn: (x) => x, dActivationFn: (x) => x },
    weights,
    bias,
  });
  const input = createMatrix([[1]]);
  const result = layer.predict(input);
  expect(result.data()).toStrictEqual([[2], [1], [1], [0]]);
});
