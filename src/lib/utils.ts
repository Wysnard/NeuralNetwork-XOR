import * as R from "ramda";

export function sigmoid(x: number) {
  return 1 / (1 + Math.exp(-x));
}

export function dsigmoid(x: number) {
  // return y * (1 - y);
  return sigmoid(x) * (1 - sigmoid(x));
}

/**
 * Shuffles array in place. ES6 version
 * @param {Array} a items An array containing the items.
 */
export function shuffle<T>(a: T[]): T[] {
  let result = R.clone(a);
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}
