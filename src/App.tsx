import React, { useState } from "react";
import Sketch from "react-p5";
import p5Types from "p5";

import { createMatrix } from "./lib/matrix";
import { createLayer } from "./lib/layer";
import { createNeuralNetwork, INeuralNetwork } from "./lib/nn";

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

function App() {
  const [nn, setNN] = useState<INeuralNetwork>(
    createNeuralNetwork([
      createLayer({ inputDimension: 2, units: 2 }),
      createLayer({ inputDimension: 2, units: 1 }),
    ])
  );

  function setup(p5: p5Types, canvasParentRef: Element) {
    p5.createCanvas(400, 400).parent(canvasParentRef);
    setNN(
      createNeuralNetwork([
        createLayer({ inputDimension: 2, units: 4 }),
        createLayer({ inputDimension: 4, units: 1 }),
      ])
    );
  }

  function draw(p5: p5Types) {
    p5.background(0);

    setNN(nn.train({ inputs, targets, epochs: 10, learning_rate: 1 }));
    const resolution = 10;
    const cols = p5.width / resolution;
    const rows = p5.height / resolution;
    for (let i = 0; i < cols; i++) {
      for (let j = 0; j < rows; j++) {
        const x1 = i / cols;
        const x2 = j / rows;
        const input = [[x1], [x2]];
        const y = nn.predict(createMatrix(input)).sum();
        p5.noStroke();
        p5.fill(y * 255);
        p5.rect(i * resolution, j * resolution, resolution, resolution);
      }
    }
  }

  return <Sketch setup={setup} draw={draw} />;
}

export default App;
