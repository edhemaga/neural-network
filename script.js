class Layer {
	input;
	output;
	weights;
	biases;

	constructor(input, outputSize) {
		this.input = input;
		this.initializeOutputsAndBiases(outputSize);
		this.initializeWeights();
	}

	initializeOutputsAndBiases(outputSize) {
		this.output = [];
		this.biases = [];
		for (let i = 0; i < outputSize; i++) {
			this.output = [...this.output, 0];
			this.biases = [...this.biases, 0.1];
		}
	}

	initializeWeights() {
		this.weights = [];
		for (let i = 0; i < this.input?.length; i++) {
			this.weights = [...this.weights, []];

			for (let j = 0; j < this.output?.length; j++) {
				const rnd = Math.random() - 0.5;
				this.weights[i] = [
					...this.weights[i],
					Math.round(rnd * 1000) / 1000,
				];
			}
		}
	}
}

class Network {
	// TODO add some input
	layers;

	constructor(input, hiddenLayersOutputSize, targetOutputs) {
		this.layers = [new Layer(input, 5)];

		const allLayers = [...hiddenLayersOutputSize, targetOutputs.length];

		allLayers.forEach((outputSize) => {
			const length = this.layers?.length - 1;
			this.layers = [
				...this.layers,
				new Layer(this.layers[length]?.output, outputSize),
			];
		});

		this.layers.forEach((layer) => {
			this.feedforward(layer);
		});
	}

	feedforward(layer) {
		for (let i = 0; i < layer.input?.length; i++) {
			for (let j = 0; j < layer.output?.length; j++) {
				const value =
					Math.round(layer.weights[i][j] * layer.input[i] * 1000) /
					1000;
				layer.output[j] += value;
				if (j === layer.output.length - 1)
					layer.output[j] = this.sigmoid(
						layer.output[j] + layer.biases[j]
					);
			}
		}
	}

	backpropagate() {}

	sigmoid(value) {
		return 1 / (1 + Math.exp(-value));
	}

	sigmoidDerivative(value) {
		const sigmoidValue = sigmoid(value);
		return sigmoidValue * (1 - sigmoidValue);
	}

	ReLu(value) {
		if (value < 0) return 0;
		return value;
	}

	calculateCost(value, expectedValue) {
		return (expectedValue - value) * (expectedValue - value);
	}

	constHelper(value, expected) {
		const valueToActivationDiff = this.sigmoidDerivative(value);
		const expectedToValueDiff = 2 * (value - expected);
		return valueToActivationDiff * expectedToValueDiff;
	}

	// Partial derivatives are to be used here
	calculateCostByWeight(previousValue, value, expected) {
		const previousToActivationDiff = previousValue;
		return previousToActivationDiff * constHelper(value, expected);
	}
	calculateCostByBias(value, expected) {
		// Since derivative of bias is 1
		return constHelper(value, expected);
	}

	calculateMSE(outputs, expectedOutputs) {
		return (
			outputs.reduce((acc, curr, i) => {
				acc += this.calculateCost(curr, expectedOutputs[i]);
			}) / outputs.length
		);
	}
}
// Parameters order: Input data, hidden layer configuration, expected values
var network = new Network([1, 2], [3], [0.2, 0.5]);

console.log(network.layers);
