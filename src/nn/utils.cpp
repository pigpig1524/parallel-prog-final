#include <random>
#include <algorithm>
#include <iostream>

namespace nn {

	float *random(size_t elementSize) {
		float *result = new float[elementSize];
		for (size_t i = 0; i < elementSize; i++) {
			result[i] = ((float)rand() / (RAND_MAX));
			// result[i] = -1 + 2 * ((float)rand()) / RAND_MAX;
		}
		return result;
	}

	float *randomGaussian(size_t elementSize, float mean, float sigma) {
		std::default_random_engine generator;
		std::normal_distribution<float> distribution(mean, sigma);
		float *result = new float[elementSize];
		for (size_t i = 0; i < elementSize; i++) {
			result[i] = distribution(generator);
		}
		return result;
	}

	float squareError(float d1, float d2) {
		return pow((d1 - d2), 2);
	}

	float sigmoid(float d) {
		return 1.0 / (1.0 + exp(-d));
	}

	float sigmoidDerivation(float d) {
		return d * (1.0 - d);
	}

	float relu(float d) {
		return std::fmax(0, d);
	}

	float reluDerivation(float d) {
		return d >= 0.0 ? 0.0 : 1.0;
	}
};

