#pragma once

namespace nn {

	float *random(size_t elementSize);

	float squareError(float d1, float d2);

	float sigmoid(float d);

	float sigmoidDerivation(float d);

	float relu(float d);

	float reluDerivation(float d);
};

