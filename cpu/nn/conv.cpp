#include "conv.h"
#include "utils.cpp"
#include <cstring>


ConvLayer::ConvLayer(int inputChannels, int outputChannels, int filterWidth, int stride, int padding)
    : _inputChannels(inputChannels), _outputChannels(outputChannels), _filterWidth(filterWidth),
      _stride(stride), _padding(padding) {
    // Allocate memory for weights and biases
    _weights = new float[outputChannels * inputChannels * filterWidth * filterWidth];
    _biases = new float[outputChannels];
    _gradWeights = new float[outputChannels * inputChannels * filterWidth * filterWidth];
    _gradBiases = new float[outputChannels];

    // Initialize weights and biases (e.g., with random values)
    // ...
    nn::utils::heInitConv(_weights, outputChannels, filterWidth, inputChannels);
    nn::utils::zeros(_biases, outputChannels);
    nn::utils::zeros(_gradWeights, outputChannels * inputChannels * filterWidth * filterWidth);
    nn::utils::zeros(_gradBiases, outputChannels);
}

ConvLayer::~ConvLayer() {
    delete[] _weights;
    delete[] _biases;
    delete[] _gradWeights;
    delete[] _gradBiases;
}

void ConvLayer::updateWeights(float learningRate) {
    int weightSize = _outputChannels * _inputChannels * _filterWidth * _filterWidth;
    for (int i = 0; i < weightSize; i++) {
        _weights[i] -= learningRate * _gradWeights[i];
    }
    for (int i = 0; i < _outputChannels; i++) {
        _biases[i] -= learningRate * _gradBiases[i];
    }
    // After updating, reset gradients to zero
    nn::utils::zeros(_gradWeights, weightSize);
    nn::utils::zeros(_gradBiases, _outputChannels);
}


void ConvLayer::forward(const float* inputBatch, float* outputBatch, int inputWidth, int inputHeight, int batchSize) {
    // Logic for forward pass
    // ...
    int outputWidth = (inputWidth - _filterWidth + 2 * _padding) / _stride + 1;
    int outputHeight = (inputHeight - _filterWidth + 2 * _padding) / _stride + 1;

    int inputSize = inputWidth * inputHeight * _inputChannels;
    int outputSize = outputWidth * outputHeight * _outputChannels;

    // delete[] this->_inputCache;
    // this->_inputCache = new float[batchSize * inputSize];
    // memcpy(this->_inputCache, inputBatch, sizeof(float) * batchSize * inputSize);
    this->_inputCache = inputBatch;

    for (int b = 0; b < batchSize; b++) {
        const float* input = inputBatch + b * inputWidth * inputHeight * _inputChannels;
        float* output = outputBatch + b * outputWidth * outputHeight * _outputChannels;

        // Perform convolution operation
        // ...
        for (int oc = 0; oc < _outputChannels; oc++) {
            for (int oy = 0; oy < outputHeight; oy++) {
                for (int ox = 0; ox < outputWidth; ox++) {
                    float sum = _biases[oc];
                    for (int ic = 0; ic < _inputChannels; ic++) {
                        for (int fy = 0; fy < _filterWidth; fy++) {
                            for (int fx = 0; fx < _filterWidth; fx++) {
                                int inY = oy * _stride + fy - _padding;
                                int inX = ox * _stride + fx - _padding;
                                if (inY >= 0 && inY < inputHeight && inX >= 0 && inX < inputWidth) {
                                    sum += input[
                                        ic * inputWidth * inputHeight +
                                        inY * inputWidth +
                                        inX
                                    ] * _weights[
                                        oc * _inputChannels * _filterWidth * _filterWidth +
                                        ic * _filterWidth * _filterWidth +
                                        fy * _filterWidth +
                                        fx
                                    ];
                                }
                            }
                        }
                    }
                    output[
                        oc * outputWidth * outputHeight +
                        oy * outputWidth +
                        ox
                    ] = sum;
                }
            }
        }
    }
    this->_outputCache = outputBatch;
}

void ConvLayer::backward(const float* inputBatch, const float* gradOutput, float* gradInput, int inputWidth, int inputHeight, int batchSize) {
    // Logic for backward pass
    // ...
    int outputWidth = (inputWidth - _filterWidth + 2 * _padding) / _stride + 1;
    int outputHeight = (inputHeight - _filterWidth + 2 * _padding) / _stride + 1;

    memset(gradInput, 0, sizeof(float) * batchSize * inputWidth * inputHeight * _inputChannels);
    nn::utils::zeros(_gradWeights, _outputChannels * _inputChannels * _filterWidth * _filterWidth);
    nn::utils::zeros(_gradBiases, _outputChannels);

    for (int b = 0; b < batchSize; b++) {
        const float* input = inputBatch + b * inputWidth * inputHeight * _inputChannels;
        const float* outputGrad = gradOutput + b * outputWidth * outputHeight * _outputChannels;
        float* inputGrad = gradInput + b * inputWidth * inputHeight * _inputChannels;

        // Compute gradients
        // ...
        for (int oc = 0; oc < _outputChannels; oc++) {
            for (int oy = 0; oy < outputHeight; oy++) {
                for (int ox = 0; ox < outputWidth; ox++) {
                    float gradOutVal = outputGrad[
                        oc * outputWidth * outputHeight +
                        oy * outputWidth +
                        ox
                    ];
                    _gradBiases[oc] += gradOutVal;

                    for (int ic = 0; ic < _inputChannels; ic++) {
                        for (int fy = 0; fy < _filterWidth; fy++) {
                            for (int fx = 0; fx < _filterWidth; fx++) {
                                int inY = oy * _stride + fy - _padding;
                                int inX = ox * _stride + fx - _padding;
                                if (inY >= 0 && inY < inputHeight && inX >= 0 && inX < inputWidth) {
                                    _gradWeights[
                                        oc * _inputChannels * _filterWidth * _filterWidth +
                                        ic * _filterWidth * _filterWidth +
                                        fy * _filterWidth +
                                        fx
                                    ] += input[
                                        ic * inputWidth * inputHeight +
                                        inY * inputWidth +
                                        inX
                                    ] * gradOutVal;
                                    inputGrad[
                                        ic * inputWidth * inputHeight +
                                        inY * inputWidth +
                                        inX
                                    ] += _weights[
                                        oc * _inputChannels * _filterWidth * _filterWidth +
                                        ic * _filterWidth * _filterWidth +
                                        fy * _filterWidth +
                                        fx
                                    ] * gradOutVal;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

