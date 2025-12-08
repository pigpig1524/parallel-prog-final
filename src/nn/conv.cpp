#include <iostream>
#include <cstring>
#include "conv.h"

Conv2D::Conv2D(int filterNum, int filterWidth, int inputChannels, int stride, int padding) {
    this->filterNum = filterNum;
    this->filterWidth = filterWidth;
    this->inputChannels = inputChannels;

    this->stride = stride;
    this->padding = padding;

    // Allocate memory for filters and biases
    filters = new float[filterNum * filterWidth * filterWidth * inputChannels];
    biases = new float[filterNum];
}

Conv2D::~Conv2D() {
    // delete[] filters;
    // delete[] biases;
}

void Conv2D::setWeights(float * filterWeights, float * biasWeights) {
    int filterSize = filterNum * filterWidth * filterWidth * inputChannels;
    memcpy(filters, filterWeights, filterSize * sizeof(float));
    memcpy(biases, biasWeights, filterNum * sizeof(float));
}

void Conv2D::convolve(float * inputBatch, float * outputBatch, int inputWidth, int inputHeight, int batchSize) {
    int outputWidth = (inputWidth - filterWidth + 2 * padding) / stride + 1;
    int outputHeight = (inputHeight - filterWidth + 2 * padding) / stride + 1;

    for (int imgIdx = 0; imgIdx < batchSize; imgIdx++) {
        printf("Convolving image %d/%d\r", imgIdx + 1, batchSize);

        for (int filterIdx = 0; filterIdx < filterNum; filterIdx++) {
            for (int outY = 0; outY < outputHeight; outY++) {
                for (int outX = 0; outX < outputWidth; outX++) {
                    float sum = biases[filterIdx];
                    for (int channel = 0; channel < inputChannels; channel++) {
                        for (int filterY = 0; filterY < filterWidth; filterY++) {
                            for (int filterX = 0; filterX < filterWidth; filterX++) {
                                int inY = outY * stride + filterY - padding;
                                int inX = outX * stride + filterX - padding;
                                if (inY >= 0 && inY < inputHeight && inX >= 0 && inX < inputWidth) {
                                    int inputIndex = imgIdx * (inputWidth * inputHeight * inputChannels) +
                                                     channel * (inputWidth * inputHeight) +
                                                     inY * inputWidth +
                                                     inX;
                                    int filterIndex = filterIdx * (filterWidth * filterWidth * inputChannels) +
                                                      channel * (filterWidth * filterWidth) +
                                                      filterY * filterWidth +
                                                      filterX;
                                    sum += inputBatch[inputIndex] * filters[filterIndex];
                                }
                            }
                        }
                    }
                    int outputIndex = imgIdx * (outputWidth * outputHeight * filterNum) +
                                      filterIdx * (outputWidth * outputHeight) +
                                      outY * outputWidth +
                                      outX;
                    outputBatch[outputIndex] = sum;
                }
            }
        }
    }

}

float* Conv2D::convolve(float * inputBatch, int inputWidth, int inputHeight, int batchSize) {
    int outputWidth = (inputWidth - filterWidth + 2 * padding) / stride + 1;
    int outputHeight = (inputHeight - filterWidth + 2 * padding) / stride + 1;

    // For convenience, allocate outputBatch if not provided
    float* outputBatch = new float[batchSize * outputWidth * outputHeight * filterNum];

    for (int imgIdx = 0; imgIdx < batchSize; imgIdx++) {
        for (int filterIdx = 0; filterIdx < filterNum; filterIdx++) {
            for (int outY = 0; outY < outputHeight; outY++) {
                for (int outX = 0; outX < outputWidth; outX++) {
                    float sum = biases[filterIdx];
                    for (int channel = 0; channel < inputChannels; channel++) {
                        for (int filterY = 0; filterY < filterWidth; filterY++) {
                            for (int filterX = 0; filterX < filterWidth; filterX++) {
                                int inY = outY * stride + filterY - padding;
                                int inX = outX * stride + filterX - padding;
                                if (inY >= 0 && inY < inputHeight && inX >= 0 && inX < inputWidth) {
                                    int inputIndex = imgIdx * (inputWidth * inputHeight * inputChannels) +
                                                     channel * (inputWidth * inputHeight) +
                                                     inY * inputWidth +
                                                     inX;
                                    int filterIndex = filterIdx * (filterWidth * filterWidth * inputChannels) +
                                                      channel * (filterWidth * filterWidth) +
                                                      filterY * filterWidth +
                                                      filterX;
                                    sum += inputBatch[inputIndex] * filters[filterIndex];
                                }
                            }
                        }
                    }
                    int outputIndex = imgIdx * (outputWidth * outputHeight * filterNum) +
                                      filterIdx * (outputWidth * outputHeight) +
                                      outY * outputWidth +
                                      outX;
                    outputBatch[outputIndex] = sum;
                }
            }
        }
    }

    return outputBatch;
}

void Conv2D::backpropagate(
    float * inputBatch,
    float * outputGradients,
    float * inputGradients,
    int inputWidth, int inputHeight,
    int batchSize
) {
    // Backpropagation implementation would go here
    
}
