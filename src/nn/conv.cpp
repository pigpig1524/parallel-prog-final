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
    int filterSize = filterNum * filterWidth * filterWidth * inputChannels;

    filters = new float[filterSize];
    biases = new float[filterNum];

    filtersGradients = new float[filterSize];
    biasesGradients = new float[filterNum];
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


void Conv2D::backward(
    float *inputBatch,        // (batch, H, W, C_in)
    float *outputGradients,   // (batch, H_out, W_out, F)
    float *inputGradients,    // (batch, H, W, C_in)
    int inputHeight, int inputWidth,
    int batchSize
) {
    int H = inputHeight;
    int W = inputWidth;

    int C_in = this->inputChannels;   // số channels đầu vào
    int F    = this->filterNum;       // số filters đầu ra
    int K    = this->filterWidth;     // kernel size

    int H_out = (H + 2 * padding - K) / stride + 1;
    int W_out = (W + 2 * padding - K) / stride + 1;

    // RESET gradients
    memset(filtersGradients, 0, F * K * K * C_in * sizeof(float));
    memset(biasesGradients, 0, F * sizeof(float));
    memset(inputGradients, 0, batchSize * H * W * C_in * sizeof(float));

    // === MAIN BACKWARD LOOP ===
    for (int b = 0; b < batchSize; b++) {
        for (int oy = 0; oy < H_out; oy++) {
            for (int ox = 0; ox < W_out; ox++) {

                for (int f = 0; f < F; f++) {

                    // dL/dOutput at (b, oy, ox, f)
                    int out_idx =
                        b * (H_out * W_out * F) +
                        oy * (W_out * F) +
                        ox * F + f;

                    float dOut = outputGradients[out_idx];

                    // dL/dBias
                    biasesGradients[f] += dOut;

                    // Loop kernel
                    for (int ky = 0; ky < K; ky++) {
                        for (int kx = 0; kx < K; kx++) {

                            int in_y = oy * stride + ky - padding;
                            int in_x = ox * stride + kx - padding;

                            // skip out-of-bound
                            if (in_y < 0 || in_y >= H || in_x < 0 || in_x >= W)
                                continue;

                            for (int c = 0; c < C_in; c++) {

                                // Index input pixel
                                int in_idx =
                                    b * (H * W * C_in) +
                                    in_y * (W * C_in) +
                                    in_x * C_in + c;

                                // Index filter weight
                                int w_idx =
                                    f * (K * K * C_in) +
                                    ky * (K * C_in) +
                                    kx * C_in + c;

                                // dL/dFilter
                                filtersGradients[w_idx] += inputBatch[in_idx] * dOut;

                                // dL/dInput
                                inputGradients[in_idx] += filters[w_idx] * dOut;
                            }
                        }
                    }
                }
            }
        }
    }
}

