#include <cstring>

// Assume using nearest neighbor upsampling
class UpSample {
private:
    int scaleFactor;
public:
    UpSample(int scaleFactor=2) : scaleFactor(scaleFactor) {}

    void forward(const float* inputBatch, float* outputBatch, int inputWidth, int inputHeight, int inputChannels, int batchSize=32) {
        int outputWidth = inputWidth * scaleFactor;
        int outputHeight = inputHeight * scaleFactor;

        for (int b = 0; b < batchSize; b++) {
            const float* input = inputBatch + b * inputWidth * inputHeight * inputChannels;
            float* output = outputBatch + b * outputWidth * outputHeight * inputChannels;

            for (int c = 0; c < inputChannels; c++) {
                for (int iy = 0; iy < inputHeight; iy++) {
                    for (int ix = 0; ix < inputWidth; ix++) {
                        float val = input[
                            c * inputWidth * inputHeight +
                            iy * inputWidth +
                            ix
                        ];
                        for (int sy = 0; sy < scaleFactor; sy++) {
                            for (int sx = 0; sx < scaleFactor; sx++) {
                                int oy = iy * scaleFactor + sy;
                                int ox = ix * scaleFactor + sx;
                                output[
                                    c * outputWidth * outputHeight +
                                    oy * outputWidth +
                                    ox
                                ] = val;
                            }
                        }
                    }
                }
            }
        }
    }

    void backward(const float* gradOutput, float* gradInput, int inputWidth, int inputHeight, int inputChannels, int batchSize=32) {
        int outputWidth = inputWidth * scaleFactor;
        int outputHeight = inputHeight * scaleFactor;

        memset(gradInput, 0, sizeof(float) * batchSize * inputWidth * inputHeight * inputChannels);

        for (int b = 0; b < batchSize; b++) {
            const float* gradOut = gradOutput + b * outputWidth * outputHeight * inputChannels;
            float* gradIn = gradInput + b * inputWidth * inputHeight * inputChannels;

            for (int c = 0; c < inputChannels; c++) {
                for (int iy = 0; iy < inputHeight; iy++) {
                    for (int ix = 0; ix < inputWidth; ix++) {
                        float sumGrad = 0.0f;
                        for (int sy = 0; sy < scaleFactor; sy++) {
                            for (int sx = 0; sx < scaleFactor; sx++) {
                                int oy = iy * scaleFactor + sy;
                                int ox = ix * scaleFactor + sx;
                                sumGrad += gradOut[
                                    c * outputWidth * outputHeight +
                                    oy * outputWidth +
                                    ox
                                ];
                            }
                        }
                        gradIn[
                            c * inputWidth * inputHeight +
                            iy * inputWidth +
                            ix
                        ] = sumGrad;
                    }
                }
            }
        }
    }
};