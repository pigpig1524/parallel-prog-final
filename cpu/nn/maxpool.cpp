#include <cfloat>
#include <cstring>

class MaxPool {
private:
    int poolSize;
    int stride;
public:
    MaxPool(int poolSize=2, int stride=2) : poolSize(poolSize), stride(stride) {}
    
    void forward(const float* inputBatch, float* outputBatch, int inputWidth, int inputHeight, int inputChannels, int batchSize=32) {
        int outputWidth = (inputWidth - poolSize) / stride + 1;
        int outputHeight = (inputHeight - poolSize) / stride + 1;

        for (int b = 0; b < batchSize; b++) {
            const float* input = inputBatch + b * inputWidth * inputHeight * inputChannels;
            float* output = outputBatch + b * outputWidth * outputHeight * inputChannels;

            for (int c = 0; c < inputChannels; c++) {
                for (int oy = 0; oy < outputHeight; oy++) {
                    for (int ox = 0; ox < outputWidth; ox++) {
                        float maxVal = -FLT_MAX;
                        for (int py = 0; py < poolSize; py++) {
                            for (int px = 0; px < poolSize; px++) {
                                int inY = oy * stride + py;
                                int inX = ox * stride + px;
                                float val = input[
                                    c * inputWidth * inputHeight +
                                    inY * inputWidth +
                                    inX
                                ];
                                if (val > maxVal) {
                                    maxVal = val;
                                }
                            }
                        }
                        output[
                            c * outputWidth * outputHeight +
                            oy * outputWidth +
                            ox
                        ] = maxVal;
                    }
                }
            }
        }
    }

    void backward(const float* gradOutput, const float* inputBatch, float* gradInput, int inputWidth, int inputHeight, int inputChannels, int batchSize=32) {
        int outputWidth = (inputWidth - poolSize) / stride + 1;
        int outputHeight = (inputHeight - poolSize) / stride + 1;

        memset(gradInput, 0, sizeof(float) * batchSize * inputWidth * inputHeight * inputChannels);

        for (int b = 0; b < batchSize; b++) {
            const float* input = inputBatch + b * inputWidth * inputHeight * inputChannels;
            const float* gradOut = gradOutput + b * outputWidth * outputHeight * inputChannels;
            float* gradIn = gradInput + b * inputWidth * inputHeight * inputChannels;

            for (int c = 0; c < inputChannels; c++) {
                for (int oy = 0; oy < outputHeight; oy++) {
                    for (int ox = 0; ox < outputWidth; ox++) {
                        float maxVal = -FLT_MAX;
                        int maxY = -1, maxX = -1;
                        for (int py = 0; py < poolSize; py++) {
                            for (int px = 0; px < poolSize; px++) {
                                int inY = oy * stride + py;
                                int inX = ox * stride + px;
                                float val = input[
                                    c * inputWidth * inputHeight +
                                    inY * inputWidth +
                                    inX
                                ];
                                if (val > maxVal) {
                                    maxVal = val;
                                    maxY = inY;
                                    maxX = inX;
                                }
                            }
                        }
                        gradIn[
                            c * inputWidth * inputHeight +
                            maxY * inputWidth +
                            maxX
                        ] += gradOut[
                            c * outputWidth * outputHeight +
                            oy * outputWidth +
                            ox
                        ];
                    }
                }
            }
        }
    }
};