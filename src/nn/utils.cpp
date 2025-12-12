#include <algorithm>
#include <random>
#include <iostream>
#include <cstring>

/**
 * 
 * This class provides utility functions for neural network operations.
 */
class NNUtils {
    public:
        /**
         * 
         * This function normalizes an image's pixels value from [0, 255] to float [0, 1].
         *
         * @param pixels Pointer to the input array of pixel values (unsigned char).
         * @param out Pointer to the output array where normalized float values will be stored.
         * @param size The number of pixels to normalize. Default is 3072.
         */
        static void normalizePixels(unsigned char * pixels, float* out, int size = 3072) {
            for (int i = 0; i < size; i++) {
                out[i] = static_cast<float>(pixels[i]) / 255.0f;
            }
        }

        /**
         * 
         * This function initializes convolutional layer weights using He initialization.
         *
         * @param filterNum Number of filters.
         * @param filterWidth Width and height of each filter (assumed square).
         * @param inputChannels Number of input channels.
         * @return Pointer to the initialized weights array (float).
         */
        static float* heInitConv(int filterNum, int filterWidth, int inputChannels) {
            int size = filterNum * filterWidth * filterWidth * inputChannels;
            float* arr = new float[size];

            float fanIn = filterWidth * filterWidth * inputChannels;
            float std = sqrt(2.0f / fanIn);

            for (int i = 0; i < size; i++) {
                // random số float từ -1 → 1
                float r = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
                arr[i] = r * std;
            }
            return arr;
        }

        static float* zeroBias(int filterNum) {
            float* arr = new float[filterNum];
            memset(arr, 0, filterNum * sizeof(float));
            return arr;
        }

        static float* randomFloatArray(int size) {
            float* arr = new float[size];
            for (int i = 0; i < size; i++) {
                arr[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            }
            return arr;
        }

        static float* maxPooling(float* input, int inputWidth, int inputHeight, int channels, int poolSize=2, int stride=2) {
            int outputWidth = (inputWidth - poolSize) / stride + 1;
            int outputHeight = (inputHeight - poolSize) / stride + 1;
            float* output = new float[channels * outputWidth * outputHeight];

            for (int c = 0; c < channels; c++) {
                for (int h = 0; h < outputHeight; h++) {
                    for (int w = 0; w < outputWidth; w++) {
                        float maxVal = -std::numeric_limits<float>::infinity();
                        for (int ph = 0; ph < poolSize; ph++) {
                            for (int pw = 0; pw < poolSize; pw++) {
                                int inputH = h * stride + ph;
                                int inputW = w * stride + pw;
                                maxVal = std::max(maxVal, input[(c * inputHeight + inputH) * inputWidth + inputW]);
                            }
                        }
                        output[(c * outputHeight + h) * outputWidth + w] = maxVal;
                    }
                }
            }
            return output;
        }

        /**
         * 
         * This function performs batch max pooling on a 3D input array with multiple channels.
         *
         * @param input Pointer to the input array (float).
         * @param output Pointer to the output array where pooled values will be stored (float).
         * @param inputWidth The width of the input array.
         * @param inputHeight The height of the input array.
         * @param channels The number of channels in the input array.
         * @param poolSize The size of the pooling window.
         * @param stride The stride of the pooling operation.
         */
        static float* batchMaxPooling(
            float* inputBatch,
            int inputWidth,
            int inputHeight,
            int channels,
            int poolSize = 2,
            int stride = 2,
            int batchSize = 1
        ) {
            int oneOutputSize = 
                ((inputWidth - poolSize) / stride + 1) * 
                ((inputHeight - poolSize) / stride + 1) * 
                channels;

            float* outputBatch = new float[batchSize * oneOutputSize];

            for (int b = 0; b < batchSize; b++) {
                float* input = inputBatch + b * inputWidth * inputHeight * channels;
                float* output = maxPooling(input, inputWidth, inputHeight, channels, poolSize, stride);
                memcpy(
                    outputBatch + b * oneOutputSize,
                    output,
                    oneOutputSize * sizeof(float)
                );
                delete[] output;
            }

            return outputBatch;
        }

        /**
         * 
         * This function performs batch max pooling on a 3D input array with multiple channels.
         *
         * @param input Pointer to the input array (float).
         * @param output Pointer to the output array where pooled values will be stored (float).
         * @param inputWidth The width of the input array.
         * @param inputHeight The height of the input array.
         * @param channels The number of channels in the input array.
         * @param poolSize The size of the pooling window.
         * @param stride The stride of the pooling operation.
         */
        static void batchMaxPooling(
            float* inputBatch,
            float* outputBatch,
            int inputWidth,
            int inputHeight,
            int channels,
            int poolSize = 2,
            int stride = 2,
            int batchSize = 1
        ) {
            // int oneOutputSize = 
            //     ((inputWidth - poolSize) / stride + 1) * 
            //     ((inputHeight - poolSize) / stride + 1) * 
            //     channels;

            // for (int b = 0; b < batchSize; b++) {
            //     float* input = inputBatch + b * inputWidth * inputHeight * channels;
            //     float* output = maxPooling(input, inputWidth, inputHeight, channels, poolSize, stride);
            //     memcpy(
            //         outputBatch + b * oneOutputSize,
            //         output,
            //         oneOutputSize * sizeof(float)
            //     );
            //     delete[] output;
            // }
            int outW = (inputWidth - poolSize) / stride + 1;
            int outH = (inputHeight - poolSize) / stride + 1;

            for (int b = 0; b < batchSize; b++) {
                float* input = inputBatch + b * inputWidth * inputHeight * channels;
                float* output = outputBatch + b * outW * outH * channels;

                for (int c = 0; c < channels; c++) {
                    for (int oy = 0; oy < outH; oy++) {
                        for (int ox = 0; ox < outW; ox++) {
                            float maxVal = -std::numeric_limits<float>::infinity();
                            for (int py = 0; py < poolSize; py++) {
                                for (int px = 0; px < poolSize; px++) {
                                    int iy = oy * stride + py;
                                    int ix = ox * stride + px;
                                    float val = input[c * inputWidth * inputHeight + iy * inputWidth + ix];
                                    if (val > maxVal) maxVal = val;
                                }
                            }
                            output[c * outW * outH + oy * outW + ox] = maxVal;
                        }
                    }
                }
            }
        }

        /**
         * 
         * This function performs batch upsampling on a 3D input array with multiple channels.
         *
         * @param input Pointer to the input array (float).
         * @param output Pointer to the output array where upsampled values will be stored (float).
         * @param inputWidth The width of the input array.
         * @param inputHeight The height of the input array.
         * @param channels The number of channels in the input array.
         * @param scaleFactor The factor by which to upsample the input.
         */
        static float* upSample(float* input, int inputWidth, int inputHeight, int channels, int scaleFactor) {
            int outputWidth = inputWidth * scaleFactor;
            int outputHeight = inputHeight * scaleFactor;

            float* output = new float[channels * outputWidth * outputHeight];

            for (int c = 0; c < channels; c++) {
                for (int h = 0; h < outputHeight; h++) {
                    for (int w = 0; w < outputWidth; w++) {
                        int inputH = h / scaleFactor;
                        int inputW = w / scaleFactor;
                        output[c * outputWidth * outputHeight + h * outputWidth + w] = input[c * inputWidth * inputHeight + inputH * inputWidth + inputW];
                    }
                }
            }
            return output;
        }

        static float* batchUpSampling(
            float* inputBatch,
            int inputWidth,
            int inputHeight,
            int channels,
            int scaleFactor,
            int batchSize = 1
        ) {
            int oneOutputSize = 
                (inputWidth * scaleFactor) * 
                (inputHeight * scaleFactor) * 
                channels;

            float* outputBatch = new float[batchSize * oneOutputSize];

            for (int b = 0; b < batchSize; b++) {
                float* input = inputBatch + b * inputWidth * inputHeight * channels;
                float* output = upSample(input, inputWidth, inputHeight, channels, scaleFactor);
                memcpy(
                    outputBatch + b * oneOutputSize,
                    output,
                    oneOutputSize * sizeof(float)
                );
                delete[] output;
            }

            return outputBatch;
        }


        static void batchUpSampling(
            float* inputBatch,
            float* outputBatch,
            int inputWidth,
            int inputHeight,
            int channels,
            int scaleFactor,
            int batchSize = 1
        ) {
            int outW = inputWidth * scaleFactor;
            int outH = inputHeight * scaleFactor;

            for (int b = 0; b < batchSize; b++) {
                float* input = inputBatch + b * inputWidth * inputHeight * channels;
                float* output = outputBatch + b * outW * outH * channels;

                for (int c = 0; c < channels; c++) {
                    for (int iy = 0; iy < inputHeight; iy++) {
                        for (int ix = 0; ix < inputWidth; ix++) {
                            float val = input[c * inputWidth * inputHeight + iy * inputWidth + ix];
                            // copy val vào scaleFactor x scaleFactor block trong output
                            for (int dy = 0; dy < scaleFactor; dy++) {
                                for (int dx = 0; dx < scaleFactor; dx++) {
                                    int oy = iy * scaleFactor + dy;
                                    int ox = ix * scaleFactor + dx;
                                    output[c * outW * outH + oy * outW + ox] = val;
                                }
                            }
                        }
                    }
                }
            }
        }

    static void backwardMaxPoolBatch(
        float* inputBatch,
        float* outputGradBatch,
        float* inputGradBatch,
        int inputWidth,
        int inputHeight,
        int channels,
        int batchSize,
        int poolSize = 2,
        int stride = 2
    ) {
        memset(inputGradBatch, 0, sizeof(float) * batchSize * inputWidth * inputHeight * channels);

        int outputWidth = (inputWidth - poolSize) / stride + 1;
        int outputHeight = (inputHeight - poolSize) / stride + 1;

        for (int b = 0; b < batchSize; ++b) {
            float* input = inputBatch + b*inputWidth*inputHeight*channels;
            float* outGrad = outputGradBatch + b*outputWidth*outputHeight*channels;
            float* inGrad = inputGradBatch + b*inputWidth*inputHeight*channels;

            for (int c = 0; c < channels; ++c) {
                for (int h = 0; h < outputHeight; ++h) {
                    for (int w = 0; w < outputWidth; ++w) {
                        float maxVal = -std::numeric_limits<float>::infinity();
                        int maxH = -1, maxW = -1;
                        for (int ph = 0; ph < poolSize; ++ph) {
                            for (int pw = 0; pw < poolSize; ++pw) {
                                int ih = h*stride + ph;
                                int iw = w*stride + pw;
                                float val = input[(c*inputHeight + ih)*inputWidth + iw];
                                if (val > maxVal) { maxVal = val; maxH = ih; maxW = iw; }
                            }
                        }
                        inGrad[(c*inputHeight + maxH)*inputWidth + maxW] += outGrad[(c*outputHeight + h)*outputWidth + w];
                    }
                }
            }
        }
    }

    static void batchUpSamplingBackward(
        float* inputBatch,     // gradient từ output upsample (bigger)
        float* outputBatch,    // gradient trả về cho tensor trước upsample (smaller)
        int outputWidth,       // width của tensor nhỏ BEFORE upsample
        int outputHeight,      // height của tensor nhỏ BEFORE upsample
        int channels,
        int scaleFactor,
        int batchSize = 1
    ) {
        int inW = outputWidth * scaleFactor;   // width của tensor lớn (after upsample)
        int inH = outputHeight * scaleFactor;

        for (int b = 0; b < batchSize; b++) {
            float* gradBig  = inputBatch  + b * inW * inH * channels;
            float* gradSmall = outputBatch + b * outputWidth * outputHeight * channels;

            for (int c = 0; c < channels; c++) {
                for (int oy = 0; oy < outputHeight; oy++) {
                    for (int ox = 0; ox < outputWidth; ox++) {

                        float sumGrad = 0.0f;

                        // gom gradient từ block scale x scale
                        for (int dy = 0; dy < scaleFactor; dy++) {
                            for (int dx = 0; dx < scaleFactor; dx++) {
                                int iy = oy * scaleFactor + dy; 
                                int ix = ox * scaleFactor + dx;

                                sumGrad += gradBig[
                                    c * inW * inH +
                                    iy * inW +
                                    ix
                                ];
                            }
                        }

                        gradSmall[
                            c * outputWidth * outputHeight +
                            oy * outputWidth +
                            ox
                        ] = sumGrad;
                    }
                }
            }
        }
    }

    
};


class LossFunctions {
    public:
        /**
         * 
         * This function computes the Mean Squared Error (MSE) between two arrays.
         *
         * @param predicted Pointer to the array of predicted values (float).
         * @param actual Pointer to the array of actual values (float).
         * @param size The number of elements in the arrays. Default is 3072.
         * @return The computed Mean Squared Error as a float.
         */
        static float meanSquaredError(float* predicted, float* actual, int size = 3072) {
            float mse = 0.0f;
            for (int i = 0; i < size; i++) {
                float diff = predicted[i] - actual[i];
                mse += diff * diff;
            }
            return mse / static_cast<float>(size);
        }

        static void mse_loss_backward(float* output, float* target, float* grad, int size) {
            for (int i = 0; i < size; i++)
                grad[i] = (output[i] - target[i]) * 2.0f / size;
        }
};


class ActivationFunctions {
    public:
        /**
         * 
         * This function applies the ReLU activation function to an array.
         *
         * @param input Pointer to the input array (float).
         * @param output Pointer to the output array where activated values will be stored (float).
         * @param size The number of elements in the arrays. Default is 3072.
         */
        static void relu(float* input, float* output, int size = 3072) {
            for (int i = 0; i < size; i++) {
                output[i] = input[i] > 0.0f ? input[i] : 0.0f;
            }
        }

        static void relu(float* data, int size = 3072) {
            for (int i = 0; i < size; i++) {
                data[i] = data[i] > 0.0f ? data[i] : 0.0f;
            }
        }
};
