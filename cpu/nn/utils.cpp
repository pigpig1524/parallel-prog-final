#include <math.h>

namespace nn {
namespace utils {
// Utility functions for neural network operations can be implemented here.

    void heInitConv(float* arr, int filterNum, int filterWidth, int inputChannels) {
        int size = filterNum * filterWidth * filterWidth * inputChannels;

        float fanIn = filterWidth * filterWidth * inputChannels;
        float std = sqrt(2.0f / fanIn);

        for (int i = 0; i < size; i++) {
            // random số float từ -1 → 1
            float r = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
            arr[i] = r * std;
        }
    }

    void zeros(float* arr, int size) {
        for (int i = 0; i < size; i++) {
            arr[i] = 0.0f;
        }
    }

    void mse_loss_backward(float* output, float* target, float* grad, int size) {
        for (int i = 0; i < size; i++)
            grad[i] = (output[i] - target[i]) * 2.0f / size;
    }

    void calcMSE(float* predicted, float* actual, int size, float& mse) {
        mse = 0.0f;
        for (int i = 0; i < size; i++) {
            float diff = predicted[i] - actual[i];
            mse += diff * diff;
        }
        mse /= static_cast<float>(size);
    }
}
}
