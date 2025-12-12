namespace nn {
namespace activation {

    void relu(const float* input, float* output, int size) {
        for (int i = 0; i < size; i++) {
            output[i] = input[i] > 0 ? input[i] : 0;
        }
    }

    void reluBackward(const float* input, const float* gradOutput, float* gradInput, int size) {
        for (int i = 0; i < size; i++) {
            gradInput[i] = input[i] > 0 ? gradOutput[i] : 0;
        }
    }

} // namespace activation
}