class ConvLayer {
public:
    ConvLayer(int inputChannels, int outputChannels, int filterWidth, int stride = 1, int padding = 2);
    ~ConvLayer();

    void forward(const float* inputBatch, float* outputBatch, int inputWidth, int inputHeight, int batchSize=32);
    void backward(const float* inputBatch, const float* gradOutput, float* gradInput, int inputWidth, int inputHeight, int batchSize=32);
    void updateWeights(float learningRate);

private:
    int _inputChannels;
    int _outputChannels;
    int _filterWidth;
    int _stride;
    int _padding;
    float* _weights;
    float* _biases;
    float* _gradWeights;
    float* _gradBiases;

    const float* _inputCache;
    float* _outputCache;
};
