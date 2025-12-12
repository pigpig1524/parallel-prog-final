class ConvLayer {
public:
    ConvLayer(int inputChannels, int outputChannels, int filterWidth, int stride = 1, int padding = 2);
    ~ConvLayer();

    // int getOutputWidth(int inputWidth);
    int getOutChannels() { return _outputChannels; };
    void setWeights(const float* weights, const float* biases) {
        int weightSize = _outputChannels * _inputChannels * _filterWidth * _filterWidth;
        int biasSize = _outputChannels;
        memcpy(_weights, weights, sizeof(float) * weightSize);
        memcpy(_biases, biases, sizeof(float) * biasSize);
    }

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
    float* _outputCache = nullptr;
};
