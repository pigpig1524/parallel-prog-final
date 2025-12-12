#include <cstring>
#include <vector>
#include "conv.cpp"
#include "maxpool.cpp"
#include "upsample.cpp"

class AutoEncoder {
    private:
        ConvLayer** encoderConvs;
        MaxPool** pools;
        UpSample** upsamples;
        ConvLayer** decoderConvs;

        int inputWidth = 32;
        int inputHeight = 32;
        int inputChannels = 3;
        int latentWidth = 8;
        int latentHeight = 8;
        int latentChannels = 128;

        float** encoderWeights;
        float** encoderBiases;
        float** decoderWeights;
        float** decoderBiases;

    public:
        AutoEncoder(int inputW, int inputH, int inputC, int latentW, int latentH, int latentC);
        ~AutoEncoder();
        void loadModel(const char* modelPath);
        void saveModel(const char* modelPath);
        void initialize(int inputW, int inputH, int inputC, int latentW, int latentH, int latentC);
        void forward(const float* inputBatch, float* outputBatch, int batchSize=32);
        void backward(const float* inputBatch, const float* outputBatch, float* gradInput, int batchSize=32);
        void updateWeights(float learningRate);
        void train(const float* inputBatch, int batchSize=32, float learningRate=0.001f);
};