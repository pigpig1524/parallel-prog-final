#ifndef AUTOENCODER
#define AUTOENCODER_H

#include "conv.cpp"
#include <vector>
#include "../dataset.cpp"
#include "utils.cpp"


/**
 * 
 * Assumptions:
 *  * Input images are of size 32x32 with 3 channels (RGB).
 *  * The autoencoder consists of multiple convolutional layers for both encoding and decoding.
 *  * The architecture and parameters (like number of filters, filter sizes, etc.) are predefined.
 */
class Autoencoder {
private:
    // Add private members for encoder and decoder layers
    std::vector<Conv2D> encoderLayers;
    std::vector<Conv2D> decoderLayers;

    float** encoderWeights;
    float** encoderBiases;
    float** decoderWeights;
    float** decoderBiases;

    Dataset* dataset;
    float* inputValue = nullptr;
    float** hiddenEncoderValue = nullptr;
    float** upSampleValue = nullptr;
    float** maxPoolValue = nullptr;
    float* latentValue = nullptr;
    float** hiddenDecoderValue = nullptr;
    float* outputValue = nullptr;

    float error = 0;
    std::vector<float> errorHistory;
    std::vector<float> lossHistory;

    void feedforward(float* inputBatch, int batchSize);
    void backpropagate(int batchSize, float learningRate = 0.001f);

    void applyCurrentWeights();
    void updateWeights(float learningRate = 0.001f);

public:
    Autoencoder(Dataset * dataset);
    ~Autoencoder();

    void train(int epochs = 10, int batchSize = 32, float learningRate = 0.001f);
    void test();
};


#endif // AUTOENCODER.H