#include "autoencoder.h"
#include <cstring>

AutoEncoder::AutoEncoder(int inputW, int inputH, int inputC, int latentW, int latentH, int latentC) {
    this->inputWidth = inputW;
    this->inputHeight = inputH;
    this->inputChannels = inputC;
    this->latentWidth = latentW;
    this->latentHeight = latentH;
    this->latentChannels = latentC;

    // Init for encoder layers
    encoderConvs = new ConvLayer*[2];
    pools = new MaxPool*[2];
    encoderConvs[0] = new ConvLayer(inputChannels, 128, 3, 1, 1);
    pools[0] = new MaxPool(2, 2);
    encoderConvs[1] = new ConvLayer(128, latentChannels, 3, 1, 1);
    pools[1] = new MaxPool(2, 2);

    // Init decoder layers
    decoderConvs = new ConvLayer*[3];
    upsamples = new UpSample*[2];
    decoderConvs[0] = new ConvLayer(latentChannels, 128, 3, 1, 1);
    upsamples[0] = new UpSample(2);
    decoderConvs[1] = new ConvLayer(128, 64, 3, 1, 1);
    upsamples[1] = new UpSample(2);
    decoderConvs[2] = new ConvLayer(64, inputChannels, 3, 1, 1);
    

    pools = nullptr;
    upsamples = nullptr;
    decoderConvs = nullptr;

    encoderWeights = nullptr;
    encoderBiases = nullptr;
    decoderWeights = nullptr;
    decoderBiases = nullptr;

    // initialize();
}

AutoEncoder::~AutoEncoder() {
    if (encoderConvs) {
        delete[] encoderConvs;
    }
    if (pools) {
        delete[] pools;
    }
    if (upsamples) {
        delete[] upsamples;
    }
    if (decoderConvs) {
        delete[] decoderConvs;
    }
}

// void AutoEncoder::initialize() {
//     encoderConvs = new ConvLayer*[2];
//     encoderWeights = new float*[2];

//     pools = new MaxPool*[2];
//     upsamples = new UpSample*[2];
//     decoderConvs = new ConvLayer*[3];
//     decoderWeights = new float*[3];

//     // Init weights with He initialization
//     for (int i = 0; i < 2; i++) {
//         encoderConvs[i] = nullptr;
//         encoderWeights[i] = nullptr;

//     }

//     // encoderConv = new ConvLayer(inputChannels, 16, 3, 1, 1);
//     // pool = new MaxPool(2, 2);
//     // upsample = new UpSample(2);
//     // decoderConv = new ConvLayer(16, inputChannels, 3, 1, 1);
// }
