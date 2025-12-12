#include "autoencoder.h"
#include <cstring>
#include <iostream>

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
    encoderConvs[0] = new ConvLayer(inputChannels, 256, 3);
    pools[0] = new MaxPool(2, 2);
    encoderConvs[1] = new ConvLayer(128, latentChannels, 3);
    pools[1] = new MaxPool(2, 2);

    encoderOutputs = std::vector<float*>(2);
    poolOutputs = std::vector<float*>(2);

    // Init decoder layers
    decoderConvs = new ConvLayer*[3];
    upsamples = new UpSample*[2];
    decoderConvs[0] = new ConvLayer(latentChannels, 128, 3);
    upsamples[0] = new UpSample(2);
    decoderConvs[1] = new ConvLayer(128, 256, 3);
    upsamples[1] = new UpSample(2);
    decoderConvs[2] = new ConvLayer(256, inputChannels, 3);
    decoderOutputs = std::vector<float*>(3);
    upsampleOutputs = std::vector<float*>(2);
    

    // pools = nullptr;
    // upsamples = nullptr;
    // // decoderConvs = nullptr;

    // encoderWeights = nullptr;
    // encoderBiases = nullptr;
    // decoderWeights = nullptr;
    // decoderBiases = nullptr;

    this->latentCache = nullptr;
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

void AutoEncoder::initialize(int inputW, int inputH, int inputC, int latentW, int latentH, int latentC) {
    // init encoder layers
    encoderWeights = new float*[2];
    encoderBiases = new float*[2];
    encoderWeights[0] = new float[3 * 3 * 3 * 256];
    nn::utils::heInitConv(encoderWeights[0], 256, 3, 3);

    encoderWeights[1] = new float[256 * 3 * 3 * 128];
    nn::utils::heInitConv(encoderWeights[1], 128, 3, 256);

    encoderBiases[0] = new float[256];
    nn::utils::zeros(encoderBiases[0], 256);

    encoderBiases[1] = new float[128];
    nn::utils::zeros(encoderBiases[1], 128);

    // init decoder layers
    decoderWeights = new float*[3];
    decoderBiases = new float*[3];

    decoderWeights[0] = new float[128 * 3 * 3 * 128];
    nn::utils::heInitConv(decoderWeights[0], 128, 3, 128);

    decoderWeights[1] = new float[128 * 3 * 3 * 256];
    nn::utils::heInitConv(decoderWeights[1], 128, 3, 256);

    decoderWeights[2] = new float[256 * 3 * 3 * 3];
    nn::utils::heInitConv(decoderWeights[2], 3, 3, 256);

    decoderBiases[0] = new float[128];
    nn::utils::zeros(decoderBiases[0], 128);

    decoderBiases[1] = new float[256];
    nn::utils::zeros(decoderBiases[1], 256);

    decoderBiases[2] = new float[3];
    nn::utils::zeros(decoderBiases[2], 3);

    for (int i = 0; i < 2; i++) {
        encoderConvs[i]->setWeights(encoderWeights[i], encoderBiases[i]);
    }

    for (int i = 0; i < 3; i++) {
        decoderConvs[i]->setWeights(decoderWeights[i], decoderBiases[i]);
    }
}


// void AutoEncoder::encode(const float* inputBatch, float* latentBatch, int batchSize=32)
// {

// }

void AutoEncoder::forward(const float* inputBatch, float* outputBatch, int batchSize)
{
    // int inputH = this->inputHeight;
    int inputH;
    memcpy(&inputH, &this->inputHeight, sizeof(int));
    // int inputW = this->inputWidth;
    int inputW;
    memcpy(&inputW, &this->inputWidth, sizeof(int));

    printf("AutoEncoder forward pass with input size: %d x %d\n", inputW, inputH);

    float* temp = nullptr;

    printf("Encoder forward pass\n");

    delete[] this->encoderOutputs[0];
    this->encoderOutputs[0] = new float[batchSize * inputW * inputH * this->encoderConvs[0]->getOutChannels()];
    encoderConvs[0]->forward(inputBatch, this->encoderOutputs[0], inputW, inputH, batchSize);

    delete[] temp;
    temp = new float[batchSize * inputW * inputH * this->encoderConvs[0]->getOutChannels()];
    nn::activation::relu(this->encoderOutputs[0], temp, batchSize * inputW * inputH * this->encoderConvs[0]->getOutChannels());

    delete[] this->poolOutputs[0];
    this->poolOutputs[0] = new float[batchSize * (inputW/2) * (inputH/2) * this->encoderConvs[0]->getOutChannels()];
    pools[0]->forward(temp, this->poolOutputs[0], inputW, inputH, this->encoderConvs[0]->getOutChannels(), batchSize);

    inputH /= 2;
    inputW /= 2;

    delete[] this->encoderOutputs[1];
    this->encoderOutputs[1] = new float[batchSize * inputW * inputH * this->encoderConvs[1]->getOutChannels()];
    encoderConvs[1]->forward(this->poolOutputs[0], this->encoderOutputs[1], inputW, inputH, batchSize);

    delete[] temp;
    temp = new float[batchSize * inputW * inputH * this->encoderConvs[1]->getOutChannels()];
    nn::activation::relu(this->encoderOutputs[1], temp, batchSize * inputW * inputH * this->encoderConvs[1]->getOutChannels());

    delete[] this->poolOutputs[1];
    this->poolOutputs[1] = new float[batchSize * (inputW/2) * (inputH/2) * this->encoderConvs[1]->getOutChannels()];
    pools[1]->forward(temp, this->poolOutputs[1], inputW, inputH, this->encoderConvs[1]->getOutChannels(), batchSize);

    inputH /= 2;
    inputW /= 2;
    
    delete[] this->latentCache;
    this->latentCache = new float[batchSize * inputW * inputH * this->encoderConvs[1]->getOutChannels()];
    memcpy(this->latentCache, this->poolOutputs[1], sizeof(float) * batchSize * inputW * inputH * this->encoderConvs[1]->getOutChannels());

    // Decoder forward pass
    delete[] this->decoderOutputs[0];
    this->decoderOutputs[0] = new float[batchSize * inputW * inputH * this->decoderConvs[0]->getOutChannels()];
    this->decoderConvs[0]->forward(this->poolOutputs[1], this->decoderOutputs[0], inputW, inputH, batchSize);

    delete[] temp;
    temp = new float[batchSize * inputW * inputH * this->decoderConvs[0]->getOutChannels()];
    nn::activation::relu(this->decoderOutputs[0], temp, batchSize * inputW * inputH * this->decoderConvs[0]->getOutChannels());

    delete[] this->upsampleOutputs[0];
    this->upsampleOutputs[0] = new float[batchSize * (inputW*2) * (inputH*2) * this->decoderConvs[0]->getOutChannels()];
    upsamples[0]->forward(temp, this->upsampleOutputs[0], inputW, inputH, this->decoderConvs[0]->getOutChannels(), batchSize);

    inputH *= 2;
    inputW *= 2;

    delete[] this->decoderOutputs[1];
    this->decoderOutputs[1] = new float[batchSize * inputW * inputH * this->decoderConvs[1]->getOutChannels()];
    this->decoderConvs[1]->forward(this->upsampleOutputs[0], this->decoderOutputs[1], inputW, inputH, batchSize);
    
    delete[] temp;
    temp = new float[batchSize * inputW * inputH * this->decoderConvs[1]->getOutChannels()];
    nn::activation::relu(this->decoderOutputs[1], temp, batchSize * inputW * inputH * this->decoderConvs[1]->getOutChannels());

    delete[] this->upsampleOutputs[1];
    this->upsampleOutputs[1] = new float[batchSize * (inputW*2) * (inputH*2) * this->decoderConvs[1]->getOutChannels()];
    upsamples[1]->forward(temp, this->upsampleOutputs[1], inputW, inputH, this->decoderConvs[1]->getOutChannels(), batchSize);

    inputH *= 2;
    inputW *= 2;

    this->decoderConvs[2]->forward(this->upsampleOutputs[1], outputBatch, inputW, inputH, batchSize);
}

void AutoEncoder::backward(const float* inputBatch, const float* outputBatch, float* gradInput, int batchSize)
{

}

void AutoEncoder::train(const float* inputBatch, int batchSize, float learningRate)
{
    float* outputBatch = new float[batchSize * this->inputWidth * this->inputHeight * this->inputChannels];
    this->forward(inputBatch, outputBatch, batchSize);

    // float* gradOutput = new float[batchSize * this->inputWidth * this->inputHeight * this->inputChannels];
    // nn::utils::mse_loss_backward(outputBatch, inputBatch, gradOutput, batchSize * this->inputWidth * this->inputHeight * this->inputChannels);

    // this->backward(inputBatch, gradOutput, nullptr, batchSize);

    // this->updateWeights(learningRate);

    // delete[] outputBatch;
    // delete[] gradOutput;
}

