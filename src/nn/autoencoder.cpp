#include "autoencoder.h"


Autoencoder::Autoencoder(Dataset * dataset) :
    dataset(dataset)
{
    this->encoderWeights = nullptr;
    this->decoderWeights = nullptr;

    this->encoderLayers = std::vector<Conv2D>();
    this->decoderLayers = std::vector<Conv2D>();

    // Initialize weights (randomly for now)
    // Encoder weights
    this->encoderLayers.push_back(Conv2D(256, 3, 3));
    this->encoderLayers.push_back(Conv2D(128, 3, 256));
    encoderWeights = new float*[this->encoderLayers.size()];
    encoderBiases = new float*[this->encoderLayers.size()];
    for (int i = 0; i < this->encoderLayers.size(); i++) {
        auto & layer = this->encoderLayers[i];
        int filterSize = layer.getFilterNum() * layer.getFilterWidth() * layer.getFilterWidth() * layer.getInputChannels();


        // encoderWeights[i] = NNUtils::randomFloatArray(filterSize);
        // encoderBiases[i] = NNUtils::randomFloatArray(layer.getFilterNum());
        encoderWeights[i] = NNUtils::heInitConv(
            layer.getFilterNum(),
            layer.getFilterWidth(),
            layer.getInputChannels()
        );
        encoderBiases[i] = NNUtils::zeroBias(layer.getFilterNum());

        layer.setWeights(encoderWeights[i], encoderBiases[i]);
        printf("First encoder layer weights sample: %f\n", encoderWeights[i][0]);
    }
    // Decoder weights
    this->decoderLayers.push_back(Conv2D(128, 3, 128));
    this->decoderLayers.push_back(Conv2D(256, 3, 128));
    this->decoderLayers.push_back(Conv2D(3, 3, 256));

    decoderWeights = new float*[this->decoderLayers.size()];
    decoderBiases = new float*[this->decoderLayers.size()];
    for (int i = 0; i < this->decoderLayers.size(); i++) {
        auto & layer = this->decoderLayers[i];
        int filterSize = layer.getFilterNum() * layer.getFilterWidth() * layer.getFilterWidth() * layer.getInputChannels();

        // decoderWeights[i] = NNUtils::randomFloatArray(filterSize);
        // decoderBiases[i] = NNUtils::randomFloatArray(layer.getFilterNum());
        decoderWeights[i] = NNUtils::heInitConv(
            layer.getFilterNum(),
            layer.getFilterWidth(),
            layer.getInputChannels()
        );
        decoderBiases[i] = NNUtils::zeroBias(layer.getFilterNum());

        layer.setWeights(decoderWeights[i], decoderBiases[i]);
    }


}

Autoencoder::~Autoencoder()
{
    // Free encoder weights
    for (int i = 0; i < this->encoderLayers.size(); i++) {
        delete[] encoderWeights[i];
        delete[] encoderBiases[i];
    }
    delete[] encoderWeights;
    delete[] encoderBiases;

    // Free decoder weights
    for (int i = 0; i < this->decoderLayers.size(); i++) {
        delete[] decoderWeights[i];
        delete[] decoderBiases[i];
    }
    delete[] decoderWeights;
    delete[] decoderBiases;

    // Free latent and output values
    delete[] this->inputValue;
    delete[] this->latentValue;
    delete[] this->outputValue;
}

void Autoencoder::applyCurrentWeights()
{
    for (int i = 0; i < this->encoderLayers.size(); i++) {
        this->encoderLayers[i].setWeights(encoderWeights[i], encoderBiases[i]);
    }
    for (int i = 0; i < this->decoderLayers.size(); i++) {
        this->decoderLayers[i].setWeights(decoderWeights[i], decoderBiases[i]);
    }
}

void Autoencoder::updateWeights(float learningRate) {


    for (auto &layer : encoderLayers) {
        int filterSize = layer.getFilterNum() * layer.getFilterWidth() * layer.getFilterWidth() * layer.getInputChannels();
        float lr = learningRate;

        float* layerFilters = layer.getFilters();
        float* layerBiases = layer.getBiases();
        float* layerFiltersGradients = layer.getFiltersGradients();
        float* layerBiasesGradients = layer.getBiasesGradients();

        printf("Encoder layer 0 grad sample: %f\n", layerFiltersGradients[0]);
        // printf("Decoder layer 0 grad sample: %f\n", currentGrad[0]);
    
        for (int i = 0; i < filterSize; i++)
            layerFilters[i] -= lr * layerFiltersGradients[i];
        for (int i = 0; i < layer.getFilterNum(); i++)
            layerBiases[i] -= lr * layerBiasesGradients[i];
    }

    for (auto &layer : decoderLayers) {
        int filterSize = layer.getFilterNum() * layer.getFilterWidth() * layer.getFilterWidth() * layer.getInputChannels();
        float lr = learningRate;
        float* layerFilters = layer.getFilters();
        float* layerBiases = layer.getBiases();
        float* layerFiltersGradients = layer.getFiltersGradients();
        float* layerBiasesGradients = layer.getBiasesGradients();
        for (int i = 0; i < filterSize; i++)
            layerFilters[i] -= lr * layerFiltersGradients[i];
        for (int i = 0; i < layer.getFilterNum(); i++)
            layerBiases[i] -= lr * layerBiasesGradients[i];
    }
}

void Autoencoder::feedforward(float* inputBatch, int batchSize)
{
    int inputWidth = 32;  // Assuming input width is 32
    int inputHeight = 32; // Assuming input height is 32
    float* output = new float[batchSize * inputWidth * inputHeight * 3];
    hiddenEncoderValue = new float*[2];
    hiddenDecoderValue = new float*[2];
    maxPoolValue = new float*[2];
    upSampleValue = new float*[2];


    memcpy(output, inputBatch, batchSize * inputWidth * inputHeight * 3 * sizeof(float));

    /*=== ENCODER ===*/
    printf("Starting ecoder:\n");
    for (auto & layer : this->encoderLayers) {
        // Step 1. Convolution
        // Step 1a. Calculate convolution dimensions
        printf("Encoder Layer: Convolution with %d filters of size %dx%d\n",
            layer.getFilterNum(),
            layer.getFilterWidth(),
            layer.getFilterWidth()
        );

        int convW = (inputWidth - layer.getFilterWidth() + 2 * layer.getPadding()) / layer.getStride() + 1;
        int convH = (inputHeight - layer.getFilterWidth() + 2 * layer.getPadding()) / layer.getStride() + 1;
        // Step 1b. Allocate convolution output
        float* convOutput = new float[batchSize * convW * convH * layer.getFilterNum()];

        // Step 1c. Call convolution
        layer.convolve(
            output,
            convOutput,
            inputWidth, inputHeight,
            batchSize
        );
        this->hiddenEncoderValue[&layer - &this->encoderLayers[0]] = new float[batchSize * convW * convH * layer.getFilterNum()];
        memcpy(
            this->hiddenEncoderValue[&layer - &this->encoderLayers[0]],
            convOutput,
            batchSize * convW * convH * layer.getFilterNum() * sizeof(float)
        );

        ActivationFunctions::relu(convOutput, batchSize * convW * convH * layer.getFilterNum());

        // Step 2. Max Pooling
        // Step 2a. Calulate pooling output dimensions
        int poolW = convW / 2;
        int poolH = convH / 2;
        // Step 2b. Allocate pooled output
        float* pooledOutput = new float[batchSize * poolW * poolH * layer.getFilterNum()];

        // Step 2c. Call max pooling
        NNUtils::batchMaxPooling(
            convOutput, pooledOutput,
            convW, convH,
            layer.getFilterNum(),
            2, 2,
            batchSize
        );
        this->maxPoolValue[&layer - &this->encoderLayers[0]] = new float[batchSize * poolW * poolH * layer.getFilterNum()];

        // Step 3. Update output pointer
        delete[] output; // Free previous output
        output = pooledOutput;

        // Step 4. Update input dimensions for the next layer
        inputWidth = poolW;
        inputHeight = poolH;

        printf("Encoder Layer: Output dimensions: %dx%dx%d\n",
            inputWidth,
            inputHeight,
            layer.getFilterNum()
        );

        // Step 5. Free intermediate convolution output
        delete[] convOutput;
    }

    // Store latent representation
    if (this->latentValue != nullptr) {
        delete[] this->latentValue;
    }
    // this->latentValue = output;
    int latentSize = inputWidth * inputHeight * this->encoderLayers.back().getFilterNum() * batchSize;
    this->latentValue = new float[latentSize];
    memcpy(this->latentValue, output, latentSize * sizeof(float));

    // === DECODER ===
    printf("Starting decoder:\n");
    for (auto & layer : this->decoderLayers) {
        // Step 1. Calculate convolution dimensions
        printf("Decoder Layer: Convolution with %d filters of size %dx%d\n",
            layer.getFilterNum(),
            layer.getFilterWidth(),
            layer.getFilterWidth()
        );

        int convW = (inputWidth - layer.getFilterWidth() + 2 * layer.getPadding()) / layer.getStride() + 1;
        int convH = (inputHeight - layer.getFilterWidth() + 2 * layer.getPadding()) / layer.getStride() + 1;

        float* convOutput = new float[batchSize * convW * convH * layer.getFilterNum()];

        layer.convolve(output, convOutput, inputWidth, inputHeight, batchSize);
        this->hiddenDecoderValue[&layer - &this->decoderLayers[0]] = new float[batchSize * convW * convH * layer.getFilterNum()];
        memcpy(
            this->hiddenDecoderValue[&layer - &this->decoderLayers[0]],
            convOutput,
            batchSize * convW * convH * layer.getFilterNum() * sizeof(float)
        );

        if (&layer == &this->decoderLayers.back()) {
            // If it's the last layer, no activation
            delete[] output; // Free previous output
            output = convOutput;
            continue;
        }

        ActivationFunctions::relu(convOutput, batchSize * convW * convH * layer.getFilterNum());

        // Step 2. Up-sampling
        int upW = convW * 2;
        int upH = convH * 2;

        float* upSampledOutput = new float[batchSize * upW * upH * layer.getFilterNum()];

        printf("Decoder Layer: Up-sampling from %dx%d to %dx%d\n",
            convW, convH,
            upW, upH
        );
        NNUtils::batchUpSampling(
            convOutput, upSampledOutput,
            convW, convH,
            layer.getFilterNum(),
            2,
            batchSize
        );
        this->upSampleValue[&layer - &this->decoderLayers[0]] = new float[batchSize * upW * upH * layer.getFilterNum()];

        // Step 3. Update output pointer
        delete[] output; // Free previous output
        output = upSampledOutput;

        // Step 4. Update input dimensions for the next layer
        inputWidth = upW;
        inputHeight = upH;

        // Step 5. Free intermediate convolution output
        delete[] convOutput;

        printf("Decoder Layer: Output dimensions: %dx%dx%d\n",
            inputWidth,
            inputHeight,
            layer.getFilterNum()
        );
    }

    // Store final output
    if (this->outputValue != nullptr) {
        delete[] this->outputValue;
    }
    int outputSize = inputWidth * inputHeight * this->decoderLayers.back().getFilterNum() * batchSize;
    this->outputValue = new float[outputSize];
    memcpy(this->outputValue, output, outputSize * sizeof(float));

    // Free the last output
    delete[] output;
}

// void Autoencoder::backpropagate(int batchSize, float learningRate)
// {
//     // Backpropagation logic to be implemented
//     // Step 1. Compute gradients at output layer
//     int inputWidth = 32;
//     int inputHeight = 32;

//     int outputSize = batchSize * inputWidth * inputHeight * 3;

//     // === 1. Gradient at output (loss gradient) ===
//     float* outputGradients = NNUtils::zeroBias(outputSize);
//     LossFunctions::mse_loss_backward(outputValue, inputValue, outputGradients, outputSize);

//     // === 2. Backprop qua decoder ===
//     float* currentGrad = outputGradients;

//     for (int i = decoderLayers.size() - 1; i >= 0; i--) {
//         Conv2D& layer = decoderLayers[i];

//         // ⭐ LẤY input forward của decoder (KHÔNG XOÁ)
//         float* layerInput = (i == 0)
//             ? latentValue                    // input layer đầu tiên của decoder
//             : upSampleValue[i-1];            // output forward của upsample trước đó

//         // ⭐ allocate input grad của layer này
//         int inC = layer.getInputChannels();
//         int inSize = batchSize * inputWidth * inputHeight * inC;
//         float* newGrad = NNUtils::zeroBias(inSize);

//         printf("Backpropagating decoder layer %d\n", i);

//         // 1. Backprop qua ReLU
//         if (i != decoderLayers.size() - 1) { // không phải layer cuối
//             for (int j = 0; j < inSize; j++) {
//                 currentGrad[j] *= (layerInput[j] > 0 ? 1.0f : 0.0f);
//             }
//         }

//         layer.backward(
//             layerInput,      // ⭐ input forward
//             currentGrad,     // dL/dOut
//             newGrad,         // dL/dIn
//             inputWidth,
//             inputHeight,
//             batchSize
//         );

//         delete[] currentGrad;
//         currentGrad = newGrad;

//         // === ⭐ Backprop qua Upsampling (nếu không phải lớp cuối) ===
//         if (i < decoderLayers.size() - 1) {
//             printf("Backpropagating through upsampling\n");

//             int nextW = inputWidth / 2;   // width trước conv
//             int nextH = inputHeight / 2;

//             int upSize = batchSize * nextW * nextH * inC;
//             float* upSampleGrad = NNUtils::zeroBias(upSize);

//             // ⭐ dùng đúng upsample backward
//             NNUtils::batchUpSamplingBackward(
//                 currentGrad,
//                 upSampleGrad,
//                 nextW, nextH,
//                 inC,
//                 2,                // scale
//                 batchSize
//             );

//             delete[] currentGrad;
//             currentGrad = upSampleGrad;

//             // ⭐ restore lại input spatial size
//             inputWidth  = nextW;
//             inputHeight = nextH;
//         }

//         // ⭐ Nếu còn layer phía trước => kích thước input của layer trước lớn hơn
//         else {
//             // layer cuối decoder => conv output nhỏ hơn latent mapping
//             inputWidth  /= 2;
//             inputHeight /= 2;
//         }

//     }

//     float* latentGrad = currentGrad;

//     // === 3. Backprop qua encoder (ngược lại) ===
//     currentGrad = latentGrad;
//     for (int i = encoderLayers.size() - 1; i >= 0; --i) {
//         Conv2D& layer = encoderLayers[i];

//         // --- 1. MaxPool backward
//         int pooledW = inputWidth;   // spatial size hiện tại (sau maxpool)
//         int pooledH = inputHeight;
//         int channels = layer.getFilterNum();

//         int beforeW = pooledW * 2;  // size trước maxpool (stride=2)
//         int beforeH = pooledH * 2;

//         size_t poolGradSize = (size_t)batchSize * beforeW * beforeH * channels;
//         float* poolGrad = NNUtils::zeroBias((int)poolGradSize);

//         // float* poolInput = (i == 0) ? inputValue : maxPoolValue[i - 1];
//         float* poolInput = hiddenEncoderValue[i];

//         NNUtils::backwardMaxPoolBatch(
//             poolInput,          // input forward
//             currentGrad,        // gradient từ output maxpool
//             poolGrad,           // gradient wrt input maxpool
//             beforeW,
//             beforeH,
//             channels,
//             batchSize,
//             2,                  // poolSize
//             2                   // stride
//         );

//         delete[] currentGrad;
//         currentGrad = poolGrad;



//         // restore input spatial size
//         inputWidth = beforeW;
//         inputHeight = beforeH;

//         // --- 2. Conv backward
//         float* layerInput = (i == 0) ? inputValue : maxPoolValue[i - 1];
//         int inC = layer.getInputChannels();
//         size_t inSize = (size_t)batchSize * inputWidth * inputHeight * inC;
//         float* newGrad = NNUtils::zeroBias((int)inSize);

//         // 2. Backprop qua ReLU
//         for (int j = 0; j < inSize; j++) {
//             currentGrad[j] *= (layerInput[j] > 0 ? 1.0f : 0.0f);
//         }

//         layer.backward(
//             layerInput,
//             currentGrad,  // dL/dOut
//             newGrad,      // dL/dIn
//             inputWidth,
//             inputHeight,
//             batchSize
//         );

//         delete[] currentGrad;
//         currentGrad = newGrad;
//     }

//     // currentGrad là gradient wrt input gốc nếu cần
//     delete[] currentGrad;
//     currentGrad = nullptr;
// }


void Autoencoder::backpropagate(int batchSize, float learningRate)
{
    // Backpropagation logic to be implemented
    // Step 1. Compute gradients at output layer
    int inputWidth = 32;
    int inputHeight = 32;

    int outputSize = batchSize * inputWidth * inputHeight * 3;

    // === 1. Gradient at output (loss gradient) ===
    float* outputGradients = NNUtils::zeroBias(outputSize);
    LossFunctions::mse_loss_backward(outputValue, inputValue, outputGradients, outputSize);

    // DECODER BACKWARD
    float* currentGrad = outputGradients;
    for (int i = decoderLayers.size() - 1; i >= 0; i--) {
        Conv2D& layer = decoderLayers[i];
        
        // Backward convolution with relu (except last layer)
        float* layerInput = (i == 0)
            ? latentValue
            : upSampleValue[i - 1];
        
        // Allocate input gradients
        int inC = layer.getInputChannels();
        int inSize = batchSize * inputWidth * inputHeight * inC;
        float* newGrad = NNUtils::zeroBias(inSize);
        printf("Backpropagating decoder layer %d\n", i);
        // Backprop through ReLU
        if (i != decoderLayers.size() - 1) {
            for (int j = 0; j < inSize; j++) {
                currentGrad[j] *= (layerInput[j] > 0 ? 1.0f : 0.0f);
            }
        }
        layer.backward(layerInput, currentGrad, newGrad, inputWidth, inputHeight, batchSize);

        delete[] currentGrad;
        currentGrad = newGrad;

        // Backprop through upsampling (if not first layer)
        
    }

    delete[] outputGradients;
}


void Autoencoder::train(int epochs, int batchSize, float learningRate)
{
    // Training loop
    for (int epoch = 0; epoch < epochs; epoch++) {
        printf("Starting epoch %d/%d\n", epoch + 1, epochs);

        // Shuffle dataset at the beginning of each epoch
        dataset->shuffle();

        // Get batches
        Batch* batches = dataset->getBatches(batchSize, true);

        // Iterate over all batches
        for (int batchIndex = 0; batchIndex < 50000 / batchSize; batchIndex++) {
            printf("Processing batch %d/%d\n", batchIndex + 1, 50000 / batchSize);

            Batch batch = batches[batchIndex];
            int inputSize = batchSize * 32 * 32 * 3;

            // Set input value
            if (this->inputValue != nullptr) {
                delete[] this->inputValue;
            }
            inputValue = new float[inputSize];
            memcpy(inputValue, batch.images, inputSize * sizeof(float));


            // Feedforward
            this->feedforward(batch.images, 32);

            // Compute loss (to be implemented)
            float loss = LossFunctions::meanSquaredError(this->outputValue, batch.images, batchSize * 32 * 32 * 3);
            printf("Epoch %d, Batch %d, Loss: %f\n", epoch + 1, batchIndex + 1, loss);
            this->errorHistory.push_back(loss);

            // Backpropagate
            printf("Backpropagating...\n");
            this->backpropagate(32);

            // Update weights
            printf("Updating weights...\n");
            this->updateWeights(learningRate);
        }

        float avgError = 0;
        for (float e : this->errorHistory) {
            avgError += e;
        }
        avgError /= this->errorHistory.size();
        printf("Epoch %d completed. Average Loss: %f\n", epoch + 1, avgError);

        // Clear error history for next epoch
        this->errorHistory.clear();
    }

}

