#include <iostream>
#include <vector>
#include <fstream>

#include "features_extractor.h"
#include "dataset.h" 

void extractLatentFeatures(const char* dataPath, const char* modelPath, const char* outputPath) {
    Dataset dataset(dataPath);
    
    
    /*
    dataset.loadData();
    dataset.shuffle();

    GPUAutoencoder autoencoder(1,1);
    autoencoder.load_weights(modelPath);*/

    //const int batchSize = 128;
    //const int nTrain = 50000;  // CIFAR-10 train size
    //const int nBatches = (nTrain + batchSize - 1) / batchSize;

    //std::cout << "Extracting features from " << nTrain << " samples..." << std::endl;

    //std::ofstream outFile(outputPath, std::ios::binary);
    //if (!outFile.is_open()) {
    //    std::cerr << "Error: Cannot create " << outputPath << std::endl;
    //    return;
    //}

    //std::vector<float> inputBatch, latentBatch;

    //for (int batchIdx = 0; batchIdx < nBatches; ++batchIdx) {
    //    Batch batch = dataset.getTrainBatch(batchSize, batchIdx);

    //    // Flatten batch for autoencoder (CIFAR-10: 32x32x3 = 3072)
    //    inputBatch.resize(batchSize * 3072);
    //    for (int i = 0; i < batchSize; ++i) {
    //        std::memcpy(&inputBatch[i * 3072],
    //            &batch.images[i * 3072],
    //            3072 * sizeof(float));
    //    }

    //    // Extract latent features
    //    autoencoder.getLatent(inputBatch, latentBatch, batchSize);

    //    // Write latent features to binary file
    //    int actualSize = std::min(batchSize, nTrain - batchIdx * batchSize);
    //    outFile.write(reinterpret_cast<const char*>(latentBatch.data()),
    //        latentBatch.size() * sizeof(float));

    //    std::cout << "Batch " << (batchIdx + 1) << "/" << nBatches << std::endl;

    //    // Cleanup batch memory
    //    delete[] batch.images;
    //    delete[] batch.labels;
    //}

    //outFile.close();
    std::cout << "Features saved: " << outputPath << std::endl;
}
