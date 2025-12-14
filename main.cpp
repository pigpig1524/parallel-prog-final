// main.cpp
#include "src/features_extractor.h"
#include <iostream>

int main() {
    const char* dataPath = "data/cifar-10-batches-bin/";
    const char* modelPath = "weights/gpu_train.bin";
    const char* featuresPath = "data/train_latent_features.bin";

    extractLatentFeatures(dataPath, modelPath, featuresPath);
    std::cout << "Extraction completed!" << std::endl;
    return 0;
}
