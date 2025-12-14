#ifndef EXTRACT_FEATURES_H
#define EXTRACT_FEATURES_H

#include "dataset.cpp"
#include "nn/gpu_autoencoder.h"

void extractLatentFeatures(const char* dataPath, const char* modelPath, const char* outputPath);

#endif // EXTRACT_FEATURES_H