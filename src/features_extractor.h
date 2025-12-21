#ifndef EXTRACT_FEATURES_H
#define EXTRACT_FEATURES_H

#include "dataset.h"
#include "nn/autoencoder.h"

void extractLatentFeatures(const char* dataPath, const char* modelPath, const char* outputPath);

#endif // EXTRACT_FEATURES_H