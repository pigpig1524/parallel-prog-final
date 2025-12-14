#include <random>
#include <algorithm>
#include <string>
#include <cstring>

#include "dataset.h"
#include "helper.cpp"


Dataset::Dataset(const char* dataPath) {
    this->dataPath = new char[strlen(dataPath) + 1];
    strcpy(this->dataPath, dataPath);
}


Dataset::~Dataset() {
    delete[] this->dataPath;
    for (int i = 0; i < this->nTrain; i++) {
        delete[] this->trainSplit.images[i];
    }
    delete[] this->trainSplit.images;
    delete[] this->trainSplit.labels;
}


void Dataset::loadTrain() {
    this->trainSplit.labels = new unsigned char[this->nTrain];
    this->trainSplit.images = new float * [this->nTrain];

    for (int batchIndex = 1; batchIndex <= 5; batchIndex++) {
        std::string trainFilePath = std::string(this->dataPath) + "data_batch_" + std::to_string(batchIndex) + ".bin";
        FILE * file = fopen(trainFilePath.c_str(), "rb");
        if (!file) {
            continue;
        }

        for (int i = 0; i < 10000; i++) {
            int globalIndex = (batchIndex - 1) * 10000 + i;

            this->trainSplit.images[globalIndex] = new float[3072];

            unsigned char buffer[3072];

            fread(&this->trainSplit.labels[globalIndex], sizeof(unsigned char), 1, file);
            fread(buffer, sizeof(unsigned char), 3072, file);

            // Normalize pixels from [0, 255] to [0, 1]
            Helper::normalizePixels(buffer, this->trainSplit.images[globalIndex], 3072);
        }
        fclose(file);
    }
}

/*
* Bring in the test data from the binary file
*
*/
void Dataset::loadTest() {
    std::string testFilePath = std::string(this->dataPath) + "test_batch.bin";
    FILE * file = fopen(testFilePath.c_str(), "rb");
    if (!file) {
        return;
    }

    this->testSplit.labels = new unsigned char[this->nTest];
    this->testSplit.images = new float * [this->nTest];

    unsigned char buffer[3072];

    for (int i = 0; i < this->nTest; i++) {
        this->testSplit.images[i] = new float[32 * 32 * 3];

        fread(&this->testSplit.labels[i], sizeof(unsigned char), 1, file);
        fread(buffer, sizeof(unsigned char), 32 * 32 * 3, file);

        // Normalize pixels from [0, 255] to [0, 1]
        Helper::normalizePixels(buffer, this->testSplit.images[i], 3072);
    }

    fclose(file);
}


void Dataset::loadData() {
    this->loadTrain();
    this->loadTest();
}


void Dataset::shuffle() {
    this->shuffledIndices.resize(this->nTrain);
    for (unsigned int i = 0; i < this->nTrain; i++) {
        this->shuffledIndices[i] = i;
    }
    std::random_shuffle(this->shuffledIndices.begin(), this->shuffledIndices.end());
}


/** 
 * @brief Generate a batch of training data
 * 
 * Note: This function assumes that shuffle() has been called beforehand
 * @param batchSize Size of the batch
 * @param batchIndex Index of the batch
 * @return Batch containing images and labels
*/
Batch Dataset::getTrainBatch(unsigned int batchSize, unsigned int batchIndex) {
    // Initialize batch
    Batch batch;
    batch.images = new float[batchSize * 3072];
    batch.labels = new unsigned char[batchSize];

    // If shuffledIndices is not initialized, initialize it
    // Default to sequential order (no shuffling)
    if (this->shuffledIndices.size() != this->nTrain) {
        this->shuffledIndices.resize(this->nTrain);
        for (unsigned int i = 0; i < this->nTrain; i++) {
            this->shuffledIndices[i] = i;
        }
    }

    // Fill batch with data
    for (unsigned int i = 0; i < batchSize; i++) {
        // Get the actual index from shuffled indices
        unsigned int index = shuffledIndices[batchIndex * batchSize + i];
        // Copy label
        batch.labels[i] = this->trainSplit.labels[index];
        // Copy image
        std::memcpy(&batch.images[i * 3072], this->trainSplit.images[index], 3072 * sizeof(float));
    }

    return batch;
}


Batch Dataset::getTestBatch(unsigned int batchSize, unsigned int batchIndex) {
    Batch batch;
    batch.images = new float[batchSize * 3072];
    batch.labels = new unsigned char[batchSize];

    for (unsigned int i = 0; i < batchSize; i++) {
        unsigned int index = batchIndex * batchSize + i;
        batch.labels[i] = this->testSplit.labels[index];
        std::memcpy(&batch.images[i * 3072], this->testSplit.images[index], 3072 * sizeof(float));
    }
    return batch;
}


/**
 * @brief Generate a batch of data (train or test)
 * @param batchSize Size of the batch
 * @param batchIndex Index of the batch
 * @param isTrain Boolean indicating whether to get training data or test data (default: true)
 * @return Batch containing images and labels
 */
Batch Dataset::getBatch(unsigned int batchSize, unsigned int batchIndex, bool isTrain) {
    if (isTrain) {
        return this->getTrainBatch(batchSize, batchIndex);
    } else {
        return this->getTestBatch(batchSize, batchIndex);
    }
}