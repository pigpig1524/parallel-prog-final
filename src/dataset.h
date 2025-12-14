#ifndef DATASET_H
#define DATASET_H


#include <vector>
#include "models.h"


class Dataset {
    private:
        char * dataPath;
        int nTrain = 50000;
        int nTest = 10000;

        std::vector<unsigned int> shuffledIndices;

        // Hard code for convenience
        const char* labelNames[10] = {
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        };

        Split trainSplit;
        Split testSplit;

        Batch getTrainBatch(unsigned int batchSize, unsigned int batchIndex);
        Batch getTestBatch(unsigned int batchSize, unsigned int batchIndex);

        void loadTrain();
        void loadTest();

    public:
        Dataset(const char* dataPath);
        ~Dataset();

        void loadData();
        void shuffle();

        Split getTrainSplit() const { return this->trainSplit; }
        Split getTestSplit() const { return this->testSplit; }

        Batch getBatch(unsigned int batchSize, unsigned int batchIndex, bool isTrain = true);
};


#endif // DATASET_H