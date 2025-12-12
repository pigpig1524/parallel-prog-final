#include "nn/autoencoder.cpp"
#include "dataset.cpp"

int main()
{
    Dataset* dataset = new Dataset("../data/");
    dataset->loadData();

    AutoEncoder autoencoder(32, 32, 3, 8, 8, 128);
    autoencoder.initialize(32, 32, 3, 8, 8, 128);
    const unsigned int epochs = 1;
    const unsigned int batchSize = 32;
    const float learningRate = 0.001f;
    const unsigned int nBatches = 50000 / batchSize;
    for (unsigned int epoch = 0; epoch < epochs; epoch++) {
        printf("Epoch %d/%d,", epoch + 1, epochs);
        dataset->shuffle();
        float epochLoss = 0.0f;

        Batch* batches = dataset->getBatches(batchSize);

        for (unsigned int batchIndex = 0; batchIndex < nBatches; batchIndex++) {
            printf("Batch %d/%d\n", batchIndex + 1, nBatches);
            // Batch batch = dataset.getBatches(batchSize, batchIndex);
            Batch batch = batches[batchIndex];

            float* outputBatch = new float[batchSize * 32 * 32 * 3];
            autoencoder.forward(batch.images, outputBatch, batchSize);

            float batchLoss = 0.0f;
            nn::utils::calcMSE(outputBatch, batch.images, batchSize * 32 * 32 * 3, batchLoss);
            printf(" Batch %d/%d, Loss: %.6f\n", batchIndex + 1, nBatches, batchLoss);

            epochLoss += batchLoss;

            // autoencoder.train(batch.images, batchSize, learningRate);

            delete[] batch.images;
            delete[] batch.labels;
            delete outputBatch;
        }

        epochLoss /= nBatches;
        printf("Epoch %d/%d, Loss: %.6f\n", epoch + 1, epochs, epochLoss);
    }

    return 0;
}