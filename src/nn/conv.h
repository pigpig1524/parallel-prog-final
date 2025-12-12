#include <iostream>


class Conv2D {
    private:
        int filterNum;
        int filterWidth;
        int inputChannels;

        int stride;
        int padding;

        // Filter dim : (filterNum, filterWidth, filterWidth, inputChannels)
        float * filters = nullptr;
        float * filtersGradients = nullptr;

        // Biases dim : (filterNum)
        float * biases = nullptr;
        float * biasesGradients = nullptr;

    public:
        // This section for constructor and destructor

        /**
         * Constructor for Conv2D layer.
         * @param filterNum Number of filters.
         * @param filterWidth Width and height of each filter (assumed square).
         * @param inputChannels Number of input channels.
         * @param stride Stride for the convolution. default is 1.
         * @param padding Padding size. default is 1.
         */
        Conv2D(int filterNum, int filterWidth, int inputChannels, int stride=1, int padding=1);

        /**
         * Destructor for Conv2D layer.
         */
        ~Conv2D();

        int getFilterNum() const { return this->filterNum; }
        int getFilterWidth() const { return this->filterWidth; }
        int getInputChannels() const { return this->inputChannels; }
        int getStride() const { return this->stride; }
        int getPadding() const { return this->padding; }
        float* getFilters() const { return this->filters; }
        float* getBiases() const { return this->biases; }
        float* getFiltersGradients() const { return this->filtersGradients; }
        float* getBiasesGradients() const { return this->biasesGradients; }

        // This section for forward and backward propagation

        /**
         * Sets the weights for filters and biases.
         * @param filterWeights Pointer to filter weights array.
         * @param biasWeights Pointer to bias weights array.
         */
        void setWeights(float * filterWeights, float * biasWeights);

        /**
         * Performs the convolution operation on a batch of input images.
         * @param inputBatch Pointer to input batch array.
         * @param outputBatch Pointer to output batch array.
         * @param inputWidth Width of the input images.
         * @param inputHeight Height of the input images.
         * @param batchSize Number of images in the batch.
         * 
         */
        void convolve(float * inputBatch, float * outputBatch, int inputWidth, int inputHeight, int batchSize);

        float* convolve(float * inputBatch, int inputWidth, int inputHeight, int batchSize);

        void backward(float * inputBatch, float * outputGradients, float * inputGradients, int inputWidth, int inputHeight, int batchSize);
};
