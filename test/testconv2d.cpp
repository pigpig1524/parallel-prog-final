#include <iostream>
#include "../src/nn/conv.cpp"


void printImageChannel(float* input, int height, int width, int channel) {
    std::cout << "Channel " << channel << ":\n";
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            std::cout << input[channel * height * width + h * width + w] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    float * input = new float[2 * 4 * 4 * 3]{
        // Image 1
        0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
        0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
        0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
        // Image 2 
        0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
        0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
        0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3
    };

    // for (int c = 0; c < 3; ++c) {
    //     printImageChannel(input, 4, 4, c);
    // }

    for (int imgIdx = 0; imgIdx < 2; ++imgIdx) {
        std::cout << "\nImage " << imgIdx + 1 << ":\n";
        for (int c = 0; c < 3; ++c) {
            printImageChannel(&input[imgIdx * 4 * 4 * 3], 4, 4, c);
        }
    }

    float * filterWeights = new float[1 * 3 * 3 * 3]{
        1, 0, -1, 0, 1, 0, -1, 0, 1,
        1, 0, -1, 0, 1, 0, -1, 0, 1,
        1, 0, -1, 0, 1, 0, -1, 0, 1
    };

    printf("\nFilter Weights:\n");
    for (int c = 0; c < 1; ++c) {
        printImageChannel(filterWeights, 3, 3, c);
    }

    float * biasWeights = new float[1]{1};

    Conv2D convLayer(1, 3, 3);
    convLayer.setWeights(filterWeights, biasWeights);
    float * output = new float[2 * 4 * 4 * 2]; // batchSize=2, outputWidth=4, outputHeight=4, filterNum=2
    convLayer.convolve(input, output, 4, 4, 2);

    for (int imgIdx = 0; imgIdx < 2; ++imgIdx) {
        std::cout << "\nOutput Image " << imgIdx + 1 << ":\n";
        for (int c = 0; c < 1; ++c)
            printImageChannel(&output[imgIdx * 4 * 4 * 1], 4, 4, c);
    }

    delete[] input;
    delete[] filterWeights;
    delete[] biasWeights;
    delete[] output;

    return 0;
}
