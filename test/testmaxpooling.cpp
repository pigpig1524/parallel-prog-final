#include "../src/nn/utils.cpp"

int main()
{

    float* fakeImage = NNUtils::randomFloatArray(2 * 3 * 4 * 4); // 3 channels, 4x4 image

    // float* pooledOutput = new float[3 * 2 * 2]; // After 2x2 pooling with stride 2, output size is 2x2

    // float* pooledOutput = NNUtils::maxPooling(fakeImage, 4, 4, 3);
    float* pooledOutput = new float [2 * 3 * 2 * 2];
    // float* pooledOutput = NNUtils::batchMaxPooling(fakeImage, p 4, 4, 3, 2, 2, 2);
    NNUtils::batchMaxPooling(
        fakeImage,
        pooledOutput,
        4,
        4,
        3,
        2,
        2,
        2
    );


    for (int imgIdx = 0; imgIdx < 2; imgIdx++) {
        printf("Image %d:\n", imgIdx);
        for (int c = 0; c < 3; c++) {
            printf(" Channel %d:\n", c);
            // Print source image for reference
            printf(" Source Image:\n");
            for (int h = 0; h < 4; h++) {
                for (int w = 0; w < 4; w++) {
                    printf("%f ", fakeImage[
                        imgIdx * (4 * 4 * 3) + 
                        c * (4 * 4) + 
                        h * 4 + 
                        w
                    ]);
                }
                printf("\n");
            }
            // Print the pooled output for verification
            printf(" Pooled Output:\n");
            for (int h = 0; h < 2; h++) {
                for (int w = 0; w < 2; w++) {
                    printf("%f ", pooledOutput[
                        imgIdx * (2 * 2 * 3) + 
                        c * (2 * 2) + 
                        h * 2 + 
                        w
                    ]);
                }
                printf("\n");
            }
            printf("\n");
        }
    }

    return 0;
}