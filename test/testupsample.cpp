#include "../src/nn/utils.cpp"

int main()
{
    float* fakeImage = NNUtils::randomFloatArray(2 * 3 * 2 * 2); // 3 channels, 4x4 image

    float* pooledOutput = new float [2 * 3 * 4 * 4];
    // float* pooledOutput = NNUtils::batchUpSampling(fakeImage, 2, 2, 3, 2, 2);
    NNUtils::batchUpSampling(
        fakeImage,
        pooledOutput,
        2,
        2,
        3,
        2,
        2
    );

    // Print source image for reference
    for (int imgIdx = 0; imgIdx < 2; imgIdx++) {
        printf("Image %d:\n", imgIdx);

        for (int c=0 ; c < 3; c++) {
            printf(" Channel %d:\n", c);
            printf(" Source Image:\n");
            for (int h = 0; h < 2; h++) {
                for (int w = 0; w < 2; w++) {
                    printf("%f ", fakeImage[
                        imgIdx * (2 * 2 * 3) + 
                        c * (2 * 2) + 
                        h * 2 + 
                        w
                    ]);
                }
                printf("\n");
            }

            // Print the pooled output for verification
            printf(" Upsampled Output:\n");
            for (int h = 0; h < 4; h++) {
                for (int w = 0; w < 4; w++) {
                    printf("%f ", pooledOutput[
                        imgIdx * (4 * 4 * 3) + 
                        c * (4 * 4) + 
                        h * 4 + 
                        w
                    ]);
                }
                printf("\n");
            }
            printf("\n");
        }    
    }
}