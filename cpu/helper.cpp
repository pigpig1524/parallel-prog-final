#include <iostream>

class Helper {
    public:
        static void printMessage(const std::string& message) {
            std::cout << message << std::endl;
        }
        
        /**
         * 
         * This function normalizes an image's pixels value from [0, 255] to float [0, 1].
         *
         * @param pixels Pointer to the input array of pixel values (unsigned char).
         * @param out Pointer to the output array where normalized float values will be stored.
         * @param size The number of pixels to normalize. Default is 3072.
         */
        static void normalizePixels(unsigned char * pixels, float* out, int size = 3072) {
            for (int i = 0; i < size; i++) {
                out[i] = static_cast<float>(pixels[i]) / 255.0f;
            }
        }
};