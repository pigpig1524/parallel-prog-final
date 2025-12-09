#include "nn/autoencoder.h"
#include <cmath>
#include <iostream>

int main(){
    Autoencoder ae = Autoencoder(0.001, 0.9);
    // ae.initWeights();
    ae.save_weights("../weights/test_weights.bin");
    return 0;
}