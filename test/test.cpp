#include <iostream>
#include "../src/nn/autoencoder.cpp"

int main()
{
    std::cout << "Running Autoencoder Test..." << std::endl;

    Dataset* dataset = new Dataset("../data/");
    dataset->loadData();

    Autoencoder* autoencoder = new Autoencoder(dataset);

    autoencoder->train(1, 32);

    return 0;
}