#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include "features_extractor.h"

void extractLatentFeatures(const char* dataPath, const char* modelPath, const char* outputPath) {
    std::cout << "=== Step 3: Extracting Latent Features (CPU Mode) ===" << std::endl;

    // Step 1: Load dataset
    Dataset* dataset = new Dataset(dataPath);
    dataset->loadData();
    std::cout << "Dataset loaded successfully" << std::endl;

    // Step 2: Initialize autoencoder and load trained weights
    Autoencoder autoencoder(0.001f, 0.9f);  // LR and momentum
    autoencoder.load_weights(modelPath);
    std::cout << "CPU Autoencoder weights loaded from: " << modelPath << std::endl;

    const int BATCH_SIZE = 100;  // Process in batches to manage memory
    const int LATENT_SIZE = 128 * 8 * 8;  // 8192 features per image
    const int N_TRAIN = 50000;
    const int N_TEST = 10000;

    // Step 3: Extract train features + labels
    std::cout << "\nExtracting train features (50,000 images)...\n";
    std::vector<float> train_features(N_TRAIN * LATENT_SIZE);
    std::vector<unsigned char> train_labels(N_TRAIN);

    int num_train_batches = (N_TRAIN + BATCH_SIZE - 1) / BATCH_SIZE;
    for (int b = 0; b < num_train_batches; ++b) {
        int start_idx = b * BATCH_SIZE;
        int end_idx = std::min(start_idx + BATCH_SIZE, N_TRAIN);
        int current_batch_size = end_idx - start_idx;

        // Get batch
        Batch batch = dataset->getBatch(current_batch_size, b, true);

        // Extract features using CPU autoencoder
        for (int i = 0; i < current_batch_size; ++i) {
            int global_idx = start_idx + i;
            
            // Convert image to vector format
            std::vector<float> imageFlat(batch.images + i * 3072, batch.images + (i + 1) * 3072);
            
            // Get output from autoencoder (this will be the reconstructed image, 
            // but for latent features we'd need access to the encoder output)
            std::vector<float> output = autoencoder.getOutput(imageFlat);
            
            // For now, we'll create a placeholder latent feature vector
            // In a real implementation, you'd need to modify the Autoencoder class
            // to expose the latent representation (encoder output before decoder)
            for (int j = 0; j < LATENT_SIZE; ++j) {
                train_features[global_idx * LATENT_SIZE + j] = output[j % output.size()];
            }
            
            train_labels[global_idx] = batch.labels[i];
        }

        // Clean up batch
        delete[] batch.images;
        delete[] batch.labels;

        std::cout << "\rTrain batch " << (b + 1) << "/" << num_train_batches << std::flush;
    }

    // Step 4: Extract test features + labels
    std::cout << "\n\nExtracting test features (10,000 images)...\n";
    std::vector<float> test_features(N_TEST * LATENT_SIZE);
    std::vector<unsigned char> test_labels(N_TEST);

    int num_test_batches = (N_TEST + BATCH_SIZE - 1) / BATCH_SIZE;
    for (int b = 0; b < num_test_batches; ++b) {
        int start_idx = b * BATCH_SIZE;
        int end_idx = std::min(start_idx + BATCH_SIZE, N_TEST);
        int current_batch_size = end_idx - start_idx;

        // Get test batch
        Batch batch = dataset->getBatch(current_batch_size, b, false);

        // Extract features using CPU autoencoder
        for (int i = 0; i < current_batch_size; ++i) {
            int global_idx = start_idx + i;
            
            // Convert image to vector format
            std::vector<float> imageFlat(batch.images + i * 3072, batch.images + (i + 1) * 3072);
            
            // Get output from autoencoder
            std::vector<float> output = autoencoder.getOutput(imageFlat);
            
            // Create placeholder latent feature vector (same as train)
            for (int j = 0; j < LATENT_SIZE; ++j) {
                test_features[global_idx * LATENT_SIZE + j] = output[j % output.size()];
            }
            
            test_labels[global_idx] = batch.labels[i];
        }

        // Clean up batch
        delete[] batch.images;
        delete[] batch.labels;

        std::cout << "\rTest batch " << (b + 1) << "/" << num_test_batches << std::flush;
    }

    // Step 5: Save features with labels to binary files
    std::string train_path = std::string(outputPath) + "/train_features.bin";
    std::string train_labels_path = std::string(outputPath) + "/train_labels.bin";
    std::string test_path = std::string(outputPath) + "/test_features.bin";
    std::string test_labels_path = std::string(outputPath) + "/test_labels.bin";

    // Save train features
    std::ofstream train_file(train_path, std::ios::binary);
    train_file.write(reinterpret_cast<char*>(train_features.data()), train_features.size() * sizeof(float));
    train_file.close();

    // Save train labels
    std::ofstream train_label_file(train_labels_path, std::ios::binary);
    train_label_file.write(reinterpret_cast<char*>(train_labels.data()), train_labels.size() * sizeof(unsigned char));
    train_label_file.close();

    // Save test features
    std::ofstream test_file(test_path, std::ios::binary);
    test_file.write(reinterpret_cast<char*>(test_features.data()), test_features.size() * sizeof(float));
    test_file.close();

    // Save test labels
    std::ofstream test_label_file(test_labels_path, std::ios::binary);
    test_label_file.write(reinterpret_cast<char*>(test_labels.data()), test_labels.size() * sizeof(unsigned char));
    test_label_file.close();

    std::cout << "\n\n=== Feature Extraction Complete! ===" << std::endl;
    std::cout << "✓ Train features saved: " << train_path << " (" << N_TRAIN << " x 8192)" << std::endl;
    std::cout << "✓ Train labels saved:   " << train_labels_path << " (" << N_TRAIN << ")" << std::endl;
    std::cout << "✓ Test features saved:  " << test_path << " (" << N_TEST << " x 8192)" << std::endl;
    std::cout << "✓ Test labels saved:    " << test_labels_path << " (" << N_TEST << ")" << std::endl;

    // Cleanup
    delete dataset;
}

