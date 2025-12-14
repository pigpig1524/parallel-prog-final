#include "gpu_autoencoder.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <algorithm>
#include <iomanip>
// #include <cuda_runtime.h>

// Hàm đọc một file binary CIFAR-10 và trả về vector<pair<label, image_pixels>>
std::vector<std::pair<int, std::vector<float>>> readBinaryFile(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }

    // CIFAR-10 format: mỗi image = 1 byte label + 3072 bytes pixel data
    const int IMAGE_SIZE = 3072; // 32 * 32 * 3 (RGB)
    const int RECORD_SIZE = 1 + IMAGE_SIZE; // 1 byte label + 3072 bytes pixels
    
    // Tính số lượng images trong file
    file.seekg(0, std::ios::end);
    std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    
    if (fileSize % RECORD_SIZE != 0) {
        throw std::runtime_error("Invalid file size for CIFAR-10 format: " + filepath);
    }
    
    int numImages = fileSize / RECORD_SIZE;
    std::vector<std::pair<int, std::vector<float>>> imageData;
    imageData.reserve(numImages);
    
    for (int i = 0; i < numImages; i++) {
        // Đọc label (1 byte)
        unsigned char label;
        file.read(reinterpret_cast<char*>(&label), 1);
        
        // Đọc pixel data (3072 bytes)
        std::vector<unsigned char> pixelBytes(IMAGE_SIZE);
        file.read(reinterpret_cast<char*>(pixelBytes.data()), IMAGE_SIZE);
        
        if (!file) {
            throw std::runtime_error("Error reading image " + std::to_string(i) + " from file: " + filepath);
        }
        
        // Convert unsigned char (0-255) to float (0.0-1.0) for GPU
        std::vector<float> pixels;
        pixels.reserve(IMAGE_SIZE);
        for (unsigned char pixel : pixelBytes) {
            pixels.push_back(static_cast<float>(pixel) / 255.0f);
        }
        
        // Add to result as pair<label, pixels>
        imageData.emplace_back(static_cast<int>(label), std::move(pixels));
    }
    
    return imageData;
}

// Hàm đọc tất cả file .bin trong folder và load vào train_data
std::vector<std::vector<float>> loadData(const std::string& folderPath, const std::string& type="train") {
    std::vector<std::vector<float>> train_data;
    printf("== Loading %s data...\n", type.c_str());
    try {
        // Kiểm tra folder có tồn tại không
        if (!std::filesystem::exists(folderPath)) {
            throw std::runtime_error("Folder does not exist: " + folderPath);
        }
        
        // Lấy danh sách tất cả file .bin trong folder
        std::vector<std::string> allFiles;
        for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
            if (entry.is_regular_file() && entry.path().extension() == ".bin") {
                allFiles.push_back(entry.path().string());
            }
        }
        std::vector<std::string> binFiles;

        for (const auto& filepath : allFiles) {
            std::string filename = std::filesystem::path(filepath).filename().string();
            if (type == "train" && filename.find("data_batch_1") != std::string::npos) {
                binFiles.push_back(filepath);
            } else if (type == "test" && filename.find("test_batch") != std::string::npos) {
                binFiles.push_back(filepath);
            }
        }
        // Sắp xếp file theo tên để đảm bảo thứ tự
        std::sort(binFiles.begin(), binFiles.end());
        
        std::cout << "Found " << binFiles.size() << " binary files" << std::endl;
        
        // Đọc từng file
        for (const auto& filepath : binFiles) {
            try {
                auto imageData = readBinaryFile(filepath);
                for (const auto& imagePair : imageData) {
                    train_data.push_back(imagePair.second); // Extract only pixels
                }
                
            } catch (const std::exception& e) {
                std::cerr << "Error loading file " << filepath << ": " << e.what() << std::endl;
            }
        }
    
    } catch (const std::exception& e) {
        std::cerr << "Error loading train data: " << e.what() << std::endl;
    }
    
    std::cout << "Total images loaded: " << train_data.size() << std::endl;
    
    return train_data;
}

// Function to export vector<float> to binary file
void exportVectorToBinary(const std::vector<float>& data, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    
    // Write the size of the vector first (optional, for reading back later)
    size_t size = data.size();
    // file.write(reinterpret_cast<const char*>(&size), sizeof(size));
    
    // Write the vector data
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    
    if (!file.good()) {
        throw std::runtime_error("Error writing to file: " + filename);
    }
    
    file.close();
    std::cout << "Successfully exported " << data.size() << " floats to: " << filename << std::endl;
}

int main() {
    // Đường dẫn đến folder chứa các file batch
    std::string dataFolder = "../data/cifar-10-binary/cifar-10-batches-bin";
    std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;
    std::cout << "Looking for data at: " << std::filesystem::absolute(dataFolder) << std::endl;
    
    std::vector<std::vector<float>> train_data = loadData(dataFolder, "train");
    std::vector<std::vector<float>> test_data = loadData(dataFolder, "test");

    std::cout << "Loaded " << train_data.size() << " samples" << std::endl;
    
    if (train_data.empty()) {
        std::cerr << "No training data loaded! Check your data folder path." << std::endl;
        return -1;
    }
    
    // Training parameters
    int EPOCHS = 1;
    int BATCH_SIZE = 32; // Reduced for GPU memory
    float LR = 0.001;
    float MOMENTUM = 0.9;
    
    // Print training configuration
    std::cout << "==== Training Configuration:" << std::endl;
    std::cout << "Epochs: " << EPOCHS << std::endl;
    std::cout << "Batch Size: " << BATCH_SIZE << std::endl;
    std::cout << "Learning Rate: " << LR << std::endl;
    std::cout << "Momentum: " << MOMENTUM << std::endl;

    int total_train_samples = train_data.size();
    int total_train_batches = (total_train_samples + BATCH_SIZE - 1) / BATCH_SIZE;

    int total_test_samples = test_data.size();
    int total_test_batches = (total_test_samples + BATCH_SIZE - 1) / BATCH_SIZE;


    // Initialize GPU Autoencoder
    GPUAutoencoder ae(LR, MOMENTUM);
    
    ae.load_weights("../weights/test_weights.bin"); // Load initial weights if available
    // CUDA Events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Training loop for multiple epochs
    // int ok = 0;
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        float epoch_loss = 0.0f;
        int processed_batches = 0;
        float train_milliseconds = 0, test_milliseconds = 0;
        std::cout << "\n=== Epoch " << (epoch + 1) << "/" << EPOCHS << " ===" << std::endl;
        
        // Start epoch timing
        cudaEventRecord(start);
        // Process data in batches
        printf("\nStarting training batches...\n");

        ae.setTrain();
        
        for (int i = 0; i < total_train_samples; i += BATCH_SIZE) {
            int current_batch_size = std::min(BATCH_SIZE, total_train_samples - i);
            int current_batch_num = (i / BATCH_SIZE) + 1;

            std::cout << "\rProcessing batch " << current_batch_num << "/" << total_train_batches 
                    << " (samples " << i << " to " << (i + current_batch_size - 1) << ")"<<std::flush;
                    
                    
            
            // Prepare batch data - flatten to single vector
            std::vector<float> batch_data;
            batch_data.reserve(current_batch_size * 3072); // 32*32*3 = 3072
            
            for (int j = 0; j < current_batch_size; j++) {
                const auto& image = train_data[i + j];
                batch_data.insert(batch_data.end(), image.begin(), image.end());
            }
            
            // Train on batch using GPU
            ae.train_batch(batch_data, current_batch_size);
            if ((processed_batches+1) % 10 == 0)
                std::cout<<"|   Single batch Loss: "<< std::fixed << std::setprecision(6) << ae.getLoss() << std::endl;

            epoch_loss += ae.getLoss();
            processed_batches++;
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        cudaEventElapsedTime(&train_milliseconds, start, stop);

        printf("\nStarting testing batches...\n");

        ae.setEval();
        float test_loss = 0.0f;
        cudaEventRecord(start);
        
        for (int i = 0; i < total_test_samples; i += BATCH_SIZE) {
            int current_batch_size = std::min(BATCH_SIZE, total_test_samples - i);
            int current_batch_num = (i / BATCH_SIZE) + 1;

            // ok ++;
            // if (ok == 5 && BATCH_SIZE == 3){
            //     ae.save_weights("../weights/gpu_trained_weights_checking.bin");
            //     break;
            // }
            std::cout << "\rProcessing batch " << current_batch_num << "/" << total_test_batches 
                    << " (samples " << i << " to " << (i + current_batch_size - 1) << ")"<<std::flush;
                    
                    
            
            // Prepare batch data - flatten to single vector
            std::vector<float> batch_data;
            batch_data.reserve(current_batch_size * 3072); // 32*32*3 = 3072
            
            for (int j = 0; j < current_batch_size; j++) {
                const auto& image = test_data[i + j];
                batch_data.insert(batch_data.end(), image.begin(), image.end());
            }
            
            // Train on batch using GPU
            ae.train_batch(batch_data, current_batch_size);
                
            test_loss += ae.getLoss();
        }
        // Stop epoch timing
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        cudaEventElapsedTime(&test_milliseconds, start, stop);

        // Calculate epoch statistics
        float avg_epoch_loss = epoch_loss / processed_batches; 
        // float avg_test_loss = test_loss / total_test_batches;
        
        std::cout << "\n--- Epoch " << (epoch + 1) << " Summary ---" << std::endl;
        std::cout << "Batches processed: " << processed_batches << std::endl;
        std::cout << "Average train loss: " << std::fixed << std::setprecision(6) << avg_epoch_loss << std::endl;
        // std::cout << "Average test loss: " << std::fixed << std::setprecision(6) << avg_test_loss << std::endl;
        std::cout << "Epoch train time: " << std::fixed << std::setprecision(2) << (train_milliseconds / 1000.0f) << " seconds" << std::endl;
        // std::cout << "Inference time: " << std::fixed << std::setprecision(2) << (test_milliseconds / 1000.0f) << " seconds" << std::endl;

        // Save weights periodically
        // if ((epoch + 1) % 2 == 0) {
        //     std::cout << "Saving weights checkpoint..." << std::endl;
        //     ae.get_weights_to_host();
        // }
    }
    printf("=== Time summary ===\n");
    printf("Total Kernel Time: %.2f ms\n", ae.getTotalKernelTime());
    printf("Convolution Forward Time: %.2f ms||| Ratio: %.2f%%\n", ae.getConvForwardTime(), (ae.getConvForwardTime() / ae.getTotalKernelTime()) * 100.0);
    printf("Convolution Backward Time: %.2f ms||| Ratio: %.2f%%\n", ae.getConvBackwardTime(), (ae.getConvBackwardTime() / ae.getTotalKernelTime()) * 100.0);
    float conv_time = ae.getConvForwardTime() + ae.getConvBackwardTime();
    printf("Convolution Time: %.2f ms||| Ratio: %.2f%%\n", conv_time, (conv_time / ae.getTotalKernelTime()) * 100.0);
    float relu_time = ae.getReluForwardTime() + ae.getReluBackwardTime();
    
    printf("ReLU Time: %.2f ms||| Ratio: %.2f%%\n", relu_time, (relu_time / ae.getTotalKernelTime()) * 100.0);
    float pool_time = ae.getPoolForwardTime() + ae.getPoolBackwardTime(); 
    printf("Pooling Time: %.2f ms||| Ratio: %.2f%%\n", pool_time, (pool_time / ae.getTotalKernelTime()) * 100.0);
    

    // Cleanup CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    std::cout << "\n=== GPU Training Completed ===" << std::endl;
    return 0;
}