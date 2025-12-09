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
std::vector<std::pair<int, std::vector<double>>> readBinaryFile(const std::string& filepath) {
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
    std::vector<std::pair<int, std::vector<double>>> imageData;
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
        
        // Convert unsigned char (0-255) to double (0.0-1.0) for GPU
        std::vector<double> pixels;
        pixels.reserve(IMAGE_SIZE);
        for (unsigned char pixel : pixelBytes) {
            pixels.push_back(static_cast<double>(pixel) / 255.0f);
        }
        
        // Add to result as pair<label, pixels>
        imageData.emplace_back(static_cast<int>(label), std::move(pixels));
    }
    
    return imageData;
}

// Hàm đọc tất cả file .bin trong folder và load vào train_data
std::vector<std::vector<double>> loadTrainData(const std::string& folderPath) {
    std::vector<std::vector<double>> train_data;
    
    try {
        // Kiểm tra folder có tồn tại không
        if (!std::filesystem::exists(folderPath)) {
            throw std::runtime_error("Folder does not exist: " + folderPath);
        }
        
        // Lấy danh sách tất cả file .bin trong folder
        std::vector<std::string> binFiles;
        for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
            if (entry.is_regular_file() && entry.path().extension() == ".bin" && 
                entry.path().filename().string().find("data") != std::string::npos) {
                binFiles.push_back(entry.path().string());
            }
        }
        
        // Sắp xếp file theo tên để đảm bảo thứ tự
        std::sort(binFiles.begin(), binFiles.end());
        
        std::cout << "Found " << binFiles.size() << " binary files" << std::endl;
        
        // Đọc từng file
        for (const auto& filepath : binFiles) {
            try {
                auto imageData = readBinaryFile(filepath);
                // printf("  Loaded %zu images\n", imageData.size());
                
                // Extract only the pixel data (second element of pair) for training
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

int main() {
    // Đường dẫn đến folder chứa các file batch
    std::string dataFolder = "../data/cifar-10-binary/cifar-10-batches-bin";
    std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;
    std::cout << "Looking for data at: " << std::filesystem::absolute(dataFolder) << std::endl;
    
    std::cout << "Loading training data..." << std::endl;
    std::vector<std::vector<double>> train_data = loadTrainData(dataFolder);
    std::cout << "Loaded " << train_data.size() << " samples" << std::endl;
    
    if (train_data.empty()) {
        std::cerr << "No training data loaded! Check your data folder path." << std::endl;
        return -1;
    }
    
    // Training parameters
    int EPOCHS = 5;
    int BATCH_SIZE = 3; // Reduced for GPU memory
    double LR = 0.001;
    double MOMENTUM = 0.9;
    
    int total_samples = train_data.size();
    int total_batches = (total_samples + BATCH_SIZE - 1) / BATCH_SIZE;

    std::cout << "\n=== Starting GPU Training ===" << std::endl;
    std::cout << "Total samples: " << total_samples << std::endl;
    std::cout << "Batch size: " << BATCH_SIZE << std::endl;
    std::cout << "Total batches per epoch: " << total_batches << std::endl;
    std::cout << "Epochs: " << EPOCHS << std::endl;
    std::cout << "================================" << std::endl;

    // Initialize GPU Autoencoder
    GPUAutoencoder ae(LR, MOMENTUM);
    ae.setTrain();
    ae.load_weights("../weights/test_weights.bin"); // Load initial weights if available
    // CUDA Events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Training loop for multiple epochs
    int ok = 0;
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        double epoch_loss = 0.0f;
        int processed_batches = 0;
        
        std::cout << "\n=== Epoch " << (epoch + 1) << "/" << EPOCHS << " ===" << std::endl;
        
        // Start epoch timing
        cudaEventRecord(start);
        if (ok > 4){
            break;
        }
        // Process data in batches
        for (int i = 0; i < total_samples; i += BATCH_SIZE) {
            int current_batch_size = std::min(BATCH_SIZE, total_samples - i);
            int current_batch_num = (i / BATCH_SIZE) + 1;

            ok ++;
            if (ok > 4){
                ae.save_weights("../weights/gpu_trained_weights.bin");
                break;
            }
            std::cout << "\rProcessing batch " << current_batch_num << "/" << total_batches 
                      << " (samples " << i << " to " << (i + current_batch_size - 1) << ")" << std::flush;
            
            // Prepare batch data - flatten to single vector
            std::vector<double> batch_data;
            batch_data.reserve(current_batch_size * 3072); // 32*32*3 = 3072
            
            for (int j = 0; j < current_batch_size; j++) {
                const auto& image = train_data[i + j];
                batch_data.insert(batch_data.end(), image.begin(), image.end());
            }
            
            // Train on batch using GPU
            ae.train_batch(batch_data, current_batch_size);
            
            epoch_loss += ae.getLoss();
            processed_batches++;
        }

        // Stop epoch timing
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        // Calculate epoch statistics
        double avg_epoch_loss = epoch_loss / processed_batches;
        
        std::cout << "\n--- Epoch " << (epoch + 1) << " Summary ---" << std::endl;
        std::cout << "Batches processed: " << processed_batches << std::endl;
        std::cout << "Average loss: " << std::fixed << std::setprecision(6) << avg_epoch_loss << std::endl;
        std::cout << "Epoch time: " << std::fixed << std::setprecision(2) << (milliseconds / 1000.0f) << " seconds" << std::endl;
        std::cout << "Samples/second: " << std::fixed << std::setprecision(1) << (total_samples / (milliseconds / 1000.0f)) << std::endl;
                  
        // Save weights periodically
        // if ((epoch + 1) % 2 == 0) {
        //     std::cout << "Saving weights checkpoint..." << std::endl;
        //     ae.get_weights_to_host();
        // }
    }

    // Cleanup CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    std::cout << "\n=== GPU Training Completed ===" << std::endl;
    return 0;
}