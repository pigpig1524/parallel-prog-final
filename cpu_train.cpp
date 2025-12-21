#include "src/nn/autoencoder.h"
#include <cmath>
#include <iostream>
#include <random>
#include <filesystem>
#include <fstream>
#include <string>
#include <algorithm>
#include <chrono>
#include <iomanip>


std::vector<std::pair<int, std::vector<float>>> readBinaryFile(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }

    // CIFAR-10 format: mỗi image = 1 byte label + 3072 bytes pixel data
    const int IMAGE_SIZE = 3072; 
    const int RECORD_SIZE = 1 + IMAGE_SIZE; 
    
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
        unsigned char label;
        file.read(reinterpret_cast<char*>(&label), 1);
        
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

std::vector<std::vector<float>> loadTrainData(const std::string& folderPath) {
    std::vector<std::vector<float>> train_data;
    
    try {
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
    std::string dataFolder = "../data/cifar-10-binary/cifar-10-batches-bin"; // Thay đổi đường dẫn này
    std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;
    std::cout << "Looking for data at: " << std::filesystem::absolute(dataFolder) << std::endl;
    std::cout << "Loading training data..." << std::endl;
    std::vector<std::vector<float>> train_data = loadTrainData(dataFolder);
    std::cout << "Loaded " << train_data.size() << " samples" << std::endl;
    
    if (train_data.empty()) {
        std::cerr << "No training data loaded! Check your data folder path." << std::endl;
        return -1;
    }
    int EPOCHS = 1;
    int BATCH_SIZE = 32;
    float LR = 0.001;
    float MOMENTUM = 0.9;
    float total_epoch_loss = 0;
    int total_samples = train_data.size();
    int total_batches = (total_samples + BATCH_SIZE - 1) / BATCH_SIZE; // Round up division

    std::cout << "\n=== Starting Training ===" << std::endl;
    std::cout << "Total samples: " << total_samples << std::endl;
    std::cout << "Batch size: " << BATCH_SIZE << std::endl;
    std::cout << "Total batches: " << total_batches << std::endl;
    std::cout << "=========================" << std::endl;
    
    // Bắt đầu đo thời gian training
    auto training_start_time = std::chrono::high_resolution_clock::now();

    Autoencoder ae = Autoencoder(LR, MOMENTUM); // Reduced learning rate 
    ae.load_weights("../weights/test_weights.bin");
    int ok = 0;
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        // Bắt đầu đo thời gian cho epoch
        auto epoch_start_time = std::chrono::high_resolution_clock::now();
        
        float epoch_loss = 0.0f;
        int processed_batches = 0;
        
        std::cout << "\n=== Epoch " << (epoch + 1) << "/" << EPOCHS << " ===" << std::endl;
        
        if (ok > 4){
            break;
        }
        // Process data in batches
        for (int i = 0; i < total_samples; i += BATCH_SIZE) {
            int current_batch_size = std::min(BATCH_SIZE, total_samples - i);
            int current_batch_num = (i / BATCH_SIZE) + 1;
            
            ok ++;
            if (ok > 1){
                // ae.save_weights("../weights/cpu_trained_weights_checking.bin");
                break;
            }
            std::cout << "\rProcessing batch " << current_batch_num << "/" << total_batches 
                      << " (samples " << i << " to " << (i + current_batch_size - 1) << ")" << std::flush;
            
            
            // Train on batch using CPU
            ae.avg_grad = 0.0;
            for (int j = 0; j < current_batch_size; j++) {
                const auto& image = train_data[i + j];
                ae.train_sample(image);
                epoch_loss += ae.getLoss()/current_batch_size;
            }
            // std::cout << "\n| Avg Grad: " << ae.avg_grad / current_batch_size<<std::endl;
            ae.update_weights(current_batch_size);

            processed_batches++;
        }


        // Tính thời gian epoch
        auto epoch_end_time = std::chrono::high_resolution_clock::now();
        auto epoch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end_time - epoch_start_time);
        
        // Calculate epoch statistics
        float avg_epoch_loss = epoch_loss / processed_batches;
        
        std::cout << "\n--- Epoch " << (epoch + 1) << " Summary ---" << std::endl;
        std::cout << "Batches processed: " << processed_batches << std::endl;
        std::cout << "Average loss: " << std::fixed << std::setprecision(6) << avg_epoch_loss << std::endl;
        std::cout << "Epoch time: " << epoch_duration.count() << " ms (" << std::fixed << std::setprecision(2) << epoch_duration.count() / 1000.0 << " seconds)" << std::endl;
        
        // Hiển thị thời gian training tổng cộng
        auto current_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - training_start_time);
        std::cout << "Total training time so far: " << total_duration.count() << " ms (" << std::fixed << std::setprecision(2) << total_duration.count() / 1000.0 << " seconds)" << std::endl;
                  
        // Save weights periodically
        // if ((epoch + 1) % 2 == 0) {
        //     std::cout << "Saving weights checkpoint..." << std::endl;
        //     ae.save_weights("../weights/cpu_trained_weights_epoch" + std::to_string(epoch + 1) + ".bin");
        // }
    }
    
    auto training_end_time = std::chrono::high_resolution_clock::now();
    auto total_training_duration = std::chrono::duration_cast<std::chrono::milliseconds>(training_end_time - training_start_time);
    
    std::cout << "\n=== Training Completed ===" << std::endl;
    std::cout << "Total training time: " << total_training_duration.count() << " ms (" 
              << std::fixed << std::setprecision(2) << total_training_duration.count() / 1000.0 << " seconds)" << std::endl;
    if (total_training_duration.count() > 60000) {
        std::cout << "Total training time: " << std::fixed << std::setprecision(2) << total_training_duration.count() / 60000.0 << " minutes" << std::endl;
    }
    std::cout << "=========================" << std::endl;
    
    return 0;
}