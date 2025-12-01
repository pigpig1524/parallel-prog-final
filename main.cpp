#include "src/nn/autoencoder.h"
#include <cmath>
#include <iostream>
#include <random>
#include <filesystem>
#include <fstream>
#include <string>
#include <algorithm>

// Hàm đọc một file binary và trả về vector<double>
// Hàm đọc một file binary CIFAR-10 và trả về vector<double> (chỉ pixel data, bỏ qua label)
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
    
    // std::cout << "  Reading " << numImages << " images from file..." << std::endl;
    
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
        
        // Convert unsigned char (0-255) to double (0.0-1.0) for neural network
        std::vector<double> pixels;
        pixels.reserve(IMAGE_SIZE);
        for (unsigned char pixel : pixelBytes) {
            pixels.push_back(static_cast<double>(pixel) / 255.0);
        }
        
        // if (i == 0) {
        //     std::cout << "    First image - Label: " << static_cast<int>(label) 
        //               << ", First few pixels: " << pixels[0] << ", " 
        //               << pixels[1] << ", " << pixels[2] << std::endl;
        // }
        
        // Add to result as pair<label, pixels>
        imageData.emplace_back(static_cast<int>(label), std::move(pixels));
        // printf("pixel[1] value: %f\n", imageData.back().second[1]);
        // Debug: Print first image info
        
    }
    
    return imageData;
}
// Hàm đọc tất cả file .bin trong folder và load vào train_data
// Hàm đọc tất cả file .bin trong folder và load vào train_data
std::vector< std::vector<double>> loadTrainData(const std::string& folderPath) {
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
            // std::cout << "Loading: " << filepath << std::endl;
            
            try {
                // readBinaryFile now returns vector<pair<int, vector<double>>>
                auto imageData = readBinaryFile(filepath);
                printf("  Loaded %zu images\n", imageData.size());
                
                // Extract only the pixel data (second element of pair) for training
                for (const auto& imagePair : imageData) {
                    train_data.push_back(imagePair.second); // Extract only pixels
                }
                
            } catch (const std::exception& e) {
                std::cerr << "Error loading file " << filepath << ": " << e.what() << std::endl;
                // Tiếp tục với file khác
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
    std::string dataFolder = "..\\data\\cifar-10-binary\\cifar-10-batches-bin"; // Thay đổi đường dẫn này
    std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;
    std::cout << "Looking for data at: " << std::filesystem::absolute(dataFolder) << std::endl;
    // Load dữ liệu training
    std::cout << "Loading training data..." << std::endl;
    std::vector<std::vector<double>> train_data = loadTrainData(dataFolder);
    std::cout << "Loaded " << train_data.size() << " samples" << std::endl;
    
    if (train_data.empty()) {
        std::cerr << "No training data loaded! Check your data folder path." << std::endl;
        return -1;
    }
    
    int BATCH_SIZE = 16;
    double total_epoch_loss = 0;
    int total_samples = train_data.size();
    int total_batches = (total_samples + BATCH_SIZE - 1) / BATCH_SIZE; // Round up division

    std::cout << "\n=== Starting Training ===" << std::endl;
    std::cout << "Total samples: " << total_samples << std::endl;
    std::cout << "Batch size: " << BATCH_SIZE << std::endl;
    std::cout << "Total batches: " << total_batches << std::endl;
    std::cout << "=========================" << std::endl;

    Autoencoder ae = Autoencoder(0.001, 0.9); // Reduced learning rate 

    for (int i = 0; i < train_data.size(); i += BATCH_SIZE) {
        int current_batch_num = (i / BATCH_SIZE) + 1;
        
        // 1. Chạy qua từng ảnh trong batch để tích lũy gradient
        double batch_loss = 0;
        int current_batch_size = 0;
        
        std::cout << "\n--- Batch " << current_batch_num << "/" << total_batches << " ---" << std::endl;
        std::cout << "Processing samples " << i << " to " << std::min(i + BATCH_SIZE - 1, (int)train_data.size() - 1) << std::endl;
        
        for (int j = 0; j < BATCH_SIZE && (i + j) < train_data.size(); j++) {
            int sample_idx = i + j;
            
            // Train on single sample (accumulate gradients)
            ae.train_sample(train_data[sample_idx]);
            double sample_loss = ae.getLoss();
            batch_loss += sample_loss;
            current_batch_size++;
            
            // Progress indicator for large batches
        }
        
        // 2. Cập nhật trọng số MỘT LẦN cho cả batch
            std::cout << "Updating weights for batch of " << current_batch_size << " samples..." << std::endl;
            ae.update_weights(current_batch_size); // Update & Reset gradients
            
            // Calculate batch statistics
            double avg_batch_loss = batch_loss / current_batch_size;
            total_epoch_loss += batch_loss;
            double avg_epoch_loss = total_epoch_loss / (i + current_batch_size);
            
            std::cout << "Batch " << current_batch_num << " Summary:" << std::endl;
            std::cout << "  Samples processed: " << current_batch_size << std::endl;
            std::cout << "  Batch total loss: " << std::fixed << std::setprecision(6) << batch_loss << std::endl;
            std::cout << "  Batch avg loss: " << std::fixed << std::setprecision(6) << avg_batch_loss << std::endl;
            std::cout << "  Running epoch avg loss: " << std::fixed << std::setprecision(6) << avg_epoch_loss << std::endl;
            
            // Progress bar
            float progress = (float)(i + current_batch_size) / total_samples * 100;
            std::cout << "  Progress: [";
            int bar_width = 20;
            int filled = (int)(progress / 100 * bar_width);
            for (int k = 0; k < bar_width; k++) {
                if (k < filled) std::cout << "=";
                else std::cout << " ";
            }
            std::cout << "] " << std::fixed << std::setprecision(1) << progress << "%" << std::endl;
        }

        // Final epoch summary
        std::cout << "\n=== Epoch Complete ===" << std::endl;
        std::cout << "Total samples processed: " << total_samples << std::endl;
        std::cout << "Total batches processed: " << total_batches << std::endl;
        std::cout << "Total epoch loss: " << std::fixed << std::setprecision(6) << total_epoch_loss << std::endl;
        std::cout << "Average loss per sample: " << std::fixed << std::setprecision(6) << (total_epoch_loss / total_samples) << std::endl;
        std::cout << "======================" << std::endl;

    return 0;
}