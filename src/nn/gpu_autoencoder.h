#pragma once
#include <vector>
#include <string>
#include <functional>
// #include <cuda_runtime.h>

// Macro kiểm tra lỗi CUDA
#define CHECK(call)\
{\
	const cudaError_t error = call;\
	if (error != cudaSuccess)\
	{\
		fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
		fprintf(stderr, "code: %d, reason: %s\n", error,\
				cudaGetErrorString(error));\
		exit(EXIT_FAILURE);\
	}\
}


class GPUAutoencoder {
    
public:
    
    GPUAutoencoder(float learningRate, float momentum);
    ~GPUAutoencoder();

    // Requirement 2.3: Copy input batch from host to device
    // Requirement 2.3: Copy final output back to host (Optional via output pointer)
    void train_batch(const std::vector<float>& h_inputBatch, int batchSize);
    
    // Requirement 2.1: Implement functions to copy weights between host and device
    // Lưu trọng số xuống file nhị phân
    void save_weights(const std::string& filepath);
    
    // Load trọng số từ file (để phục vụ Phase 4 - Feature Extraction)
    void load_weights(const std::string& filepath);

    float getLoss() const { return m_loss; }
    
    // Timing getter functions
    float getConvForwardTime() const { return conv_forward_time; }
    float getConvBackwardTime() const { return conv_backward_time; }
    float getReluForwardTime() const { return relu_forward_time; }
    float getReluBackwardTime() const { return relu_backward_time; }
    float getPoolForwardTime() const { return pool_forward_time; }
    float getPoolBackwardTime() const { return pool_backward_time; }
    float getTotalKernelTime() const { return total_kernel_time; }
    void resetTimers() { 
        conv_forward_time = conv_backward_time = relu_forward_time = relu_backward_time = pool_forward_time = pool_backward_time = total_kernel_time = 0.0; 
    }
    void get_weights_to_host();
    void getOutput(const std::vector<float>& h_inputBatch, std::vector<float>& h_output, int batchSize);
    void getLatent(const std::vector<float>& h_inputBatch, std::vector<float>& h_latent, int batchSize);
    void setTrain() { train = true; }
    void setEval() { train = false; }
    bool isTraining() const { return train; }
    template<typename Func>
    void measureKernelTime(Func&& kernelFunc, float& timeVariable, const char* kernelName);
private:
    float m_learningRate;
    float m_momentum;
    float m_loss;
    float avg_grad;
    bool allocated = false;
    bool train = true;
    // Timing variables for performance measurement
    float conv_forward_time, conv_backward_time;
    float relu_forward_time, relu_backward_time;
    float pool_forward_time, pool_backward_time;
    float total_kernel_time;
    // --- DEVICE POINTERS (WEIGHTS & BIAS) ---
    float *d_w_enc_conv1, *d_b_enc_conv1;
    float *d_w_enc_conv2, *d_b_enc_conv2;
    float *d_w_dec_conv3, *d_b_dec_conv3;
    float *d_w_dec_conv4, *d_b_dec_conv4;
    float *d_w_dec_conv5, *d_b_dec_conv5;

    // --- DEVICE POINTERS (GRADIENTS) ---
    float *d_dw_enc_conv1, *d_db_enc_conv1;
    float *d_dw_enc_conv2, *d_db_enc_conv2;
    float *d_dw_dec_conv3, *d_db_dec_conv3;
    float *d_dw_dec_conv4, *d_db_dec_conv4;
    float *d_dw_dec_conv5, *d_db_dec_conv5;

    // --- DEVICE POINTERS (VELOCITY) ---
    float *d_v_enc_conv1, *d_v_enc_conv1_b;
    float *d_v_enc_conv2, *d_v_enc_conv2_b;
    float *d_v_dec_conv3, *d_v_dec_conv3_b;
    float *d_v_dec_conv4, *d_v_dec_conv4_b;
    float *d_v_dec_conv5, *d_v_dec_conv5_b;

    // --- ACTIVATION BUFFERS ---
    float *d_input;         
    float *d_enc_conv1, *d_enc_pool1, *d_enc_conv2, *d_latent;
    float *d_dec_conv3, *d_dec_ups1, *d_dec_conv4, *d_dec_ups2, *d_output;

    // --- GRADIENT BUFFERS ---
    float *d_grad_output, *d_grad_dec_ups2, *d_grad_dec_conv4, *d_grad_dec_ups1;
    float *d_grad_dec_conv3, *d_grad_latent, *d_grad_enc_conv2, *d_grad_enc_pool1;
    float *d_grad_enc_conv1, *d_grad_input;

    // Helper Functions
    void allocateMemory(int maxBatchSize);
    void initWeightsRandomly();
    void forwardPass(int batchSize);
    void backwardPass(int batchSize);
    void updateWeights();
    
    // Helpers for Save/Load
    void copy_layer_to_host(float* d_data, std::vector<float>& h_data, int size);
    void copy_layer_to_device(float* d_data, const std::vector<float>& h_data, int size);
    
    
};