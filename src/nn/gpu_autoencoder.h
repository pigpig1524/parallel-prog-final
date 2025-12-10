#pragma once
#include <vector>
#include <string>
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
    GPUAutoencoder(double learningRate, double momentum);
    ~GPUAutoencoder();

    // Requirement 2.3: Copy input batch from host to device
    // Requirement 2.3: Copy final output back to host (Optional via output pointer)
    void train_batch(const std::vector<double>& h_inputBatch, int batchSize);
    
    // Requirement 2.1: Implement functions to copy weights between host and device
    // Lưu trọng số xuống file nhị phân
    void save_weights(const std::string& filepath);
    
    // Load trọng số từ file (để phục vụ Phase 4 - Feature Extraction)
    void load_weights(const std::string& filepath);

    double getLoss() const { return m_loss; }
    void get_weights_to_host();
    void getOutput(const std::vector<double>& h_inputBatch, std::vector<double>& h_output, int batchSize);
    void setTrain() { train = true; }
    void setEval() { train = false; }
private:
    double m_learningRate;
    double m_momentum;
    double m_loss;
    bool train = true;
    double avg_grad;
    // --- DEVICE POINTERS (WEIGHTS & BIAS) ---
    double *d_w_enc_conv1, *d_b_enc_conv1;
    double *d_w_enc_conv2, *d_b_enc_conv2;
    double *d_w_dec_conv3, *d_b_dec_conv3;
    double *d_w_dec_conv4, *d_b_dec_conv4;
    double *d_w_dec_conv5, *d_b_dec_conv5;

    // --- DEVICE POINTERS (GRADIENTS) ---
    double *d_dw_enc_conv1, *d_db_enc_conv1;
    double *d_dw_enc_conv2, *d_db_enc_conv2;
    double *d_dw_dec_conv3, *d_db_dec_conv3;
    double *d_dw_dec_conv4, *d_db_dec_conv4;
    double *d_dw_dec_conv5, *d_db_dec_conv5;

    // --- DEVICE POINTERS (VELOCITY) ---
    double *d_v_enc_conv1, *d_v_enc_conv1_b;
    double *d_v_enc_conv2, *d_v_enc_conv2_b;
    double *d_v_dec_conv3, *d_v_dec_conv3_b;
    double *d_v_dec_conv4, *d_v_dec_conv4_b;
    double *d_v_dec_conv5, *d_v_dec_conv5_b;

    // --- ACTIVATION BUFFERS ---
    double *d_input;         
    double *d_enc_conv1, *d_enc_pool1, *d_enc_conv2, *d_latent;
    double *d_dec_conv3, *d_dec_ups1, *d_dec_conv4, *d_dec_ups2, *d_output;

    // --- GRADIENT BUFFERS ---
    double *d_grad_output, *d_grad_dec_ups2, *d_grad_dec_conv4, *d_grad_dec_ups1;
    double *d_grad_dec_conv3, *d_grad_latent, *d_grad_enc_conv2, *d_grad_enc_pool1;
    double *d_grad_enc_conv1, *d_grad_input;

    // Helper Functions
    void allocateMemory(int maxBatchSize);
    void initWeightsRandomly();
    void forwardPass(int batchSize);
    void backwardPass(int batchSize);
    void updateWeights();
    
    // Helpers for Save/Load
    void copy_layer_to_host(double* d_data, std::vector<double>& h_data, int size);
    void copy_layer_to_device(double* d_data, const std::vector<double>& h_data, int size);
    
    
};