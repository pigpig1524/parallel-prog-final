#pragma once
#include <vector>
#include <string>

struct Tensor {
    int c, h, w;
    std::vector<double> data;

    Tensor(int c = 0, int h = 0, int w = 0) : c(c), h(h), w(w) {
        data.resize(c * h * w, 0.0);
    }
    
    void zero() { std::fill(data.begin(), data.end(), 0.0); }
    
    double& at(int ch, int y, int x) {
        return data[(ch * h * w) + (y * w) + x];
    }
    
    const double& at(int ch, int y, int x) const {
        return data[(ch * h * w) + (y * w) + x];
    }
};

class Autoencoder {
public:
    Autoencoder(double learningRate, double momentum);
    ~Autoencoder();

    // Thay đổi: train chỉ tính toán và tích lũy gradient, KHÔNG update ngay
    void train_sample(const std::vector<double>& imageFlat);
    
    // Hàm mới: Thực hiện update weights sau khi chạy xong 1 batch
    void update_weights(int batchSize);

    std::vector<double> getLatent(const std::vector<double>& imageFlat);
    std::vector<double> reconstruct(const std::vector<double>& imageFlat);
    double getLoss() const { return m_loss; }

private:
    double m_learningRate;
    double m_momentum;
    double m_loss;

    // --- WEIGHTS & BIAS ---
    std::vector<double> w_enc_conv1, b_enc_conv1; 
    std::vector<double> w_enc_conv2, b_enc_conv2;
    std::vector<double> w_dec_conv3, b_dec_conv3;
    std::vector<double> w_dec_conv4, b_dec_conv4;
    std::vector<double> w_dec_conv5, b_dec_conv5;

    // --- GRADIENT ACCUMULATORS (Tích lũy qua batch) ---
    // Được chuyển từ biến cục bộ thành biến thành viên
    std::vector<double> d_w_enc_conv1, d_b_enc_conv1;
    std::vector<double> d_w_enc_conv2, d_b_enc_conv2;
    std::vector<double> d_w_dec_conv3, d_b_dec_conv3;
    std::vector<double> d_w_dec_conv4, d_b_dec_conv4;
    std::vector<double> d_w_dec_conv5, d_b_dec_conv5;

    // --- MOMENTUM VELOCITY ---
    std::vector<double> v_w_enc_conv1, v_b_enc_conv1;
    std::vector<double> v_w_enc_conv2, v_b_enc_conv2;
    std::vector<double> v_w_dec_conv3, v_b_dec_conv3;
    std::vector<double> v_w_dec_conv4, v_b_dec_conv4;
    std::vector<double> v_w_dec_conv5, v_b_dec_conv5;

    // --- ACTIVATIONS ---
    Tensor a_input;
    Tensor a_enc_conv1, a_enc_pool1, a_enc_conv2, a_latent;
    Tensor a_dec_conv3, a_dec_ups1, a_dec_conv4, a_dec_ups2, a_output;

    void initWeights();
    void forwardPass();
    void backwardPass(); // Chỉ tích lũy gradient
    
    // Helper function cho update
    void apply_update(std::vector<double>& weights, std::vector<double>& d_weights, std::vector<double>& velocity, int batchSize);

    // Kernels
    void conv2d_forward(const Tensor& input, const std::vector<double>& weights, const std::vector<double>& bias, Tensor& output, int stride=1, int padding=1);
    void relu_forward(Tensor& t);
    void maxpool2d_forward(const Tensor& input, Tensor& output);
    void upsample2d_forward(const Tensor& input, Tensor& output);
    
    void conv2d_backward(const Tensor& input, const Tensor& d_output, const std::vector<double>& weights, Tensor& d_input, std::vector<double>& d_weights, std::vector<double>& d_bias, int stride=1, int padding=1);
    void relu_backward(const Tensor& input, Tensor& d_input);
    void maxpool2d_backward(const Tensor& input, const Tensor& output, const Tensor& d_output, Tensor& d_input);
    void upsample2d_backward(const Tensor& d_output, Tensor& d_input);
};