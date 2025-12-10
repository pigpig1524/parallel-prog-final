#include "autoencoder.h"
#include <cmath>
#include <iostream>
#include <random>
#include <algorithm>
#include <fstream>

#define W_IDX(out, in, ky, kx, InC, K) (((out) * (InC) * (K) * (K)) + ((in) * (K) * (K)) + ((ky) * (K)) + (kx))

using namespace std;

void random_init(vector<double>& vec, int fan_in) {
    // Xavier/Glorot initialization for better stability
    double limit = sqrt(2.0 / (fan_in + vec.size()));
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<double> dist(-limit, limit);
    for (double& val : vec) val = dist(generator);
}

Autoencoder::Autoencoder(double learningRate, double momentum) 
    : m_learningRate(learningRate), m_momentum(momentum), m_loss(0.0) 
{
    initWeights();
}

Autoencoder::~Autoencoder() {}

void Autoencoder::initWeights() {
    int K = 3; 
    
    // Hàm lambda để init vector gradient về 0
    auto init_params = [&](std::vector<double>& w, std::vector<double>& b, 
                           std::vector<double>& dw, std::vector<double>& db,
                           std::vector<double>& vw, std::vector<double>& vb,
                           int outC, int inC) {
        w.resize(outC * inC * K * K); random_init(w, inC * K * K);
        b.resize(outC, 0.0);
        
        dw.assign(w.size(), 0.0); // Init gradient = 0
        db.assign(b.size(), 0.0);
        
        vw.assign(w.size(), 0.0);
        vb.assign(b.size(), 0.0);
    };

    init_params(w_enc_conv1, b_enc_conv1, d_w_enc_conv1, d_b_enc_conv1, v_w_enc_conv1, v_b_enc_conv1, 256, 3);
    init_params(w_enc_conv2, b_enc_conv2, d_w_enc_conv2, d_b_enc_conv2, v_w_enc_conv2, v_b_enc_conv2, 128, 256);
    init_params(w_dec_conv3, b_dec_conv3, d_w_dec_conv3, d_b_dec_conv3, v_w_dec_conv3, v_b_dec_conv3, 128, 128);
    init_params(w_dec_conv4, b_dec_conv4, d_w_dec_conv4, d_b_dec_conv4, v_w_dec_conv4, v_b_dec_conv4, 256, 128);
    init_params(w_dec_conv5, b_dec_conv5, d_w_dec_conv5, d_b_dec_conv5, v_w_dec_conv5, v_b_dec_conv5, 3, 256); 
}

void Autoencoder::train_sample(const std::vector<double>& imageFlat) {
    a_input = Tensor(3, 32, 32);
    a_input.data = imageFlat;

    forwardPass();

    // Tính loss cho sample này
    m_loss = 0.0;
    int size = a_output.data.size();
    // Sử dụng MSE Loss
    for(int i=0; i<size; i++) {
        double diff = a_output.data[i] - a_input.data[i];
        m_loss += diff * diff;
    }
    m_loss /= size;

    // Backward sẽ CỘNG DỒN vào biến thành viên d_w_...
    d_output = Tensor(3, 32, 32);
    for(int i=0; i<size; i++) {
        // Match GPU version: dL/dy = (2/N) * (target - predicted)
        double diff = a_input.data[i] - a_output.data[i]; // target - predicted
        d_output.data[i] = -(2.0f / size) * diff;
        avg_grad += fabs(d_output.data[i]);
    }
    
    backwardPass();
}


// Hàm mới: Gọi hàm này sau khi chạy xong 1 batch (ví dụ 32 ảnh)
void Autoencoder::update_weights(int batchSize) {
    if (batchSize <= 0) return;

    apply_update(w_enc_conv1, d_w_enc_conv1, v_w_enc_conv1, batchSize);
    apply_update(b_enc_conv1, d_b_enc_conv1, v_b_enc_conv1, batchSize);

    apply_update(w_enc_conv2, d_w_enc_conv2, v_w_enc_conv2, batchSize);
    apply_update(b_enc_conv2, d_b_enc_conv2, v_b_enc_conv2, batchSize);

    apply_update(w_dec_conv3, d_w_dec_conv3, v_w_dec_conv3, batchSize);
    apply_update(b_dec_conv3, d_b_dec_conv3, v_b_dec_conv3, batchSize);

    apply_update(w_dec_conv4, d_w_dec_conv4, v_w_dec_conv4, batchSize);
    apply_update(b_dec_conv4, d_b_dec_conv4, v_b_dec_conv4, batchSize);

    apply_update(w_dec_conv5, d_w_dec_conv5, v_w_dec_conv5, batchSize);
    apply_update(b_dec_conv5, d_b_dec_conv5, v_b_dec_conv5, batchSize);
}

void Autoencoder::apply_update(std::vector<double>& weights, std::vector<double>& d_weights, std::vector<double>& velocity, int batchSize) {
    for(size_t i=0; i<weights.size(); i++) {
        // Chia gradient cho batch size để lấy trung bình
        double avg_grad = d_weights[i] / batchSize;
        
        // Gradient clipping to prevent exploding gradients
        const double grad_clip = 5.0;
        if (avg_grad > grad_clip) avg_grad = grad_clip;
        else if (avg_grad < -grad_clip) avg_grad = -grad_clip;
        
        // SGD with Momentum
        double v = m_momentum * velocity[i] - m_learningRate * avg_grad;
        velocity[i] = v;
        weights[i] += v;
        
        
        // Quan trọng: Reset gradient về 0 sau khi update để dùng cho batch sau
        d_weights[i] = 0.0;
    }
}

std::vector<double> Autoencoder::getOutput(const std::vector<double>& imageFlat) {
    a_input = Tensor(3, 32, 32);
    a_input.data = imageFlat;
    forwardPass();
    return a_latent.data;
}


void Autoencoder::forwardPass() {
    // ... (Code forwardPass giữ nguyên như cũ) ...
    // Copy lại nội dung forwardPass từ phiên bản trước
    
    // 1. Conv1 + ReLU
    a_enc_conv1 = Tensor(256, 32, 32);
    conv2d_forward(a_input, w_enc_conv1, b_enc_conv1, a_enc_conv1);
    relu_forward(a_enc_conv1);

    // 2. MaxPool1
    a_enc_pool1 = Tensor(256, 16, 16);
    maxpool2d_forward(a_enc_conv1, a_enc_pool1);

    // 3. Conv2 + ReLU
    a_enc_conv2 = Tensor(128, 16, 16);
    conv2d_forward(a_enc_pool1, w_enc_conv2, b_enc_conv2, a_enc_conv2);
    relu_forward(a_enc_conv2);

    // 4. MaxPool2 -> LATENT
    a_latent = Tensor(128, 8, 8);
    maxpool2d_forward(a_enc_conv2, a_latent);

    // --- DECODER ---
    // 5. Conv3 + ReLU
    a_dec_conv3 = Tensor(128, 8, 8);
    conv2d_forward(a_latent, w_dec_conv3, b_dec_conv3, a_dec_conv3);
    relu_forward(a_dec_conv3);

    // 6. UpSample1
    a_dec_ups1 = Tensor(128, 16, 16);
    upsample2d_forward(a_dec_conv3, a_dec_ups1);

    // 7. Conv4 + ReLU
    a_dec_conv4 = Tensor(256, 16, 16);
    conv2d_forward(a_dec_ups1, w_dec_conv4, b_dec_conv4, a_dec_conv4);
    relu_forward(a_dec_conv4);

    // 8. UpSample2
    a_dec_ups2 = Tensor(256, 32, 32);
    upsample2d_forward(a_dec_conv4, a_dec_ups2);

    // 9. Conv5 (Output)
    a_output = Tensor(3, 32, 32);
    conv2d_forward(a_dec_ups2, w_dec_conv5, b_dec_conv5, a_output);
}

void Autoencoder::backwardPass() {


    // --- BACKWARD DECODER ---
    // Lưu ý: d_w_dec... và d_b_dec... là biến thành viên, nên hàm conv2d_backward sẽ CỘNG DỒN (+=) vào nó
    
    // 1. Back Conv5
    Tensor d_dec_ups2(256, 32, 32);
    conv2d_backward(a_dec_ups2, d_output, w_dec_conv5, d_dec_ups2, d_w_dec_conv5, d_b_dec_conv5);

    // 2. Back UpSample2
    Tensor d_dec_conv4(256, 16, 16);
    upsample2d_backward(d_dec_ups2, d_dec_conv4);

    // 3. Back ReLU + Conv4
    relu_backward(a_dec_conv4, d_dec_conv4);
    Tensor d_dec_ups1(128, 16, 16);
    conv2d_backward(a_dec_ups1, d_dec_conv4, w_dec_conv4, d_dec_ups1, d_w_dec_conv4, d_b_dec_conv4);

    // 4. Back UpSample1
    Tensor d_dec_conv3(128, 8, 8);
    upsample2d_backward(d_dec_ups1, d_dec_conv3);

    // 5. Back ReLU + Conv3
    relu_backward(a_dec_conv3, d_dec_conv3);
    Tensor d_latent(128, 8, 8);
    conv2d_backward(a_latent, d_dec_conv3, w_dec_conv3, d_latent, d_w_dec_conv3, d_b_dec_conv3);

    // --- BACKWARD ENCODER ---
    // 6. Back MaxPool2
    Tensor d_enc_conv2(128, 16, 16);
    maxpool2d_backward(a_enc_conv2, a_latent, d_latent, d_enc_conv2);

    // 7. Back ReLU + Conv2
    relu_backward(a_enc_conv2, d_enc_conv2);
    Tensor d_enc_pool1(256, 16, 16);
    conv2d_backward(a_enc_pool1, d_enc_conv2, w_enc_conv2, d_enc_pool1, d_w_enc_conv2, d_b_enc_conv2);

    // 8. Back MaxPool1
    Tensor d_enc_conv1(256, 32, 32);
    maxpool2d_backward(a_enc_conv1, a_enc_pool1, d_enc_pool1, d_enc_conv1);

    // 9. Back ReLU + Conv1
    relu_backward(a_enc_conv1, d_enc_conv1);
    Tensor d_dummy_input(3, 32, 32);
    conv2d_backward(a_input, d_enc_conv1, w_enc_conv1, d_dummy_input, d_w_enc_conv1, d_b_enc_conv1);
}

// ... (Các hàm helper conv2d_forward, conv2d_backward... giữ nguyên logic như phiên bản trước) ...
// Để code gọn, tôi không paste lại toàn bộ nội dung helper functions nếu chúng không đổi logic.
// Tuy nhiên, logic conv2d_backward đã được viết để dùng += (accumulate), nên nó tương thích hoàn toàn.
// Chỉ cần đảm bảo hàm gọi d_input.zero() ở đầu hàm backward như phiên bản trước là đúng.

void Autoencoder::conv2d_forward(const Tensor& input, const std::vector<double>& weights, const std::vector<double>& bias, Tensor& output, int stride, int padding) {
    int K = 3;
    // ... Logic forward giữ nguyên ...
    // Để đảm bảo file chạy được, tôi chép lại đoạn code quan trọng
    
    #pragma omp parallel for 
    for(int oc = 0; oc < output.c; oc++) {
        for(int oh = 0; oh < output.h; oh++) {
            for(int ow = 0; ow < output.w; ow++) {
                double sum = bias[oc];
                for(int ic = 0; ic < input.c; ic++) {
                    for(int kh = 0; kh < K; kh++) {
                        for(int kw = 0; kw < K; kw++) {
                            int ih = oh * stride + kh - padding;
                            int iw = ow * stride + kw - padding;
                            if (ih >= 0 && ih < input.h && iw >= 0 && iw < input.w) {
                                sum += input.at(ic, ih, iw) * weights[W_IDX(oc, ic, kh, kw, input.c, K)];
                            }
                        }
                    }
                }
                output.at(oc, oh, ow) = sum;
            }
        }
    }
}

void Autoencoder::relu_forward(Tensor& t) {
    for(double& v : t.data) if (v < 0) v = 0;
}

void Autoencoder::maxpool2d_forward(const Tensor& input, Tensor& output) {
    int stride = 2; int kernel = 2;
    for(int c = 0; c < input.c; c++) {
        for(int oh = 0; oh < output.h; oh++) {
            for(int ow = 0; ow < output.w; ow++) {
                double max_val = -1e9;
                for(int kh=0; kh<kernel; kh++) {
                    for(int kw=0; kw<kernel; kw++) {
                        int ih = oh * stride + kh;
                        int iw = ow * stride + kw;
                        double val = input.at(c, ih, iw);
                        if(val > max_val) max_val = val;
                    }
                }
                output.at(c, oh, ow) = max_val;
            }
        }
    }
}

void Autoencoder::upsample2d_forward(const Tensor& input, Tensor& output) {
    int scale = 2;
    for(int c = 0; c < input.c; c++) {
        for(int ih = 0; ih < input.h; ih++) {
            for(int iw = 0; iw < input.w; iw++) {
                double val = input.at(c, ih, iw);
                output.at(c, ih*scale, iw*scale) = val;
                output.at(c, ih*scale, iw*scale+1) = val;
                output.at(c, ih*scale+1, iw*scale) = val;
                output.at(c, ih*scale+1, iw*scale+1) = val;
            }
        }
    }
}

void Autoencoder::conv2d_backward(const Tensor& input, const Tensor& d_output, const std::vector<double>& weights, Tensor& d_input, std::vector<double>& d_weights, std::vector<double>& d_bias, int stride, int padding) {
    int K = 3;
    d_input.zero(); 
    // Logic backward tích lũy (+=)
    for(int oc = 0; oc < d_output.c; oc++) {
        double sum_bias = 0.0;
        for(int oh = 0; oh < d_output.h; oh++) {
            for(int ow = 0; ow < d_output.w; ow++) {
                double d_val = d_output.at(oc, oh, ow);
                sum_bias += d_val;
                for(int ic = 0; ic < input.c; ic++) {
                    for(int kh = 0; kh < K; kh++) {
                        for(int kw = 0; kw < K; kw++) {
                            int ih = oh * stride + kh - padding;
                            int iw = ow * stride + kw - padding;
                            if (ih >= 0 && ih < input.h && iw >= 0 && iw < input.w) {
                                // Tích lũy gradient weights
                                d_weights[W_IDX(oc, ic, kh, kw, input.c, K)] += input.at(ic, ih, iw) * d_val;
                                // Tích lũy gradient input
                                d_input.at(ic, ih, iw) += weights[W_IDX(oc, ic, kh, kw, input.c, K)] * d_val;
                            }
                        }
                    }
                }
            }
        }
        // Tích lũy gradient bias
        d_bias[oc] += sum_bias;
    }
}

void Autoencoder::relu_backward(const Tensor& input, Tensor& d_input) {
    int size = input.data.size();
    for(int i=0; i<size; i++) if(input.data[i] <= 0) d_input.data[i] = 0;
}

void Autoencoder::maxpool2d_backward(const Tensor& input, const Tensor& output, const Tensor& d_output, Tensor& d_input) {
    d_input.zero();
    int stride = 2;
    for(int c = 0; c < input.c; c++) {
        for(int oh = 0; oh < output.h; oh++) {
            for(int ow = 0; ow < output.w; ow++) {
                double d_val = d_output.at(c, oh, ow);
                int ih_start = oh * stride; int iw_start = ow * stride;
                double max_val = output.at(c, oh, ow);
                for(int kh=0; kh<2; kh++) {
                    for(int kw=0; kw<2; kw++) {
                        int ih = ih_start + kh; int iw = iw_start + kw;
                        if(abs(input.at(c, ih, iw) - max_val) < 1e-9) {
                            d_input.at(c, ih, iw) += d_val;
                        }
                    }
                }
            }
        }
    }
}

void Autoencoder::upsample2d_backward(const Tensor& d_output, Tensor& d_input) {
    int scale = 2;
    for(int c = 0; c < d_input.c; c++) {
        for(int ih = 0; ih < d_input.h; ih++) {
            for(int iw = 0; iw < d_input.w; iw++) {
                double sum = 0;
                sum += d_output.at(c, ih*scale, iw*scale);
                sum += d_output.at(c, ih*scale, iw*scale+1);
                sum += d_output.at(c, ih*scale+1, iw*scale);
                sum += d_output.at(c, ih*scale+1, iw*scale+1);
                d_input.at(c, ih, iw) = sum;
            }
        }
    }
}

void Autoencoder::save_weights(const std::string& filepath) {
    ofstream file(filepath, ios::binary);
    if (!file.is_open()) {
        cerr << "Error opening file for saving weights: " << filepath << endl;
        return;
    }

    // Save raw data without size metadata (compatible with GPU version)
    auto save_layer = [&](const vector<double>& weights, const vector<double>& bias) {
        file.write(reinterpret_cast<const char*>(weights.data()), weights.size() * sizeof(double));
        file.write(reinterpret_cast<const char*>(bias.data()), bias.size() * sizeof(double));
    };

    save_layer(w_enc_conv1, b_enc_conv1);
    save_layer(w_enc_conv2, b_enc_conv2);
    save_layer(w_dec_conv3, b_dec_conv3);
    save_layer(w_dec_conv4, b_dec_conv4);
    save_layer(w_dec_conv5, b_dec_conv5);

    cout<<"Weights saved to " << filepath << endl;
    file.close();
}

void Autoencoder::load_weights(const std::string& filepath) {
    ifstream ifs(filepath, ios::binary);
    if (!ifs.is_open()) {
        cerr << "Error opening file for loading weights: " << filepath << endl;
        return;
    }

    // Load raw data without size metadata (compatible with GPU version)
    auto load_layer = [&](vector<double>& weights, vector<double>& bias, int outC, int inC, int K) {
        int w_size = outC * inC * K * K;
        int b_size = outC;
        
        weights.resize(w_size);
        bias.resize(b_size);
        
        ifs.read(reinterpret_cast<char*>(weights.data()), w_size * sizeof(double));
        ifs.read(reinterpret_cast<char*>(bias.data()), b_size * sizeof(double));
        
        if (ifs.fail()) {
            cerr << "Error: Failed to read weights from file" << endl;
            return;
        }
    };

    load_layer(w_enc_conv1, b_enc_conv1, 256, 3, 3);
    load_layer(w_enc_conv2, b_enc_conv2, 128, 256, 3);
    load_layer(w_dec_conv3, b_dec_conv3, 128, 128, 3);
    load_layer(w_dec_conv4, b_dec_conv4, 256, 128, 3);
    load_layer(w_dec_conv5, b_dec_conv5, 3, 256, 3);

    ifs.close();
}

double Autoencoder::getLoss() const { return m_loss; }