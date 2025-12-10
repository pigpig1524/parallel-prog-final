#include "gpu_autoencoder.h"
#include <cstdio>
#include <cmath>
#include <random>
#include <iostream>
#include <fstream>

// ================= CUDA KERNELS (NAIVE & REDUCTION) =================

// --- FORWARD ---
__global__ void k_conv2d_forward(double* input, double* weights, double* bias, double* output,
                                 int inC, int outC, int H, int W, int K, int padding, int stride) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int b_oc = blockIdx.z;

    if (ow >= W || oh >= H) return;

    int oc = b_oc % outC;
    int b = b_oc / outC;

    double sum = bias[oc];

    for (int ic = 0; ic < inC; ++ic) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                int ih = oh * stride + kh - padding;
                int iw = ow * stride + kw - padding;
                ih = max(min(ih, H - 1), 0);
                iw = max(min(iw, W - 1), 0);
                // Tính flat index dựa vào batch, in_Channel, H và W
                int in_idx = ((b * inC + ic) * H + ih) * W + iw;
                int w_idx = ((oc * inC + ic) * K + kh) * K + kw;
                sum += input[in_idx] * weights[w_idx];
                
            }
        }
    }

    int out_idx = ((b * outC + oc) * H + oh) * W + ow;
    output[out_idx] = sum;
}

// Kernel Forward ReLU
__global__ void k_relu_forward(double* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (data[idx] < 0) data[idx] = 0;
    }
}

// Kernel Forward MaxPool
__global__ void k_maxpool_forward(double* input, double* output, int C, int inH, int inW, int outH, int outW) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int bc = blockIdx.z;
    int K = 2; // kernel_size = 2

    if (ow >= outW || oh >= outH) return;
    
    int ih_start = oh * K;
    int iw_start = ow * K;

    double max_val = -1e30;

    for (int kh = 0; kh < K; ++kh) {
        for (int kw = 0; kw < K; ++kw) {
            int hh = ih_start + kh;
            int ww = iw_start + kw;
            hh = max(min(hh, inH - 1), 0);
            ww = max(min(ww, inW - 1), 0);

            int idx = (bc * inH + hh) * inW + ww;
            double val = input[idx];
            if (val > max_val) max_val = val;
        }
    }
    int out_idx = (bc * outH + oh) * outW + ow;
    output[out_idx] = max_val;
}

// Kernel Forward UpSample
__global__ void k_upsample_forward(double* input, double* output, int C, int inH, int inW, int outH, int outW) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int bc = blockIdx.z;

    if (ow >= outW || oh >= outH) return;

    int ih = oh / 2;
    int iw = ow / 2;
    int in_idx = (bc * inH + ih) * inW + iw;
    int out_idx = (bc * outH + oh) * outW + ow;

    output[out_idx] = input[in_idx];
}

// --- UPDATED: MSE Loss Kernel with SHARED MEMORY Reduction ---
// Requirement 2.2: Use shared memory for partial sums
__global__ void k_mse_loss_backward_input(double* predicted, double* target, double* d_grad, double* avg_grad_sum, double* loss_val, int size) {
    // Shared memory để lưu tổng cục bộ của block
    extern __shared__ double sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    double sq_diff = 0.0f;
    double grad_abs = 0.0f;
    
    // 1. Tính toán diff và gradient cho pixel-wise, đồng thời tính squared error
    if (idx < size) {
        double diff = target[idx] - predicted[idx];
        d_grad[idx] = 2.0f * diff / size; // Gradient: dL/dOutput
        sq_diff = diff * diff / size; // Chia cho tổng số elements để có MSE
        grad_abs = fabs(d_grad[idx]);
        
        // Atomic add to global gradient sum
        atomicAdd(avg_grad_sum, grad_abs);
    }
    
    // 2. Load vào Shared Memory
    sdata[tid] = sq_diff;
    __syncthreads();

    // 3. Parallel Reduction trong Shared Memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 4. Thread 0 của mỗi block sẽ atomicAdd vào biến toàn cục
    if (tid == 0) {
        atomicAdd(loss_val, sdata[0]);
    }
}

// --- BACKWARD ---

__global__ void k_relu_backward(double* input_val, double* d_output, double* d_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_input[idx] = (input_val[idx] > 0) ? d_output[idx] : 0.0f;
    }
}

__global__ void k_maxpool_backward(double* input, double* output, double* d_output, double* d_input, 
                                   int inH, int inW, int outH, int outW) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int bc = blockIdx.z;

    if (ow >= outW || oh >= outH) return;

    int out_idx = (bc * outH + oh) * outW + ow;
    double max_val = output[out_idx];
    double d_val = d_output[out_idx];

    int ih_start = oh * 2;
    int iw_start = ow * 2;

    for (int kh = 0; kh < 2; ++kh) {
        for (int kw = 0; kw < 2; ++kw) {
            int in_idx = (bc * inH + (ih_start + kh)) * inW + (iw_start + kw);
            if (fabs(input[in_idx] - max_val) < 1e-6) {
                d_input[in_idx] = d_val; 
            } else {
                d_input[in_idx] = 0; 
            }
        }
    }
}

__global__ void k_upsample_backward(double* d_output, double* d_input, int inH, int inW, int outH, int outW) {
    int iw = blockIdx.x * blockDim.x + threadIdx.x;
    int ih = blockIdx.y * blockDim.y + threadIdx.y;
    int bc = blockIdx.z;

    if (iw >= inW || ih >= inH) return;

    double sum = 0.0f;
    int oh_start = ih * 2;
    int ow_start = iw * 2;

    for(int kh=0; kh<2; ++kh) {
        for(int kw=0; kw<2; ++kw) {
            int out_idx = (bc * outH + (oh_start+kh)) * outW + (ow_start+kw);
            sum += d_output[out_idx];
        }
    }
    int in_idx = (bc * inH + ih) * inW + iw;
    d_input[in_idx] = sum;
}

__global__ void k_conv2d_backward_input(double* d_output, double* weights, double* d_input,
                                        int inC, int outC, int H, int W, int K, int padding, int stride) {
    int iw = blockIdx.x * blockDim.x + threadIdx.x;
    int ih = blockIdx.y * blockDim.y + threadIdx.y;
    int b_ic = blockIdx.z;

    if (iw >= W || ih >= H) return;

    int ic = b_ic % inC;
    int b = b_ic / inC;
    double sum = 0.0f;

    for(int oc=0; oc<outC; ++oc) {
        for(int kh=0; kh<K; ++kh) {
            for(int kw=0; kw<K; ++kw) {
                int oh_num = ih + padding - kh;
                int ow_num = iw + padding - kw;
                if (oh_num % stride == 0 && ow_num % stride == 0) {
                    int oh = oh_num / stride;
                    int ow = ow_num / stride;
                    if (oh >= 0 && oh < H && ow >= 0 && ow < W) {
                        int w_idx = ((oc * inC + ic) * K + kh) * K + kw;
                        int dout_idx = ((b * outC + oc) * H + oh) * W + ow;
                        sum += d_output[dout_idx] * weights[w_idx];
                    }
                }
            }
        }
    }
    int din_idx = ((b * inC + ic) * H + ih) * W + iw;
    d_input[din_idx] = sum;
}

__global__ void k_conv2d_backward_weights(double* input, double* d_output, double* d_weights, double* d_bias,
                                          int inC, int outC, int H, int W, int K, int padding, int stride) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int b_oc = blockIdx.z;

    if (ow >= W || oh >= H) return;

    int oc = b_oc % outC;
    int b = b_oc / outC;
    
    int dout_idx = ((b * outC + oc) * H + oh) * W + ow;
    double d_val = d_output[dout_idx];

    atomicAdd(&d_bias[oc], d_val);

    for(int ic=0; ic<inC; ++ic) {
        for(int kh=0; kh<K; ++kh) {
            for(int kw=0; kw<K; ++kw) {
                int ih = oh * stride + kh - padding;
                int iw = ow * stride + kw - padding;

                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    int in_idx = ((b * inC + ic) * H + ih) * W + iw;
                    double val = input[in_idx];
                    int w_idx = ((oc * inC + ic) * K + kh) * K + kw;
                    atomicAdd(&d_weights[w_idx], val * d_val);
                }
            }
        }
    }
}

__global__ void apply_update(double* weights, double* d_weights, double* velocity, 
                                 double lr, double momentum, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        double grad = d_weights[idx]; 
        
        const double grad_clip = 5.0;
        if (grad > grad_clip) grad = grad_clip;
        else if (grad < -grad_clip) grad = -grad_clip;

        double v = momentum * velocity[idx] - lr * grad;
        velocity[idx] = v;
        weights[idx] += v;
        d_weights[idx] = 0.0f;
    }
}

// ================= CLASS IMPLEMENTATION =================

GPUAutoencoder::GPUAutoencoder(double learningRate, double momentum) 
    : m_learningRate(learningRate), m_momentum(momentum) 
{
    initWeightsRandomly();
}

void random_init(std::vector<double>& vec, int fan_in) {
    // Xavier/Glorot initialization for better stability
    double limit = sqrt(2.0f / (fan_in + vec.size()));
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<double> dist(-limit, limit);
    for (double& val : vec) val = dist(generator);
}

void GPUAutoencoder::initWeightsRandomly() {
    int K = 3;
    auto init_layer = [&](double** d_w, double** d_b, double** d_dw, double** d_db, double** d_vw, double** d_vb, int outC, int inC) {
        int w_size = outC * inC * K * K;
        int b_size = outC;
        
        std::vector<double> h_w(w_size);
        std::vector<double> h_b(b_size, 0.0f);
        
        random_init(h_w, inC * K * K);

        CHECK(cudaMalloc(d_w, w_size * sizeof(double)));
        CHECK(cudaMalloc(d_b, b_size * sizeof(double)));
        CHECK(cudaMalloc(d_dw, w_size * sizeof(double)));
        CHECK(cudaMalloc(d_db, b_size * sizeof(double)));
        CHECK(cudaMalloc(d_vw, w_size * sizeof(double)));
        CHECK(cudaMalloc(d_vb, b_size * sizeof(double)));

        CHECK(cudaMemcpy(*d_w, h_w.data(), w_size * sizeof(double), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(*d_b, h_b.data(), b_size * sizeof(double), cudaMemcpyHostToDevice));
        
        CHECK(cudaMemset(*d_dw, 0, w_size * sizeof(double)));
        CHECK(cudaMemset(*d_db, 0, b_size * sizeof(double)));
        CHECK(cudaMemset(*d_vw, 0, w_size * sizeof(double)));
        CHECK(cudaMemset(*d_vb, 0, b_size * sizeof(double)));
    };

    init_layer(&d_w_enc_conv1, &d_b_enc_conv1, &d_dw_enc_conv1, &d_db_enc_conv1, &d_v_enc_conv1, &d_v_enc_conv1_b, 256, 3);
    init_layer(&d_w_enc_conv2, &d_b_enc_conv2, &d_dw_enc_conv2, &d_db_enc_conv2, &d_v_enc_conv2, &d_v_enc_conv2_b, 128, 256);
    init_layer(&d_w_dec_conv3, &d_b_dec_conv3, &d_dw_dec_conv3, &d_db_dec_conv3, &d_v_dec_conv3, &d_v_dec_conv3_b, 128, 128);
    init_layer(&d_w_dec_conv4, &d_b_dec_conv4, &d_dw_dec_conv4, &d_db_dec_conv4, &d_v_dec_conv4, &d_v_dec_conv4_b, 256, 128);
    init_layer(&d_w_dec_conv5, &d_b_dec_conv5, &d_dw_dec_conv5, &d_db_dec_conv5, &d_v_dec_conv5, &d_v_dec_conv5_b, 3, 256);
    allocateMemory(32);
}

void GPUAutoencoder::allocateMemory(int batchSize) {
    auto malloc_tensor = [&](double** ptr, int c, int h, int w) {
        CHECK(cudaMalloc(ptr, batchSize * c * h * w * sizeof(double)));
    };
    //Khởi tạo bộ nhớ cho các tensor hiddenState và gradients
    malloc_tensor(&d_input, 3, 32, 32);
    malloc_tensor(&d_enc_conv1, 256, 32, 32);
    malloc_tensor(&d_enc_pool1, 256, 16, 16);
    malloc_tensor(&d_enc_conv2, 128, 16, 16);
    malloc_tensor(&d_latent, 128, 8, 8);
    malloc_tensor(&d_dec_conv3, 128, 8, 8);
    malloc_tensor(&d_dec_ups1, 128, 16, 16);
    malloc_tensor(&d_dec_conv4, 256, 16, 16);
    malloc_tensor(&d_dec_ups2, 256, 32, 32);
    malloc_tensor(&d_output, 3, 32, 32);

    // Gradients dim = dim of corresponding tensors
    malloc_tensor(&d_grad_output, 3, 32, 32);
    malloc_tensor(&d_grad_dec_ups2, 256, 32, 32);
    malloc_tensor(&d_grad_dec_conv4, 256, 16, 16);
    malloc_tensor(&d_grad_dec_ups1, 128, 16, 16);
    malloc_tensor(&d_grad_dec_conv3, 128, 8, 8);
    malloc_tensor(&d_grad_latent, 128, 8, 8);
    malloc_tensor(&d_grad_enc_conv2, 128, 16, 16);
    malloc_tensor(&d_grad_enc_pool1, 256, 16, 16);
    malloc_tensor(&d_grad_enc_conv1, 256, 32, 32);
    malloc_tensor(&d_grad_input, 3, 32, 32);
}

// Requirement 2.1: Implement functions to copy weights between host and device
void GPUAutoencoder::get_weights_to_host() {
    // Demo copy một lớp về để debug/save. Cần làm tương tự cho tất cả các lớp.
    // Trong thực tế, bạn sẽ truyền con trỏ host buffer vào hàm này.
    std::cout << "Weights copy functionality is ready but requires Host buffers." << std::endl;
}

void GPUAutoencoder::getOutput(const std::vector<double>& h_inputBatch, std::vector<double>& h_output, int batchSize) {
    train_batch(h_inputBatch, batchSize);
    int size = batchSize * 3 * 32 * 32;
    h_output.resize(size);
    CHECK(cudaMemcpy(h_output.data(), d_output, size * sizeof(double), cudaMemcpyDeviceToHost));
}



void GPUAutoencoder::train_batch(const std::vector<double>& h_inputBatch, int batchSize) {
    CHECK(cudaMemcpy(d_input, h_inputBatch.data(), h_inputBatch.size() * sizeof(double), cudaMemcpyHostToDevice));

    forwardPass(batchSize);

    CHECK(cudaDeviceSynchronize());

    double* d_loss; 
    double* d_avg_grad_sum;
    CHECK(cudaMalloc(&d_loss, sizeof(double)));
    CHECK(cudaMalloc(&d_avg_grad_sum, sizeof(double)));
    CHECK(cudaMemset(d_loss, 0, sizeof(double)));
    CHECK(cudaMemset(d_avg_grad_sum, 0, sizeof(double)));
    

    int size = batchSize * 3 * 32 * 32; //total numbers of pixels of both input and output
    int blockSize = 32;
    int gridSize = (size - 1) / blockSize + 1;

    k_mse_loss_backward_input<<<gridSize, blockSize, blockSize * sizeof(double)>>>(d_output, d_input, d_grad_output, d_avg_grad_sum, d_loss, size);
    
    CHECK(cudaMemcpy(&m_loss, d_loss, sizeof(double), cudaMemcpyDeviceToHost));
    
    double avg_grad_sum;
    CHECK(cudaMemcpy(&avg_grad_sum, d_avg_grad_sum, sizeof(double), cudaMemcpyDeviceToHost));
    double avg_grad = avg_grad_sum; // Calculate average

    if (train) {
        backwardPass(batchSize);
        updateWeights();
    }
    printf("\nAverage gradient: %f\n", avg_grad);
    CHECK(cudaFree(d_loss));
    CHECK(cudaFree(d_avg_grad_sum));
}

void GPUAutoencoder::forwardPass(int batchSize){
    int blockSize = 32;
    dim3 block(blockSize, blockSize);
    int H_in = 32, W_in = 32, outC = 256, inC = 3;
    int filterWidth = 3, padding = 1, stride = 1;
    dim3 grid((W_in-1)/blockSize + 1, (H_in-1)/blockSize + 1, batchSize*outC);
    k_conv2d_forward<<<grid, block>>>(d_input, d_w_enc_conv1, d_b_enc_conv1, d_enc_conv1, inC, outC, H_in, W_in, filterWidth, padding, stride);    

    k_relu_forward<<<(batchSize*outC*H_in*W_in-1)/256 + 1, 256>>>(d_enc_conv1, batchSize*outC*H_in*W_in);

    inC = 256, outC = 256, filterWidth = 2;
    grid = dim3((W_in-1)/blockSize + 1, (H_in-1)/blockSize + 1, batchSize*outC);
    k_maxpool_forward<<<grid, block>>>(d_enc_conv1, d_enc_pool1, inC, H_in, W_in, H_in/2, W_in/2);

    H_in = 16; W_in = 16; inC = 256; outC = 128; filterWidth = 3;
    grid = dim3((W_in-1)/blockSize + 1, (H_in-1)/blockSize + 1, batchSize*outC);    
    k_conv2d_forward<<<grid, block>>>(d_enc_pool1, d_w_enc_conv2, d_b_enc_conv2, d_enc_conv2, inC, outC, H_in, W_in, filterWidth, padding, stride);
    k_relu_forward<<<(batchSize*outC*H_in*W_in-1)/256 + 1, 256>>>(d_enc_conv2, batchSize*outC*H_in*W_in);

    filterWidth = 2; inC=128; outC=128;
    grid = dim3((W_in-1)/blockSize + 1, (H_in-1)/blockSize + 1, batchSize*outC);
    k_maxpool_forward<<<grid, block>>>(d_enc_conv2, d_latent, inC, H_in, W_in, H_in/2, W_in/2);

    H_in=8; W_in=8; inC=128; outC=128; filterWidth=3;
    grid = dim3((W_in-1)/blockSize + 1, (H_in-1)/blockSize + 1, batchSize*outC);
    k_conv2d_forward<<<grid, block>>>(d_latent, d_w_dec_conv3, d_b_dec_conv3, d_dec_conv3, inC, outC, H_in, W_in, filterWidth, padding, stride);

    k_relu_forward<<<(batchSize*outC*H_in*W_in-1)/256 + 1, 256>>>(d_dec_conv3, batchSize*outC*H_in*W_in);

    filterWidth=2; inC=128; outC=128;
    grid = dim3((W_in-1)/blockSize + 1, (H_in-1)/blockSize + 1, batchSize*outC);
    k_upsample_forward<<<grid, block>>>(d_dec_conv3, d_dec_ups1, inC, H_in, W_in, H_in*2, W_in*2);

    H_in = 16; W_in = 16; inC = 128; outC = 256; filterWidth = 3;
    grid = dim3((W_in-1)/blockSize + 1, (H_in-1)/blockSize + 1, batchSize*outC);
    k_conv2d_forward<<<grid, block>>>(d_dec_ups1, d_w_dec_conv4, d_b_dec_conv4, d_dec_conv4, inC, outC, H_in, W_in, filterWidth, padding, stride);

    k_relu_forward<<<(batchSize*outC*H_in*W_in-1)/256 + 1, 256>>>(d_dec_conv4, batchSize*outC*H_in*W_in);

    filterWidth=2; inC=256; outC=256;
    grid = dim3((W_in-1)/blockSize + 1, (H_in-1)/blockSize + 1, batchSize*outC);
    k_upsample_forward<<<grid, block>>>(d_dec_conv4, d_dec_ups2, inC, H_in, W_in, H_in*2, W_in*2);

    H_in = 32; W_in = 32; inC = 256; outC = 3; filterWidth = 3;
    grid = dim3((W_in-1)/blockSize + 1, (H_in-1)/blockSize + 1, batchSize*outC);
    k_conv2d_forward<<<grid, block>>>(d_dec_ups2, d_w_dec_conv5, d_b_dec_conv5, d_output, inC, outC, H_in, W_in, filterWidth, padding, stride);

}

void GPUAutoencoder::backwardPass(int batchSize) {
    int blockSize = 32;
    dim3 block(blockSize, blockSize);
    
    // Output layer (dec_conv5) - 32x32x3
    int H_out = 32, W_out = 32, outC = 3, inC = 256;
    int filterWidth = 3, padding = 1, stride = 1;
    dim3 grid((W_out-1)/blockSize + 1, (H_out-1)/blockSize + 1, batchSize*outC);
    k_conv2d_backward_weights<<<grid, block>>>(d_dec_ups2, d_grad_output, d_dw_dec_conv5, d_db_dec_conv5, inC, outC, H_out, W_out, filterWidth, padding, stride);
    
    grid = dim3((W_out-1)/blockSize + 1, (H_out-1)/blockSize + 1, batchSize*inC);
    k_conv2d_backward_input<<<grid, block>>>(d_grad_output, d_w_dec_conv5, d_grad_dec_ups2, inC, outC, H_out, W_out, filterWidth, padding, stride);
    
    // Upsample2 backward
    inC = 256; outC = 256; filterWidth = 2;
    grid = dim3((W_out-1)/blockSize + 1, (H_out-1)/blockSize + 1, batchSize*inC);
    k_upsample_backward<<<grid, block>>>(d_grad_dec_ups2, d_grad_dec_conv4, H_out/2, W_out/2, H_out, W_out);
    
    // dec_conv4 backward - 16x16x256  
    H_out = 16; W_out = 16; outC = 256; inC = 128; filterWidth = 3;
    k_relu_backward<<<(batchSize*outC*H_out*W_out-1)/256 + 1, 256>>>(d_dec_conv4, d_grad_dec_conv4, d_grad_dec_conv4, batchSize*outC*H_out*W_out);
    
    grid = dim3((W_out-1)/blockSize + 1, (H_out-1)/blockSize + 1, batchSize*outC);
    k_conv2d_backward_weights<<<grid, block>>>(d_dec_ups1, d_grad_dec_conv4, d_dw_dec_conv4, d_db_dec_conv4, inC, outC, H_out, W_out, filterWidth, padding, stride);
    
    grid = dim3((W_out-1)/blockSize + 1, (H_out-1)/blockSize + 1, batchSize*inC);
    k_conv2d_backward_input<<<grid, block>>>(d_grad_dec_conv4, d_w_dec_conv4, d_grad_dec_ups1, inC, outC, H_out, W_out, filterWidth, padding, stride);

    // Upsample1 backward
    inC = 128; filterWidth = 2;
    grid = dim3((W_out-1)/blockSize + 1, (H_out-1)/blockSize + 1, batchSize*inC);
    k_upsample_backward<<<grid, block>>>(d_grad_dec_ups1, d_grad_dec_conv3, H_out/2, W_out/2, H_out, W_out);
    
    // dec_conv3 backward - 8x8x128
    H_out = 8; W_out = 8; outC = 128; inC = 128; filterWidth = 3;
    k_relu_backward<<<(batchSize*outC*H_out*W_out-1)/256 + 1, 256>>>(d_dec_conv3, d_grad_dec_conv3, d_grad_dec_conv3, batchSize*outC*H_out*W_out);
    
    grid = dim3((W_out-1)/blockSize + 1, (H_out-1)/blockSize + 1, batchSize*outC);
    k_conv2d_backward_weights<<<grid, block>>>(d_latent, d_grad_dec_conv3, d_dw_dec_conv3, d_db_dec_conv3, inC, outC, H_out, W_out, filterWidth, padding, stride);
    
    grid = dim3((W_out-1)/blockSize + 1, (H_out-1)/blockSize + 1, batchSize*inC);
    k_conv2d_backward_input<<<grid, block>>>(d_grad_dec_conv3, d_w_dec_conv3, d_grad_latent, inC, outC, H_out, W_out, filterWidth, padding, stride);

    // MaxPool backward to enc_conv2 - 16x16x128 
    inC = 128; filterWidth = 2;
    grid = dim3((W_out*2-1)/blockSize + 1, (H_out*2-1)/blockSize + 1, batchSize*inC);
    k_maxpool_backward<<<grid, block>>>(d_enc_conv2, d_latent, d_grad_latent, d_grad_enc_conv2, H_out*2, W_out*2, H_out, W_out);
    
    // enc_conv2 backward - 16x16x128
    H_out = 16; W_out = 16; outC = 128; inC = 256; filterWidth = 3;
    k_relu_backward<<<(batchSize*outC*H_out*W_out-1)/256 + 1, 256>>>(d_enc_conv2, d_grad_enc_conv2, d_grad_enc_conv2, batchSize*outC*H_out*W_out);
    
    grid = dim3((W_out-1)/blockSize + 1, (H_out-1)/blockSize + 1, batchSize*outC);
    k_conv2d_backward_weights<<<grid, block>>>(d_enc_pool1, d_grad_enc_conv2, d_dw_enc_conv2, d_db_enc_conv2, inC, outC, H_out, W_out, filterWidth, padding, stride);
    
    grid = dim3((W_out-1)/blockSize + 1, (H_out-1)/blockSize + 1, batchSize*inC);
    k_conv2d_backward_input<<<grid, block>>>(d_grad_enc_conv2, d_w_enc_conv2, d_grad_enc_pool1, inC, outC, H_out, W_out, filterWidth, padding, stride);

    // MaxPool backward to enc_conv1 - 32x32x256
    inC = 256; filterWidth = 2;
    grid = dim3((W_out*2-1)/blockSize + 1, (H_out*2-1)/blockSize + 1, batchSize*inC);
    k_maxpool_backward<<<grid, block>>>(d_enc_conv1, d_enc_pool1, d_grad_enc_pool1, d_grad_enc_conv1, H_out*2, W_out*2, H_out, W_out);
    
    // enc_conv1 backward - 32x32x256
    H_out = 32; W_out = 32; outC = 256; inC = 3; filterWidth = 3;
    k_relu_backward<<<(batchSize*outC*H_out*W_out-1)/256 + 1, 256>>>(d_enc_conv1, d_grad_enc_conv1, d_grad_enc_conv1, batchSize*outC*H_out*W_out);
    
    grid = dim3((W_out-1)/blockSize + 1, (H_out-1)/blockSize + 1, batchSize*outC);
    k_conv2d_backward_weights<<<grid, block>>>(d_input, d_grad_enc_conv1, d_dw_enc_conv1, d_db_enc_conv1, inC, outC, H_out, W_out, filterWidth, padding, stride);
    // No need backward input for the very first layer
}

void GPUAutoencoder::updateWeights() {
    int blockSize = 256;
    int outC, inC, K, gridSize;
    
    outC=256; inC=3; K=3;
    gridSize = (outC*inC*K*K - 1) / blockSize + 1;
    apply_update<<<gridSize, blockSize>>>(d_w_enc_conv1, d_dw_enc_conv1, d_v_enc_conv1, m_learningRate, m_momentum, 256*3*3*3);
    gridSize = (outC - 1) / blockSize + 1;
    apply_update<<<gridSize, blockSize>>>(d_b_enc_conv1, d_db_enc_conv1, d_v_enc_conv1_b, m_learningRate, m_momentum, 256);

    outC=256; inC=128; K=3;
    gridSize = (outC*inC*K*K - 1) / blockSize + 1;
    apply_update<<<gridSize, blockSize>>>(d_w_enc_conv2, d_dw_enc_conv2, d_v_enc_conv2, m_learningRate, m_momentum, outC*inC*K*K);
    gridSize = (outC - 1) / blockSize + 1;
    apply_update<<<gridSize, blockSize>>>(d_b_enc_conv2, d_db_enc_conv2, d_v_enc_conv2_b, m_learningRate, m_momentum, outC);
    
    outC=128; inC=128; K=3;
    gridSize = (outC*inC*K*K - 1) / blockSize + 1;
    apply_update<<<gridSize, blockSize>>>(d_w_dec_conv3, d_dw_dec_conv3, d_v_dec_conv3, m_learningRate, m_momentum, outC*inC*K*K);
    gridSize = (outC - 1) / blockSize + 1;
    apply_update<<<gridSize, blockSize>>>(d_b_dec_conv3, d_db_dec_conv3, d_v_dec_conv3_b, m_learningRate, m_momentum, outC);

    outC=256; inC=128; K=3;
    gridSize = (outC*inC*K*K - 1) / blockSize + 1;
    apply_update<<<gridSize, blockSize>>>(d_w_dec_conv4, d_dw_dec_conv4, d_v_dec_conv4, m_learningRate, m_momentum, outC*inC*K*K);
    gridSize = (outC - 1) / blockSize + 1;
    apply_update<<<gridSize, blockSize>>>(d_b_dec_conv4, d_db_dec_conv4, d_v_dec_conv4_b, m_learningRate, m_momentum, outC);

    outC=3; inC=256; K=3;
    gridSize = (outC*inC*K*K - 1) / blockSize + 1;
    apply_update<<<gridSize, blockSize>>>(d_w_dec_conv5, d_dw_dec_conv5, d_v_dec_conv5, m_learningRate, m_momentum, outC*inC*K*K);
    gridSize = (outC - 1) / blockSize + 1;
    apply_update<<<gridSize, blockSize>>>(d_b_dec_conv5, d_db_dec_conv5, d_v_dec_conv5_b, m_learningRate, m_momentum, outC);
}


void GPUAutoencoder::load_weights(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file for reading: " << filepath << std::endl;
        return;
    }
    
    // Helper lambda to load layer weights and biases
    auto load_layer = [&](double* d_w, double* d_b, int outC, int inC, int K) {
        int w_size = outC * inC * K * K;
        int b_size = outC;
        
        // Read from file to host
        std::vector<double> h_weights(w_size);
        std::vector<double> h_bias(b_size);
        
        file.read(reinterpret_cast<char*>(h_weights.data()), w_size * sizeof(double));
        file.read(reinterpret_cast<char*>(h_bias.data()), b_size * sizeof(double));
        
        if (file.fail()) {
            std::cerr << "Error: Failed to read weights from file" << std::endl;
            return;
        }
        
        // Copy from host to device
        CHECK(cudaMemcpy(d_w, h_weights.data(), w_size * sizeof(double), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_b, h_bias.data(), b_size * sizeof(double), cudaMemcpyHostToDevice));
    };      
    
    load_layer(d_w_enc_conv1, d_b_enc_conv1, 256, 3, 3);
    load_layer(d_w_enc_conv2, d_b_enc_conv2, 128, 256, 3);
    load_layer(d_w_dec_conv3, d_b_dec_conv3, 128, 128, 3);
    load_layer(d_w_dec_conv4, d_b_dec_conv4, 256, 128, 3);
    load_layer(d_w_dec_conv5, d_b_dec_conv5, 3, 256, 3);
    
    file.close();
    std::cout << "Weights loaded from: " << filepath << std::endl;
}

void GPUAutoencoder::save_weights(const std::string& filepath) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file for writing: " << filepath << std::endl;
        return;
    }
    
    // Helper lambda to     save layer weights and biases
    auto save_layer = [&](double* d_w, double* d_b, int outC, int inC, int K) {
        int w_size = outC * inC * K * K;
        int b_size = outC;
        
        // Copy weights from device to host
        std::vector<double> h_weights(w_size);
        std::vector<double> h_bias(b_size);
        
        CHECK(cudaMemcpy(h_weights.data(), d_w, w_size * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(h_bias.data(), d_b, b_size * sizeof(double), cudaMemcpyDeviceToHost));
        
        // Write to file
        file.write(reinterpret_cast<const char*>(h_weights.data()), w_size * sizeof(double));
        file.write(reinterpret_cast<const char*>(h_bias.data()), b_size * sizeof(double));
    };
    
    // Save all layers in order
    save_layer(d_w_enc_conv1, d_b_enc_conv1, 256, 3, 3);
    save_layer(d_w_enc_conv2, d_b_enc_conv2, 128, 256, 3);
    save_layer(d_w_dec_conv3, d_b_dec_conv3, 128, 128, 3);
    save_layer(d_w_dec_conv4, d_b_dec_conv4, 256, 128, 3);
    save_layer(d_w_dec_conv5, d_b_dec_conv5, 3, 256, 3);
    
    file.close();
    std::cout << "Weights saved to: " << filepath << std::endl;
}

// Requirement 2.1: Implement proper memory cleanup
GPUAutoencoder::~GPUAutoencoder() {
    
    

    CHECK(cudaFree(d_input)); CHECK(cudaFree(d_enc_conv1)); CHECK(cudaFree(d_enc_pool1)); CHECK(cudaFree(d_enc_conv2)); CHECK(cudaFree(d_latent));
    CHECK(cudaFree(d_dec_conv3)); CHECK(cudaFree(d_dec_ups1)); CHECK(cudaFree(d_dec_conv4)); CHECK(cudaFree(d_dec_ups2)); CHECK(cudaFree(d_output));

    CHECK(cudaFree(d_grad_output)); CHECK(cudaFree(d_grad_dec_ups2)); CHECK(cudaFree(d_grad_dec_conv4)); CHECK(cudaFree(d_grad_dec_ups1));
    CHECK(cudaFree(d_grad_dec_conv3)); CHECK(cudaFree(d_grad_latent)); CHECK(cudaFree(d_grad_enc_conv2)); CHECK(cudaFree(d_grad_enc_pool1));
    CHECK(cudaFree(d_grad_enc_conv1)); CHECK(cudaFree(d_grad_input));

    CHECK(cudaFree(d_w_enc_conv1)); CHECK(cudaFree(d_b_enc_conv1)); CHECK(cudaFree(d_dw_enc_conv1)); CHECK(cudaFree(d_db_enc_conv1)); CHECK(cudaFree(d_v_enc_conv1)); CHECK(cudaFree(d_v_enc_conv1_b));
    CHECK(cudaFree(d_w_enc_conv2)); CHECK(cudaFree(d_b_enc_conv2)); CHECK(cudaFree(d_dw_enc_conv2)); CHECK(cudaFree(d_db_enc_conv2)); CHECK(cudaFree(d_v_enc_conv2)); CHECK(cudaFree(d_v_enc_conv2_b));
    CHECK(cudaFree(d_w_dec_conv3)); CHECK(cudaFree(d_b_dec_conv3)); CHECK(cudaFree(d_dw_dec_conv3)); CHECK(cudaFree(d_db_dec_conv3)); CHECK(cudaFree(d_v_dec_conv3)); CHECK(cudaFree(d_v_dec_conv3_b));
    CHECK(cudaFree(d_w_dec_conv4)); CHECK(cudaFree(d_b_dec_conv4)); CHECK(cudaFree(d_dw_dec_conv4)); CHECK(cudaFree(d_db_dec_conv4)); CHECK(cudaFree(d_v_dec_conv4)); CHECK(cudaFree(d_v_dec_conv4_b));
    CHECK(cudaFree(d_w_dec_conv5)); CHECK(cudaFree(d_b_dec_conv5)); CHECK(cudaFree(d_dw_dec_conv5)); CHECK(cudaFree(d_db_dec_conv5)); CHECK(cudaFree(d_v_dec_conv5)); CHECK(cudaFree(d_v_dec_conv5_b));
}