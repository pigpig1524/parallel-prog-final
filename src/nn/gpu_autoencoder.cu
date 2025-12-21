#include "gpu_autoencoder.h"
#include <cstdio>
#include <cmath>
#include <random>
#include <iostream>
#include <fstream>

// ======== CUDA KERNELS =========

template<typename Func>
void GPUAutoencoder::measureKernelTime(Func&& kernelFunc, float& timeVariable, const char* kernelName) {
    if (train == false) {
        kernelFunc();
        return;
    }
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    
    CHECK(cudaEventRecord(start));
    
    kernelFunc();
    
    CHECK(cudaEventRecord(stop));
    
    CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    timeVariable += milliseconds; // in milliseconds
    
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
}



// --- FORWARD ---
__global__ void k_conv2d_forward(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* output,
    int inC, int outC,
    int H, int W,
    int K, int padding, int stride
) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int b_oc = blockIdx.z;

    if (ow >= W || oh >= H) return;

    int oc = b_oc % outC;
    int b = b_oc / outC;

    float sum = bias[oc];

    for (int ic = 0; ic < inC; ++ic) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                int ih = oh * stride + kh - padding;
                int iw = ow * stride + kw - padding;
                if (ih < 0 || ih >= H || iw < 0 || iw >= W) {
                    continue; 
                }
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

// Fuse Forward Conv
__global__ void k_conv2d_forward_fuse(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* output,
    int inC, int outC,
    int H, int W,
    int K, int padding, int stride
) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int b_oc = blockIdx.z;

    if (ow >= W || oh >= H) return;

    int oc = b_oc % outC;
    int b = b_oc / outC;

    float sum = bias[oc];

    for (int ic = 0; ic < inC; ++ic) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                int ih = oh * stride + kh - padding;
                int iw = ow * stride + kw - padding;
                if (ih < 0 || ih >= H || iw < 0 || iw >= W) {
                    continue; 
                }
                // Tính flat index dựa vào batch, in_Channel, H và W
                int in_idx = ((b * inC + ic) * H + ih) * W + iw;
                int w_idx = ((oc * inC + ic) * K + kh) * K + kw;
                sum += input[in_idx] * weights[w_idx];
                
            }
        }
    }

    int out_idx = ((b * outC + oc) * H + oh) * W + ow;
    output[out_idx] = fmaxf(0.0f, sum); // ReLU activation
}

// Kernel Forward ReLU
__global__ void k_relu_forward(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (data[idx] < 0) data[idx] = 0;
    }
}

// Kernel Forward MaxPool
__global__ void k_maxpool_forward(
    const float* __restrict__ input,
    float* output,
    int C,
    int inH, int inW,
    int outH, int outW
) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int bc = blockIdx.z;
    int K = 2; // kernel_size = 2

    if (ow >= outW || oh >= outH) return;
    
    int ih_start = oh * K;
    int iw_start = ow * K;

    float max_val = -1e30;

    for (int kh = 0; kh < K; ++kh) {
        for (int kw = 0; kw < K; ++kw) {
            int hh = ih_start + kh;
            int ww = iw_start + kw;
            if (hh >= inH || ww >= inW) continue;

            int idx = (bc * inH + hh) * inW + ww;
            float val = input[idx];
            if (val > max_val) max_val = val;
        }
    }
    int out_idx = (bc * outH + oh) * outW + ow;
    output[out_idx] = max_val;
}

// Kernel Forward UpSample
__global__ void k_upsample_forward(
    const float* __restrict__ input,
    float* output,
    int C,
    int inH, int inW,
    int outH, int outW
) {
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


__global__ void k_mse_loss_backward_input(float* predicted, float* target, float* d_grad, float* loss_val, int size) {
    // Shared memory để lưu tổng cục bộ của block
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    float sq_diff = 0.0f;
    // float grad_abs = 0.0f;
    
    if (idx < size) {
        float diff = - target[idx] + predicted[idx];
        d_grad[idx] = 2.0f * diff / size; // Gradient: dL/dOutput
        sq_diff = diff * diff / size; // Chia cho tổng số elements để có MSE
        // grad_abs = fabs(d_grad[idx]);
        
    }
    
    sdata[tid] = sq_diff;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(loss_val, sdata[0]);
    }
}

// --- BACKWARD ---

__global__ void k_relu_backward(float* input_val, float* d_output, float* d_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_input[idx] = (input_val[idx] > 0) ? d_output[idx] : 0.0f;
    }
}

__global__ void k_maxpool_backward(float* input, float* output, float* d_output, float* d_input, 
                                   int inH, int inW, int outH, int outW) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int bc = blockIdx.z;

    if (ow >= outW || oh >= outH) return;

    int out_idx = (bc * outH + oh) * outW + ow;
    float max_val = output[out_idx];
    float d_val = d_output[out_idx];

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

__global__ void k_upsample_backward(float* d_output, float* d_input, int inH, int inW, int outH, int outW) {
    int iw = blockIdx.x * blockDim.x + threadIdx.x;
    int ih = blockIdx.y * blockDim.y + threadIdx.y;
    int bc = blockIdx.z;

    if (iw >= inW || ih >= inH) return;

    float sum = 0.0f;
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

__global__ void k_conv2d_backward_input(float* d_output, float* weights, float* d_input,
                                        int inC, int outC, int H, int W, int K, int padding, int stride) {
    int iw = blockIdx.x * blockDim.x + threadIdx.x;
    int ih = blockIdx.y * blockDim.y + threadIdx.y;
    int b_ic = blockIdx.z;

    if (iw >= W || ih >= H) return;

    int ic = b_ic % inC;
    int b = b_ic / inC;
    float sum = 0.0f;

    for(int oc=0; oc<outC; ++oc) {
        for(int kh=0; kh<K; ++kh) {
            for(int kw=0; kw<K; ++kw) {
                int oh = ih + padding - kh;
                int ow = iw + padding - kw;
                if (oh % stride == 0 && ow % stride == 0) {
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

__global__ void k_conv2d_backward_weights(float* input, float* d_output, float* d_weights, float* d_bias,
                                          int inC, int outC, int H, int W, int K, int padding, int stride) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int b_oc = blockIdx.z;

    if (ow >= W || oh >= H) return;

    int oc = b_oc % outC;
    int b = b_oc / outC;
    
    int dout_idx = ((b * outC + oc) * H + oh) * W + ow;
    float d_val = d_output[dout_idx];

    atomicAdd(&d_bias[oc], d_val);

    for(int ic=0; ic<inC; ++ic) {
        for(int kh=0; kh<K; ++kh) {
            for(int kw=0; kw<K; ++kw) {
                int ih = oh * stride + kh - padding;
                int iw = ow * stride + kw - padding;

                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    int in_idx = ((b * inC + ic) * H + ih) * W + iw;
                    float val = input[in_idx];
                    int w_idx = ((oc * inC + ic) * K + kh) * K + kw;
                    atomicAdd(&d_weights[w_idx], val * d_val);
                }
            }
        }
    }
}

__global__ void k_conv2d_backward_input_smem(float* d_output, float* weights, float* d_input,
                                        int inC, int outC, int H, int W, int K, int padding, int stride) {
    int iw = blockIdx.x * blockDim.x + threadIdx.x;
    int ih = blockIdx.y * blockDim.y + threadIdx.y;
    int b_ic = blockIdx.z;

    if (iw >= W || ih >= H) return;
    extern __shared__ float s_input[];
    int ic = b_ic % inC;
    int b = b_ic / inC;
    float sum = 0.0f;

    int tid_x = threadIdx.x, tid_y = threadIdx.y;
    // Load input tile into shared memory
    int s_width = blockDim.x + K - padding;

    for (int oc=0; oc<outC; ++oc){
        int s_idx;
        s_idx = (tid_y + K/2) * s_width + (tid_x + K/2);
        s_input[s_idx] = 0.0f;
        if (tid_y < K/2){ // Top halo
            s_idx = (tid_y) * s_width + (tid_x + K/2);
            s_input[s_idx] = 0.0f;
            s_idx = (tid_y + K/2 + blockDim.y) * s_width + (tid_x + K/2);
            s_input[s_idx] = 0.0f;
        }
        if (tid_x < K/2){ // left halo
            s_input[s_idx] = 0.0f;
            s_idx = (tid_y + K/2) * s_width + (tid_x);
            s_input[s_idx] = 0.0f;
            s_idx = (tid_y + K/2) * s_width + (tid_x + K/2 + blockDim.x);
        }
        if (tid_x < K/2 && tid_y < K/2){
                // Top-left
            s_input[s_idx] = 0.0f;
            s_idx = (tid_y) * s_width + (tid_x);
            s_input[s_idx] = 0.0f;
            s_idx = (tid_y) * s_width + (tid_x + K/2 + blockDim.x);
            s_input[s_idx] = 0.0f;
            s_idx = (tid_y + K/2 + blockDim.y) * s_width + (tid_x);
            s_input[s_idx] = 0.0f;
            s_idx = (tid_y + K/2 + blockDim.y) * s_width + (tid_x + K/2 + blockDim.x);
            s_input[s_idx] = 0.0f;
        }
    }
    
    for(int oc=0; oc<outC; ++oc) {
        // Center 
        int oh, ow, out_idx, s_idx;
        oh = ih * stride - K/2 + padding;
        ow = iw * stride - K/2 + padding;
        out_idx = ((b* outC  + oc) * H + oh) * W + ow;
        if (oh >= 0 && oh < H && ow >= 0 && ow < W){
            s_idx = (tid_y + K/2) * s_width + (tid_x + K/2);
            s_input[s_idx] += d_output[out_idx] ;
        }

        if (tid_y < K/2){ // Top halo
            oh = ih * stride - 0 + padding;
            ow = iw * stride - K/2 + padding;
            out_idx = ((b* outC  + oc) * H + oh) * W + ow;
            if (oh >= 0 && oh < H && ow >= 0 && ow < W){
                s_idx = (tid_y) * s_width + (tid_x + K/2);
                s_input[s_idx] += d_output[out_idx];
            }

            //bottom halo
            oh = ih * stride - K + padding;
            ow = iw * stride - K/2 + padding;
            out_idx = ((b* outC  + oc) * H + oh) * W + ow;
            if (oh >= 0 && oh < H && ow >= 0 && ow < W){
                s_idx = (tid_y + K/2 + blockDim.y) * s_width + (tid_x + K/2);
                s_input[s_idx] += d_output[out_idx];
            }
        }

        if (tid_x < K/2){ // left halo
            oh = ih * stride - K/2 + padding;
            ow = iw * stride + padding;
            out_idx = ((b* outC  + oc) * H + oh) * W + ow;
            if (oh >= 0 && oh < H && ow >= 0 && ow < W){
                s_idx = (tid_y + K/2) * s_width + (tid_x);
                s_input[s_idx] += d_output[out_idx];
            }
            // right halo
            oh = ih * stride - K/2 + padding;
            ow = iw * stride - K + padding;
            out_idx = ((b* outC  + oc) * H + oh) * W + ow;
            if (oh >= 0 && oh < H && ow >= 0 && ow < W){
                s_idx = (tid_y + K/2) * s_width + (tid_x + K/2 + blockDim.x);
                s_input[s_idx] += d_output[out_idx];
            }
        }

        if (tid_x < K/2 && tid_y < K/2){
            // Top-left
            oh = ih * stride + padding;
            ow = iw * stride + padding;
            out_idx = ((b* outC  + oc) * H + oh) * W + ow;
            if (oh >= 0 && oh < H && ow >= 0 && ow < W){
                s_idx = (tid_y) * s_width + (tid_x);
                s_input[s_idx] += d_output[out_idx];
            }
            // Top-right
            oh = ih * stride + padding;
            ow = iw * stride - K + padding;
            out_idx = ((b* outC  + oc) * H + oh) * W + ow;
            if (oh >= 0 && oh < H && ow >= 0 && ow < W){
                s_idx = (tid_y) * s_width + (tid_x + K/2 + blockDim.x);
                s_input[s_idx] += d_output[out_idx];
            }
            // Bottom-left
            oh = ih * stride - K + padding;
            ow = iw * stride + padding;
            out_idx = ((b* outC  + oc) * H + oh) * W + ow;
            if (oh >= 0 && oh < H && ow >= 0 && ow < W){
                s_idx = (tid_y + K/2 + blockDim.y) * s_width + (tid_x);
                s_input[s_idx] += d_output[out_idx];
            }
            // Bottom-right
            oh = oh * stride - K + padding;
            ow = ow * stride - K + padding;
            out_idx = ((b* outC  + oc) * H + oh) * W + ow;
            if (oh >= 0 && oh < H && ow >= 0 && ow < W){
                s_idx = (tid_y + K/2 + blockDim.y) * s_width + (tid_x + K/2 + blockDim.x);
                s_input[s_idx] += d_output[out_idx];
            }
        }
        __syncthreads();

        for(int kh=0; kh<K; ++kh) {
            for(int kw=0; kw<K; ++kw) {
                int oh = ih * stride - kh + padding;
                int ow = iw * stride - kw + padding;

                if (oh >= 0 && oh < H && ow >= 0 && ow < W) {
                    // int in_idx = ((b * inC + ic) * H + ih) * W + iw;
                    // float val = input[in_idx];
                    int w_idx = ((oc * inC + ic) * K + kh) * K + kw;
                    sum += weights[w_idx] * s_input[(tid_y + kh) * s_width + (tid_x + kw)];
                }
            }
        }
        __syncthreads();
    }
    int din_idx = ((b * inC + ic) * H + ih) * W + iw;
    d_input[din_idx] = sum;
}

__global__ void k_conv2d_backward_weights_smem(float* input, float* d_output, float* d_weights, float* d_bias,
                                          int inC, int outC, int H, int W, int K, int padding, int stride) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int b_oc = blockIdx.z;
    extern __shared__ float s_input[]; 
    if (ow >= W || oh >= H) return;
    int oc = b_oc % outC;
    int b = b_oc / outC;
    
    int dout_idx = ((b * outC + oc) * H + oh) * W + ow;
    float d_val = d_output[dout_idx];

    atomicAdd(&d_bias[oc], d_val);

    int tid_x = threadIdx.x, tid_y = threadIdx.y;
    // Load input tile into shared memory
    int s_width = blockDim.x + K - padding;
    
    for (int ic=0; ic<inC; ++ic){
        int s_idx;
        s_idx = (tid_y + K/2) * s_width + (tid_x + K/2);
        s_input[s_idx] = 0.0f;
        if (tid_y < K/2){ // Top halo
            s_idx = (tid_y) * s_width + (tid_x + K/2);
            s_input[s_idx] = 0.0f;
            s_idx = (tid_y + K/2 + blockDim.y) * s_width + (tid_x + K/2);
            s_input[s_idx] = 0.0f;
        }
        if (tid_x < K/2){ // left halo
            s_input[s_idx] = 0.0f;
            s_idx = (tid_y + K/2) * s_width + (tid_x);
            s_input[s_idx] = 0.0f;
            s_idx = (tid_y + K/2) * s_width + (tid_x + K/2 + blockDim.x);
        }
        if (tid_x < K/2 && tid_y < K/2){
                // Top-left
            s_input[s_idx] = 0.0f;
            s_idx = (tid_y) * s_width + (tid_x);
            s_input[s_idx] = 0.0f;
            s_idx = (tid_y) * s_width + (tid_x + K/2 + blockDim.x);
            s_input[s_idx] = 0.0f;
            s_idx = (tid_y + K/2 + blockDim.y) * s_width + (tid_x);
            s_input[s_idx] = 0.0f;
            s_idx = (tid_y + K/2 + blockDim.y) * s_width + (tid_x + K/2 + blockDim.x);
            s_input[s_idx] = 0.0f;
        }
    }

    for(int ic=0; ic<inC; ++ic) {
        // Center 
        int ih, iw, in_idx, s_idx;
        ih = oh * stride + K/2 - padding;
        iw = ow * stride + K/2 - padding;
        in_idx = ((b* inC + ic) * H + ih) * W + iw;
        if (ih >= 0 && ih < H && iw >= 0 && iw < W){
            s_idx = (tid_y + K/2) * s_width + (tid_x + K/2);
            s_input[s_idx] += input[in_idx] ;
        }

        if (tid_y < K/2){ // Top halo
            ih = oh * stride + 0 - padding;
            iw = ow * stride + K/2 - padding;
            in_idx = ((b* inC + ic) * H + ih) * W + iw;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W){
                s_idx = (tid_y) * s_width + (tid_x + K/2);
                s_input[s_idx] += input[in_idx];
            }

            //bottom halo
            ih = oh * stride + K - padding;
            iw = ow * stride + K/2 - padding;
            in_idx = ((b* inC + ic) * H + ih) * W + iw;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W){
                s_idx = (tid_y + K/2 + blockDim.y) * s_width + (tid_x + K/2);
                s_input[s_idx] += input[in_idx];
            }
        }

        if (tid_x < K/2){ // left halo
            ih = oh * stride + K/2 - padding;
            iw = ow * stride - padding;
            in_idx = ((b* inC + ic) * H + ih) * W + iw;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W){
                s_idx = (tid_y + K/2) * s_width + (tid_x);
                s_input[s_idx] += input[in_idx];
            }
            // right halo
            ih = oh * stride + K/2 - padding;
            iw = ow * stride + K - padding;
            in_idx = ((b* inC + ic) * H + ih) * W + iw;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W){
                s_idx = (tid_y + K/2) * s_width + (tid_x + K/2 + blockDim.x);
                s_input[s_idx] += input[in_idx];
            }
        }

        if (tid_x < K/2 && tid_y < K/2){
            // Top-left
            ih = oh * stride - padding;
            iw = ow * stride - padding;
            in_idx = ((b* inC + ic) * H + ih) * W + iw;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W){
                s_idx = (tid_y) * s_width + (tid_x);
                s_input[s_idx] += input[in_idx];
            }
            // Top-right
            ih = oh * stride - padding;
            iw = ow * stride + K - padding;
            in_idx = ((b* inC + ic) * H + ih) * W + iw;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W){
                s_idx = (tid_y) * s_width + (tid_x + K/2 + blockDim.x);
                s_input[s_idx] += input[in_idx];
            }
            // Bottom-left
            ih = oh * stride + K - padding;
            iw = ow * stride - padding;
            in_idx = ((b* inC + ic) * H + ih) * W + iw;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W){
                s_idx = (tid_y + K/2 + blockDim.y) * s_width + (tid_x);
                s_input[s_idx] += input[in_idx];
            }
            // Bottom-right
            ih = oh * stride + K - padding;
            iw = ow * stride + K - padding;
            in_idx = ((b* inC + ic) * H + ih) * W + iw;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W){
                s_idx = (tid_y + K/2 + blockDim.y) * s_width + (tid_x + K/2 + blockDim.x);
                s_input[s_idx] += input[in_idx];
            }
        }
        __syncthreads();

        for(int kh=0; kh<K; ++kh) {
            for(int kw=0; kw<K; ++kw) {
                int ih = oh * stride + kh - padding;
                int iw = ow * stride + kw - padding;

                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    // int in_idx = ((b * inC + ic) * H + ih) * W + iw;
                    // float val = input[in_idx];
                    int w_idx = ((oc * inC + ic) * K + kh) * K + kw;
                    int s_idx = (tid_y + kh) * s_width + (tid_x + kw);
                    atomicAdd(&d_weights[w_idx], s_input[s_idx] * d_val);
                }
            }
        }
        __syncthreads();
    }
    
}


__global__ void k_conv2d_backward_weights_reduction(float* input, float* d_output, float* d_weights,
                                          int inC, int outC, int H, int W, int K, int padding, int stride, int batchSize) {
    
    extern __shared__ float sdata[];                                        

    int oc = blockIdx.x;
    int ic = blockIdx.y;
    int kh = blockIdx.z / K;
    int kw = blockIdx.z % K;

    int w_idx = ((oc * inC + ic)* K + kh) * K + kw;

    float sum = 0.0;

    int total_pixels = batchSize * H * W;
    
    for(int tid = threadIdx.x; tid < total_pixels; tid += blockDim.x){
        int b = tid / (H*W);
        int oh = tid % (H*W) / W;
        int ow = tid % (H*W) % W;

        int ih = oh * stride + kh - padding;
        int iw = ow * stride + kw - padding;

        if (oh >= 0 && oh < H && ow >= 0 && ow < W
            && ih >= 0 && ih < H && iw >= 0 && iw < W){
            int in_idx = ((b * inC + ic) * H + ih) * W + iw;
            int out_idx = ((b * outC + oc) * H + oh) * W + ow;

            sum += input[in_idx] * d_output[out_idx];
        }
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
            // sdata[threadIdx.x + total_pixels] += sdata[threadIdx.x + total_pixels + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        d_weights[w_idx] = sdata[0];
    }
}

__global__ void k_conv2d_backward_bias_reduction(float* input, float* d_output, float* d_bias,
                                          int inC, int outC, int H, int W, int K, int padding, int stride, int batchSize){

    extern __shared__ float sdata[];
    
    int oc = blockIdx.x;

    float sum = 0.0;
    int total_pixels = H * W * batchSize;

    for(int tid = threadIdx.x; tid < total_pixels; tid += blockDim.x){
        int b = tid / (H*W);
        int oh = tid % (H*W) / W;
        int ow = tid % (H*W) % W;

        if (oh >= 0 && oh < H && ow >= 0 && ow < W){
            int out_idx = ((b * outC + oc) * H + oh) * W + ow;
            sum += d_output[out_idx];
        }
    }

    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x/2; s>0; s/=2){
        if (threadIdx.x < s){
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
        d_bias[oc] = sdata[0]; 
}


__global__ void apply_update(float* weights, float* d_weights, float* velocity, 
                                 float lr, float momentum, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float grad = d_weights[idx]; 
        
        const float grad_clip = 5.0;
        if (grad > grad_clip) grad = grad_clip;
        else if (grad < -grad_clip) grad = -grad_clip;

        float v = momentum * velocity[idx] - lr * grad;
        velocity[idx] = v;
        weights[idx] += v;
        d_weights[idx] = 0.0f;
    }
}


GPUAutoencoder::GPUAutoencoder(float learningRate, float momentum) 
    : m_learningRate(learningRate), m_momentum(momentum),
      conv_forward_time(0.0), conv_backward_time(0.0), relu_forward_time(0.0), relu_backward_time(0.0),
      pool_forward_time(0.0), pool_backward_time(0.0), total_kernel_time(0.0)
{
    // train = true;
    initWeightsRandomly();
}

void random_init(std::vector<float>& vec, int fan_in) {
    // Xavier/Glorot initialization 
    float limit = sqrt(2.0f / (fan_in + vec.size()));
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<float> dist(-limit, limit);
    for (float& val : vec) val = dist(generator);
}

void GPUAutoencoder::initWeightsRandomly() {
    int K = 3;
    auto init_layer = [&](float** d_w, float** d_b, float** d_dw, float** d_db, float** d_vw, float** d_vb, int outC, int inC) {
        int w_size = outC * inC * K * K;
        int b_size = outC;
        
        std::vector<float> h_w(w_size);
        std::vector<float> h_b(b_size, 0.0f);
        
        random_init(h_w, inC * K * K);

        CHECK(cudaMalloc(d_w, w_size * sizeof(float)));
        CHECK(cudaMalloc(d_b, b_size * sizeof(float)));
        CHECK(cudaMalloc(d_dw, w_size * sizeof(float)));
        CHECK(cudaMalloc(d_db, b_size * sizeof(float)));
        CHECK(cudaMalloc(d_vw, w_size * sizeof(float)));
        CHECK(cudaMalloc(d_vb, b_size * sizeof(float)));

        CHECK(cudaMemcpy(*d_w, h_w.data(), w_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(*d_b, h_b.data(), b_size * sizeof(float), cudaMemcpyHostToDevice));
        
        CHECK(cudaMemset(*d_dw, 0, w_size * sizeof(float)));
        CHECK(cudaMemset(*d_db, 0, b_size * sizeof(float)));
        CHECK(cudaMemset(*d_vw, 0, w_size * sizeof(float)));
        CHECK(cudaMemset(*d_vb, 0, b_size * sizeof(float)));
    };

    init_layer(&d_w_enc_conv1, &d_b_enc_conv1, &d_dw_enc_conv1, &d_db_enc_conv1, &d_v_enc_conv1, &d_v_enc_conv1_b, 256, 3);
    init_layer(&d_w_enc_conv2, &d_b_enc_conv2, &d_dw_enc_conv2, &d_db_enc_conv2, &d_v_enc_conv2, &d_v_enc_conv2_b, 128, 256);
    init_layer(&d_w_dec_conv3, &d_b_dec_conv3, &d_dw_dec_conv3, &d_db_dec_conv3, &d_v_dec_conv3, &d_v_dec_conv3_b, 128, 128);
    init_layer(&d_w_dec_conv4, &d_b_dec_conv4, &d_dw_dec_conv4, &d_db_dec_conv4, &d_v_dec_conv4, &d_v_dec_conv4_b, 256, 128);
    init_layer(&d_w_dec_conv5, &d_b_dec_conv5, &d_dw_dec_conv5, &d_db_dec_conv5, &d_v_dec_conv5, &d_v_dec_conv5_b, 3, 256);
    allocateMemory(32);
}

void GPUAutoencoder::allocateMemory(int batchSize) {
    auto malloc_tensor = [&](float** ptr, int c, int h, int w) {
        CHECK(cudaMalloc(ptr, batchSize * c * h * w * sizeof(float)));
    };
    allocated = true;
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


void GPUAutoencoder::getOutput(const std::vector<float>& h_inputBatch, std::vector<float>& h_output, int batchSize) {
    train_batch(h_inputBatch, batchSize);
    int size = batchSize * 3 * 32 * 32;
    h_output.resize(size);
    CHECK(cudaMemcpy(h_output.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost));
}

void GPUAutoencoder::getLatent(const std::vector<float>& h_inputBatch, std::vector<float>& h_latent, int batchSize) {
    if (! allocated)
        allocateMemory(batchSize);
    
    CHECK(cudaMemcpy(d_input, h_inputBatch.data(), h_inputBatch.size() * sizeof(float), cudaMemcpyHostToDevice));

    forwardPass(batchSize);

    int size = batchSize * 128 * 8 * 8;
    h_latent.resize(size);
    CHECK(cudaMemcpy(h_latent.data(), d_latent, size * sizeof(float), cudaMemcpyDeviceToHost));
}

void GPUAutoencoder::train_batch(const std::vector<float>& h_inputBatch, int batchSize) {
    if (! allocated)
        allocateMemory(batchSize);

    CHECK(cudaMemcpy(d_input, h_inputBatch.data(), h_inputBatch.size() * sizeof(float), cudaMemcpyHostToDevice));
    // printf("111");
    forwardPass(batchSize);
    // printf("123");
    CHECK(cudaDeviceSynchronize());

    float* d_loss; 
    CHECK(cudaMalloc(&d_loss, sizeof(float)));
    CHECK(cudaMemset(d_loss, 0, sizeof(float)));
    

    int size = batchSize * 3 * 32 * 32; //total numbers of pixels of both input and output
    int blockSize = 32;
    int gridSize = (size - 1) / blockSize + 1;

    k_mse_loss_backward_input<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_output, d_input, d_grad_output, d_loss, size);
    
    CHECK(cudaMemcpy(&m_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    
    total_kernel_time = conv_forward_time + conv_backward_time + relu_forward_time + relu_backward_time + pool_forward_time + pool_backward_time;

    if (train) {
        backwardPass(batchSize);
        // printf("456");
        updateWeights();
    }

    CHECK(cudaFree(d_loss));
}

void GPUAutoencoder::forwardPass(int batchSize){
    int blockSize = 32;
    dim3 block(blockSize, blockSize);
    int H_in = 32, W_in = 32, outC = 256, inC = 3;
    int filterWidth = 3, padding = 1, stride = 1;
    dim3 grid((W_in-1)/blockSize + 1, (H_in-1)/blockSize + 1, batchSize*outC);

    measureKernelTime([&]() {
        k_conv2d_forward_fuse<<<grid, block>>>(d_input, d_w_enc_conv1, d_b_enc_conv1, d_enc_conv1, inC, outC, H_in, W_in, filterWidth, padding, stride);
    }, conv_forward_time, "Conv2D Forward");
    CHECK(cudaGetLastError());


    inC = 256, outC = 256, filterWidth = 2;
    grid = dim3((W_in-1)/blockSize + 1, (H_in-1)/blockSize + 1, batchSize*outC);
    measureKernelTime([&]() {
        k_maxpool_forward<<<grid, block>>>(d_enc_conv1, d_enc_pool1, inC, H_in, W_in, H_in/2, W_in/2);
    }, pool_forward_time, "MaxPool");

    H_in = 16; W_in = 16; inC = 256; outC = 128; filterWidth = 3;
    grid = dim3((W_in-1)/blockSize + 1, (H_in-1)/blockSize + 1, batchSize*outC);    
    measureKernelTime([&]() {
        k_conv2d_forward_fuse<<<grid, block>>>(d_enc_pool1, d_w_enc_conv2, d_b_enc_conv2, d_enc_conv2, inC, outC, H_in, W_in, filterWidth, padding, stride);
    }, conv_forward_time, "Conv2D Forward");



    filterWidth = 2; inC=128; outC=128;
    grid = dim3((W_in-1)/blockSize + 1, (H_in-1)/blockSize + 1, batchSize*outC);
    measureKernelTime([&]() {
        k_maxpool_forward<<<grid, block>>>(d_enc_conv2, d_latent, inC, H_in, W_in, H_in/2, W_in/2);
    }, pool_forward_time, "MaxPool");

    H_in=8; W_in=8; inC=128; outC=128; filterWidth=3;
    grid = dim3((W_in-1)/blockSize + 1, (H_in-1)/blockSize + 1, batchSize*outC);
    measureKernelTime([&]() {
    k_conv2d_forward_fuse<<<grid, block>>>(d_latent, d_w_dec_conv3, d_b_dec_conv3, d_dec_conv3, inC, outC, H_in, W_in, filterWidth, padding, stride);
    }, conv_forward_time, "Conv2D Forward");


    filterWidth=2; inC=128; outC=128;
    grid = dim3((W_in-1)/blockSize + 1, (H_in-1)/blockSize + 1, batchSize*outC);
    measureKernelTime([&]() {
        k_upsample_forward<<<grid, block>>>(d_dec_conv3, d_dec_ups1, inC, H_in, W_in, H_in*2, W_in*2);
    }, pool_forward_time, "UpSample");

    H_in = 16; W_in = 16; inC = 128; outC = 256; filterWidth = 3;
    grid = dim3((W_in-1)/blockSize + 1, (H_in-1)/blockSize + 1, batchSize*outC);
    measureKernelTime([&]() {
        k_conv2d_forward_fuse<<<grid, block>>>(d_dec_ups1, d_w_dec_conv4, d_b_dec_conv4, d_dec_conv4, inC, outC, H_in, W_in, filterWidth, padding, stride);
    }, conv_forward_time, "Conv2D Forward");

    filterWidth=2; inC=256; outC=256;
    grid = dim3((W_in-1)/blockSize + 1, (H_in-1)/blockSize + 1, batchSize*outC);
    measureKernelTime([&]() {
        k_upsample_forward<<<grid, block>>>(d_dec_conv4, d_dec_ups2, inC, H_in, W_in, H_in*2, W_in*2);
    }, pool_forward_time, "UpSample");
    

    H_in = 32; W_in = 32; inC = 256; outC = 3; filterWidth = 3;
    grid = dim3((W_in-1)/blockSize + 1, (H_in-1)/blockSize + 1, batchSize*outC);
    measureKernelTime([&]() {
        k_conv2d_forward<<<grid, block>>>(d_dec_ups2, d_w_dec_conv5, d_b_dec_conv5, d_output, inC, outC, H_in, W_in, filterWidth, padding, stride);
    }, conv_forward_time, "Conv2D Forward");

}

void GPUAutoencoder::backwardPass(int batchSize) {
    dim3 grid;
    int blockSize = 32;
    dim3 block = dim3(blockSize,blockSize);
    dim3 blockConv = dim3(256);
    size_t sharedMemSize;
    int H_out = 32, W_out = 32, outC = 3, inC = 256;
    int filterWidth = 3, padding = 1, stride = 1;
    grid = dim3(outC, inC, filterWidth * filterWidth);
    measureKernelTime([&]() {   
        sharedMemSize = blockConv.x * sizeof(float);
        k_conv2d_backward_weights_reduction<<<grid, blockConv, sharedMemSize>>>(d_dec_ups2, d_grad_output, d_dw_dec_conv5, inC, outC, H_out, W_out, filterWidth, padding, stride, batchSize);
        k_conv2d_backward_bias_reduction<<<grid, blockConv, sharedMemSize>>>(d_dec_ups2, d_grad_output, d_db_dec_conv5, inC, outC, H_out, W_out, filterWidth, padding, stride, batchSize);
    }, conv_backward_time, "Conv2D Backward Weights");
    CHECK(cudaGetLastError());
    // printf("haha");
    
    grid = dim3((W_out-1)/blockSize + 1, (H_out-1)/blockSize + 1, batchSize*inC);
    measureKernelTime([&]() {
        k_conv2d_backward_input<<<grid, block>>>(d_grad_output, d_w_dec_conv5, d_grad_dec_ups2, inC, outC, H_out, W_out, filterWidth, padding, stride);
    }, conv_backward_time, "Conv2D Backward Input");
    // Upsample2 backward
    inC = 256; outC = 256; filterWidth = 2;
    grid = dim3((W_out-1)/blockSize + 1, (H_out-1)/blockSize + 1, batchSize*inC);
    measureKernelTime([&]() {
        k_upsample_backward<<<grid, block>>>(d_grad_dec_ups2, d_grad_dec_conv4, H_out/2, W_out/2, H_out, W_out);
    }, pool_backward_time, "UpSample Backward");
    
    H_out = 16; W_out = 16; outC = 256; inC = 128; filterWidth = 3;
    measureKernelTime([&]() {
        k_relu_backward<<<(batchSize*outC*H_out*W_out-1)/256 + 1, 256>>>(d_dec_conv4, d_grad_dec_conv4, d_grad_dec_conv4, batchSize*outC*H_out*W_out);
    }, relu_backward_time, "ReLU Backward");
    
    grid = dim3(outC, inC, filterWidth * filterWidth);
    measureKernelTime([&]() {
        sharedMemSize = blockConv.x * sizeof(float);
        k_conv2d_backward_weights_reduction<<<grid, blockConv, sharedMemSize>>>(d_dec_ups1, d_grad_dec_conv4, d_dw_dec_conv4, inC, outC, H_out, W_out, filterWidth, padding, stride, batchSize);
        k_conv2d_backward_bias_reduction<<<grid, blockConv, sharedMemSize>>>(d_dec_ups1, d_grad_dec_conv4, d_db_dec_conv4, inC, outC, H_out, W_out, filterWidth, padding, stride, batchSize);
    }, conv_backward_time, "Conv2D Backward Weights");

    grid = dim3((W_out-1)/blockSize + 1, (H_out-1)/blockSize + 1, batchSize*inC);
    measureKernelTime([&]() {
        k_conv2d_backward_input<<<grid, block>>>(d_grad_dec_conv4, d_w_dec_conv4, d_grad_dec_ups1, inC, outC, H_out, W_out, filterWidth, padding, stride);
    }, conv_backward_time, "Conv2D Backward Input");

    // Upsample1 backward
    inC = 128; filterWidth = 2;
    grid = dim3((W_out-1)/blockSize + 1, (H_out-1)/blockSize + 1, batchSize*inC);
    measureKernelTime([&]() {
        k_upsample_backward<<<grid, block>>>(d_grad_dec_ups1, d_grad_dec_conv3, H_out/2, W_out/2, H_out, W_out);
    }, pool_backward_time, "UpSample Backward");
    
    H_out = 8; W_out = 8; outC = 128; inC = 128; filterWidth = 3;
    measureKernelTime([&]() {
        k_relu_backward<<<(batchSize*outC*H_out*W_out-1)/256 + 1, 256>>>(d_dec_conv3, d_grad_dec_conv3, d_grad_dec_conv3, batchSize*outC*H_out*W_out);
    }, relu_backward_time, "ReLU Backward");

    grid = dim3(outC, inC, filterWidth * filterWidth);
    measureKernelTime([&]() {
        sharedMemSize = blockConv.x * sizeof(float);
        k_conv2d_backward_weights_reduction<<<grid, blockConv, sharedMemSize>>>(d_latent, d_grad_dec_conv3, d_dw_dec_conv3, inC, outC, H_out, W_out, filterWidth, padding, stride, batchSize);
        k_conv2d_backward_bias_reduction<<<grid, blockConv, sharedMemSize>>>(d_latent, d_grad_dec_conv3, d_db_dec_conv3, inC, outC, H_out, W_out, filterWidth, padding, stride, batchSize);
    }, conv_backward_time, "Conv2D Backward Weights");
    
    grid = dim3((W_out-1)/blockSize + 1, (H_out-1)/blockSize + 1, batchSize*inC);
    measureKernelTime([&]() {
        k_conv2d_backward_input<<<grid, block>>>(d_grad_dec_conv3, d_w_dec_conv3, d_grad_latent, inC, outC, H_out, W_out, filterWidth, padding, stride);
    }, conv_backward_time, "Conv2D Backward Input");

    // MaxPool backward to enc_conv2 - 16x16x128 
    inC = 128; filterWidth = 2;
    grid = dim3((W_out*2-1)/blockSize + 1, (H_out*2-1)/blockSize + 1, batchSize*inC);
    measureKernelTime([&]() {
        k_maxpool_backward<<<grid, block>>>(d_enc_conv2, d_latent, d_grad_latent, d_grad_enc_conv2, H_out*2, W_out*2, H_out, W_out);
    }, pool_backward_time, "MaxPool Backward");
    // enc_conv2 backward - 16x16x128
    H_out = 16; W_out = 16; outC = 128; inC = 256; filterWidth = 3;
    measureKernelTime([&]() {
        k_relu_backward<<<(batchSize*outC*H_out*W_out-1)/256 + 1, 256>>>(d_enc_conv2, d_grad_enc_conv2, d_grad_enc_conv2, batchSize*outC*H_out*W_out);
    }, relu_backward_time, "ReLU Backward");
    
    grid = dim3(outC, inC, filterWidth * filterWidth);
    measureKernelTime([&]() {
        sharedMemSize = blockConv.x * sizeof(float);
        k_conv2d_backward_weights_reduction<<<grid, blockConv, sharedMemSize>>>(d_enc_pool1, d_grad_enc_conv2, d_dw_enc_conv2, inC, outC, H_out, W_out, filterWidth, padding, stride, batchSize);
        k_conv2d_backward_bias_reduction<<<grid, blockConv, sharedMemSize>>>(d_enc_pool1, d_grad_enc_conv2, d_db_enc_conv2, inC, outC, H_out, W_out, filterWidth, padding, stride, batchSize);
    }, conv_backward_time, "Conv2D Backward Weights");

    grid = dim3((W_out-1)/blockSize + 1, (H_out-1)/blockSize + 1, batchSize*inC);
    measureKernelTime([&]() {
        k_conv2d_backward_input<<<grid, block>>>(d_grad_enc_conv2, d_w_enc_conv2, d_grad_enc_pool1, inC, outC, H_out, W_out, filterWidth, padding, stride);
    }, conv_backward_time, "Conv2D Backward Input");
    // MaxPool backward to enc_conv1 - 32x32x256
    inC = 256; filterWidth = 2;
    grid = dim3((W_out*2-1)/blockSize + 1, (H_out*2-1)/blockSize + 1, batchSize*inC);
    measureKernelTime([&]() {
        k_maxpool_backward<<<grid, block>>>(d_enc_conv1, d_enc_pool1, d_grad_enc_pool1, d_grad_enc_conv1, H_out*2, W_out*2, H_out, W_out);
    }, pool_backward_time, "MaxPool Backward");
    // enc_conv1 backward - 32x32x256
    H_out = 32; W_out = 32; outC = 256; inC = 3; filterWidth = 3;
    measureKernelTime([&]() {
        k_relu_backward<<<(batchSize*outC*H_out*W_out-1)/256 + 1, 256>>>(d_enc_conv1, d_grad_enc_conv1, d_grad_enc_conv1, batchSize*outC*H_out*W_out);
    }, relu_backward_time, "ReLU Backward");

    grid = dim3(outC, inC, filterWidth * filterWidth);
    measureKernelTime([&]() {
        sharedMemSize = blockConv.x * sizeof(float);
        k_conv2d_backward_weights_reduction<<<grid, blockConv, sharedMemSize>>>(d_input, d_grad_enc_conv1, d_dw_enc_conv1, inC, outC, H_out, W_out, filterWidth, padding, stride, batchSize);
        k_conv2d_backward_bias_reduction<<<grid, blockConv, sharedMemSize>>>(d_input, d_grad_enc_conv1, d_db_enc_conv1, inC, outC, H_out, W_out, filterWidth, padding, stride, batchSize);
    }, conv_backward_time, "Conv2D Backward Weights");
    
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
    auto load_layer = [&](float* d_w, float* d_b, int outC, int inC, int K) {
        int w_size = outC * inC * K * K;
        int b_size = outC;
        
        std::vector<float> h_weights(w_size);
        std::vector<float> h_bias(b_size);
        
        file.read(reinterpret_cast<char*>(h_weights.data()), w_size * sizeof(float));
        file.read(reinterpret_cast<char*>(h_bias.data()), b_size * sizeof(float));
        
        if (file.fail()) {
            std::cerr << "Error: Failed to read weights from file" << std::endl;
            return;
        }
        
        // Copy from host to device
        CHECK(cudaMemcpy(d_w, h_weights.data(), w_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_b, h_bias.data(), b_size * sizeof(float), cudaMemcpyHostToDevice));
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
    
    auto save_layer = [&](float* d_w, float* d_b, int outC, int inC, int K) {
        int w_size = outC * inC * K * K;
        int b_size = outC;
        
        // Copy weights from device to host
        std::vector<float> h_weights(w_size);
        std::vector<float> h_bias(b_size);
        
        CHECK(cudaMemcpy(h_weights.data(), d_w, w_size * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(h_bias.data(), d_b, b_size * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Write to file
        file.write(reinterpret_cast<const char*>(h_weights.data()), w_size * sizeof(float));
        file.write(reinterpret_cast<const char*>(h_bias.data()), b_size * sizeof(float));
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