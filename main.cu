#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cstdlib>

// ======================= Utility ===========================
__device__ float relu(float x) {
    return fmaxf(0.0f, x);
}
__device__ float relu_deriv(float x) {
    return x > 0 ? 1.0f : 0.0f;
}

// ======================= Kernel: Matrix Multiply ===========================
__global__ void matmul(float* A, float* B, float* C,
                       int M, int N, int K) {
    // C[MxN] = A[MxK] * B[KxN]
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float val = 0.0f;
        for (int k = 0; k < K; k++) {
            val += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = val;
    }
}

// ======================= Kernel: Add Bias + ReLU ===========================
__global__ void add_bias_relu(float* Z, float* bias, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        int idx = row * N + col;
        Z[idx] = relu(Z[idx] + bias[col]);
    }
}

// ======================= Kernel: Output Softmax ===========================
__global__ void softmax(float* Z, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float max_val = -1e9;
        for (int j = 0; j < N; j++) max_val = fmaxf(max_val, Z[row * N + j]);

        float sum = 0.0f;
        for (int j = 0; j < N; j++) sum += expf(Z[row * N + j] - max_val);

        for (int j = 0; j < N; j++)
            Z[row * N + j] = expf(Z[row * N + j] - max_val) / sum;
    }
}

// ======================= Kernel: Cross-Entropy Loss ===========================
__global__ void compute_loss(float* preds, int* labels, float* loss, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        int label = labels[row];
        float prob = preds[row * N + label];
        atomicAdd(loss, -logf(prob + 1e-8));
    }
}

// ======================= Training Step ===========================
__global__ void sgd_update(float* W, float* dW, float lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        W[idx] -= lr * dW[idx];
    }
}

// ======================= Host Code ===========================
int main() {
    // Example: XOR dataset (4 samples, 2 features, 2 classes)
    const int M = 4;   // samples
    const int input_dim = 2;
    const int hidden_dim = 4;
    const int output_dim = 2;

    float h_X[M * input_dim] = {0,0, 0,1, 1,0, 1,1};
    int   h_Y[M] = {0,1,1,0};

    // Allocate GPU memory
    float *d_X, *W1, *b1, *Z1, *W2, *b2, *Z2;
    cudaMalloc(&d_X, M * input_dim * sizeof(float));
    cudaMalloc(&W1, input_dim * hidden_dim * sizeof(float));
    cudaMalloc(&b1, hidden_dim * sizeof(float));
    cudaMalloc(&Z1, M * hidden_dim * sizeof(float));
    cudaMalloc(&W2, hidden_dim * output_dim * sizeof(float));
    cudaMalloc(&b2, output_dim * sizeof(float));
    cudaMalloc(&Z2, M * output_dim * sizeof(float));

    cudaMemcpy(d_X, h_X, M * input_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Init weights randomly
    float h_W1[input_dim * hidden_dim], h_b1[hidden_dim];
    float h_W2[hidden_dim * output_dim], h_b2[output_dim];
    for (int i = 0; i < input_dim * hidden_dim; i++) h_W1[i] = 0.01f * (rand()%100/100.0f);
    for (int i = 0; i < hidden_dim; i++) h_b1[i] = 0;
    for (int i = 0; i < hidden_dim * output_dim; i++) h_W2[i] = 0.01f * (rand()%100/100.0f);
    for (int i = 0; i < output_dim; i++) h_b2[i] = 0;

    cudaMemcpy(W1, h_W1, sizeof(h_W1), cudaMemcpyHostToDevice);
    cudaMemcpy(b1, h_b1, sizeof(h_b1), cudaMemcpyHostToDevice);
    cudaMemcpy(W2, h_W2, sizeof(h_W2), cudaMemcpyHostToDevice);
    cudaMemcpy(b2, h_b2, sizeof(h_b2), cudaMemcpyHostToDevice);

    // Training loop (simplified, forward only)
    dim3 block(16,16);
    dim3 grid1((hidden_dim+15)/16, (M+15)/16);
    dim3 grid2((output_dim+15)/16, (M+15)/16);

    for (int epoch=0; epoch<1000; epoch++) {
        // Forward: Input -> Hidden
        matmul<<<grid1, block>>>(d_X, W1, Z1, M, hidden_dim, input_dim);
        add_bias_relu<<<grid1, block>>>(Z1, b1, M, hidden_dim);

        // Forward: Hidden -> Output
        matmul<<<grid2, block>>>(Z1, W2, Z2, M, output_dim, hidden_dim);
        add_bias_relu<<<grid2, block>>>(Z2, b2, M, output_dim);

        softmax<<<1, M>>>(Z2, M, output_dim);

        float* d_loss;
        cudaMalloc(&d_loss, sizeof(float));
        cudaMemset(d_loss, 0, sizeof(float));

        int* d_Y;
        cudaMalloc(&d_Y, M * sizeof(int));
        cudaMemcpy(d_Y, h_Y, M * sizeof(int), cudaMemcpyHostToDevice);

        compute_loss<<<1, M>>>(Z2, d_Y, d_loss, M, output_dim);

        float h_loss;
        cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << " Loss: " << h_loss/M << std::endl;
        }

        cudaFree(d_loss);
        cudaFree(d_Y);
    }

    // Inference: just run forward pass same as above

    return 0;
}
