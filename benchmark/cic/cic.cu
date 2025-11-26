#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <vector>

#include "apis_cu.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Each block reduces a slice of the payload; shared memory holds partial sums.
__global__ void reduce_sum(const int32_t* in, int64_t* block_sums, int n) {
    extern __shared__ int64_t sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    int64_t v = 0;
    if (idx < n) v = static_cast<int64_t>(in[idx]);
    sdata[tid] = v;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) block_sums[blockIdx.x] = sdata[0];
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: cic_cu <idX> <idY>\n");
        return 1;
    }

    int idX = std::atoi(argv[1]);
    int idY = std::atoi(argv[2]);

    int64_t* d_len = nullptr;
    cudaMalloc((void**)&d_len, sizeof(int64_t));

    while (true) {
        // Receive length of incoming payload.
        receiveMessage(idX, idY, 0, 0, d_len, sizeof(int64_t));
        int64_t h_len = 0;
        cudaMemcpy(&h_len, d_len, sizeof(int64_t), cudaMemcpyDeviceToHost);

        if (h_len <= 0) break;  // stop signal

        int32_t* d_data = nullptr;
        cudaMalloc((void**)&d_data, sizeof(int32_t) * h_len);
        receiveMessage(idX, idY, 0, 0, d_data, sizeof(int32_t) * h_len);

        int threads = 256;
        int blocks = static_cast<int>((h_len + threads - 1) / threads);
        int shared_bytes = threads * sizeof(int64_t);

        int64_t* d_block_sums = nullptr;
        cudaMalloc((void**)&d_block_sums, sizeof(int64_t) * blocks);

        reduce_sum<<<blocks, threads, shared_bytes>>>(d_data, d_block_sums,
                                                      static_cast<int>(h_len));
        cudaDeviceSynchronize();

        std::vector<int64_t> h_block_sums(static_cast<size_t>(blocks), 0);
        cudaMemcpy(h_block_sums.data(), d_block_sums, sizeof(int64_t) * blocks,
                   cudaMemcpyDeviceToHost);

        int64_t total = 0;
        for (auto v : h_block_sums) total += v;

        printf("[GPU %d,%d] len=%ld sum=%ld\n", idX, idY, static_cast<long>(h_len),
               static_cast<long>(total));

        int64_t* d_result = nullptr;
        cudaMalloc((void**)&d_result, sizeof(int64_t));
        cudaMemcpy(d_result, &total, sizeof(int64_t), cudaMemcpyHostToDevice);

        // Echo checksum back to CPU source (0,0).
        sendMessage(0, 0, idX, idY, d_result, sizeof(int64_t));

        cudaFree(d_data);
        cudaFree(d_block_sums);
        cudaFree(d_result);
    }

    cudaFree(d_len);
    return 0;
}
