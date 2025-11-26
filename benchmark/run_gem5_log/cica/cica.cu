#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <vector>

#include "apis_cu.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Reduce a slice; block_sums holds one partial sum per block.
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
        printf("Usage: cica_cu <idX> <idY>\n");
        return 1;
    }

    int idX = std::atoi(argv[1]);
    int idY = std::atoi(argv[2]);

    int64_t* d_batch_count = nullptr;
    cudaMalloc((void**)&d_batch_count, sizeof(int64_t));

    while (true) {
        // Receive number of slices in this batch.
        receiveMessage(idX, idY, 0, 0, d_batch_count, sizeof(int64_t));
        int64_t batch_count = 0;
        cudaMemcpy(&batch_count, d_batch_count, sizeof(int64_t), cudaMemcpyDeviceToHost);
        if (batch_count <= 0) break;

        // Receive slice lengths.
        int64_t* d_lengths = nullptr;
        cudaMalloc((void**)&d_lengths, sizeof(int64_t) * batch_count);
        receiveMessage(idX, idY, 0, 0, d_lengths, sizeof(int64_t) * batch_count);
        std::vector<int64_t> h_lengths(static_cast<size_t>(batch_count), 0);
        cudaMemcpy(h_lengths.data(), d_lengths, sizeof(int64_t) * batch_count,
                   cudaMemcpyDeviceToHost);

        // Compute total payload length and max slice for scratch sizing.
        int64_t total_len = 0;
        int64_t max_len = 0;
        for (auto len : h_lengths) {
            total_len += len;
            if (len > max_len) max_len = len;
        }

        int32_t* d_payload = nullptr;
        cudaMalloc((void**)&d_payload, sizeof(int32_t) * total_len);
        receiveMessage(idX, idY, 0, 0, d_payload, sizeof(int32_t) * total_len);

        int threads = 256;
        int max_blocks = static_cast<int>((max_len + threads - 1) / threads);
        int shared_bytes = threads * sizeof(int64_t);
        int64_t* d_block_sums = nullptr;
        cudaMalloc((void**)&d_block_sums, sizeof(int64_t) * max_blocks);

        std::vector<int64_t> results(static_cast<size_t>(batch_count), 0);
        int64_t offset = 0;
        for (int64_t i = 0; i < batch_count; i++) {
            int64_t len = h_lengths[static_cast<size_t>(i)];
            int blocks = static_cast<int>((len + threads - 1) / threads);
            reduce_sum<<<blocks, threads, shared_bytes>>>(d_payload + offset, d_block_sums,
                                                          static_cast<int>(len));
            cudaDeviceSynchronize();

            std::vector<int64_t> h_block_sums(static_cast<size_t>(blocks), 0);
            cudaMemcpy(h_block_sums.data(), d_block_sums, sizeof(int64_t) * blocks,
                       cudaMemcpyDeviceToHost);
            int64_t total = 0;
            for (auto v : h_block_sums) total += v;
            results[static_cast<size_t>(i)] = total;
            offset += len;
        }

        printf("[GPU %d,%d] batches=%ld total_len=%ld\n", idX, idY,
               static_cast<long>(batch_count), static_cast<long>(total_len));

        // Send all results back together.
        int64_t* d_results = nullptr;
        cudaMalloc((void**)&d_results, sizeof(int64_t) * batch_count);
        cudaMemcpy(d_results, results.data(), sizeof(int64_t) * batch_count,
                   cudaMemcpyHostToDevice);
        sendMessage(0, 0, idX, idY, d_results, sizeof(int64_t) * batch_count);

        cudaFree(d_lengths);
        cudaFree(d_payload);
        cudaFree(d_block_sums);
        cudaFree(d_results);
    }

    cudaFree(d_batch_count);
    return 0;
}
