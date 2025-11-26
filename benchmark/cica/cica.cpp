#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "apis_c.h"

// CPU checksum for each slice.
int64_t checksum_cpu(const int32_t* data, int64_t len) {
    int64_t sum = 0;
    for (int64_t i = 0; i < len; i++) sum += static_cast<int64_t>(data[i]);
    return sum;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: cica_cpu <srcX> <srcY>\n";
        return 1;
    }

    const int srcX = std::atoi(argv[1]);
    const int srcY = std::atoi(argv[2]);

    // Neighbors in 2x2 mesh.
    struct Target {
        int dstX;
        int dstY;
    };
    std::vector<Target> targets = {{0, 1}, {1, 0}, {1, 1}};

    // Batch multiple message sizes into one aggregated transfer to reduce per-message overhead.
    std::vector<int64_t> batch_lengths = {256, 1024, 4096, 16384};  // int32 elements

    std::mt19937 rng(17);
    std::uniform_int_distribution<int32_t> dist(-2048, 2048);

    for (auto t : targets) {
        // Build one contiguous payload for all batches.
        int64_t total_len =
            std::accumulate(batch_lengths.begin(), batch_lengths.end(), int64_t{0});
        std::vector<int32_t> payload(static_cast<size_t>(total_len));
        int64_t offset = 0;
        for (auto len : batch_lengths) {
            for (int64_t i = 0; i < len; i++) {
                payload[static_cast<size_t>(offset + i)] = dist(rng);
            }
            offset += len;
        }

        // Pre-compute expected sums per slice.
        std::vector<int64_t> expected;
        expected.reserve(batch_lengths.size());
        offset = 0;
        for (auto len : batch_lengths) {
            expected.push_back(checksum_cpu(payload.data() + offset, len));
            offset += len;
        }

        int64_t batch_count = static_cast<int64_t>(batch_lengths.size());
        auto t0 = std::chrono::high_resolution_clock::now();

        // Send metadata: number of slices, then lengths array, then bulk payload.
        InterChiplet::sendMessage(t.dstX, t.dstY, srcX, srcY, &batch_count,
                                  sizeof(batch_count));
        InterChiplet::sendMessage(t.dstX, t.dstY, srcX, srcY, batch_lengths.data(),
                                  batch_lengths.size() * sizeof(int64_t));
        InterChiplet::sendMessage(t.dstX, t.dstY, srcX, srcY, payload.data(),
                                  payload.size() * sizeof(int32_t));

        // Receive all slice sums in one message.
        std::vector<int64_t> results(batch_lengths.size(), 0);
        InterChiplet::receiveMessage(srcX, srcY, t.dstX, t.dstY, results.data(),
                                     results.size() * sizeof(int64_t));

        auto t1 = std::chrono::high_resolution_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        int64_t total_bytes = sizeof(batch_count) +
                              batch_lengths.size() * sizeof(int64_t) +
                              payload.size() * sizeof(int32_t) +
                              results.size() * sizeof(int64_t);
        double bandwidth_gbps =
            (total_bytes * 1e-3) / us;  // MB/us ≈ GB/s (host-level indicator)

        std::cout << "[CPU] dst=(" << t.dstX << "," << t.dstY << ") batches="
                  << batch_lengths.size() << " total_len=" << total_len
                  << " bytes=" << total_bytes << " latency(us)=" << us
                  << " approx_bw(GB/s)=" << bandwidth_gbps << std::endl;
        for (size_t i = 0; i < batch_lengths.size(); i++) {
            std::cout << "  slice " << i << " len=" << batch_lengths[i]
                      << " sum=" << results[i] << " expected=" << expected[i] << std::endl;
        }
    }

    // Stop signal: batch_count = 0.
    int64_t stop = 0;
    for (auto t : targets) {
        InterChiplet::sendMessage(t.dstX, t.dstY, srcX, srcY, &stop, sizeof(stop));
    }
    return 0;
}
