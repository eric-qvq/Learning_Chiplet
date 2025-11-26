#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include "apis_c.h"

// Simple checksum to compare device result against host.
int64_t checksum_cpu(const std::vector<int32_t>& data) {
    int64_t sum = 0;
    for (auto v : data) sum += static_cast<int64_t>(v);
    return sum;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: cic_cpu <srcX> <srcY>\n";
        return 1;
    }

    const int srcX = std::atoi(argv[1]);
    const int srcY = std::atoi(argv[2]);

    // Stress three neighbor chiplets in the 2x2 mesh.
    struct Target {
        int dstX;
        int dstY;
    };
    std::vector<Target> targets = {{0, 1}, {1, 0}, {1, 1}};
    std::vector<int64_t> sizes = {256, 1024, 4096, 16384};  // number of int32 elements

    std::mt19937 rng(7);
    std::uniform_int_distribution<int32_t> dist(-1024, 1024);

    for (auto len : sizes) {
        for (auto t : targets) {
            std::vector<int32_t> payload(static_cast<size_t>(len));
            for (auto& v : payload) v = dist(rng);

            int64_t expected = checksum_cpu(payload);
            int64_t header = len;

            // Send length then payload.
            InterChiplet::sendMessage(t.dstX, t.dstY, srcX, srcY, &header, sizeof(header));


            InterChiplet::sendMessage(t.dstX, t.dstY, srcX, srcY, payload.data(),payload.size() * sizeof(int32_t));


            auto t0 = std::chrono::high_resolution_clock::now();
            int64_t device_sum = 0;
            InterChiplet::receiveMessage(srcX, srcY, t.dstX, t.dstY, &device_sum,
                                         sizeof(int64_t));
            auto t1 = std::chrono::high_resolution_clock::now();
            auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

            std::cout << "[CPU] dst=(" << t.dstX << "," << t.dstY << ") len=" << len
                      << " sum=" << device_sum << " expected=" << expected
                      << " latency(us)=" << us << std::endl;
        }
    }

    // Stop signal so GPU kernels exit cleanly.
    int64_t stop = 0;
    for (auto t : targets) {
        InterChiplet::sendMessage(t.dstX, t.dstY, srcX, srcY, &stop, sizeof(stop));
    }
    return 0;
}
