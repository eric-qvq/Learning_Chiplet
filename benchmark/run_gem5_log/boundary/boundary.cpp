#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "apis_c.h"

// Simple clamp to keep values within |bound|.
int64_t clamp_value(int64_t v, int64_t bound) {
    if (v > bound) return bound;
    if (v < -bound) return -bound;
    return v;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: boundary_cpu <srcX> <srcY>\n";
        return 1;
    }

    const int srcX = std::atoi(argv[1]);
    const int srcY = std::atoi(argv[2]);

    // Targets exercise the coordinate boundary of a 2x2 mesh and vary payload size.
    struct Target {
        int dstX;
        int dstY;
        int64_t len;
        int64_t clamp;
    };

    std::vector<Target> targets = {
        {0, 1, 128, 256},
        {1, 0, 512, 512},
        {1, 1, 1024, 1024},
    };

    std::mt19937_64 rng(42);
    std::uniform_int_distribution<int32_t> dist(-1500, 1500);

    for (const auto& t : targets) {
        std::vector<int32_t> payload(static_cast<size_t>(t.len));
        for (auto& v : payload) {
            v = dist(rng);
        }

        // Compute expected clamp-and-sum on CPU for verification.
        int64_t expected = 0;
        for (auto v : payload) {
            expected += clamp_value(v, t.clamp);
        }

        int64_t header[2] = {t.len, t.clamp};
        // First send metadata so the GPU knows how many elements to expect and what clamp to use.
        InterChiplet::sendMessage(t.dstX, t.dstY, srcX, srcY, (void*)header, sizeof(header));
        // Then send the data buffer.
        InterChiplet::sendMessage(t.dstX, t.dstY, srcX, srcY, payload.data(),
                                  t.len * sizeof(int32_t));

        int64_t result = 0;
        InterChiplet::receiveMessage(srcX, srcY, t.dstX, t.dstY, (void*)&result,
                                     sizeof(int64_t));

        std::cout << "[CPU] From chiplet (" << t.dstX << "," << t.dstY << ") length=" << t.len
                  << " clamp=" << t.clamp << " -> sum=" << result
                  << " (expected " << expected << ")\n";
    }

    return 0;
}
