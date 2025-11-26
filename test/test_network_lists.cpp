#include "net_bench.h"
#include "net_delay.h"
#include <gtest/gtest.h>

TEST(NetworkBenchListTest, TestInsertAndDump) {
    InterChiplet::NetworkBenchList bench_list;
    InterChiplet::NetworkBenchItem item(10.0, {1, 2}, {3, 4}, 100, 1);
    bench_list.insert(item);
    bench_list.dumpBench("bench.txt", 1.0);

    std::ifstream file("bench.txt");
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    EXPECT_NE(content.find("10 10 1 3 100 1"), std::string::npos);
}

TEST(NetworkDelayListTest, TestInsertAndDump) {
    InterChiplet::NetworkDelayList delay_list;
    InterChiplet::NetworkDelayItem item(10.0, {1, 2}, {3, 4}, 1, {1.0, 2.0});
    delay_list.insert(10.0, item);
    delay_list.dumpDelay("delay.txt");

    std::ifstream file("delay.txt");
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    EXPECT_NE(content.find("10 1 3 1 2 1.0 2.0"), std::string::npos);
}