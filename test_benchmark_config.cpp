#include "C:\Users\win10\Desktop\Code\scheme3\includes\benchmark_yaml.h"
#include <gtest/gtest.h>
#include <fstream>

TEST(BenchmarkConfigTest, TestParseConfig) {
    std::ofstream file("test_config.yaml");
    file << "cmd: sim_command\n"
         << "args: [arg1, arg2]\n"
         << "log: log.txt\n"
         << "is_to_stdout: true\n"
         << "clock_rate: 1.0\n"
         << "pre_copy: pre_copy\n"
         << "chip_frequency: 1000";
    file.close();

    BenchmarkConfig config("test_config.yaml");
    EXPECT_EQ(config.m_proc_cfg_list.size(), 1);
    EXPECT_EQ(config.m_proc_cfg_list[0].m_command, "sim_command");
    EXPECT_EQ(config.m_proc_cfg_list[0].m_args.size(), 2);
    EXPECT_EQ(config.m_proc_cfg_list[0].m_log_file, "log.txt");
    EXPECT_EQ(config.m_proc_cfg_list[0].m_to_stdout, true);
    EXPECT_EQ(config.m_proc_cfg_list[0].m_clock_rate, 1.0);
    EXPECT_EQ(config.m_proc_cfg_list[0].m_pre_copy, "pre_copy");
    EXPECT_EQ(config.m_proc_cfg_list[0].m_chip_frequency, 1000);
}