#include "interchiplets.cpp"
#include <gtest/gtest.h>

TEST(ProcessStructTest, TestProcessStruct) {
    ProcessConfig config("sim_command", {"arg1", "arg2"}, "log.txt", true, 1.0, "pre_copy", 1000);
    ProcessStruct proc_struct(config);
    EXPECT_EQ(proc_struct.m_command, "sim_command");
    EXPECT_EQ(proc_struct.m_args.size(), 2);
    EXPECT_EQ(proc_struct.m_log_file, "log.txt");
    EXPECT_EQ(proc_struct.m_to_stdout, true);
    EXPECT_EQ(proc_struct.m_clock_rate, 1.0);
    EXPECT_EQ(proc_struct.m_pre_copy, "pre_copy");
    EXPECT_EQ(proc_struct.m_chip_frequency, 1000);
}