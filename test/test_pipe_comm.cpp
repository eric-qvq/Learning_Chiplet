#include "pipe_comm.h"
#include <gtest/gtest.h>
#include <unistd.h>

TEST(PipeCommTest, TestPipeCommunication) {
    const char* pipe_name = "/tmp/test_pipe";
    InterChiplet::PipeComm pipe_comm;

    // Write data to pipe
    std::string data = "Hello, World!";
    pipe_comm.write_data(pipe_name, data.c_str(), data.size());

    // Read data from pipe
    char buffer[1024];
    int bytes_read = pipe_comm.read_data(pipe_name, buffer, sizeof(buffer) - 1);
    buffer[bytes_read] = '\0';
    EXPECT_EQ(std::string(buffer), data);
}