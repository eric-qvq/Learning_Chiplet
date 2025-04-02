#include "cmd_handler.h"
#include "global_manager.h"
#include <gtest/gtest.h>

class CmdHandlerTest : public ::testing::Test {
protected:
    GlobalManager gm;

    void SetUp() override {
        gm = GlobalManager();
    }
};

TEST_F(CmdHandlerTest, TestHandleSendCmd) {
    InterChiplet::SyncCommand cmd;
    cmd.m_type = InterChiplet::SC_SEND;
    cmd.m_src = {"Sim1"};
    cmd.m_dst = {"Sim2"};
    cmd.m_cycle = 10.0;
    cmd.m_clock_rate = 1.0;
    cmd.m_nbytes = 100;
    handle_send_cmd(cmd, &gm);
    EXPECT_EQ(gm.requestList.size(), 1);
}

TEST_F(CmdHandlerTest, TestHandleReceiveCmd) {
    InterChiplet::SyncCommand cmd;
    cmd.m_type = InterChiplet::SC_RECEIVE;
    cmd.m_src = {"Sim2"};
    cmd.m_dst = {"Sim1"};
    cmd.m_cycle = 10.0;
    cmd.m_clock_rate = 1.0;
    cmd.m_nbytes = 100;
    handle_receive_cmd(cmd, &gm);
    EXPECT_EQ(gm.requestList.size(), 1);
}