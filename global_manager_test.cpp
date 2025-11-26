#include "global_manager.h"
#include <gtest/gtest.h>

class GlobalManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        gm = std::make_unique<GlobalManager>();
    }

    void TearDown() override {
        gm.reset();
    }

    std::unique_ptr<GlobalManager> gm;
};

TEST_F(GlobalManagerTest, TestIsFinish) {
    EXPECT_TRUE(gm->IsFinish());
}

TEST_F(GlobalManagerTest, TestAddRequest) {
    InterChiplet::AddrType sender = {0, 0};
    InterChiplet::AddrType receiver = {1, 1};
    std::string data = "Hello";
    double senderClock = 100.0;
    double frequency = 1.0;
    InterChiplet::SyncCommType behavior = InterChiplet::SC_SEND;

    gm->AddRequest(sender, receiver, data, senderClock, frequency, behavior);
    EXPECT_FALSE(gm->IsFinish());
}

TEST_F(GlobalManagerTest, TestCheckPair) {
    InterChiplet::AddrType sender = {0, 0};
    InterChiplet::AddrType receiver = {1, 1};
    std::string data = "Hello";
    double senderClock = 100.0;
    double frequency = 1.0;

    gm->AddRequest(sender, receiver, data, senderClock, frequency, InterChiplet::SC_SEND);
    gm->AddRequest(receiver, sender, data, senderClock, frequency, InterChiplet::SC_RECEIVE);

    EXPECT_TRUE(gm->CheckPair());
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}