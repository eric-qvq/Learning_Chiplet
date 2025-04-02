#include "C:\Users\win10\Desktop\Code\scheme3\includes\global_manager.h"
#include <gtest/gtest.h>

class GlobalManagerTest : public ::testing::Test {
protected:
    GlobalManager gm;

    void SetUp() override {
        gm = GlobalManager();
    }
};

TEST_F(GlobalManagerTest, TestAddRequest) {
    gm.AddRequest("Sim1", "Sim2", "Data", 10.0, 1.0, Behavior::SEND);
    gm.AddRequest("Sim2", "Sim1", "Data", 10.0, 1.0, Behavior::RECEIVE);
    EXPECT_EQ(gm.requestList.size(), 2);
}

TEST_F(GlobalManagerTest, TestCheckPair) {
    gm.AddRequest("Sim1", "Sim2", "Data", 10.0, 1.0, Behavior::SEND);
    gm.AddRequest("Sim2", "Sim1", "Data", 10.0, 1.0, Behavior::RECEIVE);
    EXPECT_TRUE(gm.CheckPair());
}

TEST_F(GlobalManagerTest, TestUpdateWaterline) {
    gm.processClocks["Sim1"] = 5.0;
    gm.processClocks["Sim2"] = 10.0;
    gm.updateWaterline();
    EXPECT_EQ(gm.waterLine, 5.0);
}

TEST_F(GlobalManagerTest, TestCalculateNetworkDelay) {
    gm.processClocks["Sim1"] = 0.0;
    gm.processClocks["Sim2"] = 0.0;
    gm.calculateNetworkDelay("Sim1", "Sim2", 100);
    EXPECT_GT(gm.processClocks["Sim1"], 0.0);
    EXPECT_GT(gm.processClocks["Sim2"], 0.0);
}