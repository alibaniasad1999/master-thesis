// src/hello_torch_node.cpp

#include <rclcpp/rclcpp.hpp>
#include <torch/torch.h>
#include <iostream>
#include <memory>
#include <sstream>

class HelloTorchNode : public rclcpp::Node {
public:
    HelloTorchNode() : Node("hello_torch_node") {
        RCLCPP_INFO(this->get_logger(), "Hello, World from LibTorch within ROS 2!");

        // Create a random tensor
        torch::Tensor tensor = torch::rand({2, 3});

        // Convert tensor to string for logging
        std::stringstream ss;
        ss << tensor;

        RCLCPP_INFO(this->get_logger(), "Random Tensor:\n%s", ss.str().c_str());
    }
};

int main(int argc, char **argv) {
    // Initialize ROS 2
    rclcpp::init(argc, argv);

    // Create and spin the node
    rclcpp::spin(std::make_shared<HelloTorchNode>());

    // Shutdown ROS 2
    rclcpp::shutdown();
    return 0;
}
