#include "rclcpp/rclcpp.hpp"
#include "tbp_interface/srv/control_command.hpp"
#include <torch/script.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <chrono>
#include "src/ModelLocator.h"

using namespace std::chrono;

class ControllerServiceNode : public rclcpp::Node {
public:
    ControllerServiceNode() : Node("controller_service_node"), model_path_("/home/ali/Documents/University/master-thesis/Code/ROS2/src/tbp_sac_controler/src/model/sac_model.pt") {
        RCLCPP_INFO(this->get_logger(), "Initializing Controller Service Node");

        // Load the Torch model
        if (!model_file_exists(model_path_)) {
            RCLCPP_ERROR(this->get_logger(), "Model file not found: %s", model_path_.c_str());
            rclcpp::shutdown();
            return;
        }

        try {
            module_ = torch::jit::load(model_path_);
            // warmup model with 100 iteration
            for (int i = 0; i < 100; ++i) {
              // random number
              torch::Tensor input_tensor = torch::rand({1, 4});
              torch::jit::IValue output_tensor = module_.forward({input_tensor});
            }
            RCLCPP_INFO(this->get_logger(), "Model loaded successfully.");
        } catch (const c10::Error& e) {
            RCLCPP_ERROR(this->get_logger(), "Error loading the model: %s", e.what());
            rclcpp::shutdown();
            return;
        }

        // Create the service server
        service_ = this->create_service<tbp_interface::srv::ControlCommand>(
            "compute_control_force", std::bind(&ControllerServiceNode::compute_control_callback, this, std::placeholders::_1, std::placeholders::_2));

        RCLCPP_INFO(this->get_logger(), "Controller Service Node is ready to compute control forces using Torch model.");
    }

private:
    bool model_file_exists(const std::string& path) {
        std::ifstream f(path.c_str());
        return f.good();
    }

    void compute_control_callback(
        const std::shared_ptr<tbp_interface::srv::ControlCommand::Request> request,
        std::shared_ptr<tbp_interface::srv::ControlCommand::Response> response)
    {
        // Extract the state from the request
        double current_position_x = request->position_x;
        double current_position_y = request->position_y;
        double current_velocity_x = request->velocity_x;
        double current_velocity_y = request->velocity_y;

        RCLCPP_INFO(this->get_logger(), "Received state -> Position x: %f, Position y: %f, Velocity x: %f, Velocity y: %f",
            current_position_x, current_position_y, current_velocity_x, current_velocity_y);

        // Prepare the input tensor for the model
        // We assume the model expects a 1D tensor with position and velocity as inputs
        at::Tensor input_tensor = torch::tensor({{current_position_x, current_position_y, current_velocity_x, current_velocity_y}});

        // Perform inference
        try {
            RCLCPP_INFO(this->get_logger(), "Computing control force from Torch model");
            std::cout << "Input tensor: " << input_tensor << std::endl;
            auto start = high_resolution_clock::now();
            torch::jit::IValue output = module_.forward({input_tensor});
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(stop - start);
            RCLCPP_INFO(this->get_logger(), "Time taken by inference: %ld microseconds", duration.count());
            // print the output tensor in ros
            std::cout << "Output tensor: " << output << std::endl;
            auto outputs = output.toTuple();
            // Extract the control force from the model output
            // at::Tensor output_tensor = output.toTensor();
            at::Tensor output_tensor = outputs->elements()[0].toTensor();
            float control_force_x = output_tensor[0][0].item<float>();
            float control_force_y = output_tensor[0][1].item<float>();

            RCLCPP_INFO(this->get_logger(), "Computed control force from Torch model: control force x: %f, control force y: %f", control_force_x, control_force_y);


            // Set the response
            response->control_force_x = control_force_x;
            response->control_force_y = control_force_y;
        } catch (const c10::Error& e) {
            RCLCPP_ERROR(this->get_logger(), "Error during inference: %s", e.what());
            response->control_force_x = 0.0;
            response->control_force_y = 0.0;
        }
    }

    std::string model_path_;
    torch::jit::script::Module module_;
    rclcpp::Service<tbp_interface::srv::ControlCommand>::SharedPtr service_;
};

int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ControllerServiceNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
