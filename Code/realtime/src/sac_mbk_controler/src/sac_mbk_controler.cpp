#include "rclcpp/rclcpp.hpp"
#include "mbk_pid_controler_interface/srv/control_command.hpp"
#include <torch/script.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <chrono>

using namespace std::chrono;

class ControllerServiceNode : public rclcpp::Node {
public:
    ControllerServiceNode() : Node("controller_service_node"), model_path_("/home/ali/Documents/University/master-thesis/Code/realtime/src/sac_mbk_controler/src/model/sac_mbk_pi_model.pt") {
        RCLCPP_INFO(this->get_logger(), "Initializing Controller Service Node");

        // Load the Torch model
        if (!model_file_exists(model_path_)) {
            RCLCPP_ERROR(this->get_logger(), "Model file not found: %s", model_path_.c_str());
            rclcpp::shutdown();
            return;
        }

        try {
            module_ = torch::jit::load(model_path_);
            RCLCPP_INFO(this->get_logger(), "Model loaded successfully.");
        } catch (const c10::Error& e) {
            RCLCPP_ERROR(this->get_logger(), "Error loading the model: %s", e.what());
            rclcpp::shutdown();
            return;
        }

        // Create the service server
        service_ = this->create_service<mbk_pid_controler_interface::srv::ControlCommand>(
            "compute_control_force", std::bind(&ControllerServiceNode::compute_control_callback, this, std::placeholders::_1, std::placeholders::_2));

        RCLCPP_INFO(this->get_logger(), "Controller Service Node is ready to compute control forces using Torch model.");
    }

private:
    bool model_file_exists(const std::string& path) {
        std::ifstream f(path.c_str());
        return f.good();
    }

    void compute_control_callback(
        const std::shared_ptr<mbk_pid_controler_interface::srv::ControlCommand::Request> request,
        std::shared_ptr<mbk_pid_controler_interface::srv::ControlCommand::Response> response)
    {
        // Extract the state from the request
        double current_position = request->position;
        double current_velocity = request->velocity;

        RCLCPP_INFO(this->get_logger(), "Received state -> Position: %f, Velocity: %f", current_position, current_velocity);

        // Prepare the input tensor for the model
        // We assume the model expects a 1D tensor with position and velocity as inputs
        at::Tensor input_tensor = torch::tensor({current_position, current_velocity}, at::kFloat);

        // Perform inference
        try {
            torch::jit::IValue output = module_.forward({input_tensor});
            // print the output tensor
            std::cout << "Output tensor: " << output << std::endl;
            auto outputs = output.toTuple();
            // Extract the control force from the model output
            // at::Tensor output_tensor = output.toTensor();
            at::Tensor output_tensor = outputs->elements()[0].toTensor();
            float control_force = output_tensor.item<float>();

            RCLCPP_INFO(this->get_logger(), "Computed control force from Torch model: %f", control_force);

            // Set the response
            response->control_force = control_force;
        } catch (const c10::Error& e) {
            RCLCPP_ERROR(this->get_logger(), "Error during inference: %s", e.what());
            response->control_force = 0.0;
        }
    }

    std::string model_path_;
    torch::jit::script::Module module_;
    rclcpp::Service<mbk_pid_controler_interface::srv::ControlCommand>::SharedPtr service_;
};

int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ControllerServiceNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
