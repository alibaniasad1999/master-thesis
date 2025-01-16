// src/torch_inference_node.cpp

#include <rclcpp/rclcpp.hpp>
#include <torch/script.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <chrono>

using namespace std::chrono;

class TorchInferenceNode : public rclcpp::Node {
public:
    TorchInferenceNode() : Node("torch_inference_node") {
        RCLCPP_INFO(this->get_logger(), "Initializing Torch Inference Node");

        // Load the model
        if (!model_file_exists(model_path_)) {
            RCLCPP_ERROR(this->get_logger(), "Model file not found: %s", model_path_.c_str());
            rclcpp::shutdown();
            return;
        }

        auto start = high_resolution_clock::now();

        try {
            module_ = torch::jit::load(model_path_);
            RCLCPP_INFO(this->get_logger(), "Model loaded successfully.");
        } catch (const c10::Error& e) {
            RCLCPP_ERROR(this->get_logger(), "Error loading the model: %s", e.what());
            rclcpp::shutdown();
            return;
        }

        // Prepare the input tensor
        input_tensor_ = torch::randn({1, 4});
        RCLCPP_INFO(this->get_logger(), "Input Tensor: %s", input_tensor_.toString().c_str());

        // Perform inference
        perform_inference();

        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        RCLCPP_INFO(this->get_logger(), "Time taken by function: %ld microseconds", duration.count());

        // Optionally, you can set up a timer to perform inference periodically
        // timer_ = this->create_wall_timer(
        //     std::chrono::seconds(1),
        //     std::bind(&TorchInferenceNode::perform_inference, this)
        // );
    }

private:
    bool model_file_exists(const std::string& path) {
        std::ifstream f(path.c_str());
        return f.good();
    }

    void perform_inference() {
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor_);

        try {
            torch::jit::IValue output = module_.forward(inputs);

            at::Tensor output_tensor;
            if (output.isTensor()) {
                output_tensor = output.toTensor();
            }
            else if (output.isTuple()) {
                auto output_tuple = output.toTuple();
                if (output_tuple->elements().size() > 0) {
                    output_tensor = output_tuple->elements()[0].toTensor();
                } else {
                    RCLCPP_ERROR(this->get_logger(), "Output tuple is empty.");
                    return;
                }
            }
            else {
                RCLCPP_ERROR(this->get_logger(), "Unexpected output type.");
                return;
            }

            RCLCPP_INFO(this->get_logger(), "Inference completed successfully.");
            RCLCPP_INFO(this->get_logger(), "Output Tensor: %s", output_tensor.toString().c_str());

            // Convert the tensor to float(s)
            if (output_tensor.dim() == 0) {
                float output_float = output_tensor.item<float>();
                RCLCPP_INFO(this->get_logger(), "Output (Scalar): %f", output_float);
            }
            else if (output_tensor.dim() == 1) {
                std::ostringstream oss;
                oss << "Output (1D Tensor): ";
                for (int64_t i = 0; i < output_tensor.size(0); ++i) {
                    float value = output_tensor[i].item<float>();
                    oss << value << " ";
                }
                RCLCPP_INFO(this->get_logger(), "%s", oss.str().c_str());
            }
            else if (output_tensor.dim() == 2) {
                std::ostringstream oss;
                oss << "Output (2D Tensor):\n";
                for (int64_t i = 0; i < output_tensor.size(0); ++i) {
                    for (int64_t j = 0; j < output_tensor.size(1); ++j) {
                        float value = output_tensor[i][j].item<float>();
                        oss << value << " ";
                    }
                    oss << "\n";
                }
                RCLCPP_INFO(this->get_logger(), "%s", oss.str().c_str());
            }
            else {
                RCLCPP_INFO(this->get_logger(), "Output has %d dimensions. Conversion not implemented.",
                            output_tensor.dim());
            }

        } catch (const c10::Error& e) {
            RCLCPP_ERROR(this->get_logger(), "Error during inference: %s", e.what());
        }
    }

    // Member variables
    // Inside your TorchInferenceNode class
    std::string model_path_ = "/home/ali/Documents/University/master-thesis/Code/realtime/src/torch_interface_node/model/pi_model_traced.pt";

    torch::jit::script::Module module_;
    at::Tensor input_tensor_;
    // rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<TorchInferenceNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
