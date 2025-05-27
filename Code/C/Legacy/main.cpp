#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <chrono>

int main() {
    // Check if CUDA is available
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available!" << std::endl;
    } else {
        std::cout << "CUDA is not available." << std::endl;
    }

    // Simple tensor example
    torch::Tensor tensor = torch::rand({2, 2});
    std::cout << tensor << std::endl;

    // load model
    std::string model_path = "model/sac_mbk_pi_model.pt";

    // Create an ifstream object to check if the file exists
    std::ifstream file(model_path.c_str());

    if (file.good()) {
            std::cout << "Model file found: " << model_path << std::endl;
        } else {
            std::cout << "Model file not found: " << model_path << std::endl;
    }


    // Declare the module variable to hold the loaded model
    torch::jit::script::Module module;

    try {
        module = torch::jit::load(model_path);
        std::cout << "Model loaded!" << std::endl;
    } catch (const c10::Error& e) {
        std::cout << "Error loading the model: " << e.what() << std::endl;
    }

    // Create a tensor with random values
    torch::Tensor input_tensor = torch::rand({1, 2})*1000;

    auto start = std::chrono::high_resolution_clock::now();
    // Perform inference
    torch::jit::IValue output_tensor = module.forward({input_tensor});
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << "Output tensor: " << output_tensor << std::endl;

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Inference time: " << duration.count() << " microseconds" << std::endl;

    float kp = 10.0;
    float ki = 0.1;
    float kd = 0.01;

    float force = 0.0;

    for (int i = 0; i<=100; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        // Perform inference
        torch::jit::IValue output_tensor = module.forward({input_tensor});
        auto stop = std::chrono::high_resolution_clock::now();
        std::cout << "Output tensor: " << output_tensor << std::endl;

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "Inference time: " << duration.count() << " microseconds" << std::endl;

        auto start_pid = std::chrono::high_resolution_clock::now();
        // Perform inference
        force += 1 * kp + 1 * ki + 1 * kd;
        auto stop_pid = std::chrono::high_resolution_clock::now();
        auto duration_ = std::chrono::duration_cast<std::chrono::microseconds>(stop_pid - start_pid);
        std::cout << "PID time: " << duration_.count() << " microseconds" << std::endl;
    }

    // output tuple to tensor
    torch::Tensor output = output_tensor.toTuple()->elements()[0].toTensor();
    std::cout << "Output tensor: " << output << std::endl;


    return 0;
}
