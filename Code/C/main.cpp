#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <fstream>

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
    std::string model_path = "../model/sac_mbk_pi_model.pt";

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
    torch::Tensor input_tensor = torch::rand({1, 2});

    // Perform inference
    torch::jit::IValue output_tensor = module.forward({input_tensor});
    std::cout << "Output tensor: " << output_tensor << std::endl;

    // output tuple to tensor
    torch::Tensor output = output_tensor.toTuple()->elements()[0].toTensor();
    std::cout << "Output tensor: " << output << std::endl;


    return 0;
}
