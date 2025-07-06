#include "src/ModelLocator.h"
#include "torch/script.h"
#include <iostream>

int main() {
    try {
        // Only need to supply the .pt filename:
        auto modelPath = ModelLocator::locateModel("zs_sac_traced_deterministic.pt");
        std::cout << "Loading model from: " << modelPath << "\n";

        auto module = torch::jit::load(modelPath.string());
        std::cout << "Model loaded successfully!\n";

        // (Your inference code here...)
        torch::Tensor input = torch::rand({1, 4});
        auto output = module.forward({input});
        std::cout << "Output tensor: " << output << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return -1;
    }
    return 0;
}
