#include <torch/script.h>                // One-stop header.
#include <iostream>
#include <filesystem>                    // C++17 filesystem
#include <memory>

int main(int argc, const char* argv[]) {
    try {
        // 1. Print current working directory
        auto cwd = std::filesystem::current_path();
        std::cout << "Current working directory: " << cwd << "\n";

        // 2. Build relative path: go up one level, then into "model"
        std::filesystem::path model_rel = cwd / ".." / "model" / "sac_mbk_pi_model.pt";

        // 3. Canonicalize (resolves '..' and symlinks)
        std::filesystem::path model_path = std::filesystem::canonical(model_rel);
        std::cout << "Loading model from: " << model_path << "\n";

        // // 4. Load the TorchScript module
        torch::jit::script::Module module = torch::jit::load(model_path.string());
        std::cout << "Model loaded successfully!\n";

        torch::Tensor input = torch::rand({2, 2});

        std::cout << "Input tensor: " << input << "\n";

        // // (Optional) Example dummy inference:
        // /*
        // torch::Tensor input = torch::rand({1, /* your feature dims */});
        // auto output = module.forward({input}).toTensor();
        // std::cout << "Output tensor: " << output << "\n";
        //
        torch::jit::IValue output_tensor = module.forward({input});
        std::cout << "Output tensor: " << output_tensor << "\n";

    }
    catch (const c10::Error& e) {
        std::cerr << "Torch error while loading the model:\n"
                  << e.what() << "\n";
        return -1;
    }
    catch (const std::exception& e) {
        std::cerr << "Standard exception:\n"
                  << e.what() << "\n";
        return -1;
    }

    return 0;
}
