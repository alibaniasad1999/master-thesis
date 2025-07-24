//
// Created by Ali Baniasad on 6/9/2025 A.
//

#pragma once

#include <filesystem>
#include <string>

namespace ModelLocator {

/// Locate a TorchScript model living under the "master-thesis/Code/C/model" tree.
/// \param filename  e.g. "sac_mbk_pi_model.pt"
/// \returns          absolute, canonical path to that model file
    std::filesystem::path locateModel(const std::string& filename);

} // namespace ModelLocator
