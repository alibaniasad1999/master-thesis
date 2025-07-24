//
// Created by Ali Baniasad on 6/9/2025 A.
//
#include "ModelLocator.h"
#include <cstdlib>
#include <filesystem>
#include <stdexcept>

namespace fs = std::filesystem;

namespace ModelLocator {

// 1) Find the <master-thesis> root dir via:
//    a) MASTER_THESIS_ROOT env var
//    b) walking up from CWD
//    c) scanning HOME
    static fs::path findMasterThesisRoot() {
        if (auto* e = std::getenv("MASTER_THESIS_ROOT")) {
            if (fs::is_directory(e))
                return fs::path(e);
        }
        for (fs::path p = fs::current_path(); ; p = p.parent_path()) {
            if (p.filename() == "master-thesis")
                return p;
            if (!p.has_parent_path())
                break;
        }
#ifdef _WIN32
        const char* homeEnv = std::getenv("USERPROFILE");
#else
        const char* homeEnv = std::getenv("HOME");
#endif
        fs::path home = homeEnv ? homeEnv : fs::current_path();
        for (auto& e : fs::recursive_directory_iterator(home)) {
            if (e.is_directory() && e.path().filename() == "master-thesis")
                return e.path();
        }
        throw std::runtime_error(
                "Could not locate 'master-thesis' directory; "
                "set MASTER_THESIS_ROOT or run inside the tree.");
    }

// 2) Append the fixed "Code/C/model" path + filename, verify and canonicalize
    std::filesystem::path locateModel(const std::string& filename) {
        fs::path root = findMasterThesisRoot();
        fs::path p = root / "Code" / "C" / "model" / filename;
        if (!fs::exists(p) || !fs::is_regular_file(p)) {
            throw std::runtime_error("Model file not found: " + p.string());
        }
        return fs::canonical(p);
    }

} // namespace ModelLocator
