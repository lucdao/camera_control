#pragma once
#include "ptz/algorithms.hpp"
#include <filesystem>

namespace ptz {
std::vector<GalleryIdentity> load_gallery(const std::filesystem::path& root);
void save_embedding(const std::filesystem::path& root,const std::string& identity,
                    const std::string& sample,const std::vector<float>& embedding);
}
