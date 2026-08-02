#include "ptz/gallery.hpp"
#include <algorithm>
#include <fstream>
#include <cctype>
#include <stdexcept>

namespace ptz {
namespace {bool safe_token(const std::string& value){return !value.empty()&&std::all_of(value.begin(),value.end(),[](unsigned char c){return std::isalnum(c)||c=='_'||c=='-';});}}

std::vector<GalleryIdentity> load_gallery(const std::filesystem::path& root){
  std::vector<GalleryIdentity> out;
  if(!std::filesystem::exists(root)) return out;
  for(const auto& idir:std::filesystem::directory_iterator(root)){
    if(!idir.is_directory())continue;
    GalleryIdentity id;id.name=idir.path().filename().string();
    for(const auto& file:std::filesystem::directory_iterator(idir.path())){
      if(!file.is_regular_file()||file.path().extension()!=".f32")continue;
      std::ifstream in(file.path(),std::ios::binary|std::ios::ate);if(!in)continue;
      const auto bytes=in.tellg();if(bytes<=0||bytes%static_cast<std::streamoff>(sizeof(float))!=0)continue;
      in.seekg(0);std::vector<float> e(static_cast<std::size_t>(bytes)/sizeof(float));
      in.read(reinterpret_cast<char*>(e.data()),bytes);
      if(in&&e.size()==512)id.templates.push_back(l2_normalize(e));
    }
    if(!id.templates.empty())out.push_back(std::move(id));
  }
  return out;
}

void save_embedding(const std::filesystem::path& root,const std::string& identity,
                    const std::string& sample,const std::vector<float>& embedding){
  if(!safe_token(identity)||!safe_token(sample))
    throw std::invalid_argument("invalid identity or sample");
  const auto normalized=l2_normalize(embedding);
  if(normalized.size()!=512)throw std::invalid_argument("ArcFace embedding must contain 512 floats");
  const auto dir=root/identity;std::filesystem::create_directories(dir);
  const auto destination=dir/(sample+".f32"),temporary=dir/(sample+".tmp");
  {std::ofstream out(temporary,std::ios::binary|std::ios::trunc);if(!out)throw std::runtime_error("cannot create embedding");
   out.write(reinterpret_cast<const char*>(normalized.data()),static_cast<std::streamsize>(normalized.size()*sizeof(float)));
   if(!out)throw std::runtime_error("cannot write embedding");}
  std::filesystem::rename(temporary,destination);
}

} // namespace ptz
