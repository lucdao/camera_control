#include "ptz/engine_builder.hpp"
#include <NvInferVersion.h>
#include <cuda_runtime_api.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace ptz {
namespace {
void run(const std::vector<std::string>& args){std::vector<char*> argv;for(const auto& a:args)argv.push_back(const_cast<char*>(a.c_str()));argv.push_back(nullptr);pid_t pid=fork();if(pid<0)throw std::runtime_error("fork failed");if(pid==0){execv(argv[0],argv.data());_exit(127);}int status=0;if(waitpid(pid,&status,0)<0||!WIFEXITED(status)||WEXITSTATUS(status)!=0)throw std::runtime_error("trtexec failed");}
std::string fingerprint(const std::filesystem::path& p){
  // FNV-1a identifies the exact ONNX bytes for cache invalidation; the manifest
  // also records TensorRT/CUDA versions emitted by trtexec itself.
  std::ifstream in(p,std::ios::binary);if(!in)throw std::runtime_error("cannot open "+p.string());std::uint64_t h=1469598103934665603ULL;char buf[65536];
  while(in){in.read(buf,sizeof(buf));for(std::streamsize i=0;i<in.gcount();++i){h^=static_cast<unsigned char>(buf[i]);h*=1099511628211ULL;}}
  cudaDeviceProp prop{};if(cudaGetDeviceProperties(&prop,0)==cudaSuccess){const std::string abi=std::to_string(NV_TENSORRT_MAJOR)+"."+std::to_string(NV_TENSORRT_MINOR)+"."+std::to_string(NV_TENSORRT_PATCH)+":"+std::to_string(prop.major)+"."+std::to_string(prop.minor)+":"+prop.name;for(char c:abi){h^=static_cast<unsigned char>(c);h*=1099511628211ULL;}}
  char out[17];std::snprintf(out,sizeof(out),"%016llx",static_cast<unsigned long long>(h));return out;
}}
std::filesystem::path EngineBuilder::engine_path(const std::filesystem::path& onnx,const std::filesystem::path& cache){return cache/(onnx.stem().string()+"-"+fingerprint(onnx)+".fp16.engine");}
void EngineBuilder::build_all(const std::filesystem::path& models,const std::filesystem::path& cache){
  std::filesystem::create_directories(cache);const char* env=std::getenv("TRTEXEC");const std::string trtexec=env?env:"/usr/src/tensorrt/bin/trtexec";
  for(const char* name:{"det_10g.onnx","w600k_r50.onnx","yolo11n-pose.onnx","osnet_x0_25.onnx"}){
    const auto input=models/name;if(!std::filesystem::exists(input))throw std::runtime_error("missing model: "+input.string());const auto output=engine_path(input,cache);if(std::filesystem::exists(output))continue;const auto partial=output.string()+".partial";std::filesystem::remove(partial);
    run({trtexec,"--onnx="+input.string(),"--saveEngine="+partial,"--fp16","--builderOptimizationLevel=5","--memPoolSize=workspace:1024","--timingCacheFile="+(cache/"timing.cache").string(),"--skipInference"});std::filesystem::rename(partial,output);
  }
}
}
