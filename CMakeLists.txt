cmake_minimum_required(VERSION 3.20)
project(ptz_control VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(PTZ_WITH_ONVIF "Build the libcurl/pugixml ONVIF backend" ON)
option(PTZ_WITH_GSTREAMER "Build the GStreamer RTSP source" OFF)
option(PTZ_WITH_TENSORRT "Build direct CUDA/TensorRT inference" OFF)
option(PTZ_WITH_JETSON_NVMM "Use Jetson NVMM/EGL CUDA zero-copy frames" OFF)
option(PTZ_BUILD_TESTS "Build unit tests" ON)

add_library(ptz_core
    src/algorithms.cpp
    src/config.cpp
    src/controller.cpp
    src/gallery.cpp
    src/health.cpp
    src/perception.cpp
    src/runtime.cpp
    src/tracker.cpp
)
target_include_directories(ptz_core PUBLIC include)
target_compile_options(ptz_core PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
target_link_libraries(ptz_core PUBLIC pthread)

if(PTZ_WITH_ONVIF)
  find_package(CURL REQUIRED)
  find_package(pugixml CONFIG REQUIRED)
  target_sources(ptz_core PRIVATE src/onvif.cpp)
  target_link_libraries(ptz_core PUBLIC CURL::libcurl pugixml::pugixml)
  target_compile_definitions(ptz_core PUBLIC PTZ_WITH_ONVIF=1)
endif()

if(PTZ_WITH_TENSORRT)
  find_package(CUDAToolkit REQUIRED)
  find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs dnn calib3d videoio)
  find_library(NVINFER_LIBRARY nvinfer REQUIRED)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  target_sources(ptz_core PRIVATE src/tensorrt_inference.cpp src/engine_builder.cpp src/cuda_preprocess.cu src/replay.cpp)
  target_include_directories(ptz_core PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(ptz_core PUBLIC CUDA::cudart ${NVINFER_LIBRARY} ${OpenCV_LIBS})
  target_compile_definitions(ptz_core PUBLIC PTZ_WITH_TENSORRT=1)
endif()

if(PTZ_WITH_JETSON_NVMM)
  if(NOT PTZ_WITH_GSTREAMER OR NOT PTZ_WITH_TENSORRT)
    message(FATAL_ERROR "PTZ_WITH_JETSON_NVMM requires GStreamer and TensorRT")
  endif()
  find_path(NVBUFSURFACE_INCLUDE_DIR nvbufsurface.h PATHS /usr/src/jetson_multimedia_api/include /opt/nvidia/deepstream/deepstream/sources/includes REQUIRED)
  find_library(NVBUFSURFACE_LIBRARY nvbufsurface REQUIRED)
  find_library(EGL_LIBRARY EGL REQUIRED)
  target_include_directories(ptz_core PRIVATE ${NVBUFSURFACE_INCLUDE_DIR})
  target_link_libraries(ptz_core PUBLIC ${NVBUFSURFACE_LIBRARY} ${EGL_LIBRARY} CUDA::cuda_driver)
  target_compile_definitions(ptz_core PUBLIC PTZ_WITH_JETSON_NVMM=1)
endif()

if(PTZ_WITH_GSTREAMER)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(GST REQUIRED IMPORTED_TARGET gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)
  target_sources(ptz_core PRIVATE src/gstreamer_source.cpp)
  target_link_libraries(ptz_core PUBLIC PkgConfig::GST)
  target_compile_definitions(ptz_core PUBLIC PTZ_WITH_GSTREAMER=1)
endif()

if(PTZ_WITH_GSTREAMER AND PTZ_WITH_TENSORRT)
  target_sources(ptz_core PRIVATE src/benchmark.cpp)
endif()

add_executable(ptz-control src/main.cpp)
target_link_libraries(ptz-control PRIVATE ptz_core)

if(PTZ_BUILD_TESTS)
  enable_testing()
  add_executable(ptz-tests tests/test_main.cpp)
  target_link_libraries(ptz-tests PRIVATE ptz_core)
  add_test(NAME ptz-tests COMMAND ptz-tests)
  if(PTZ_WITH_ONVIF)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_executable(ptz-onvif-test tests/test_onvif.cpp)
    target_link_libraries(ptz-onvif-test PRIVATE ptz_core)
    add_test(NAME ptz-onvif-integration COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/mock_onvif_server.py $<TARGET_FILE:ptz-onvif-test>)
  endif()
endif()

install(TARGETS ptz-control RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
