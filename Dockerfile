ARG BASE_IMAGE=nvcr.io/nvidia/l4t-jetpack:r36.4.0
FROM ${BASE_IMAGE} AS build
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential cmake pkg-config libcurl4-openssl-dev libpugixml-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libopencv-dev && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DPTZ_WITH_ONVIF=ON -DPTZ_WITH_GSTREAMER=ON -DPTZ_WITH_TENSORRT=ON \
      -DPTZ_WITH_JETSON_NVMM=ON \
      -DPTZ_BUILD_TESTS=ON && \
    cmake --build build -j"$(nproc)" && ctest --test-dir build --output-on-failure

FROM ${BASE_IMAGE} AS runtime
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libcurl4 libpugixml1v5 gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad libopencv-dev && \
    rm -rf /var/lib/apt/lists/*
COPY --from=build /src/build/ptz-control /usr/local/bin/ptz-control
EXPOSE 8080
ENTRYPOINT ["ptz-control"]
CMD ["run", "--config", "/config/pipeline.yaml"]
