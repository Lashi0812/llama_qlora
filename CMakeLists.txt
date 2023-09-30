cmake_minimum_required(VERSION 3.26)
project(quantize CUDA CXX)

# set(CMAKE_PREFIX_PATH /root/miniconda3/envs/nlp/lib/python3.11/site-packages/torch/share/cmake)
set(CMAKE_PREFIX_PATH /pytorch/pytorch/torch/share/cmake/)
set(TORCH_CUDA_ARCH_LIST 8.9)

find_package(Torch REQUIRED)
add_subdirectory(third-party/pybind11)


add_subdirectory(csrc/quantize)

