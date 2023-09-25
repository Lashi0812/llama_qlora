#include "include/quantize.cuh"
#include <torch/extension.h>
#include <iostream>
#include <ATen/ATen.h>
#include <vector>

std::vector<int> Dummy(int const N) {
    return std::vector<int>(N,N);
}

std::vector<at::Tensor> Dummy2(int const N) {
    return std::vector<at::Tensor>{at::ones({1,2}),at::zeros({2,2})};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m)
{
    m.def("dummy",&Dummy,"pybind");
    m.def("dummy2",&Dummy2,"pybind");
    m.def("cuda_hello",&cuda_hello,"cuda bind");
}
