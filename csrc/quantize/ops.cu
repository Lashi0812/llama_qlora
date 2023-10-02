#include "include/quantize.cuh"
#include <iostream>
#include <vector>
#include <ATen/ATen.h>
#include <torch/extension.h>

// std::vector<int> Dummy(int const N) {
//     return std::vector<int>(N,N);
// }

// std::vector<at::Tensor> Dummy2(int const N) {
//     return std::vector<at::Tensor>{at::ones({1,2}),at::zeros({2,2})};
// }

std::vector<at::Tensor> quantizeNF4(const at::Tensor weights, const int blockSize)
{
    // get number of element
    auto nElems = weights.numel();

    // find the execution configuration
    unsigned int gridSize = (nElems + blockSize - 1) / blockSize;
    

    // create out and absmax tensor which we need to pass into the kernel
    at::Tensor absmax = at::zeros({
                                      gridSize,
                                  },
                                  at::TensorOptions().device(at::kCUDA, 0).dtype(at::kFloat));
    at::Tensor out = at::zeros({(nElems + 1) / 2, 1}, at::TensorOptions().device(at::kCUDA, 0).dtype(at::kByte));

    // launch the kernel
    kQuantizeNF4<__nv_bfloat16,2,64><<<gridSize, 32>>>(reinterpret_cast<__nv_bfloat16*>(weights.data_ptr()), absmax.data_ptr<float>(), out.data_ptr<unsigned char>(), nElems);
    return std::vector<at::Tensor>{absmax, out};
}

at::Tensor deQuantizeNF4(const at::Tensor quant,const at::Tensor absmax,const at::IntArrayRef outShape)
{
    at::Tensor out = at::zeros(outShape,at::TensorOptions().device(at::kCUDA,0).dtype(at::kBFloat16));
    // launch execution configuration 
    int blockSize = (quant.numel() / absmax.numel())/2;
    dim3 block(blockSize);
    dim3 grid(absmax.numel(),1);

    // launch the kernel
    kDequantizeNF4<__nv_bfloat16,2,32><<<grid,16>>>(absmax.data_ptr<float>(),quant.data_ptr<unsigned char>(),reinterpret_cast<__nv_bfloat16*>(out.data_ptr()));

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    // m.def("dummy",&Dummy,"pybind");
    // m.def("dummy2",&Dummy2,"pybind");
    // m.def("cuda_hello",&cuda_hello,"cuda bind");
    m.def("quantizeNF4", &quantizeNF4, "Quantize bf16 to NF4 format");
    m.def("dequantizeNF4", &deQuantizeNF4, "Dequantize to bf16");
}
int main()
{
    auto a = at::rand({4096,4096},at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA,0));
    auto ans = quantizeNF4(a,64);
    std::cout << ans[0] << std::endl;
    std::cout << ans[1] << std::endl;
}

