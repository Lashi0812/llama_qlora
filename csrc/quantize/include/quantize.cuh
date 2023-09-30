#ifndef quantize_h
#define quantize_h

#include <cuda_runtime.h>
#include <cuda_bf16.h>

// void cuda_hello();
template<typename T,int ITEMS_PER_TH,int BLOCK_SIZE>
__global__ void kQuantizeNF4(const T*__restrict__ A,float * absmax,unsigned char * out,const int n);

#endif