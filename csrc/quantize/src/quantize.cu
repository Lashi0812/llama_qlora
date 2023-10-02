#include "../include/quantize.cuh"
#include <iostream>
#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_store.cuh>
#include <cub/thread/thread_operators.cuh>

// void cuda_hello()
// {
//     std::cout << "cuda hello" << std::endl;
// }
__device__ unsigned char dQuantizeNF4(float x)
{
    // the values for this tree was generated by test_normal_map_tree
    // in the file tests/test_functional.py
    if (x > 0.03979014977812767f)
        if (x > 0.3893125355243683f)         // 1
            if (x > 0.6427869200706482f)     // 11
                if (x > 0.8614784181118011f) // 111
                    return 0b1111;
                else
                    return 0b1110;
            else if (x > 0.5016634166240692f) // 110
                return 0b1101;
            else
                return 0b1100;
        else if (x > 0.2035212516784668f) // 10
            if (x > 0.2920137718319893f)  // 101
                return 0b1011;
            else
                return 0b1010;
        else if (x > 0.1202552504837513f) // 100
            return 0b1001;
        else
            return 0b1000;
    else if (x > -0.33967943489551544f)     // 0
        if (x > -0.13791173323988914f)      // 01
            if (x > -0.045525018125772476f) // 011
                return 0b0111;
            else
                return 0b0110;
        else if (x > -0.23460740596055984f) // 010
            return 0b0101;
        else
            return 0b0100;
    else if (x > -0.6106329262256622f) // 00
        if (x > -0.4599952697753906f)  // 001
            return 0b0011;
        else
            return 0b0010;
    else if (x > -0.8480964004993439f) // 000
        return 0b0001;
    else
        return 0b0000;
}

__device__ float dDequantizeNF4(unsigned char val)
{

  // the values for this tree was generated by test_normal_map_tree
  // in the file tests/test_functional.py
  if((val & 0b1000) == 8)
    if((val & 0b0100) == 4) // 1
      if((val & 0b0010) == 2) // 11
        if((val & 0b0001) == 1) // 111
          return 1.0f; 
        else
          return 0.7229568362236023f;
      else
        if((val & 0b0001) == 1) // 110
          return 0.5626170039176941f; 
        else
          return 0.44070982933044434f; 
    else
      if((val & 0b0010) == 2) //10
        if((val & 0b0001) == 1) // 101
          return 0.33791524171829224f; 
        else
          return 0.24611230194568634f; 
      else 
        if((val & 0b0001) == 1) // 100
          return 0.16093020141124725f; 
        else
          return 0.07958029955625534f; 

  else
    if((val & 0b0100) == 4) // 0
      if((val & 0b0010) == 2) //01
        if((val & 0b0001) == 1) // 011
          return 0.0f; 
        else
          return -0.09105003625154495f; 
      else
        if((val & 0b0001) == 1) // 010
          return -0.18477343022823334f; 
        else
          return -0.28444138169288635f;
    else
      if((val & 0b0010) == 2) //00
        if((val & 0b0001) == 1) // 001
          return -0.39491748809814453f;
        else
          return -0.5250730514526367f; 
      else 
        if((val & 0b0001) == 1) // 000
          return -0.6961928009986877f; 
        else
          return -1.0f; 

}

template <typename T, int ITEMS_PER_TH, int BLOCK_SIZE>
__global__ void kQuantizeNF4(const T *__restrict__ A, float *absmax, unsigned char *out, const int n)
{
    //! ---HUGH ASSUMPTIONS---
    //! 1. A is Square Matrix : LLAMA has 4096
    //! 2. Size is Multiple of Block Size
    //! 3. Block Size is multiple of Warp Size
    //! 4. So All warp will have valid items so, im not taking care of edge cases.

    // const int n_full = gridDim.x * BLOCK_SIZE;
    const int base_idx = (blockIdx.x * BLOCK_SIZE);

    // Number of item process by thread --> this to increase the active warp
    T thread_data[ITEMS_PER_TH];
    float local_absmax = -FLT_MAX;
    unsigned char quant_vals[ITEMS_PER_TH / 2];

    typedef cub::BlockLoad<T, BLOCK_SIZE / ITEMS_PER_TH, ITEMS_PER_TH, cub::BLOCK_LOAD_WARP_TRANSPOSE, 1, 1, 890> BlockLoadT;
    typedef cub::BlockReduce<float, BLOCK_SIZE / ITEMS_PER_TH, cub::BLOCK_REDUCE_WARP_REDUCTIONS, 1, 1, 890> BlockReduce;
    typedef cub::BlockStore<unsigned char, BLOCK_SIZE / ITEMS_PER_TH, ITEMS_PER_TH / 2, cub::BLOCK_STORE_WARP_TRANSPOSE, 1, 1, 890> BlockStore;

    // Allocate shared memory
    __shared__ typename BlockLoadT::TempStorage tmp_load;
    __shared__ typename BlockReduce::TempStorage tmp_reduce;
    __shared__ typename BlockStore::TempStorage tmp_store;
    __shared__ float smem_absmax_value[1];

    //! I doesn't know why they are using this for loop
    //! I believe cub is take care internally
    //! So We don't this loop
    // for (unsigned int i = base_idx; i < n_full; i += gridDim.x * BLOCK_SIZE)
    // {
    //     local_absmax = -FLT_MAX;
    //     __syncthreads();
        BlockLoadT(tmp_load).Load(&A[base_idx], thread_data);
        // do the local thread Reduce
        #pragma unroll ITEMS_PER_TH
        for (int j = 0; j < ITEMS_PER_TH; j++)
        {
            local_absmax = fmaxf(local_absmax, fabsf(__bfloat162float(thread_data[j])));
        }
        // do block wise reduce
        //? Block reduce Store the result only in the thread'0
        //? We need to pass this value to all thread in the block
        //? this done using the shared memory
        local_absmax = BlockReduce(tmp_reduce).Reduce(local_absmax, cub::Max());
        if(threadIdx.x == 0)
            smem_absmax_value[0] = local_absmax;

        __syncthreads();
        if (threadIdx.x == 0)
        {
            absmax[blockIdx.x] = local_absmax;
        }

        else
            local_absmax = smem_absmax_value[0];
            
        __syncwarp();
        local_absmax = 1.0f / local_absmax;

        unsigned char packed_4bit = 0;
        #pragma unroll ITEMS_PER_TH / 2
        for (int j = 0; j < ITEMS_PER_TH / 2; j++)
        {
            packed_4bit |= dQuantizeNF4((__bfloat162float(thread_data[2 * j]) * local_absmax)) << 4;
            packed_4bit |= dQuantizeNF4((__bfloat162float(thread_data[2 * j + 1]) * local_absmax));
            quant_vals[j] = packed_4bit;
        }
        __syncthreads();

        BlockStore(tmp_store).Store(&out[base_idx/2], quant_vals);
    // }
}

template <typename T, int ITEMS_PER_TH,int BLOCK_SIZE>
__global__ void kDequantizeNF4(const float *__restrict__ absmax, const unsigned char *__restrict__ quant, T *A)
{
    const int base_idx = blockIdx.x * BLOCK_SIZE;

    unsigned char quant_vals[ITEMS_PER_TH];
    float local_absmax;
    T vals[ITEMS_PER_TH * 2];

    typedef cub::BlockLoad<unsigned char, BLOCK_SIZE, ITEMS_PER_TH, cub::BLOCK_LOAD_DIRECT, 1, 1, 890> BlockLoadQuant;
    typedef cub::BlockStore<T, BLOCK_SIZE, ITEMS_PER_TH * 2, cub::BLOCK_STORE_DIRECT, 1, 1, 890> BlockStoreA;

    __shared__ typename BlockLoadQuant::TempStorage temp_quant;
    __shared__ typename BlockStoreA::TempStorage temp_A;

    BlockLoadQuant(temp_quant).Load(&quant[base_idx], quant_vals);
    local_absmax = __ldg(&absmax[blockIdx.x]);

    #pragma unroll ITEMS_PER_TH
    for (int j = 0; j < ITEMS_PER_TH; j++)
    {
        vals[j * 2] = __float2bfloat16(dDequantizeNF4(quant_vals[j] >> 4) * local_absmax);
        vals[j * 2 + 1] = __float2bfloat16(dDequantizeNF4(quant_vals[j] & 0x0F) * local_absmax);
    }
    __syncthreads();

    BlockStoreA(temp_A).Store(&A[base_idx * 2], vals);
}

//  template Explicitly Instantiate
template __global__ void kQuantizeNF4<__nv_bfloat16, 2, 64>(const __nv_bfloat16 *__restrict__ A, float *absmax, unsigned char *out, const int n);
template __global__ void kDequantizeNF4<__nv_bfloat16, 2,32>(const float *__restrict__ absmax, const unsigned char *__restrict__ quant, __nv_bfloat16 *A);
