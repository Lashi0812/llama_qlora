include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/include 
    )
pybind11_add_module(quantize src/quantize.cu ops.cu)
# add_executable(quantize ops.cu src/quantize.cu)
set_target_properties(quantize PROPERTIES CXX_STANDARD 17 CUDA_ARCHITECTURES 89)
target_compile_options(quantize PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>)
# target_link_libraries(quantize PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARIES})


target_link_libraries(quantize PRIVATE ${TORCH_LIBRARIES} )    
# Link additional libraries (e.g., MKL, LAPACK, OpenMP)
# target_link_libraries(quantize PRIVATE
#     mkl_intel_lp64 mkl_sequential mkl_core gomp mkl_gnu_thread
# )
