include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

pybind11_add_module(quantize src/quantize.cu quantize.cpp)

target_link_libraries(quantize PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARIES})    