cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(blas_gemm)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(blas_gemm cpu_blas_examples.cpp)
target_link_libraries(blas_gemm "${TORCH_LIBRARIES}")

set_property(TARGET blas_gemm PROPERTY CXX_STANDARD 14)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET blas_gemm
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:blas_gemm>)
endif (MSVC)

