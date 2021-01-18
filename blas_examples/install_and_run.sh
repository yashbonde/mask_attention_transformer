mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/usr/local/lib/python3.9/site-packages/torch/share/cmake -DCMAKE_BUILD_TYPE=Debug ..
cmake --build . --config Release
./blas_gemm
cd ..
