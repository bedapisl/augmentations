rm augmentations_backend.cpython-36m-x86_64-linux-gnu.so

mkdir build

cd build
cmake .. -DPYBIND11_PYTHON_VERSION=3.6 -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON
make -j $1
cd ..

cp build/augmentations_backend.cpython-36m-x86_64-linux-gnu.so .
