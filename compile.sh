#bin/bash
export TORCH_PYTHON_CMAKE=`python3 -c "import torch;print(torch.utils.cmake_prefix_path)"`
echo ":: Found path to pytorch Cmake file -->" $TORCH_PYTHON_CMAKE

# now compile from 
cd csrc/
mkdir build; cd build;
cmake -DCMAKE_PREFIX_PATH=$TORCH_PYTHON_CMAKE .. && make -j

echo ":: Starting Test Runs (Library) ..."
cd ..
python3 test.py

echo ":: Installing Python Version"
cd ..
pip3 install -e . #mask_attention

echo ":: Starting Test Runs (Python) ..."
python3 test_attn.py # for testing

echo ":: ... Complete"
