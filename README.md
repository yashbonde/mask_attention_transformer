# mask_attention_transformer

Simple attention APIs for masked attention in transformers.

This package is aimed at making different attention patterns used in transformers easy to use. For eg. if you want to perform attention on only the 10% of all the tokens, then removing the compute for 90% improves speed and reduces memory complexity. Consider the sparse attention used by GPT3, or the one used by BigBIRD:

<img src="https://lilianweng.github.io/lil-log/assets/images/sparse-attention.png" height=300px>
<img src="https://miro.medium.com/max/430/0*J32UHaxXZ782KGLo.png" height=200px>

Currently you need to write your own CUDA kernels or use code from [longformer](https://github.com/allenai/longformer), which does something similar. However in all the cases it is very difficult to use this and also none provides any speed up on CPUs. With this simple package we aim to solve this problem.

Currently the code should be like this:
```python
from mask_attention import attention as att

...
m = torch.Tensor([
    [[1., 1., 1., 1.],
     [1., 0., 1., 0.],
     [1., 0., 0., 1.],
     [0., 1., 0., 0.]],

    [[1., 1., 1., 1.],
     [1., 0., 1., 1.],
     [0., 0., 1., 0.],
     [1., 1., 1., 1.]]
]))
# pass the query and key vectors and attend where m == 0
w = att.mask_attn(tensor1 = q, tensor2 = k, mask_to_attend_to = m == 0, fill = -1e6)
```

Our inspiration is [`pytorch_geometric`](https://github.com/rusty1s/pytorch_scatter/tree/master) and our aim is to make it even more simpler while writing good tests.

### Installation

Before running the code ensure that you have `pytorch` installed as you will be using path from there. Though you can run directly from `libtorch` (C++ bindings of `pytorch`) our aim is to use this in python.

To get the path to libtorch cmake on your system run:
```
python3 -c "import torch;print(torch.utils.cmake_prefix_path)"
# /usr/local/lib/python3.9/site-packages/torch/share/cmake
```

For practice goto `libtorch_example/` and run the following commands:
```
mkdir build; cd build;
# where PATH_TO_LIBTORCH comes from above
cmake -DCMAKE_PREFIX_PATH=$PATH_TO_LIBTORCH ..
cmake --build . --config Release
./attn

>>>
Inputs
 0.5423  0.9763  0.7522
 0.4946  0.7281  0.4401
[ CPUFloatType{2,3} ]
-0.8206 -0.5334 -0.1927
-0.4966 -0.5969 -0.9106
[ CPUFloatType{2,3} ]
output:
-1.1108
-1.0810
[ CPUFloatType{2} ]
```

To build dylib file run the command:
```
cd csrc/
mkdir build; cd build;
cmake -DCMAKE_PREFIX_PATH=/usr/local/lib/python3.9/site-packages/torch/share/cmake .. && make -j
python3 test.py

>>>
<built-in method dot_prod of PyCapsule object at 0x1023cae40>
Inputs
-1.3643  1.5106 -2.6652
-0.4383 -0.1192  2.7443
[ CPUFloatType{2,3} ]
-0.3753 -0.5521 -2.0461
-0.0443 -0.7246 -0.3306
[ CPUFloatType{2,3} ]
---> tensor([ 5.1314, -0.8015])
```

<!-- ### Tests -->

<!-- To run the tests please install `pytest` and run in CLI `pytest`. -->
