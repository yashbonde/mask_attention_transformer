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

There are two steps to get the code working:
1. Clone repo:
```
git clone https://github.com/yashbonde/mask_attention_transformer.git
cd csrc/
mkdir build; cd build;
cmake -DCMAKE_PREFIX_PATH=/usr/local/lib/python3.9/site-packages/torch/share/cmake .. && make -j
python3 test.py

# if everything works you should get an output like
<built-in method dot_prod of PyCapsule object at 0x107008e40>
Inputs
 0.7300 -0.9675 -1.1057
 0.1083  0.5945  0.5429
[ CPUFloatType{2,3} ]
-2.2314  0.5905 -1.0162
 0.7856  0.2872  0.2024
[ CPUFloatType{2,3} ]
---> tensor([-1.0767,  0.3657])
```

2. The above code will build the linked library. The next step is super straight forward, use `pip` to install this package:
```
pip3 install -e . #mask_attention
python3 test_attn.py # for testing
```
<!-- ### Tests -->

<!-- To run the tests please install `pytest` and run in CLI `pytest`. -->
