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
Columns 1 to 10 0.9489  0.5948  0.2950  0.8394  0.1390  0.6728  0.8467  0.3469  0.2457  0.4685
 0.0725  0.2992  0.5331  0.3450  0.2367  0.9305  0.7477  0.8554  0.8183  0.5501
 0.6110  0.5360  0.9743  0.2603  0.2684  0.0092  0.3239  0.4426  0.0680  0.2002
 0.4944  0.1413  0.3813  0.5926  0.9954  0.0951  0.1282  0.0628  0.0917  0.0770
 0.3558  0.9315  0.4816  0.1149  0.6061  0.4210  0.5779  0.8021  0.0316  0.8200

Columns 11 to 19 0.7140  0.2207  0.2590  0.5393  0.9614  0.7566  0.5626  0.3252  0.6659
 0.6935  0.8160  0.1549  0.7256  0.6942  0.6180  0.2956  0.1381  0.0285
 0.2845  0.0835  0.8574  0.4455  0.3755  0.1914  0.6825  0.1695  0.1018
 0.2605  0.1782  0.8071  0.9277  0.8383  0.4415  0.1615  0.7248  0.2368
 0.9317  0.4312  0.5307  0.3011  0.5757  0.1056  0.6304  0.0648  0.6872
[ CPUFloatType{5,19} ]

```

<!-- ### Tests -->

<!-- To run the tests please install `pytest` and run in CLI `pytest`. -->
