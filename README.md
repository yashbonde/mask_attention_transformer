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

<!-- ### Installation -->

<!-- ### Tests -->

<!-- To run the tests please install `pytest` and run in CLI `pytest`. -->
