# main file where API code exists

from torch import Tensor, BoolTensor
from ____ import _c_mask_attn

def mask_attn(tensor1: Tensor, tensor2: Tensor, mask: BoolTensor, fill: float = -1e6):
    """performs matrix multiplication of tensor1 and tensor2 on indexes where mask is True
    and fills the rest of values with fill."""
    w = _c_mask_attn(tensor1, tensor2, mask, fill)
    return w
