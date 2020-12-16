import torch
import mask_attention as ma

print(ma.dot_prod)
out = ma.dot_prod(torch.randn(2, 3), torch.randn(2, 3))
print("out", out)

