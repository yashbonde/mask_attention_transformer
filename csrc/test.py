import platform
import torch

if platform.system() == "Darwin":
    # macos uses *.dylib
    torch.ops.load_library("build/libtorch_mask_attention.dylib")
    mo = torch.ops.my_ops
elif platform.system() == "Linux":
    # linux platform uses *.so as dylib
    torch.ops.load_library("build/libtorch_mask_attention.so")
    mo = torch.ops.my_ops
else:
    print("Only MacOS is supported")
    exit()

print(mo.dot_prod)

t1 = torch.randn(2, 3)
t2 = torch.randn(2, 3)

out = mo.dot_prod(t1, t2)
print("--->", out)
