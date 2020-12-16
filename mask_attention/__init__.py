"""
MIT License

Copyright (c) 2020 Yash Bonde

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import platform
import os.path as osp

path_to_dynamic_lib = osp.join(
    osp.dirname(osp.dirname(__file__)), # to repo level
    "csrc",
    "build",
    "libtorch_mask_attention"
)
if platform.system() == "Darwin":
    # macos uses *.dylib
    path_to_dynamic_lib = path_to_dynamic_lib + ".dylib"
    if not osp.exists(path_to_dynamic_lib):
        raise ImportError(f"dylib file not found at path: {path_to_dynamic_lib} please build from README")
    torch.ops.load_library(path_to_dynamic_lib)
    mo = torch.ops.my_ops
elif platform.system() == "Linux":
    # linux platform uses *.so as dylib
    path_to_dynamic_lib = path_to_dynamic_lib + ".so"
    if not osp.exists(path_to_dynamic_lib):
        raise ImportError(f"so file not found at path: {path_to_dynamic_lib} please build from README")
    torch.ops.load_library(path_to_dynamic_lib)
    mo = torch.ops.my_ops
else:
    print("Only MacOS is supported")
    exit()

# definations
def dot_prod(t1: torch.Tensor, t2: torch.Tensor, verbose: bool = False):
    """simple function with fully built doc string and wrapper for Cpp-object
    :param t1: tensor 1
    :param t2: tensor 2
    :param verbose: bool whether to allow C++ code to print
    """
    assert t1.size() == t2.size(), "Sizes for dot-product must match"
    return mo.dot_prod(t1, t2, verbose)

# __all__
__all__ = [
    "dot_prod",
]
