import platform
from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import CppExtension, CUDAExtension, CUDA_HOME

TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

# extra_compile_args = []
# if platform.system() != 'Windows':
#     extra_compile_args += ['-Wno-unused-variable']

# if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 2):
#     extra_compile_args += ['-DVERSION_GE_1_3']

# ext_modules = [
#     CppExtension('torch_scatter.scatter_cpu', ['cpu/scatter.cpp'],
#                  extra_compile_args=extra_compile_args)
# ]
# cmdclass = {'build_ext': torch.utils.cpp_extension.BuildExtension}

# if CUDA_HOME is not None:
#     ext_modules += [
#         CUDAExtension('torch_scatter.scatter_cuda',
#                       ['cuda/scatter.cpp', 'cuda/scatter_kernel.cu'])
#     ]

__version__ = '0.0.1'
url = 'https://github.com/yashbonde/mask_attention_transformer'

install_requires = []
setup_requires = ['pytest-runner']
tests_require = ['pytest', 'pytest-cov']

setup(
    name='mask_attention',
    version=__version__,
    description='Simple PyTorch APIs for Masked Attention in Transformers',
    author='Yash Bonde',
    author_email='bonde.yash97@gmail.com',
    url=url,
    download_url='{}/archive/{}.tar.gz'.format(url, __version__),
    keywords=[
        'pytorch',
        'mask_attention',
        'transformer',
    ],
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    # ext_modules=ext_modules,
    # cmdclass=cmdclass,
    packages=find_packages(),
)
