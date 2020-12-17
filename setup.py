import os
import os.path as osp
import glob
from setuptools import setup, find_packages

# import torch
# from torch.utils.cpp_extension import BuildExtension
# from torch.utils.cpp_extension import CppExtension, CUDAExtension, CUDA_HOME

# WITH_CUDA = torch.cuda.is_available() and CUDA_HOME is not None
# if os.getenv('FORCE_CUDA', '0') == '1':
#     WITH_CUDA = True
# if os.getenv('FORCE_CPU', '0') == '1':
#     WITH_CUDA = False

# BUILD_DOCS = os.getenv('BUILD_DOCS', '0') == '1'


# def get_extensions():
#     Extension = CppExtension
#     define_macros = []
#     extra_compile_args = {'cxx': []}

#     if WITH_CUDA:
#         Extension = CUDAExtension
#         define_macros += [('WITH_CUDA', None)]
#         nvcc_flags = os.getenv('NVCC_FLAGS', '')
#         nvcc_flags = [] if nvcc_flags == '' else nvcc_flags.split(' ')
#         nvcc_flags += ['-arch=sm_35', '--expt-relaxed-constexpr']
#         extra_compile_args['nvcc'] = nvcc_flags

#     extensions_dir = osp.join(osp.dirname(osp.abspath(__file__)), 'csrc')
#     main_files = glob.glob(osp.join(extensions_dir, '*.cpp'))
#     extensions = []
#     for main in main_files:
#         name = main.split(os.sep)[-1][:-4]

#         sources = [main]

#         path = osp.join(extensions_dir, 'cpu', f'{name}_cpu.cpp')
#         if osp.exists(path):
#             sources += [path]

#         path = osp.join(extensions_dir, 'cuda', f'{name}_cuda.cu')
#         if WITH_CUDA and osp.exists(path):
#             sources += [path]

#         extension = Extension(
#             'mask_attention._' + name,
#             sources,
#             include_dirs=[extensions_dir],
#             define_macros=define_macros,
#             extra_compile_args=extra_compile_args,
#         )
#         extensions += [extension]

#     print("Extensions:", extensions)
#     return extensions

__version__ = '0.0.1'
url = 'https://github.com/yashbonde/mask_attention_transformer'

# install_requires = []
# setup_requires = ['pytest-runner']
# tests_require = ['pytest', 'pytest-cov']

# get_extensions()

setup(
    name='mask_attention',
    version=__version__,
    description='Simple PyTorch APIs for Masked Attention in Transformers',
    author='Yash Bonde',
    author_email='bonde.yash97@gmail.com',
    url=url,
    keywords=[
        'pytorch',
        'mask_attention',
        'transformer',
    ],
    download_url='{}/archive/{}.tar.gz'.format(url, __version__),
    packages=find_packages(),
)
