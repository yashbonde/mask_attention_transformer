import os
import os.path as osp
import glob
from setuptools import setup, find_packages

__version__ = '0.0.1'
url = 'https://github.com/yashbonde/mask_attention_transformer'

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
