from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='compactlogic',
    version='0.1.0',
    author='Shengpu Wang',
    author_email='wangshen@ethz.ch',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ShengpuWang1/compact-logic',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    package_dir={'compactlogic': 'compactlogic'},
    packages=['compactlogic'],
    ext_modules=[
        CUDAExtension(
            'difflogic_cuda',
            [
                'compactlogic/cuda/difflogic.cpp',
                'compactlogic/cuda/difflogic_kernel.cu',
            ],
            extra_compile_args={'nvcc': ['-lineinfo']}
        ),

        CUDAExtension(
            'convlogic_cuda',
            [
                'compactlogic/cuda/convlogic.cpp',
                'compactlogic/cuda/convlogic_kernel.cu',
            ],
            extra_compile_args={'nvcc': ['-lineinfo']}
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
    python_requires='>=3.6',
    install_requires=[
        'torch>=1.6.0',
        'numpy',
    ],
)
