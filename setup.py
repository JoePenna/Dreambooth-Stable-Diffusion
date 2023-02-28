from setuptools import setup, find_packages

setup(
    name='dreambooth-stable-diffusion',
    version='1.0.0',
    description='',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],
)