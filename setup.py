import torch
from setuptools import setup, find_packages

setup(
    name="transformer",
    version="0.1",
    packages=find_packages(),
    install_requires=[torch],
)
