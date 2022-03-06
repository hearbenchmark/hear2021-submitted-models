#!/usr/bin/env python3
from setuptools import find_packages, setup

long_description = open("README.md", "r").read()

setup(
    name="udons",
    description="(HEAR) 2021 -- UDONS Baseline Model",
    author="Fabian-Robert Stöter",
    author_email="mail@faroit.com",
    url="https://github.com/faroit/udons",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("tests",)),
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.8.1",
        "torchvision>=0.9.1",
        "pytorch-lightning>=1.3.8",
        "torchaudio>=0.9.0",
        "einops",
    ],
    extras_require={
        "test": [
            "pytest",
            "pytest-cov",
            "pytest-env",
        ],
        "dev": [
            "pre-commit",
            "black",  # Used in pre-commit hooks
            "pytest",
            "pytest-cov",
            "pytest-env",
        ],
    },
)