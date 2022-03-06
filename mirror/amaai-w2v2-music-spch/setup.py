#!/usr/bin/env python3
from setuptools import find_packages, setup

long_description = open("README.md", "r").read()

setup(
    name="MusicSpeechWav2vec",
    description="Wav2vec model trained on both music and speech data",
    author="Kin Wai Cheuk",
    author_email="kinwai_cheuk@mymail.sutd.edu.sg",
    url="https://github.com/KinWaiCheuk/my_package",
    license="Apache-2.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Bug Tracker": "https://github.com/KinWaiCheuk/my_package/issues",
        "Source Code": "https://github.com/KinWaiCheuk/my_package",
    },
    packages=find_packages(exclude=("tests",)),
    python_requires=">=3.7",
    install_requires=[
        "librosa",
        # otherwise librosa breaks
        "numba==0.48",
        # tf 2.6.0
        "numpy==1.19.2",
        "torch==1.9.0",
        # "numba>=0.49.0", # not directly required, pinned by Snyk to avoid a vulnerability
        "scikit-learn>=0.24.2",  # not directly required, pinned by Snyk to avoid a vulnerability
        "fairseq@git+https://github.com/pytorch/fairseq"
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
