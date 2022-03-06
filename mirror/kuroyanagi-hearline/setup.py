#!/usr/bin/env python3
from setuptools import find_packages, setup

long_description = open("README.md", "r", encoding="utf-8").read()

setup(
    name="hearline",
    version="0.0.0",
    description="HEAR 2021 -- Model",
    author="ibkuroyagi",
    author_email="ibkuroyagi@gmail.com",
    url="https://github.com/ibkuroyagi/hear2021-submit.git",
    license="Apache-2.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Bug Tracker": "https://github.com/ibkuroyagi/hear2021-submit/issues",
        "Source Code": "https://github.com/ibkuroyagi/hear2021-submit",
    },
    packages=find_packages(exclude=("tests",)),
    python_requires=">=3.6",
    install_requires=[
        "numpy==1.19.2",
        "torch",
        "torchaudio",
        "efficientnet-pytorch",
        "PyYAML",
        "hearvalidator",
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
