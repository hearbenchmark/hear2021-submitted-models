from setuptools import find_packages
from setuptools import setup

with open("README.md") as f:
    long_description = f.read()

setup(
    name="wav2clip-hear",
    version="0.1.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Topic :: Artistic Software",
        "Topic :: Multimedia",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Multimedia :: Sound/Audio :: Editors",
        "Topic :: Software Development :: Libraries",
    ],
    description="Wav2CLIP HEAR Challenges",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Ho-Hsiang Wu",
    author_email="hohsiangwu@gmail.com",
    url="https://github.com/hohsiangwu/wav2clip-hear",
    license="MIT",
    packages=find_packages(),
    keywords=[
        "audio",
        "representation",
        "learning",
        "music",
        "sound",
        "representation learning",
        "wav2clip",
    ],
    install_requires=[
        "wav2clip",
    ],
    extras_require={
        "tests": ["pytest", "pytest-cov"],
    },
)
