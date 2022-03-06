import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='panns_hear',
    version='0.2.1',
    description='PANNs embeddings for HEAR 2021 Challenge.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/qiuqiangkong/HEAR2021_Challenge_PANNs',
    packages=setuptools.find_packages(),
    author='Yin Cao',
    author_email='yin.k.cao@gmail.com',
    license='Apache License 2.0',
    install_requires=['numpy',
                      'torchlibrosa'
                      ],
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.7',
)