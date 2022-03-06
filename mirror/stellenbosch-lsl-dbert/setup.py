from setuptools import setup

setup(name='audio_dbert',
      version='0.5',
      description='Entry for Stellenbosch LSL team to the HEAR 2021 challenge',
      url='',
      author='M Baas',
      author_email='',
      license='MIT',
      packages=['audio_dbert'],
      install_requires=[
          'omegaconf',
          'hydra-core>=1.1',
          'librosa',
          'numpy',
          'torch>=1.9',
          'fairseq'
      ],
      zip_safe=False)