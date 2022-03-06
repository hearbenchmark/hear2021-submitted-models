from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os.path
import setuptools

version = "0.0.1"

here = os.path.dirname(__file__)
project_dir = os.path.abspath(here)

def read_requirements():
    requirements_txt = os.path.join(project_dir, 'requirements.txt')
    with open(requirements_txt, encoding='utf-8') as f:
        contents = f.read()

    specifiers = [s for s in contents.split('\n') if s]
    return specifiers

def read_readme():
    readme = os.path.join(project_dir, 'README.md')
    with open(readme, encoding='utf-8') as f:
        long_description = f.read()

    return long_description


setup(
    name='openl3-hear',
    version=version,
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    packages=['openl3_hear'],
    install_requires=read_requirements(),
    zip_safe=True,
)
