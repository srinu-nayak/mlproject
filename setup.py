from typing import List
from setuptools import setup, find_packages



def get_requirements(filename: str) -> List[str]:
    with open('requirements.txt') as f:
        return f.read().splitlines()


setup(
    name='mlproject',
    version='0.0.1',
    packages=find_packages(),
    include_package_data=True,
    author='srinu',
    author_email='srinunayakk7@gmail.com',
    install_requires= get_requirements('requirements.txt')
)