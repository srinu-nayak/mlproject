from typing import List
from setuptools import setup, find_packages

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    requirements = []

    with open(file_path) as f:
        requirements = f.readlines()
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

        requirements = [req.replace("\n", "") for req in requirements]
    return requirements




setup(
    name='mlproject',
    version='0.0.1',
    packages=find_packages(),
    include_package_data=True,
    author='srinu',
    author_email='srinunayakk7@gmail.com',
    install_requires= get_requirements('requirements.txt')
)