from setuptools import setup, find_packages
from typing import List
def get_requirements(file_path: str) -> List[str]:
    with open(file_path, 'r') as file:
        requirements = file.read().splitlines()
    return requirements
setup(
    name="Fault Detection",
    version="0.0.1",
    author="Ujjval Dwivedi",
    author_email="ujjvaldwivedi26@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)