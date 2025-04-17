from setuptools import find_packages, setup  # Package discovery and setup tools
from typing import List  # Type hints support

# Constant for editable installation marker
# -e stand for specifies that the package should be installed in editable mode.
# and . for current directory
HYPHEN_E_DOT = '-e .'


def get_requirements(file_path: str) -> List[str]:
    """
    Read and parse requirements from a text file.

    Args:
        file_path (str): Path to requirements.txt file

    Returns:
        List[str]: Clean list of requirements excluding empty lines and -e .
    """
    requirements = []
    with open(file_path) as file_obj:
        # Read all lines and remove whitespace/newline characters
        requirements = [req.strip() for req in file_obj.readlines()]

        # Remove editable installation marker if present
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


# Package configuration
setup(
    name="hatefultext",  # Package name
    version='0.0.1',  # Initial version
    author='Ved Prakash',  # Author name
    author_email='vihastvideos21@gmail.com',  # not real gmail
    packages=find_packages(),  # Automatically discover Python packages in directory
    install_requires=get_requirements('requirements.txt'),  # Runtime dependencies
)

