from setuptools import find_packages, setup
from typing import List

requirement_file_name = "requirements.txt"
REMOVE_PACKAGE = "-e ."


def get_requirements_file() -> List[str]:
    with open(requirement_file_name) as requirement_file:
        requirement_list = requirement_file.readlines()
    requirement_list = [
        requirement_name.replace("\n", "") for requirement_name in requirement_list
    ]

    if REMOVE_PACKAGE in requirement_list:
        requirement_list.remove(REMOVE_PACKAGE)
    return requirement_list


setup(
    name="Insurance",
    version="0.0.1",
    description="Insurance Industry level Project",
    author="Nilutpal Das",
    author_email="nilutpaldas992@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements_file(),
)
