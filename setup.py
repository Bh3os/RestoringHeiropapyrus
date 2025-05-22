#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="hierolm",
    version="0.1.0",
    author="Sab3awy",
    author_email="your.email@example.com",
    description="A hierarchical language model for hieroglyphic data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/HieroLM",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
)
