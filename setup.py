#!/usr/bin/env python
"""Setup script for Germinal Center Analysis Pipeline"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements_pip.txt"
if requirements_path.exists():
    requirements = [
        line.strip() for line in requirements_path.read_text().splitlines()
        if line.strip() and not line.startswith('#') and not line.startswith('git+')
    ]
else:
    requirements = []

setup(
    name="germinal-center-analysis",
    version="0.1.0",
    author="MUSA Lab",
    description="Chromatin analysis pipeline for germinal center cell populations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GVS-Lab/germinal_center",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements + [
        "pyyaml>=5.4",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
    },
    entry_points={
        "console_scripts": [
            "gc-pipeline=cli.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["configs/*.yaml"],
    },
)

