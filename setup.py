#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="climb",
    version="0.1.0",
    description="Symbolic Regression over Binary Functions",
    author="Lauren Moos",
    author_email="lauren@special-circumstanc.es",
    download_url="https://github.com/REFUSR/REFUSR/tree/lucca",
    packages=find_packages(exclude=["tests*"]),
    include_package_data=True,
    zip_safe=False,
    keywords=["symbolic regression", "reinforcement learning", "AI"],
    python_requires=">=3.7",
    setup_requires=[],
    entry_points={
        'console_scripts': ['project=climb.climb:train']
    },
    classifiers=[
        "Symbolic Regression",
        "Reinforcement Learning",
        "PyTorch Lightning",
    ],
)
