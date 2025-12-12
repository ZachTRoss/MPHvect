from setuptools import setup, find_packages

setup(
    name="MPHvect",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "scikit-learn",
        "plotly",
        "gudhi",
        "ripser",
        # optional:
        # "multipers",
        # "persim",
    ],
)
