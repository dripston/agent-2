from setuptools import setup, find_packages

setup(
    name="weather-prediction",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",  # Replace tensorflow with torch
        "requests"
    ]
)