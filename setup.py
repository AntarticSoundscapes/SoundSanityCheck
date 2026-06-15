from setuptools import setup, find_packages

setup(
    name="soundsanity",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
    ],
    author="Antarctic Soundscape Research Group",
    description="A Python library to perform sanity checks and quality diagnostics on Antarctic soundscape field recordings using Essentia.",
    python_requires=">=3.7",
)
