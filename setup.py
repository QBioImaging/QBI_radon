from setuptools import setup


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="QBI_radon",
    version="1.6",
    description="Radon Transformation for Pytorch 2.0 package",
    author="Minh Nhat Trinh",
    license="GNU GENERAL PUBLIC LICENSE",
    packages=["QBI_radon"],
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.25.0",
    ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
)
