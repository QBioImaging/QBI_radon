from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="qbi-radon",
    version="1.8.4",
    description="Radon Transformation for PyTorch 2",
    author="Minh Nhat Trinh",
    license_files=["LICENSE"],
    packages=["QBI_radon"],
    install_requires=[
        # Declare torch as *optional*, not mandatory.
        # PyPI will NOT auto-install it; users must install torch themselves.
        # You can document this in README.
    ],
    extras_require={
        "torch": ["torch>=2.0"],
    },
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
)