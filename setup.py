from setuptools import setup, find_packages


def readme():
    with open("README.md") as f:
        return f.read()


setup(
    name="PyInterpX",
    version="0.2.1",
    description="A highly performant, GPU compatible package for higher order interpolation in PyTorch",
    long_description=readme(),
    long_description_content_type='text/markdown',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
    keywords="Machine learning, Pytorch, Interpolation, Higher order interpolation, 3D interpolation, GPU compatible",
    author="ThomasHelfer",
    author_email="thomashelfer@live.de",
    license="MIT",  # Updated to MIT License
    packages=find_packages(exclude=["tests"]),
    install_requires=[
        "torch",
        "black",
        "pre-commit",
        "pytest",
        "numpy",
        "tqdm",
        "matplotlib",
    ],
    python_requires=">=3.8",
    include_package_data=True,
    zip_safe=False,
)
