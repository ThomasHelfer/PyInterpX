from setuptools import setup, find_packages


def readme():
    with open("README.md") as f:
        return f.read()


def read_requirements():
    with open("requirements.txt") as f:
        return f.read().splitlines()


setup(
    name="HigherOrderInterpolation3DTorch",
    version="0.1",
    description="A highly performant, GPU compatible package for higher order interpolation in PyTorch",
    long_description=readme(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Topic :: Pytorch :: Interpolation :: Machine learning",
    ],
    keywords="Machine learning, Pytorch, Interpolation, Higher order interpolation, 3D interpolation, GPU compatible",
    author="ThomasHelfer",
    author_email="thomashelfer@live.de",
    license="MIT",  # Updated to MIT License
    packages=find_packages(exclude=["tests"]),
    install_requires=read_requirements(),
    python_requires=">=3.8",
    include_package_data=True,
    zip_safe=False,
)
