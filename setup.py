import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    required_packages = f.read().splitlines()

setuptools.setup(
    name="tpsimilarity",
    version="0.3.1",
    author="Filipi N. Silva and Sadamori Kojaku and Attila Varga",
    author_email="filsilva@iu.edu",
    description="Package to compute TP similarities between nodes in a network.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/filipinascimento/tpsimilarity",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.0',
    install_requires=required_packages
)