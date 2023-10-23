import setuptools

with open("README.md", 'r') as f:
    long_description = f.read()

with open("requirements.txt", 'r') as f:
    install_requires = [line.strip() for line in f.readlines()]

setuptools.setup(
    name="yotse",
    version="0.2.0",
    author="SURFQuantum",
    # author_email="",
    description="Your Optimization Tool for Scientific Experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SURFQuantum/yotse",
    project_urls={
        "Bug Tracker": "https://github.com/SURFQuantum/yotse/issues",
    },
    include_package_data=True,
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    python_requires='>=3.9,<3.13',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
