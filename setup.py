import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytetris", # Replace with your own username
    version="0.0.1",
    author="tokheim",
    author_email="tokheim@github.com",
    description="Minimal tetris implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tokheim/pytetris",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'pygame>=2.0.0',
        'numpy>=1.13.1'],
    python_requires='>=3',
)
