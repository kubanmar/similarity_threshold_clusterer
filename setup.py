import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="threshold_clusterer",
    version="1.0",
    author="Martin Kuban",
    author_email="kuban@physik.hu-berlin.de",
    description="A clustering method for small clusters in high dimensional descriptor spaces.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.physik.hu-berlin.de/kuban/threshold-clusterer",
    python_requires='>=3.6',
    install_requires = ['numpy'],
    packages=['threshold_clusterer']
)
