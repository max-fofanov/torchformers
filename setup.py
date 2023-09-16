from setuptools import setup, find_packages


setup(
    name="torchformers",
    version="0.1",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=["torch"],
)
