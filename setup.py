from setuptools import setup, find_packages


setup(
    name="torchformers",
    version="1.0.1",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=["torch"],
    author="Max Fofanov",
    author_email="max.fofanov@gmail.com",
    description="Torchformers is a sleek and efficient Python library, providing essential and customizable transformer\
     models and components for rapid development and experimentation in Natural Language Processing, all with the\
      simplicity and power of PyTorch.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/max-fofanov/torchformers",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
