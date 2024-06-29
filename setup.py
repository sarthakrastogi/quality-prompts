from setuptools import setup, find_packages

with open("README.MD", "r") as f:
    readme_content = f.read()

setup(
    name="nebulousai",
    version="0.1.1",
    packages=find_packages(),
    long_description=readme_content,
    long_description_content_type="text/markdown",
    install_requires=[
        "litellm==1.34.0",
        "wikipedia==1.4.0",
        "duckduckgo_search==6.1.0",

    ]
)