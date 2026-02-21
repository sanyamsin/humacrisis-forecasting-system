from setuptools import setup, find_packages

setup(
    name="humacrisis",
    version="1.0.0",
    author="Tresor",
    description="Humanitarian Crisis Forecasting System",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=open("requirements.txt").read().splitlines(),
)
