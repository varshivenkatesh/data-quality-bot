from setuptools import setup, find_packages

setup(
    name="dqbot",
    version="0.1.4",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "pyyaml",
        "openpyxl",
        "azure-storage-blob",
        "argparse",
    ],
    entry_points={
        "console_scripts": [
            "dqbot=dqbot.cli:main",
        ],
    },
)
