import os

from setuptools import find_packages, setup

version_contents = {}
version_path = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "openai/version.py"
)
with open(version_path, "rt") as f:
    exec(f.read(), version_contents)

setup(
    name="openai",
    description="Python client library for the OpenAI API",
    version=version_contents["VERSION"],
    install_requires=[
        "requests>=2.20",  # to get the patch for CVE-2018-18074
        "tqdm",  # Needed for progress bars
        "pandas>=1.2.3",  # Needed for CLI fine-tuning data preparation tool
        "pandas-stubs>=1.1.0.11",  # Needed for type hints for mypy
        "openpyxl>=3.0.7",  # Needed for CLI fine-tuning data preparation tool xlsx format
        "numpy>=1.22.0",  # To address a vuln in <1.21.6
    ],
    extras_require={
        "dev": ["black~=21.6b0", "pytest==6.*"],
        "embeddings": [
            "scikit-learn>=1.0.2", # Needed for embedding utils, versions >= 1.1 require python 3.8
            "tenacity>=8.0.1"
        ],
    },
    python_requires=">=3.7.1",
    entry_points={
        "console_scripts": [
            "openai=openai._openai_scripts:main",
        ],
    },
    packages=find_packages(exclude=["tests", "tests.*"]),
    package_data={
        "openai": [
            "py.typed",
        ]
    },
    author="OpenAI",
    author_email="support@openai.com",
    url="https://github.com/openai/openai-python",
)
