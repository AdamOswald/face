# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Simple check list from AllenNLP repo: https://github.com/allenai/allennlp/blob/main/setup.py

To create the package for pypi.

1. Run `make pre-release` (or `make pre-patch` for a patch release) then run `make fix-copies` to fix the index of the
   documentation.

   If releasing on a special branch, copy the updated README.md on the main branch for your the commit you will make
   for the post-release and run `make fix-copies` on the main branch as well.

2. Run Tests for Amazon Sagemaker. The documentation is located in `./tests/sagemaker/README.md`, otherwise @philschmid.

3. Unpin specific versions from setup.py that use a git install.

4. Checkout the release branch (v<RELEASE>-release, for example v4.19-release), and commit these changes with the
   message: "Release: <RELEASE>" and push.

5. Wait for the tests on main to be completed and be green (otherwise revert and fix bugs)

6. Add a tag in git to mark the release: "git tag v<RELEASE> -m 'Adds tag v<RELEASE> for pypi' "
   Push the tag to git: git push --tags origin v<RELEASE>-release

7. Build both the sources and the wheel. Do not change anything in setup.py between
   creating the wheel and the source distribution (obviously).

   For the wheel, run: "python setup.py bdist_wheel" in the top level directory.
   (this will build a wheel for the python version you use to build it).

   For the sources, run: "python setup.py sdist"
   You should now have a /dist directory with both .whl and .tar.gz source versions.

8. Check that everything looks correct by uploading the package to the pypi test server:

   twine upload dist/* -r pypitest
   (pypi suggest using twine as other methods upload files via plaintext.)
   You may have to specify the repository url, use the following command then:
   twine upload dist/* -r pypitest --repository-url=https://test.pypi.org/legacy/

   Check that you can install it in a virtualenv by running:
   pip install -i https://testpypi.python.org/pypi diffusers

   Check you can run the following commands:
   python -c "from diffusers import pipeline; classifier = pipeline('text-classification'); print(classifier('What a nice release'))"
   python -c "from diffusers import *"

9. Upload the final version to actual pypi:
   twine upload dist/* -r pypi

10. Copy the release notes from RELEASE.md to the tag in github once everything is looking hunky-dory.

11. Run `make post-release` (or, for a patch release, `make post-patch`). If you were on a branch for the release,
    you need to go back to main before executing this.
"""

import os
import re
from distutils.core import Command

from setuptools import find_packages, setup


# IMPORTANT:
# 1. all dependencies should be listed here with their version requirements if any
# 2. once modified, run: `make deps_table_update` to update src/diffusers/dependency_versions_table.py
_deps = [
    "Pillow<10.0",  # keep the PIL.Image.Resampling deprecation away
    "accelerate>=0.11.0",
    "black==22.8",
    "datasets",
    "filelock",
    "flake8>=3.8.3",
    "flax>=0.4.1",
    "hf-doc-builder>=0.3.0",
    "huggingface-hub>=0.10.0",
    "importlib_metadata",
    "isort>=5.5.4",
    "jax>=0.2.8,!=0.3.2,<=0.3.6",
    "jaxlib>=0.1.65,<=0.3.6",
    "modelcards>=0.1.4",
    "numpy",
    "onnxruntime",
    "pytest",
    "pytest-timeout",
    "pytest-xdist",
    "scipy",
    "regex!=2019.12.17",
    "requests",
    "tensorboard",
    "torch>=1.4",
    "torchvision",
    "transformers>=4.21.0",
]

# this is a lookup table with items like:
#
# tokenizers: "huggingface-hub==0.8.0"
# packaging: "packaging"
#
# some of the values are versioned whereas others aren't.
deps = {b: a for a, b in (re.findall(r"^(([^!=<>~]+)(?:[!=<>~].*)?$)", x)[0] for x in _deps)}

# since we save this data in src/diffusers/dependency_versions_table.py it can be easily accessed from
# anywhere. If you need to quickly access the data from this table in a shell, you can do so easily with:
#
# python -c 'import sys; from diffusers.dependency_versions_table import deps; \
# print(" ".join([ deps[x] for x in sys.argv[1:]]))' tokenizers datasets
#
# Just pass the desired package names to that script as it's shown with 2 packages above.
#
# If diffusers is not yet installed and the work is done from the cloned repo remember to add `PYTHONPATH=src` to the script above
#
# You can then feed this for example to `pip`:
#
# pip install -U $(python -c 'import sys; from diffusers.dependency_versions_table import deps; \
# print(" ".join([ deps[x] for x in sys.argv[1:]]))' tokenizers datasets)
#


def deps_list(*pkgs):
    return [deps[pkg] for pkg in pkgs]


class DepsTableUpdateCommand(Command):
    """
    A custom distutils command that updates the dependency table.
    usage: python setup.py deps_table_update
    """

    description = "build runtime dependency table"
    user_options = [
        # format: (long option, short option, description).
        ("dep-table-update", None, "updates src/diffusers/dependency_versions_table.py"),
    ]

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        entries = "\n".join([f'    "{k}": "{v}",' for k, v in deps.items()])
        content = [
            "# THIS FILE HAS BEEN AUTOGENERATED. To update:",
            "# 1. modify the `_deps` dict in setup.py",
            "# 2. run `make deps_table_update``",
            "deps = {",
            entries,
            "}",
            "",
        ]
        target = "src/diffusers/dependency_versions_table.py"
        print(f"updating {target}")
        with open(target, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(content))


extras = {}


extras = {"quality": deps_list("black", "isort", "flake8", "hf-doc-builder")}
extras["docs"] = deps_list("hf-doc-builder")
extras["training"] = deps_list("accelerate", "datasets", "tensorboard", "modelcards")
extras["test"] = deps_list(
    "accelerate",
    "datasets",
    "onnxruntime",
    "pytest",
    "pytest-timeout",
    "pytest-xdist",
    "scipy",
    "torchvision",
    "transformers"
)
extras["torch"] = deps_list("torch")

extras["flax"] = [] if os.name == "nt" else deps_list("jax", "jaxlib", "flax")
extras["dev"] = (
    extras["quality"] + extras["test"] + extras["training"] + extras["docs"] + extras["torch"] + extras["flax"]
)

install_requires = [
    deps["importlib_metadata"],
    deps["filelock"],
    deps["huggingface-hub"],
    deps["numpy"],
    deps["regex"],
    deps["requests"],
    deps["Pillow"],
]

setup(
    name="diffusers",
    version="0.5.0.dev0",  # expected format is one of x.y.z.dev0, or x.y.z.rc1 or x.y.z (no to dashes, yes to dots)
    description="Diffusers",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="deep learning",
    license="Apache",
    author="The HuggingFace team",
    author_email="patrick@huggingface.co",
    url="https://github.com/huggingface/diffusers",
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
    python_requires=">=3.7.0",
    install_requires=install_requires,
    extras_require=extras,
    entry_points={"console_scripts": ["diffusers-cli=diffusers.commands.diffusers_cli:main"]},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    cmdclass={"deps_table_update": DepsTableUpdateCommand},
)

# Release checklist
# 1. Change the version in __init__.py and setup.py.
# 2. Commit these changes with the message: "Release: Release"
# 3. Add a tag in git to mark the release: "git tag RELEASE -m 'Adds tag RELEASE for pypi' "
#    Push the tag to git: git push --tags origin main
# 4. Run the following commands in the top-level directory:
#      python setup.py bdist_wheel
#      python setup.py sdist
# 5. Upload the package to the pypi test server first:
#      twine upload dist/* -r pypitest
#      twine upload dist/* -r pypitest --repository-url=https://test.pypi.org/legacy/
# 6. Check that you can install it in a virtualenv by running:
#      pip install -i https://testpypi.python.org/pypi diffusers
#      diffusers env
#      diffusers test
# 7. Upload the final version to actual pypi:
#      twine upload dist/* -r pypi
# 8. Add release notes to the tag in github once everything is looking hunky-dory.
# 9. Update the version in __init__.py, setup.py to the new version "-dev" and push to master

# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import setuptools
import inspect
import sys
import os
import re

with open('requirements.txt') as f:
    REQUIREMENTS = f.read().splitlines()

if not hasattr(setuptools, 'find_namespace_packages') or not inspect.ismethod(setuptools.find_namespace_packages):
    print(
        f"Your setuptools version:'{setuptools.__version__}' does not support PEP 420 (find_namespace_packages). Upgrade it to version >='40.1.0' and repeat install."
    )

    sys.exit(1)

VERSION_PATH = os.path.join(os.path.dirname(__file__), "qiskit_machine_learning", "VERSION.txt")
with open(VERSION_PATH, "r") as version_file:
    VERSION = version_file.read().strip()

# Read long description from README.
README_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.md")
with open(README_PATH) as readme_file:
    README = re.sub(
        "<!--- long-description-skip-begin -->.*<!--- long-description-skip-end -->",
        "",
        readme_file.read(),
        flags=re.S | re.M,
    )

setuptools.setup(
    name='qiskit-machine-learning',
    version=VERSION,
    description='Qiskit Machine Learning: A library of quantum computing machine learning experiments',
    long_description=README,
    long_description_content_type="text/markdown",
    url='https://github.com/Qiskit/qiskit-machine-learning',
    author='Qiskit Machine Learning Development Team',
    author_email='hello@qiskit.org',
    license='Apache-2.0',
    classifiers=[
        "Environment :: Console",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering"
    ],
    keywords='qiskit sdk quantum machine learning ml',
    packages=setuptools.find_packages(include=['qiskit_machine_learning','qiskit_machine_learning.*']),
    install_requires=REQUIREMENTS,
    include_package_data=True,
    python_requires=">=3.7",
    extras_require={
        'torch': ["torch; python_version < '3.10'"],
        'sparse': ["sparse"],
    },
    zip_safe=False
)

from setuptools import setup, find_packages

setup(name='torchquantum',
      version='0.1.2',
      description='A PyTorch-centric hybrid classical-quantum dynamic '
                  'neural networks framework.',
      url='https://github.com/mit-han-lab/torchquantum',
      author='Hanrui Wang',
      author_email='hanruiwang.hw@gmail.com',
      license='MIT',
      install_requires=[
            'numpy>=1.19.2',
            'torchvision>=0.9.0.dev20210130',
            'tqdm>=4.56.0',
            'setuptools>=52.0.0',
            'torch>=1.8.0',
            'torchquantum>=0.1',
            'torchpack>=0.3.0',
            'qiskit==0.32.1',
            'matplotlib>=3.3.2',
            'pathos>=0.2.7',
            'pylatexenc>=2.10',
            # 'qiskit-nature>=0.4.4'
      ],
      extras_require = {
            'doc': [
                  'nbsphinx',
                  'recommonmark'
            ]
      },
      python_requires='>=3.5',
      include_package_data=True,
      packages=find_packages()
)
import sys
from numpy.distutils.core import Extension, setup

from mkldiscover import mkl_exists

__author__ = "Anders S. Christensen"
__copyright__ = "Copyright 2016"
__credits__ = ["Anders S. Christensen et al. (2016) https://github.com/qmlcode/qml"]
__license__ = "MIT"
__version__ = "0.4.0.12"
__maintainer__ = "Anders S. Christensen"
__email__ = "andersbiceps@gmail.com"
__status__ = "Beta"
__description__ = "Quantum Machine Learning"
__url__ = "https://github.com/qmlcode/qml"


FORTRAN = "f90"

# GNU (default)
COMPILER_FLAGS = ["-O3", "-fopenmp", "-m64", "-march=native", "-fPIC",
                    "-Wno-maybe-uninitialized", "-Wno-unused-function", "-Wno-cpp"]
LINKER_FLAGS = ["-lgomp"]
MATH_LINKER_FLAGS = ["-lblas", "-llapack"]

# UNCOMMENT TO FORCE LINKING TO MKL with GNU compilers:
if mkl_exists(verbose=True):
    LINKER_FLAGS = ["-lgomp", " -lpthread", "-lm", "-ldl"]
    MATH_LINKER_FLAGS = ["-L${MKLROOT}/lib/intel64", "-lmkl_rt"]

# For clang without OpenMP: (i.e. most Apple/mac system)
if sys.platform == "darwin" and all("gnu" not in arg for arg in sys.argv):
    COMPILER_FLAGS = ["-O3", "-m64", "-march=native", "-fPIC"]
    LINKER_FLAGS = []
    MATH_LINKER_FLAGS = ["-lblas", "-llapack"]


# Intel
if any("intelem" in arg for arg in sys.argv):
    COMPILER_FLAGS = ["-xHost", "-O3", "-axAVX", "-qopenmp"]
    LINKER_FLAGS = ["-liomp5", " -lpthread", "-lm", "-ldl"]
    MATH_LINKER_FLAGS = ["-L${MKLROOT}/lib/intel64", "-lmkl_rt"]




ext_ffchl_module = Extension(name = 'ffchl_module',
                          sources = [
                                'qml/ffchl_module.f90',
                                'qml/ffchl_scalar_kernels.f90',
                            ],
                          extra_f90_compile_args = COMPILER_FLAGS,
                          extra_f77_compile_args = COMPILER_FLAGS,
                          extra_compile_args = COMPILER_FLAGS ,
                          extra_link_args = LINKER_FLAGS + MATH_LINKER_FLAGS,
                          language = FORTRAN,
                          f2py_options=['--quiet'])

ext_ffchl_scalar_kernels = Extension(name = 'ffchl_scalar_kernels',
                          sources = ['qml/ffchl_scalar_kernels.f90'],
                          extra_f90_compile_args = COMPILER_FLAGS,
                          extra_f77_compile_args = COMPILER_FLAGS,
                          extra_compile_args = COMPILER_FLAGS,
                          extra_link_args = LINKER_FLAGS + MATH_LINKER_FLAGS,
                          language = FORTRAN,
                          f2py_options=['--quiet'])

ext_ffchl_vector_kernels = Extension(name = 'ffchl_vector_kernels',
                          sources = ['qml/ffchl_vector_kernels.f90'],
                          extra_f90_compile_args = COMPILER_FLAGS,
                          extra_f77_compile_args = COMPILER_FLAGS,
                          extra_compile_args = COMPILER_FLAGS,
                          extra_link_args = LINKER_FLAGS + MATH_LINKER_FLAGS,
                          language = FORTRAN,
                          f2py_options=['--quiet'])

ext_farad_kernels = Extension(name = 'farad_kernels',
                          sources = ['qml/farad_kernels.f90'],
                          extra_f90_compile_args = COMPILER_FLAGS,
                          extra_f77_compile_args = COMPILER_FLAGS,
                          extra_compile_args = COMPILER_FLAGS,
                          extra_link_args = LINKER_FLAGS,
                          language = FORTRAN,
                          f2py_options=['--quiet'])

ext_fcho_solve = Extension(name = 'fcho_solve',
                          sources = ['qml/fcho_solve.f90'],
                          extra_f90_compile_args = COMPILER_FLAGS,
                          extra_f77_compile_args = COMPILER_FLAGS,
                          extra_compile_args = COMPILER_FLAGS,
                          extra_link_args = MATH_LINKER_FLAGS + LINKER_FLAGS,
                          language = FORTRAN,
                          f2py_options=['--quiet'])

ext_fdistance = Extension(name = 'fdistance',
                          sources = ['qml/fdistance.f90'],
                          extra_f90_compile_args = COMPILER_FLAGS,
                          extra_f77_compile_args = COMPILER_FLAGS,
                          extra_compile_args = COMPILER_FLAGS,
                          extra_link_args = LINKER_FLAGS,
                          language = FORTRAN,
                          f2py_options=['--quiet'])

ext_fkernels = Extension(name = 'fkernels',
                          sources = ['qml/fkernels.f90'],
                          extra_f90_compile_args = COMPILER_FLAGS,
                          extra_f77_compile_args = COMPILER_FLAGS,
                          extra_compile_args = COMPILER_FLAGS,
                          extra_link_args = LINKER_FLAGS,
                          language = FORTRAN,
                          f2py_options=['--quiet'])

ext_frepresentations = Extension(name = 'frepresentations',
                          sources = ['qml/frepresentations.f90'],
                          extra_f90_compile_args = COMPILER_FLAGS,
                          extra_f77_compile_args = COMPILER_FLAGS,
                          extra_compile_args = COMPILER_FLAGS,
                          extra_link_args = MATH_LINKER_FLAGS + LINKER_FLAGS,
                          language = FORTRAN,
                          f2py_options=['--quiet'])

ext_fslatm = Extension(name = 'fslatm',
                          sources = ['qml/fslatm.f90'],
                          extra_f90_compile_args = COMPILER_FLAGS,
                          extra_f77_compile_args = COMPILER_FLAGS,
                          extra_compile_args = COMPILER_FLAGS,
                          extra_link_args = LINKER_FLAGS,
                          language = FORTRAN,
                          f2py_options=['--quiet'])

# use README.md as long description
def readme():
    with open('README.md') as f:
        return f.read()

def setup_pepytools():

    setup(

        name="qml",
        packages=['qml'],

        # metadata
        version=__version__,
        author=__author__,
        author_email=__email__,
        platforms = 'Any',
        description = __description__,
        long_description = readme(),
        keywords = ['Machine Learning', 'Quantum Chemistry'],
        classifiers = [],
        url = __url__,

        # set up package contents

        ext_package = 'qml',
        ext_modules = [
              ext_ffchl_module,
              ext_farad_kernels,
              ext_fcho_solve,
              ext_fdistance,
              ext_fkernels,
              ext_fslatm,
              ext_frepresentations,
        ],
)

if __name__ == '__main__':

    setup_pepytools()
      
try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

import re
from pathlib import Path

def version(root_path):
    """Returns the version taken from __init__.py
    Parameters
    ----------
    root_path : pathlib.Path
        path to the root of the package
    Reference
    ---------
    https://packaging.python.org/guides/single-sourcing-package-version/
    """
    version_path = root_path.joinpath('tlquantum', '__init__.py')
    with version_path.open() as f:
        version_file = f.read()
    if version_match := re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M
    ):
        return version_match[1]
    raise RuntimeError("Unable to find version string.")


def readme(root_path):
    """Returns the text content of the README.rst of the package
    Parameters
    ----------
    root_path : pathlib.Path
        path to the root of the package
    """
    with root_path.joinpath('README.rst').open(encoding='UTF-8') as f:
        return f.read()


root_path = Path(__file__).parent
README = readme(root_path)
VERSION = version(root_path)


config = {
    'name': 'tensorly-quantum',
    'packages': find_packages(exclude=['doc']),
    'description': 'Tensor-Based Quantum Machine Learning',
    'long_description': README,
    'long_description_content_type' : 'text/x-rst',
    'author': 'TensorLy-Quantum developers',
    'version': VERSION,
    'install_requires': ['numpy', 'scipy', 'nose', 'tensorly', 'tensorly-torch', 'opt-einsum'],
    'license': 'Modified BSD',
    'scripts': [],
    'classifiers': [
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3'
    ],
}

setup(**config)

# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Simple check list from AllenNLP repo: https://github.com/allenai/allennlp/blob/main/setup.py

To create the package for pypi.

1. Run `make pre-release` (or `make pre-patch` for a patch release) then run `make fix-copies` to fix the index of the
   documentation.

   If releasing on a special branch, copy the updated README.md on the main branch for your the commit you will make
   for the post-release and run `make fix-copies` on the main branch as well.

2. Run Tests for Amazon Sagemaker. The documentation is located in `./tests/sagemaker/README.md`, otherwise @philschmid.

3. Unpin specific versions from setup.py that use a git install.

4. Checkout the release branch (v<RELEASE>-release, for example v4.19-release), and commit these changes with the
   message: "Release: <RELEASE>" and push.

5. Wait for the tests on main to be completed and be green (otherwise revert and fix bugs)

6. Add a tag in git to mark the release: "git tag v<RELEASE> -m 'Adds tag v<RELEASE> for pypi' "
   Push the tag to git: git push --tags origin v<RELEASE>-release

7. Build both the sources and the wheel. Do not change anything in setup.py between
   creating the wheel and the source distribution (obviously).

   For the wheel, run: "python setup.py bdist_wheel" in the top level directory.
   (this will build a wheel for the python version you use to build it).

   For the sources, run: "python setup.py sdist"
   You should now have a /dist directory with both .whl and .tar.gz source versions.

8. Check that everything looks correct by uploading the package to the pypi test server:

   twine upload dist/* -r pypitest
   (pypi suggest using twine as other methods upload files via plaintext.)
   You may have to specify the repository url, use the following command then:
   twine upload dist/* -r pypitest --repository-url=https://test.pypi.org/legacy/

   Check that you can install it in a virtualenv by running:
   pip install -i https://testpypi.python.org/pypi diffusers

   Check you can run the following commands:
   python -c "from diffusers import pipeline; classifier = pipeline('text-classification'); print(classifier('What a nice release'))"
   python -c "from diffusers import *"

9. Upload the final version to actual pypi:
   twine upload dist/* -r pypi

10. Copy the release notes from RELEASE.md to the tag in github once everything is looking hunky-dory.

11. Run `make post-release` (or, for a patch release, `make post-patch`). If you were on a branch for the release,
    you need to go back to main before executing this.
"""

import os
import re
from distutils.core import Command

from setuptools import find_packages, setup


# IMPORTANT:
# 1. all dependencies should be listed here with their version requirements if any
# 2. once modified, run: `make deps_table_update` to update src/diffusers/dependency_versions_table.py
_deps = [
    "Pillow<10.0",  # keep the PIL.Image.Resampling deprecation away
    "accelerate>=0.11.0",
    "black==22.8",
    "datasets",
    "filelock",
    "flake8>=3.8.3",
    "flax>=0.4.1",
    "hf-doc-builder>=0.3.0",
    "huggingface-hub>=0.10.0",
    "importlib_metadata",
    "isort>=5.5.4",
    "jax>=0.2.8,!=0.3.2,<=0.3.6",
    "jaxlib>=0.1.65,<=0.3.6",
    "modelcards>=0.1.4",
    "numpy",
    "onnxruntime",
    "pytest",
    "pytest-timeout",
    "pytest-xdist",
    "scipy",
    "regex!=2019.12.17",
    "requests",
    "tensorboard",
    "torch>=1.4",
    "torchvision",
    "transformers>=4.21.0",
]

# this is a lookup table with items like:
#
# tokenizers: "huggingface-hub==0.8.0"
# packaging: "packaging"
#
# some of the values are versioned whereas others aren't.
deps = {b: a for a, b in (re.findall(r"^(([^!=<>~]+)(?:[!=<>~].*)?$)", x)[0] for x in _deps)}

# since we save this data in src/diffusers/dependency_versions_table.py it can be easily accessed from
# anywhere. If you need to quickly access the data from this table in a shell, you can do so easily with:
#
# python -c 'import sys; from diffusers.dependency_versions_table import deps; \
# print(" ".join([ deps[x] for x in sys.argv[1:]]))' tokenizers datasets
#
# Just pass the desired package names to that script as it's shown with 2 packages above.
#
# If diffusers is not yet installed and the work is done from the cloned repo remember to add `PYTHONPATH=src` to the script above
#
# You can then feed this for example to `pip`:
#
# pip install -U $(python -c 'import sys; from diffusers.dependency_versions_table import deps; \
# print(" ".join([ deps[x] for x in sys.argv[1:]]))' tokenizers datasets)
#


def deps_list(*pkgs):
    return [deps[pkg] for pkg in pkgs]


class DepsTableUpdateCommand(Command):
    """
    A custom distutils command that updates the dependency table.
    usage: python setup.py deps_table_update
    """

    description = "build runtime dependency table"
    user_options = [
        # format: (long option, short option, description).
        ("dep-table-update", None, "updates src/diffusers/dependency_versions_table.py"),
    ]

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        entries = "\n".join([f'    "{k}": "{v}",' for k, v in deps.items()])
        content = [
            "# THIS FILE HAS BEEN AUTOGENERATED. To update:",
            "# 1. modify the `_deps` dict in setup.py",
            "# 2. run `make deps_table_update``",
            "deps = {",
            entries,
            "}",
            "",
        ]
        target = "src/diffusers/dependency_versions_table.py"
        print(f"updating {target}")
        with open(target, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(content))


extras = {}


extras = {"quality": deps_list("black", "isort", "flake8", "hf-doc-builder")}
extras["docs"] = deps_list("hf-doc-builder")
extras["training"] = deps_list("accelerate", "datasets", "tensorboard", "modelcards")
extras["test"] = deps_list(
    "accelerate",
    "datasets",
    "onnxruntime",
    "pytest",
    "pytest-timeout",
    "pytest-xdist",
    "scipy",
    "torchvision",
    "transformers"
)
extras["torch"] = deps_list("torch")

extras["flax"] = [] if os.name == "nt" else deps_list("jax", "jaxlib", "flax")
extras["dev"] = (
    extras["quality"] + extras["test"] + extras["training"] + extras["docs"] + extras["torch"] + extras["flax"]
)

install_requires = [
    deps["importlib_metadata"],
    deps["filelock"],
    deps["huggingface-hub"],
    deps["numpy"],
    deps["regex"],
    deps["requests"],
    deps["Pillow"],
]

setup(
    name="diffusers",
    version="0.5.0.dev0",  # expected format is one of x.y.z.dev0, or x.y.z.rc1 or x.y.z (no to dashes, yes to dots)
    description="Diffusers",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="deep learning",
    license="Apache",
    author="The HuggingFace team",
    author_email="patrick@huggingface.co",
    url="https://github.com/huggingface/diffusers",
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
    python_requires=">=3.7.0",
    install_requires=install_requires,
    extras_require=extras,
    entry_points={"console_scripts": ["diffusers-cli=diffusers.commands.diffusers_cli:main"]},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    cmdclass={"deps_table_update": DepsTableUpdateCommand},
)

# Release checklist
# 1. Change the version in __init__.py and setup.py.
# 2. Commit these changes with the message: "Release: Release"
# 3. Add a tag in git to mark the release: "git tag RELEASE -m 'Adds tag RELEASE for pypi' "
#    Push the tag to git: git push --tags origin main
# 4. Run the following commands in the top-level directory:
#      python setup.py bdist_wheel
#      python setup.py sdist
# 5. Upload the package to the pypi test server first:
#      twine upload dist/* -r pypitest
#      twine upload dist/* -r pypitest --repository-url=https://test.pypi.org/legacy/
# 6. Check that you can install it in a virtualenv by running:
#      pip install -i https://testpypi.python.org/pypi diffusers
#      diffusers env
#      diffusers test
# 7. Upload the final version to actual pypi:
#      twine upload dist/* -r pypi
# 8. Add release notes to the tag in github once everything is looking hunky-dory.
# 9. Update the version in __init__.py, setup.py to the new version "-dev" and push to master

from setuptools import find_packages, setup

setup(
    name="luna",
    version="0.0.1",
    description="Stable diffusion in tensorflow",
    author="arfy slowy",
    author_email="slowy.arfy@gmail.com",
    platform=["any"],
    url="https://github.com/slowy07/luna",
    packages=find_packages(),
)

#!/usr/bin/env python3
""" Install packages for faceswap.py """
# pylint: disable=too-many-lines

import logging
import ctypes
import json
import locale
import platform
import operator
import os
import re
import sys
from shutil import which
from subprocess import list2cmdline, PIPE, Popen, run, STDOUT
from typing import Any, Dict, List, Optional, Set, Tuple, Type

from pkg_resources import parse_requirements, Requirement

from lib.logger import log_setup

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
_INSTALL_FAILED = False
# Revisions of tensorflow GPU and cuda/cudnn requirements. These relate specifically to the
# Tensorflow builds available from pypi
_TENSORFLOW_REQUIREMENTS = {">=2.7.0,<2.10.0": ["11.2", "8.1"]}
# Packages that are explicitly required for setup.py
_INSTALLER_REQUIREMENTS = [("pexpect>=4.8.0", "!Windows"), ("pywinpty==2.0.2", "Windows")]

# Mapping of Python packages to their conda names if different from pip or in non-default channel
_CONDA_MAPPING: Dict[str, Tuple[str, str]] = {
    # "opencv-python": ("opencv", "conda-forge"),  # Periodic issues with conda-forge opencv
    "fastcluster": ("fastcluster", "conda-forge"),
    "imageio-ffmpeg": ("imageio-ffmpeg", "conda-forge"),
    "scikit-learn": ("scikit-learn", "conda-forge"),  # Exists in Default but is dependency hell
    "tensorflow-deps": ("tensorflow-deps", "apple"),
    "libblas": ("libblas", "conda-forge")}

# Packages that should be installed first to prevent version conflicts
_PRIORITY = ["numpy"]


class Environment():
    """ The current install environment

    Parameters
    ----------
    updater: bool, Optional
        ``True`` if the script is being called by Faceswap's internal updater. ``False`` if full
        setup is running. Default: ``False``
    """
    def __init__(self, updater: bool = False) -> None:
        self.conda_required_packages: List[Tuple[str, ...]] = [("tk", )]
        self.updater = updater
        # Flag that setup is being run by installer so steps can be skipped
        self.is_installer: bool = False
        self.enable_amd: bool = False
        self.enable_apple_silicon: bool = False
        self.enable_docker: bool = False
        self.enable_cuda: bool = False
        self.required_packages: List[Tuple[str, List[Tuple[str, str]]]] = []
        self.missing_packages: List[Tuple[str, List[Tuple[str, str]]]] = []
        self.conda_missing_packages: List[Tuple[str, ...]] = []
        self.cuda_cudnn = ["", ""]

        self._process_arguments()
        self._check_permission()
        self._check_system()
        self._check_python()
        self._output_runtime_info()
        self._check_pip()
        self._upgrade_pip()
        self._set_env_vars()

        self.installed_packages = self.get_installed_packages()
        self.installed_packages.update(self.get_installed_conda_packages())

    @property
    def encoding(self) -> str:
        """ Get system encoding """
        return locale.getpreferredencoding()

    @property
    def os_version(self) -> Tuple[str, str]:
        """ Get OS Version """
        return platform.system(), platform.release()

    @property
    def py_version(self) -> Tuple[str, str]:
        """ Get Python Version """
        return platform.python_version(), platform.architecture()[0]

    @property
    def is_conda(self) -> bool:
        """ Check whether using Conda """
        return ("conda" in sys.version.lower() or
                os.path.exists(os.path.join(sys.prefix, 'conda-meta')))

    @property
    def is_admin(self) -> bool:
        """ Check whether user is admin """
        try:
            retval = os.getuid() == 0  # type: ignore
        except AttributeError:
            retval = ctypes.windll.shell32.IsUserAnAdmin() != 0  # type: ignore
        return retval

    @property
    def cuda_version(self) -> str:
        """ str: The detected globally installed Cuda Version """
        return self.cuda_cudnn[0]

    @property
    def cudnn_version(self) -> str:
        """ str: The detected globally installed cuDNN Version """
        return self.cuda_cudnn[1]

    @property
    def is_virtualenv(self) -> bool:
        """ Check whether this is a virtual environment """
        if not self.is_conda:
            return hasattr(sys, "real_prefix") or (
                hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
            )

        prefix = os.path.dirname(sys.prefix)
        return (os.path.basename(prefix) == "envs")

    def _process_arguments(self) -> None:
        """ Process any cli arguments and dummy in cli arguments if calling from updater. """
        args = list(sys.argv)
        if self.updater:
            from lib.utils import get_backend  # pylint:disable=import-outside-toplevel
            args.append(f"--{get_backend()}")

        logger.debug(args)
        for arg in args:
            if arg == "--amd":
                self.enable_amd = True
            elif arg == "--apple-silicon":
                self.enable_apple_silicon = True
            elif arg == "--installer":
                self.is_installer = True
            elif arg == "--nvidia":
                self.enable_cuda = True

    def get_required_packages(self) -> None:
        """ Load requirements list """
        if self.enable_amd:
            suffix = "amd.txt"
        elif self.enable_cuda:
            suffix = "nvidia.txt"
        elif self.enable_apple_silicon:
            suffix = "apple_silicon.txt"
        else:
            suffix = "cpu.txt"
        req_files = ["_requirements_base.txt", f"requirements_{suffix}"]
        pypath = os.path.dirname(os.path.realpath(__file__))
        requirements = []
        for req_file in req_files:
            requirements_file = os.path.join(pypath, "requirements", req_file)
            with open(requirements_file, encoding="utf8") as req:
                for package in req:
                    package = package.strip()
                    if package and (not package.startswith(("#", "-r"))):
                        requirements.append(package)

        # Add required installer packages
        for pkg, plat in _INSTALLER_REQUIREMENTS:
            if self.os_version[0] == plat or (plat[0] == "!" and self.os_version[0] != plat[1:]):
                requirements.insert(0, pkg)

        self.required_packages = [(pkg.unsafe_name, pkg.specs)
                                  for pkg in parse_requirements(requirements)
                                  if pkg.marker is None or pkg.marker.evaluate()]
        logger.debug(self.required_packages)

    def _check_permission(self) -> None:
        """ Check for Admin permissions """
        if self.updater:
            return
        if self.is_admin:
            logger.info("Running as Root/Admin")
        else:
            logger.info("Running without root/admin privileges")

    def _check_system(self) -> None:
        """ Check the system """
        if not self.updater:
            logger.info("The tool provides tips for installation and installs required python "
                        "packages")
        logger.info("Setup in %s %s", self.os_version[0], self.os_version[1])
        if not self.updater and self.os_version[0] not in [
            "Windows",
            "Linux",
            "Darwin",
        ]:
            logger.error("Your system %s is not supported!", self.os_version[0])
            sys.exit(1)
        if self.os_version[0].lower() == "darwin" and platform.machine() == "arm64":
            self.enable_apple_silicon = True

            if not self.updater and not self.is_conda:
                logger.error("Setting up Faceswap for Apple Silicon outside of a Conda "
                             "environment is unsupported")
                sys.exit(1)

    def _check_python(self) -> None:
        """ Check python and virtual environment status """
        logger.info("Installed Python: %s %s", self.py_version[0], self.py_version[1])

        if self.updater:
            return

        if not ((3, 7) <= sys.version_info < (3, 10) and self.py_version[1] == "64bit"):
            logger.error("Please run this script with Python version 3.7 to 3.9 64bit and try "
                         "again.")
            sys.exit(1)
        if self.enable_amd and sys.version_info >= (3, 9):
            logger.error("The AMD version of Faceswap cannot be installed on versions of Python "
                         "higher than 3.8")
            sys.exit(1)

    def _output_runtime_info(self) -> None:
        """ Output run time info """
        if self.is_conda:
            logger.info("Running in Conda")
        if self.is_virtualenv:
            logger.info("Running in a Virtual Environment")
        logger.info("Encoding: %s", self.encoding)

    def _check_pip(self) -> None:
        """ Check installed pip version """
        if self.updater:
            return
        try:
            import pip  # noqa pylint:disable=unused-import,import-outside-toplevel
        except ImportError:
            logger.error("Import pip failed. Please Install python3-pip and try again")
            sys.exit(1)

    def _upgrade_pip(self) -> None:
        """ Upgrade pip to latest version """
        if not self.is_conda:
            # Don't do this with Conda, as we must use Conda version of pip
            logger.info("Upgrading pip...")
            pipexe = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "-qq",
                "--upgrade",
            ]

            if not self.is_admin and not self.is_virtualenv:
                pipexe.append("--user")
            pipexe.append("pip")
            run(pipexe, check=True)
        import pip  # pylint:disable=import-outside-toplevel
        pip_version = pip.__version__
        logger.info("Installed pip: %s", pip_version)

    def get_installed_packages(self) -> Dict[str, str]:
        """ Get currently installed packages """
        installed_packages = {}
        with Popen(f"\"{sys.executable}\" -m pip freeze --local", shell=True, stdout=PIPE) as chk:
            installed = chk.communicate()[0].decode(self.encoding).splitlines()

        for pkg in installed:
            if "==" not in pkg:
                continue
            item = pkg.split("==")
            installed_packages[item[0]] = item[1]
        logger.debug(installed_packages)
        return installed_packages

    def get_installed_conda_packages(self) -> Dict[str, str]:
        """ Get currently installed conda packages """
        if not self.is_conda:
            return {}
        chk = os.popen("conda list").read()
        installed = [re.sub(" +", " ", line.strip())
                     for line in chk.splitlines() if not line.startswith("#")]
        retval = {}
        for pkg in installed:
            item = pkg.split(" ")
            retval[item[0]] = item[1]
        logger.debug(retval)
        return retval

    def update_tf_dep(self) -> None:
        """ Update Tensorflow Dependency """
        if self.is_conda or not self.enable_cuda:
            # CPU/AMD doesn't need Cuda and Conda handles Cuda and cuDNN so nothing to do here
            return

        tf_ver = None
        cudnn_inst = self.cudnn_version.split(".")
        for key, val in _TENSORFLOW_REQUIREMENTS.items():
            cuda_req = val[0]
            cudnn_req = val[1].split(".")
            if cuda_req == self.cuda_version and (cudnn_req[0] == cudnn_inst[0] and
                                                  cudnn_req[1] <= cudnn_inst[1]):
                tf_ver = key
                break
        if tf_ver:
            # Remove the version of tensorflow in requirements file and add the correct version
            # that corresponds to the installed Cuda/cuDNN versions
            self.required_packages = [pkg for pkg in self.required_packages
                                      if not pkg[0].startswith("tensorflow-gpu")]
            tf_ver = f"tensorflow-gpu{tf_ver}"

            tf_ver = f"tensorflow-gpu{tf_ver}"
            self.required_packages.append(("tensorflow-gpu",
                                           next(parse_requirements(tf_ver)).specs))
            return

        logger.warning(
            "The minimum Tensorflow requirement is 2.4 \n"
            "Tensorflow currently has no official prebuild for your CUDA, cuDNN combination.\n"
            "Either install a combination that Tensorflow supports or build and install your own "
            "tensorflow-gpu.\r\n"
            "CUDA Version: %s\r\n"
            "cuDNN Version: %s\r\n"
            "Help:\n"
            "Building Tensorflow: https://www.tensorflow.org/install/install_sources\r\n"
            "Tensorflow supported versions: "
            "https://www.tensorflow.org/install/source#tested_build_configurations",
            self.cuda_version, self.cudnn_version)

        custom_tf = input("Location of custom tensorflow-gpu wheel (leave "
                          "blank to manually install): ")
        if not custom_tf:
            return

        custom_tf = os.path.realpath(os.path.expanduser(custom_tf))
        global _INSTALL_FAILED  # pylint:disable=global-statement
        if not os.path.isfile(custom_tf):
            logger.error("%s not found", custom_tf)
            _INSTALL_FAILED = True
        elif os.path.splitext(custom_tf)[1] != ".whl":
            logger.error("%s is not a valid pip wheel", custom_tf)
            _INSTALL_FAILED = True
        elif custom_tf:
            self.required_packages.append((custom_tf, [(custom_tf, "")]))

    def set_config(self) -> None:
        """ Set the backend in the faceswap config file """
        if self.enable_amd:
            backend = "amd"
        elif self.enable_cuda:
            backend = "nvidia"
        elif self.enable_apple_silicon:
            backend = "apple_silicon"
        else:
            backend = "cpu"
        config = {"backend": backend}
        pypath = os.path.dirname(os.path.realpath(__file__))
        config_file = os.path.join(pypath, "config", ".faceswap")
        with open(config_file, "w", encoding="utf8") as cnf:
            json.dump(config, cnf)
        logger.info("Faceswap config written to: %s", config_file)

    def _set_env_vars(self) -> None:
        """ There are some foibles under Conda which need to be worked around in different
        situations.

        Linux:
        Update the LD_LIBRARY_PATH environment variable when activating a conda environment
        and revert it when deactivating.

        Windows + AMD + Python 3.8:
        Add CONDA_DLL_SEARCH_MODIFICATION_ENABLE=1 environment variable to get around a bug which
        prevents SciPy from loading in this config: https://github.com/scipy/scipy/issues/14002

        Notes
        -----
        From Tensorflow 2.7, installing Cuda Toolkit from conda-forge and tensorflow from pip
        causes tensorflow to not be able to locate shared libs and hence not use the GPU.
        We update the environment variable for all instances using Conda as it shouldn't hurt
        anything and may help avoid conflicts with globally installed Cuda
        """
        if not self.is_conda:
            return

        linux_update = self.os_version[0].lower() == "linux" and self.enable_cuda
        windows_update = (self.os_version[0].lower() == "windows" and
                          self.enable_amd and (3, 8) <= sys.version_info < (3, 9))

        if not linux_update and not windows_update:
            return

        conda_prefix = os.environ["CONDA_PREFIX"]
        activate_folder = os.path.join(conda_prefix, "etc", "conda", "activate.d")
        deactivate_folder = os.path.join(conda_prefix, "etc", "conda", "deactivate.d")
        os.makedirs(activate_folder, exist_ok=True)
        os.makedirs(deactivate_folder, exist_ok=True)

        ext = ".bat" if windows_update else ".sh"
        activate_script = os.path.join(conda_prefix, activate_folder, f"env_vars{ext}")
        deactivate_script = os.path.join(conda_prefix, deactivate_folder, f"env_vars{ext}")

        if os.path.isfile(activate_script):
            # Only create file if it does not already exist. There may be instances where people
            # have created their own scripts, but these should be few and far between and those
            # people should already know what they are doing.
            return

        if linux_update:
            conda_libs = os.path.join(conda_prefix, "lib")
            activate = ["#!/bin/sh\n\n",
                        "export OLD_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}\n",
                        f"export LD_LIBRARY_PATH='{conda_libs}':${{LD_LIBRARY_PATH}}\n"]
            deactivate = ["#!/bin/sh\n\n",
                          "export LD_LIBRARY_PATH=${OLD_LD_LIBRARY_PATH}\n",
                          "unset OLD_LD_LIBRARY_PATH\n"]
            logger.info("Cuda search path set to '%s'", conda_libs)

        if windows_update:
            activate = ["@ECHO OFF\n",
                        "set CONDA_DLL_SEARCH_MODIFICATION_ENABLE=1\n"]
            deactivate = ["@ECHO OFF\n",
                          "set CONDA_DLL_SEARCH_MODIFICATION_ENABLE=\n"]
            logger.verbose("CONDA_DLL_SEARCH_MODIFICATION_ENABLE set to 1")  # type: ignore

        with open(activate_script, "w", encoding="utf8") as afile:
            afile.writelines(activate)
        with open(deactivate_script, "w", encoding="utf8") as afile:
            afile.writelines(deactivate)


class Checks():  # pylint:disable=too-few-public-methods
    """ Pre-installation checks

    Parameters
    ----------
    environment: :class:`Environment`
        Environment class holding information about the running system
    """
    def __init__(self, environment: Environment) -> None:
        self._env:  Environment = environment
        self._tips: Tips = Tips()
    # Checks not required for installer
        if self._env.is_installer:
            return
    # Checks not required for Apple Silicon
        if self._env.enable_apple_silicon:
            return
        self._user_input()
        self._check_cuda()
        self._env.update_tf_dep()
        if self._env.os_version[0] == "Windows":
            self._tips.pip()

    def _user_input(self) -> None:
        """ Get user input for AMD/Cuda/Docker """
        self._amd_ask_enable()
        if not self._env.enable_amd:
            self._docker_ask_enable()
            self._cuda_ask_enable()
        if self._env.os_version[0] != "Linux" and (self._env.enable_docker
                                                   and self._env.enable_cuda):
            self._docker_confirm()
        if self._env.enable_docker:
            self._docker_tips()
            self._env.set_config()
            sys.exit(0)

    def _amd_ask_enable(self) -> None:
        """ Enable or disable Plaidml for AMD"""
        logger.info("AMD Support: AMD GPU support is currently limited.\r\n"
                    "Nvidia Users MUST answer 'no' to this option.")
        i = input("Enable AMD Support? [y/N] ")
        if i in ("Y", "y"):
            logger.info("AMD Support Enabled")
            self._env.enable_amd = True
        else:
            logger.info("AMD Support Disabled")
            self._env.enable_amd = False

    def _docker_ask_enable(self) -> None:
        """ Enable or disable Docker """
        i = input("Enable  Docker? [y/N] ")
        if i in ("Y", "y"):
            logger.info("Docker Enabled")
            self._env.enable_docker = True
        else:
            logger.info("Docker Disabled")
            self._env.enable_docker = False

    def _docker_confirm(self) -> None:
        """ Warn if nvidia-docker on non-Linux system """
        logger.warning("Nvidia-Docker is only supported on Linux.\r\n"
                       "Only CPU is supported in Docker for your system")
        self._docker_ask_enable()
        if self._env.enable_docker:
            logger.warning("CUDA Disabled")
            self._env.enable_cuda = False

    def _docker_tips(self) -> None:
        """ Provide tips for Docker use """
        if not self._env.enable_cuda:
            self._tips.docker_no_cuda()
        else:
            self._tips.docker_cuda()

    def _cuda_ask_enable(self) -> None:
        """ Enable or disable CUDA """
        i = input("Enable  CUDA? [Y/n] ")
        if i in ("", "Y", "y"):
            logger.info("CUDA Enabled")
            self._env.enable_cuda = True
        else:
            logger.info("CUDA Disabled")
            self._env.enable_cuda = False

    def _check_cuda(self) -> None:
        """ Check for Cuda and cuDNN Locations. """
        if not self._env.enable_cuda:
            logger.debug("Skipping Cuda checks as not enabled")
            return

        if self._env.is_conda:
            logger.info("Skipping Cuda/cuDNN checks for Conda install")
            return

        if self._env.os_version[0] in ("Linux", "Windows"):
            global _INSTALL_FAILED  # pylint:disable=global-statement
            check = CudaCheck()
            if check.cuda_version:
                self._env.cuda_cudnn[0] = check.cuda_version
                logger.info("CUDA version: %s", self._env.cuda_version)
            else:
                logger.error("CUDA not found. Install and try again.\n"
                             "Recommended version:      CUDA 10.1     cuDNN 7.6\n"
                             "CUDA: https://developer.nvidia.com/cuda-downloads\n"
                             "cuDNN: https://developer.nvidia.com/rdp/cudnn-download")
                _INSTALL_FAILED = True
                return

            if check.cudnn_version:
                self._env.cuda_cudnn[1] = ".".join(check.cudnn_version.split(".")[:2])
                logger.info("cuDNN version: %s", self._env.cudnn_version)
            else:
                logger.error("cuDNN not found. See "
                             "https://github.com/deepfakes/faceswap/blob/master/INSTALL.md#"
                             "cudnn for instructions")
                _INSTALL_FAILED = True
            return

        # If we get here we're on MacOS
        self._tips.macos()
        logger.warning("Cannot find CUDA on macOS")
        self._env.cuda_cudnn[0] = input("Manually specify CUDA version: ")


class CudaCheck():  # pylint:disable=too-few-public-methods
    """ Find the location of system installed Cuda and cuDNN on Windows and Linux. """

    def __init__(self) -> None:
        self.cuda_path: Optional[str] = None
        self.cuda_version: Optional[str] = None
        self.cudnn_version: Optional[str] = None

        self._os: str = platform.system().lower()
        self._cuda_keys: List[str] = [key
                                      for key in os.environ
                                      if key.lower().startswith("cuda_path_v")]
        self._cudnn_header_files: List[str] = ["cudnn_version.h", "cudnn.h"]
        logger.debug("cuda keys: %s, cudnn header files: %s",
                     self._cuda_keys, self._cudnn_header_files)
        if self._os in {"windows", "linux"}:
            self._cuda_check()
            self._cudnn_check()

    def _cuda_check(self) -> None:
        """ Obtain the location and version of Cuda and populate :attr:`cuda_version` and
        :attr:`cuda_path`

        Initially just calls `nvcc -V` to get the installed version of Cuda currently in use.
        If this fails, drills down to more OS specific checking methods.
        """
        with Popen("nvcc -V", shell=True, stdout=PIPE, stderr=PIPE) as chk:
            stdout, stderr = chk.communicate()
        if not stderr:
            version = re.search(r".*release (?P<cuda>\d+\.\d+)",
                                stdout.decode(locale.getpreferredencoding()))
            if version is not None:
                self.cuda_version = version.groupdict().get("cuda", None)
            locate = "where" if self._os == "windows" else "which"
            if path := os.popen(f"{locate} nvcc").read():
                path = path.split("\n")[0]  # Split multiple entries and take first found
                while True:  # Get Cuda root folder
                    path, split = os.path.split(path)
                    if split == "bin":
                        break
                self.cuda_path = path
            return

        # Failed to load nvcc, manual check
        getattr(self, f"_cuda_check_{self._os}")()
        logger.debug("Cuda Version: %s, Cuda Path: %s", self.cuda_version, self.cuda_path)

    def _cuda_check_linux(self) -> None:
        """ For Linux check the dynamic link loader for libcudart. If not found with ldconfig then
        attempt to find it in LD_LIBRARY_PATH. """
        chk = os.popen("ldconfig -p | grep -P \"libcudart.so.\\d+.\\d+\" | head -n 1").read()
        if not chk and os.environ.get("LD_LIBRARY_PATH"):
            for path in os.environ["LD_LIBRARY_PATH"].split(":"):
                chk = os.popen(f"ls {path} | grep -P -o \"libcudart.so.\\d+.\\d+\" | "
                               "head -n 1").read()
                if chk:
                    break
        if not chk:  # Cuda not found
            return

        cudavers = chk.strip().replace("libcudart.so.", "")
        self.cuda_version = cudavers[:cudavers.find(" ")]
        self.cuda_path = chk[chk.find("=>") + 3:chk.find("targets") - 1]

    def _cuda_check_windows(self) -> None:
        """ Check Windows CUDA Version and path from Environment Variables"""
        if not self._cuda_keys:  # Cuda environment variable not found
            return
        self.cuda_version = self._cuda_keys[0].lower().replace("cuda_path_v", "").replace("_", ".")
        self.cuda_path = os.environ[self._cuda_keys[0][0]]

    def _cudnn_check(self):
        """ Check Linux or Windows cuDNN Version from cudnn.h and add to :attr:`cudnn_version`. """
        cudnn_checkfiles = getattr(self, f"_get_checkfiles_{self._os}")()
        cudnn_checkfile = next((hdr for hdr in cudnn_checkfiles if os.path.isfile(hdr)), None)
        logger.debug("cudnn checkfiles: %s", cudnn_checkfile)
        if not cudnn_checkfile:
            return
        found = 0
        with open(cudnn_checkfile, "r", encoding="utf8") as ofile:
            for line in ofile:
                if line.lower().startswith("#define cudnn_major"):
                    major = line[line.rfind(" ") + 1:].strip()
                    found += 1
                elif line.lower().startswith("#define cudnn_minor"):
                    minor = line[line.rfind(" ") + 1:].strip()
                    found += 1
                elif line.lower().startswith("#define cudnn_patchlevel"):
                    patchlevel = line[line.rfind(" ") + 1:].strip()
                    found += 1
                if found == 3:
                    break
        if found != 3:  # Full version could not be determined
            return
        self.cudnn_version = ".".join([str(major), str(minor), str(patchlevel)])
        logger.debug("cudnn version: %s", self.cudnn_version)

    def _get_checkfiles_linux(self) -> List[str]:
        """ Return the the files to check for cuDNN locations for Linux by querying
        the dynamic link loader.

        Returns
        -------
        list
            List of header file locations to scan for cuDNN versions
        """
        chk = os.popen("ldconfig -p | grep -P \"libcudnn.so.\\d+\" | head -n 1").read()
        chk = chk.strip().replace("libcudnn.so.", "")
        if not chk:
            return []

        cudnn_vers = chk[0]
        header_files = [f"cudnn_v{cudnn_vers}.h"] + self._cudnn_header_files

        cudnn_path = os.path.realpath(chk[chk.find("=>") + 3:chk.find("libcudnn") - 1])
        cudnn_path = cudnn_path.replace("lib", "include")
        return [os.path.join(cudnn_path, header) for header in header_files]

    def _get_checkfiles_windows(self) -> List[str]:
        """ Return the check-file locations for Windows. Just looks inside the include folder of
        the discovered :attr:`cuda_path`

        Returns
        -------
        list
            List of header file locations to scan for cuDNN versions
        """
        # TODO A more reliable way of getting the windows location
        if not self.cuda_path:
            return []
        scandir = os.path.join(self.cuda_path, "include")
        return [os.path.join(scandir, header) for header in self._cudnn_header_files]


class Install():  # pylint:disable=too-few-public-methods
    """ Handles installation of Faceswap requirements

    Parameters
    ----------
    environment: :class:`Environment`
        Environment class holding information about the running system
    is_gui: bool, Optional
        ``True`` if the caller is the Faceswap GUI. Used to prevent output of progress bars
        which get scrambled in the GUI
     """
    def __init__(self, environment: Environment, is_gui: bool = False) -> None:
        self._operators = {"==": operator.eq,
                           ">=": operator.ge,
                           "<=": operator.le,
                           ">": operator.gt,
                           "<": operator.lt}
        self._env = environment
        self._is_gui = is_gui
        if self._env.os_version[0] == "Windows":
            self._installer: Type[Installer] = WinPTYInstaller
        else:
            self._installer = PexpectInstaller

        if not self._env.is_installer and not self._env.updater:
            self._ask_continue()
        self._env.get_required_packages()
        self._check_missing_dep()
        self._check_conda_missing_dep()
        if (self._env.updater and
                not self._env.missing_packages and not self._env.conda_missing_packages):
            logger.info("All Dependencies are up to date")
            return
        logger.info("Installing Required Python Packages. This may take some time...")
        self._install_setup_packages()
        self._install_missing_dep()
        if self._env.updater:
            return
        if not _INSTALL_FAILED:
            logger.info("All python3 dependencies are met.\r\nYou are good to go.\r\n\r\n"
                        "Enter:  'python faceswap.py -h' to see the options\r\n"
                        "        'python faceswap.py gui' to launch the GUI")
        else:
            logger.error("Some packages failed to install. This may be a temporary error which "
                         "might be fixed by re-running this script. Otherwise please install "
                         "these packages manually.")
            sys.exit(1)

    @classmethod
    def _ask_continue(cls) -> None:
        """ Ask Continue with Install """
        inp = input("Please ensure your System Dependencies are met. Continue? [y/N] ")
        if inp in ("", "N", "n"):
            logger.error("Please install system dependencies to continue")
            sys.exit(1)

    def _check_missing_dep(self) -> None:
        """ Check for missing dependencies """
        for key, specs in self._env.required_packages:

            if self._env.is_conda:  # Get Conda alias for Key
                key = _CONDA_MAPPING.get(key, (key, None))[0]

            if key not in self._env.installed_packages:
                # Add not installed packages to missing packages list
                self._env.missing_packages.append((key, specs))
                continue

            installed_vers = self._env.installed_packages.get(key, "")

            if specs and not all(self._operators[spec[0]](
                [int(s) for s in installed_vers.split(".")],
                [int(s) for s in spec[1].split(".")])
                                 for spec in specs):
                self._env.missing_packages.append((key, specs))

        for priority in reversed(_PRIORITY):
            if package := next(
                (pkg for pkg in self._env.missing_packages if pkg[0] == priority),
                None,
            ):
                idx = self._env.missing_packages.index(package)
                self._env.missing_packages.insert(0, self._env.missing_packages.pop(idx))
        logger.debug(self._env.missing_packages)

    def _check_conda_missing_dep(self) -> None:
        """ Check for conda missing dependencies """
        if not self._env.is_conda:
            return
        installed_conda_packages = self._env.get_installed_conda_packages()
        for pkg in self._env.conda_required_packages:
            key = pkg[0].split("==")[0]
            if key not in self._env.installed_packages:
                self._env.conda_missing_packages.append(pkg)
                continue
            if len(pkg[0].split("==")) > 1 and pkg[0].split("==")[
                1
            ] != installed_conda_packages.get(key):
                self._env.conda_missing_packages.append(pkg)
        logger.debug(self._env.conda_missing_packages)

    @classmethod
    def _format_package(cls, package: str, version: List[Tuple[str, str]]) -> str:
        """ Format a parsed requirement package and version string to a format that can be used by
        the installer.

        Parameters
        ----------
        package: str
            The package name
        version: list
            The parsed requirement version strings

        Returns
        -------
        str
            The formatted full package and version string
        """
        return f"{package}{','.join(''.join(spec) for spec in version)}"

    def _install_setup_packages(self) -> None:
        """ Install any packages that are required for the setup.py installer to work. This
        includes the pexpect package if it is not already installed.

        Subprocess is used as we do not currently have pexpect
        """
        pkgs = [pkg[0] for pkg in _INSTALLER_REQUIREMENTS]
        setup_packages = [(pkg.unsafe_name, pkg.specs) for pkg in parse_requirements(pkgs)]

        for pkg in setup_packages:
            if pkg not in self._env.missing_packages:
                continue
            self._env.missing_packages.pop(self._env.missing_packages.index(pkg))
            pkg_str = self._format_package(*pkg)
            if self._env.is_conda:
                cmd = ["conda", "install", "-y"]
                if any(char in pkg_str for char in (" ", "<", ">", "*", "|")):
                    pkg_str = f"\"{pkg_str}\""
            else:
                cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir"]
                if self._env.is_admin:
                    cmd.append("--user")
            cmd.append(pkg_str)

            clean_pkg = pkg_str.replace("\"", "")
            installer = SubProcInstaller(self._env, clean_pkg, cmd, self._is_gui)
            if installer() != 0:
                logger.error("Unable to install package: %s. Process aborted", clean_pkg)
                sys.exit(1)

    def _install_missing_dep(self) -> None:
        """ Install missing dependencies """
        # Install conda packages first
        if self._env.conda_missing_packages:
            self._install_conda_packages()
        if self._env.missing_packages:
            self._install_python_packages()

    def _install_python_packages(self) -> None:
        """ Install required pip packages """
        conda_only = False
        for pkg, version in self._env.missing_packages:
            if self._env.is_conda:
                mapping = _CONDA_MAPPING.get(pkg, (pkg, ""))
                channel = None if mapping[1] == "" else mapping[1]
                pkg = mapping[0]
            pkg = self._format_package(pkg, version) if version else pkg
            if self._env.is_conda:
                if pkg.startswith("tensorflow-gpu"):
                    # From TF 2.4 onwards, Anaconda Tensorflow becomes a mess. The version of 2.5
                    # installed by Anaconda is compiled against an incorrect numpy version which
                    # breaks Tensorflow. Coupled with this the versions of cudatoolkit and cudnn
                    # available in the default Anaconda channel are not compatible with the
                    # official PyPi versions of Tensorflow. With this in mind we will pull in the
                    # required Cuda/cuDNN from conda-forge, and install Tensorflow with pip
                    # TODO Revert to Conda if they get their act together

                    # Rewrite tensorflow requirement to versions from highest available cuda/cudnn
                    highest_cuda = sorted(_TENSORFLOW_REQUIREMENTS.values())[-1]
                    compat_tf = next(k for k, v in _TENSORFLOW_REQUIREMENTS.items()
                                     if v == highest_cuda)
                    pkg = f"tensorflow-gpu{compat_tf}"
                    conda_only = True

                if self._from_conda(pkg, channel=channel, conda_only=conda_only):
                    continue
            self._from_pip(pkg)

    def _install_conda_packages(self) -> None:
        """ Install required conda packages """
        logger.info("Installing Required Conda Packages. This may take some time...")
        for pkg in self._env.conda_missing_packages:
            channel = None if len(pkg) != 2 else pkg[1]
            self._from_conda(pkg[0], channel=channel, conda_only=True)

    def _from_conda(self,
                    package: str,
                    channel: Optional[str] = None,
                    conda_only: bool = False) -> bool:
        """ Install a conda package

        Parameters
        ----------
        package: str
            The full formatted package, with version, to be installed
        channel: str, optional
            The Conda channel to install from. Select ``None`` for default channel.
            Default: ``None``
        conda_only: bool, optional
            ``True`` if the package is only available in Conda. Default: ``False``

        Returns
        -------
        bool
            ``True`` if the package was succesfully installed otherwise ``False``
        """
        #  Packages with special characters need to be enclosed in double quotes
        success = True
        condaexe = ["conda", "install", "-y"]
        if channel:
            condaexe.extend(["-c", channel])

        if package.startswith("tensorflow-gpu"):
            # Here we will install the cuda/cudnn toolkits, currently only available from
            # conda-forge, but fail tensorflow itself so that it can be handled by pip.
            specs = Requirement.parse(package).specs
            for key, val in _TENSORFLOW_REQUIREMENTS.items():
                req_specs = Requirement.parse(f"foobar{key}").specs
                if all(item in req_specs for item in specs):
                    cuda, cudnn = val
                    break
            condaexe.extend(["-c", "conda-forge", f"cudatoolkit={cuda}", f"cudnn={cudnn}"])
            package = "Cuda Toolkit"
            success = False

        if package != "Cuda Toolkit":
            if any(char in package for char in (" ", "<", ">", "*", "|")):
                package = f"\"{package}\""
            condaexe.append(package)

        clean_pkg = package.replace("\"", "")
        installer = self._installer(self._env, clean_pkg, condaexe, self._is_gui)
        retcode = installer()

        if retcode != 0 and not conda_only:
            logger.info("%s not available in Conda. Installing with pip", package)
        elif retcode != 0:
            logger.warning("Couldn't install %s with Conda. Please install this package "
                           "manually", package)
        success = retcode == 0 and success
        return success

    def _from_pip(self, package: str) -> None:
        """ Install a pip package

        Parameters
        ----------
        package: str
            The full formatted package, with version, to be installed
        """
        pipexe = [sys.executable, "-u", "-m", "pip", "install", "--no-cache-dir"]
        # install as user to solve perm restriction
        if not self._env.is_admin and not self._env.is_virtualenv:
            pipexe.append("--user")
        pipexe.append(package)

        installer = self._installer(self._env, package, pipexe, self._is_gui)
        if installer() != 0:
            logger.warning("Couldn't install %s with pip. Please install this package manually",
                           package)
            global _INSTALL_FAILED  # pylint:disable=global-statement
            _INSTALL_FAILED = True


class Installer():
    """ Parent class for package installers.

    PyWinPty is used for Windows, Pexpect is used for Linux, as these can provide us with realtime
    output.

    Subprocess is used as a fallback if any of the above fail, but this caches output, so it can
    look like the process has hung to the end user

    Parameters
    ----------
    environment: :class:`Environment`
        Environment class holding information about the running system
    package: str
        The package name that is being installed
    command: list
        The command to run
    is_gui: bool
        ``True if the process is being called from the Faceswap GUI
    """
    def __init__(self,
                 environment: Environment,
                 package: str,
                 command: List[str],
                 is_gui: bool) -> None:
        logger.info("Installing %s", package)
        logger.debug("argv: %s", command)
        self._env = environment
        self._package = package
        self._command = command
        self._is_gui = is_gui
        self._last_line_cr = False
        self._seen_lines: Set[str] = set()

    def __call__(self) -> int:
        """ Call the subclassed call function

        Returns
        -------
        int
            The return code of the package install process
        """
        try:
            returncode = self.call()
        except Exception as err:  # pylint:disable=broad-except
            logger.debug("Failed to install with %s. Falling back to subprocess. Error: %s",
                         self.__class__.__name__, str(err))
            returncode = SubProcInstaller(self._env, self._package, self._command, self._is_gui)()

        logger.debug("Package: %s, returncode: %s", self._package, returncode)
        return returncode

    def call(self) -> int:
        """ Override for package installer specific logic.

        Returns
        -------
        int
            The return code of the package install process
        """
        raise NotImplementedError()

    def _non_gui_print(self, text: str, end: Optional[str] = None) -> None:
        """ Print output to console if not running in the GUI

        Parameters
        ----------
        text: str
            The text to print
        end: str, optional
            The line ending to use. Default: ``None`` (new line)
        """
        if self._is_gui:
            return
        print(text, end=end)

    def _seen_line_log(self, text: str) -> None:
        """ Output gets spammed to the log file when conda is waiting/processing. Only log each
        unique line once.

        Parameters
        ----------
        text: str
            The text to log
        """
        if text in self._seen_lines:
            return
        logger.verbose(text)  # type:ignore
        self._seen_lines.add(text)


class PexpectInstaller(Installer):  # pylint: disable=too-few-public-methods
    """ Package installer for Linux/macOS using Pexpect

    Uses Pexpect for installing packages allowing access to realtime feedback

    Parameters
    ----------
    environment: :class:`Environment`
        Environment class holding information about the running system
    package: str
        The package name that is being installed
    command: list
        The command to run
    is_gui: bool
        ``True if the process is being called from the Faceswap GUI
    """
    def call(self) -> int:
        """ Install a package using the Pexpect module

        Returns
        -------
        int
            The return code of the package install process
        """
        import pexpect  # pylint:disable=import-outside-toplevel,import-error
        proc = pexpect.spawn(" ".join(self._command),
                             encoding=self._env.encoding, codec_errors="replace", timeout=None)
        while True:
            try:
                idx = proc.expect(["\r\n", "\r"])
                if line := proc.before.rstrip():
                    if idx == 0:
                        if self._last_line_cr:
                            self._last_line_cr = False
                            # Output last line of progress bar and go to next line
                            self._non_gui_print(line)
                        self._seen_line_log(line)
                    elif idx == 1:
                        self._last_line_cr = True
                        logger.debug(line)
                        self._non_gui_print(line, end="\r")
            except pexpect.EOF:
                break
        proc.close()
        return proc.exitstatus


class WinPTYInstaller(Installer):  # pylint: disable=too-few-public-methods
    """ Package installer for Windows using WinPTY

    Spawns a pseudo PTY for installing packages allowing access to realtime feedback

    Parameters
    ----------
    environment: :class:`Environment`
        Environment class holding information about the running system
    package: str
        The package name that is being installed
    command: list
        The command to run
    is_gui: bool
        ``True if the process is being called from the Faceswap GUI
    """
    def __init__(self,
                 environment: Environment,
                 package: str,
                 command: List[str],
                 is_gui: bool) -> None:
        super().__init__(environment, package, command, is_gui)
        self._cmd = which(command[0], path=os.environ.get('PATH', os.defpath))
        self._cmdline = list2cmdline(command)
        logger.debug("cmd: '%s', cmdline: '%s'", self._cmd, self._cmdline)

        self._pbar = re.compile(r"(?:eta\s[\d\W]+)|(?:\s+\|\s+\d+%)\Z")
        self._eof = False
        self._read_bytes = 1024

        self._lines: List[str] = []
        self._out = ""

    def _read_from_pty(self, proc: Any, winpty_error: Any) -> None:
        """ Read :attr:`_num_bytes` from WinPTY. If there is an error reading, recursively halve
        the number of bytes read until we get a succesful read. If we get down to 1 byte without a
        succesful read, assume we are at EOF.

        Parameters
        ----------
        proc: :class:`winpty.PTY`
            The WinPTY process
        winpty_error: :class:`winpty.WinptyError`
            The winpty error exception. Passed in as WinPTY is not in global scope
        """
        try:
            from_pty = proc.read(self._read_bytes)
        except winpty_error:
            # TODO Reinsert this check
            # The error message "pipe has been ended" is language specific so this check
            # fails on non english systems. For now we just swallow all errors until no
            # bytes are left to read and then check the return code
            # if any(val in str(err) for val in ["EOF", "pipe has been ended"]):
            #    # Get remaining bytes. On a comms error, the buffer remains unread so keep
            #    # halving buffer amount until down to 1 when we know we have everything
            #     if self._read_bytes == 1:
            #         self._eof = True
            #     from_pty = ""
            #     self._read_bytes //= 2
            # else:
            #     raise

            # Get remaining bytes. On a comms error, the buffer remains unread so keep
            # halving buffer amount until down to 1 when we know we have everything
            if self._read_bytes == 1:
                self._eof = True
            from_pty = ""
            self._read_bytes //= 2

        self._out += from_pty

    def _out_to_lines(self) -> None:
        """ Process the winpty output into separate lines. Roll over any semi-consumed lines to the
        next proc call. """
        if "\n" not in self._out:
            return

        self._lines.extend(self._out.split("\n"))

        if self._out.endswith("\n") or self._eof:  # Ends on newline or is EOF
            self._out = ""
        else:  # roll over semi-consumed line to next read
            self._out = self._lines[-1]
            self._lines = self._lines[:-1]

    def _parse_lines(self) -> None:
        """ Process the latest batch of lines that have been received from winPTY. """
        for line in self._lines:  # Dump the output to log
            line = line.rstrip()
            is_cr = bool(self._pbar.search(line))
            if line and not is_cr:
                if self._last_line_cr:
                    self._last_line_cr = False
                    if not self._env.is_installer:
                        # Go to next line
                        self._non_gui_print("")
                self._seen_line_log(line)
            elif line:
                self._last_line_cr = True
                logger.debug(line)
                # NSIS only updates on line endings, so force new line for installer
                self._non_gui_print(line, end=None if self._env.is_installer else "\r")
        self._lines = []

    def call(self) -> int:
        """ Install a package using the PyWinPTY module

        Returns
        -------
        int
            The return code of the package install process
        """
        import winpty  # pylint:disable=import-outside-toplevel,import-error
        # For some reason with WinPTY we need to pass in the full command. Probably a bug
        proc = winpty.PTY(
            80 if self._env.is_installer else 100,
            24,
            backend=winpty.enums.Backend.WinPTY,  # ConPTY hangs and has lots of Ansi Escapes
            agent_config=winpty.enums.AgentConfig.WINPTY_FLAG_PLAIN_OUTPUT)  # Strip all Ansi

        if not proc.spawn(self._cmd, cmdline=self._cmdline):
            del proc
            raise RuntimeError("Failed to spawn winpty")

        while True:
            self._read_from_pty(proc, winpty.WinptyError)
            self._out_to_lines()
            self._parse_lines()

            if self._eof:
                returncode = proc.get_exitstatus()
                break

        del proc
        return returncode


class SubProcInstaller(Installer):
    """ The fallback package installer if either of the OS specific installers fail.

    Uses the python Subprocess module to install packages. Feedback does not return in realtime
    so the process can look like it has hung to the end user

    Parameters
    ----------
    environment: :class:`Environment`
        Environment class holding information about the running system
    package: str
        The package name that is being installed
    command: list
        The command to run
    is_gui: bool
        ``True if the process is being called from the Faceswap GUI
    """
    def __init__(self,
                 environment: Environment,
                 package: str,
                 command: List[str],
                 is_gui: bool) -> None:
        super().__init__(environment, package, command, is_gui)
        self._shell = self._env.os_version[0] == "Windows" and command[0] == "conda"

    def __call__(self) -> int:
        """ Override default call function so we don't recursively call ourselves on failure. """
        returncode = self.call()
        logger.debug("Package: %s, returncode: %s", self._package, returncode)
        return returncode

    def call(self) -> int:
        """ Install a package using the Subprocess module

        Returns
        -------
        int
            The return code of the package install process
        """
        with Popen(self._command,
                   bufsize=0, stdout=PIPE, stderr=STDOUT, shell=self._shell) as proc:
            while True:
                if proc.stdout is not None:
                    line = proc.stdout.readline().decode(self._env.encoding, errors="replace")
                returncode = proc.poll()
                if line == "" and returncode is not None:
                    break

                is_cr = line.startswith("\r")
                line = line.rstrip()

                if line and not is_cr:
                    if self._last_line_cr:
                        self._last_line_cr = False
                        # Go to next line
                        self._non_gui_print("")
                    self._seen_line_log(line)
                elif line:
                    self._last_line_cr = True
                    logger.debug(line)
                    self._non_gui_print("", end="\r")
        return returncode


class Tips():
    """ Display installation Tips """
    @classmethod
    def docker_no_cuda(cls) -> None:
        """ Output Tips for Docker without Cuda """
        path = os.path.dirname(os.path.realpath(__file__))
        logger.info(
            "1. Install Docker\n"
            "https://www.docker.com/community-edition\n\n"
            "2. Build Docker Image For Faceswap\n"
            "docker build -t deepfakes-cpu -f Dockerfile.cpu .\n\n"
            "3. Mount faceswap volume and Run it\n"
            "# without GUI\n"
            "docker run -tid -p 8888:8888 \\ \n"
            "\t--hostname deepfakes-cpu --name deepfakes-cpu \\ \n"
            "\t-v %s:/srv \\ \n"
            "\tdeepfakes-cpu\n\n"
            "# with gui. tools.py gui working.\n"
            "## enable local access to X11 server\n"
            "xhost +local:\n"
            "## create container\n"
            "nvidia-docker run -tid -p 8888:8888 \\ \n"
            "\t--hostname deepfakes-cpu --name deepfakes-cpu \\ \n"
            "\t-v %s:/srv \\ \n"
            "\t-v /tmp/.X11-unix:/tmp/.X11-unix \\ \n"
            "\t-e DISPLAY=unix$DISPLAY \\ \n"
            "\t-e AUDIO_GID=`getent group audio | cut -d: -f3` \\ \n"
            "\t-e VIDEO_GID=`getent group video | cut -d: -f3` \\ \n"
            "\t-e GID=`id -g` \\ \n"
            "\t-e UID=`id -u` \\ \n"
            "\tdeepfakes-cpu \n\n"
            "4. Open a new terminal to run faceswap.py in /srv\n"
            "docker exec -it deepfakes-cpu bash", path, path)
        logger.info("That's all you need to do with a docker. Have fun.")

    @classmethod
    def docker_cuda(cls) -> None:
        """ Output Tips for Docker with Cuda"""
        path = os.path.dirname(os.path.realpath(__file__))
        logger.info(
            "1. Install Docker\n"
            "https://www.docker.com/community-edition\n\n"
            "2. Install latest CUDA\n"
            "CUDA: https://developer.nvidia.com/cuda-downloads\n\n"
            "3. Install Nvidia-Docker & Restart Docker Service\n"
            "https://github.com/NVIDIA/nvidia-docker\n\n"
            "4. Build Docker Image For Faceswap\n"
            "docker build -t deepfakes-gpu -f Dockerfile.gpu .\n\n"
            "5. Mount faceswap volume and Run it\n"
            "# without gui \n"
            "docker run -tid -p 8888:8888 \\ \n"
            "\t--hostname deepfakes-gpu --name deepfakes-gpu \\ \n"
            "\t-v %s:/srv \\ \n"
            "\tdeepfakes-gpu\n\n"
            "# with gui.\n"
            "## enable local access to X11 server\n"
            "xhost +local:\n"
            "## enable nvidia device if working under bumblebee\n"
            "echo ON > /proc/acpi/bbswitch\n"
            "## create container\n"
            "nvidia-docker run -tid -p 8888:8888 \\ \n"
            "\t--hostname deepfakes-gpu --name deepfakes-gpu \\ \n"
            "\t-v %s:/srv \\ \n"
            "\t-v /tmp/.X11-unix:/tmp/.X11-unix \\ \n"
            "\t-e DISPLAY=unix$DISPLAY \\ \n"
            "\t-e AUDIO_GID=`getent group audio | cut -d: -f3` \\ \n"
            "\t-e VIDEO_GID=`getent group video | cut -d: -f3` \\ \n"
            "\t-e GID=`id -g` \\ \n"
            "\t-e UID=`id -u` \\ \n"
            "\tdeepfakes-gpu\n\n"
            "6. Open a new terminal to interact with the project\n"
            "docker exec deepfakes-gpu python /srv/faceswap.py gui\n",
            path, path)

    @classmethod
    def macos(cls) -> None:
        """ Output Tips for macOS"""
        logger.info(
            "setup.py does not directly support macOS. The following tips should help:\n\n"
            "1. Install system dependencies:\n"
            "XCode from the Apple Store\n"
            "XQuartz: https://www.xquartz.org/\n\n"

            "2a. It is recommended to use Anaconda for your Python Virtual Environment as this\n"
            "will handle the installation of CUDA and cuDNN for you:\n"
            "https://www.anaconda.com/distribution/\n\n"

            "2b. If you do not want to use Anaconda you will need to manually install CUDA and "
            "cuDNN:\n"
            "CUDA: https://developer.nvidia.com/cuda-downloads"
            "cuDNN: https://developer.nvidia.com/rdp/cudnn-download\n\n")

    @classmethod
    def pip(cls) -> None:
        """ Pip Tips """
        logger.info("1. Install PIP requirements\n"
                    "You may want to execute `chcp 65001` in cmd line\n"
                    "to fix Unicode issues on Windows when installing dependencies")


if __name__ == "__main__":
    logfile = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), "faceswap_setup.log")
    log_setup("INFO", logfile, "setup")
    logger.debug("Setup called with args: %s", sys.argv)
    ENV = Environment()
    Checks(ENV)
    ENV.set_config()
    if _INSTALL_FAILED:
        sys.exit(1)
    Install(ENV)
   
from distutils.core import setup

setup(
    name='twitterbot',
    version='0.1.0',
    author='thricedotted',
    author_email='thricedotted@gmail.com',
    packages=['twitterbot'],
    description='A simple Python framework for creating Twitter bots.',
    install_requires=[
        "tweepy >= 2.3"
    ],
)

#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# pylint: disable=invalid-name, exec-used

import os
import shutil

from setuptools import find_packages, setup
from setuptools.dist import Distribution

# flake8: noqa

CURRENT_DIR = os.path.dirname(__file__)
libinfo_py = os.path.join(CURRENT_DIR, "aitemplate", "_libinfo.py")
libinfo = {}
with open(libinfo_py, "r") as f:
    exec(f.read(), libinfo)
__version__ = libinfo["__version__"]


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return False

    def is_pure(self):
        return True


# temp copy 3rdparty libs to build dir
try:
    shutil.copytree("../3rdparty", "./aitemplate/3rdparty")
except FileExistsError:
    pass
try:
    shutil.copytree("../static", "./aitemplate/static")
except FileExistsError:
    pass
try:
    shutil.copytree("../licenses", "./aitemplate/licenses")
except FileExistsError:
    pass


def gen_file_list(srcs, f_cond):
    file_list = []
    for src in srcs:
        for root, _, files in os.walk(src):
            value = []
            for file in files:
                if f_cond(file):
                    path = os.path.join(root, file)
                    value.append(path.replace("aitemplate/", ""))
            file_list.extend(value)
    return file_list


def gen_cutlass_list():
    srcs = [
        "aitemplate/3rdparty/cutlass/include",
        "aitemplate/3rdparty/cutlass/examples",
        "aitemplate/3rdparty/cutlass/tools/util/include",
    ]
    f_cond = lambda x: bool(x.endswith(".h") or x.endswith(".cuh"))
    return gen_file_list(srcs, f_cond)


def gen_cutlass_lib_list():
    srcs = ["aitemplate/3rdparty/cutlass/tools/library/scripts"]
    f_cond = lambda x: True
    return gen_file_list(srcs, f_cond)


def gen_cub_list():
    srcs = ["aitemplate/3rdparty/cub/cub"]
    f_cond = lambda x: bool(x.endswith(".h") or x.endswith(".cuh"))
    return gen_file_list(srcs, f_cond)


def gen_ck_list():
    srcs = [
        "aitemplate/3rdparty/composable_kernel/include",
        "aitemplate/3rdparty/composable_kernel/library/include/ck/library/utility",
    ]
    f_cond = lambda x: bool(x.endswith(".h") or x.endswith(".hpp"))
    return gen_file_list(srcs, f_cond)


def gen_flash_attention_list():
    srcs = [
        "aitemplate/backend/cuda/attention/src",
        "aitemplate/backend/cuda/attention/src/fmha",
    ]
    f_cond = lambda x: bool(x.endswith(".h") or x.endswith(".cuh"))
    return gen_file_list(srcs, f_cond)


def gen_static_list():
    srcs = [
        "aitemplate/static",
    ]
    f_cond = lambda x: bool(x.endswith(".h") or x.endswith(".cpp"))
    return gen_file_list(srcs, f_cond)


def gen_utils_file_list():
    srcs = ["aitemplate/utils"]
    f_cond = lambda x: bool(x.endswith(".py"))
    return gen_file_list(srcs, f_cond)


def gen_backend_common_file_list():
    srcs = ["aitemplate/backend/common"]
    f_cond = lambda x: bool(x.endswith(".py") or x.endswith(".cuh"))
    return gen_file_list(srcs, f_cond)


def gen_license_file_list():
    srcs = ["aitemplate/licenses"]
    f_cond = lambda x: True
    return gen_file_list(srcs, f_cond)


setup_kwargs = {}
include_libs = True
wheel_include_libs = True


setup(
    name="aitemplate",
    version=__version__,
    description="AITemplate: Make Templates Great for AI",
    zip_safe=True,
    install_requires=["jinja2", "numpy"],
    packages=find_packages(),
    package_data={
        "aitemplate": [
            "backend/cuda/elementwise/custom_math.cuh",
            "backend/cuda/layernorm_sigmoid_mul/layernorm_sigmoid_mul_kernel.cuh",
            "backend/cuda/groupnorm/groupnorm_kernel.cuh",
            "backend/cuda/softmax/softmax.cuh",
            "backend/cuda/vision_ops/nms/batched_nms_kernel.cuh",
            "backend/cuda/vision_ops/nms/nms_kernel.cuh",
            "backend/cuda/vision_ops/roi_ops/multi_level_roi_align.cuh",
            "backend/rocm/elementwise/custom_math.h",
        ]
        + gen_utils_file_list()
        + gen_cutlass_list()
        + gen_cutlass_lib_list()
        + gen_cub_list()
        + gen_ck_list()
        + gen_flash_attention_list()
        + gen_static_list()
        + gen_backend_common_file_list()
        + gen_license_file_list(),
    },
    python_requires=">=3.7, <4",
    distclass=BinaryDistribution,
    **setup_kwargs,
)

# remove temp
shutil.rmtree("./aitemplate/3rdparty")
shutil.rmtree("./aitemplate/static")
shutil.rmtree("./aitemplate/licenses")

import io
import re
from codecs import open
from os import path

from setuptools import setup, find_packages

with io.open("paperspace/version.py", "rt", encoding="utf8") as f:
    version = re.search(r"version = \"(.*?)\"", f.read())[1]

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
try:
    import pypandoc

    long_description = pypandoc.convert('README.md', 'rst')
except(IOError, ImportError, OSError):
    with open('README.md', encoding='utf-8') as f:
        long_description = f.read()

setup(
    name='paperspace',
    version=version,
    description='Paperspace Python',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/paperspace/paperspace-python',
    author='Paperspace Co.',
    author_email='info@paperspace.com',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
        'License :: OSI Approved :: ISC License (ISCL)',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='paperspace api development library',
    packages=find_packages(exclude=['contrib', 'docs', 'tests', 'old_tests']),
    install_requires=[
        'requests[security]',
        'six',
        'gradient-statsd',
        'click',
        'gradient-sdk',
        'terminaltables',
        'click-didyoumean',
        'click-help-colors',
        'click-completion',
        'colorama',
        'requests-toolbelt',
        'progressbar2',
    ],
    entry_points={'console_scripts': [
        'paperspace-python = paperspace:main',
    ]},
    extras_require={
        "dev": [
            'tox',
            'pytest',
            'mock',
        ],
    },
)

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='lltm_cuda',
    ext_modules=[
        CUDAExtension('lltm_cuda', [
            'lltm_cuda.cpp',
            'lltm_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='lltm_cpp',
    ext_modules=[
        CppExtension('lltm_cpp', ['lltm.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import distutils.command.clean
import glob
import os
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import List

from setuptools import setup, find_packages
from torch.utils.cpp_extension import (
    CppExtension,
    BuildExtension,
)

cwd = os.path.dirname(os.path.abspath(__file__))
try:
    sha = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
        .decode("ascii")
        .strip()
    )
except Exception:
    sha = "Unknown"


def get_version():
    version_txt = os.path.join(cwd, "version.txt")
    with open(version_txt, "r") as f:
        version = f.readline().strip()
    if os.getenv("BUILD_VERSION"):
        version = os.getenv("BUILD_VERSION")
    elif sha != "Unknown":
        version += f"+{sha[:7]}"
    return version


ROOT_DIR = Path(__file__).parent.resolve()


package_name = "torchrl"


def get_nightly_version():
    today = date.today()
    return f"{today.year}.{today.month}.{today.day}"


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="torchrl setup")
    parser.add_argument(
        "--package_name",
        type=str,
        default="torchrl",
        help="the name of this output wheel",
    )
    return parser.parse_known_args(argv)


def write_version_file(version):
    version_path = os.path.join(cwd, "torchrl", "version.py")
    with open(version_path, "w") as f:
        f.write(f"__version__ = '{version}'\n")
        f.write(f"git_version = {repr(sha)}\n")


def _get_pytorch_version():
    # if "PYTORCH_VERSION" in os.environ:
    #     return f"torch=={os.environ['PYTORCH_VERSION']}"
    return "torch"


def _get_packages():
    exclude = [
        "build*",
        "test*",
        "torchrl.csrc*",
        "third_party*",
        "tools*",
    ]
    return find_packages(exclude=exclude)


ROOT_DIR = Path(__file__).parent.resolve()


class clean(distutils.command.clean.clean):
    def run(self):
        # Run default behavior first
        distutils.command.clean.clean.run(self)

        # Remove torchrl extension
        for path in (ROOT_DIR / "torchrl").glob("**/*.so"):
            print(f"removing '{path}'")
            path.unlink()
        # Remove build directory
        build_dirs = [
            ROOT_DIR / "build",
        ]
        for path in build_dirs:
            if path.exists():
                print(f"removing '{path}' (and everything under it)")
                shutil.rmtree(str(path), ignore_errors=True)


# def _run_cmd(cmd):
#     try:
#         return subprocess.check_output(cmd, cwd=ROOT_DIR).decode("ascii").strip()
#     except Exception:
#         return None


def get_extensions():
    extension = CppExtension

    extra_link_args = []
    extra_compile_args = {
        "cxx": [
            "-O3",
            "-std=c++14",
            "-fdiagnostics-color=always",
        ]
    }
    debug_mode = os.getenv("DEBUG", "0") == "1"
    if debug_mode:
        print("Compiling in debug mode")
        extra_compile_args = {
            "cxx": [
                "-O0",
                "-fno-inline",
                "-g",
                "-std=c++14",
                "-fdiagnostics-color=always",
            ]
        }
        extra_link_args = ["-O0", "-g"]

    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "torchrl", "csrc")

    extension_sources = {
        os.path.join(extensions_dir, p)
        for p in glob.glob(os.path.join(extensions_dir, "*.cpp"))
    }

    sources = list(extension_sources)

    return [
        extension(
            "torchrl._torchrl",
            sources,
            include_dirs=[this_dir],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]


def _main(argv):
    args, unknown = parse_args(argv)
    name = args.package_name
    is_nightly = "nightly" in name

    if is_nightly:
        version = get_nightly_version()
        write_version_file(version)
        print(f"Building wheel {package_name}-{version}")
        print(f"BUILD_VERSION is {os.getenv('BUILD_VERSION')}")
    else:
        version = get_version()

    pytorch_package_dep = _get_pytorch_version()
    print("-- PyTorch dependency:", pytorch_package_dep)
    # branch = _run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    # tag = _run_cmd(["git", "describe", "--tags", "--exact-match", "@"])

    this_directory = Path(__file__).parent
    long_description = (this_directory / "README.md").read_text()
    sys.argv = [sys.argv[0]] + unknown

    setup(
        # Metadata
        name=name,
        version=version,
        author="torchrl contributors",
        author_email="vmoens@fb.com",
        url="https://github.com/pytorch/rl",
        long_description=long_description,
        long_description_content_type="text/markdown",
        license="BSD",
        # Package info
        packages=find_packages(exclude=("test", "tutorials")),
        ext_modules=get_extensions(),
        cmdclass={
            "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
            "clean": clean,
        },
        install_requires=[pytorch_package_dep, "numpy", "packaging", "cloudpickle"],
        extras_require={
            "atari": [
                "gym<=0.24",
                "atari-py",
                "ale-py",
                "gym[accept-rom-license]",
                "pygame",
            ],
            "dm_control": ["dm_control"],
            "gym_continuous": ["mujoco-py", "mujoco"],
            "rendering": ["moviepy"],
            "tests": ["pytest", "pyyaml", "pytest-instafail"],
            "utils": [
                "tensorboard",
                "wandb",
                "tqdm",
                "hydra-core>=1.1",
                "hydra-submitit-launcher",
            ],
        },
        zip_safe=False,
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: BSD License",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    )


if __name__ == "__main__":

    _main(sys.argv[1:])
      
      
