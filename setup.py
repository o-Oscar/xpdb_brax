# Copyright 2021 The Brax Authors.
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

"""setup.py for xpdb_brax.

Install for development:

  pip intall -e .
"""

from setuptools import find_packages
from setuptools import setup

setup(
    name="xpdb_brax",
    version="0.0.0",
    description=("A differentiable physics engine written in JAX."),
    author="Boutin Oscar",
    author_email="oscar.boutin@polytechnique.edu",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    # url="http://github.com/google/brax",
    # license="Apache 2.0",
    packages=find_packages(),
    include_package_data=True,
    scripts=[],
    install_requires=[
        "absl-py",
        "clu",
        "dataclasses",
        "flax",
        "gym",
        "grpcio",
        "jax",
        "jaxlib",
        "numpy",
        "optax",
        "panda3d_viewer",
        # TODO: restore this once tfp-nightly and tensorflow are compatible
        # breakage caused by https://github.com/tensorflow/probability/commit/fdbdece116a98e101420ce38e8a45aa1e7e5656f
        # "tfp-nightly[jax]<=0.13.0.dev20210422",
    ],
    extras_require={
        # "develop": ["pytest", "transforms3d"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="JAX reinforcement learning rigidbody physics",
)
