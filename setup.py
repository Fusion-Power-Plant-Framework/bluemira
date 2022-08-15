"""
Setup utility for bluemira
"""
from setuptools import find_packages, setup

import versioneer

short = "An integrated inter-disciplinary design tool for future fusion "
"reactors, incorporating several modules, some of which rely on "
"other codes, to carry out a range of typical conceptual fusion "
"reactor design activities."


with open("README.md", "r") as f:
    long = f.read()

install_requires = [
    "anytree",
    "asteval",
    "Babel",
    "click",
    "CoolProp",
    "fortranformat",
    "gmsh",
    "imageio",
    "matplotlib>=3.5",
    "neutronics-material-maker==0.1.11",  # Crash on upgrade
    "nlopt",
    "numba-scipy",
    "numba",
    "numpy",
    "pandas",
    "periodictable",
    "pint",
    "pyclipper",
    "pypet",
    "pyquaternion",
    "scikit-learn",
    "scipy",
    "seaborn",
    "tables",
    "tabulate",
    "typeguard",
    "wrapt",
]

openmoc = [
    "OpenMOC @git+https://github.com/mit-crpg/OpenMOC.git@7940c0b",
]

openmc = [
    "OpenMC @git+https://github.com/openmc-dev/openmc.git",
    "parametric-plasma-source @git+https://github.com/open-radiation-sources/parametric-plasma-source.git",
]

process = [
    "cmake>=3.13.0",
]

prominence = [
    "prominence",
]

pinned = [
    "nlopt==2.7.1",
    "numba==0.55.2",
    "numba-scipy==0.3.1",
    "numpy==1.22.4",
    "matplotlib==3.5.2",
    "scipy==1.7.3",
]

dev_requires = [
    "black",
    "flake8",
    "flake8-absolute-import",
    "flake8-bandit",
    "flake8-docstrings",
    "pep8-naming",
    "pre-commit",
    "pytest",
    "pytest-cov",
    "pytest-html",
    "pytest-metadata",
    "pytest-json-report",
    "sphinx",
    "sphinx-autoapi",
    "sphinx-rtd-theme",
    "versioneer",
]

examples = [
    "notebook",
]

extras_require = {
    "dev": dev_requires,
    "pinned": pinned,
    "examples": examples,
    "process": process,
    "openmoc": openmoc,
    "openmc": openmc,
    "prominence": prominence,
}

setup(
    name="bluemira",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description=short,
    long_description=long,
    url="git@github.com:Fusion-Power-Plant-Framework/bluemira.git",
    author="The bluemira team",
    author_email="",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)",
        "Operating System :: POSIX :: Linux",
    ],
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.7",
    zip_safe=False,
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "numba_extensions": ["init=numba_scipy:_init_extension"],
    },
)
