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
    "asteval",
    "Babel",
    "click",
    "CoolProp",
    "fortranformat",
    "imageio",
    "ipykernel",
    "matplotlib<=3.3.4",  # upgrade on BP removal
    "natsort",
    "neutronics-material-maker==0.1.11",  # Crash on upgrade
    "nlopt",
    "numba",
    "numba-scipy",
    "numpy<=1.21.5",  # numba's highest numpy
    "pandas",
    "pint",
    "periodictable",
    "pyclipper",
    "pypet",
    "pyquaternion",
    "scikit-learn",
    "seaborn",
    "sectionproperties",  # 1.0.8dev (with scipy < 1.6)
    "Shapely",
    "tables",
    "tabulate",
    "trimesh",
    "scipy<=1.5.3",
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
    "beautifulsoup4>=4.8",
    "cmake>=3.12.0",
    "graphviz>=0.13",
    "markdown>=3.2",
    "markdown_include>=0.5",
    "md-environ>=0.1",
    "toposort>=1.5",
]

dev_requires = [
    "black",
    "flake8",
    "flake8-bandit",
    "flake8-docstrings",
    "flake8-absolute-import",
    "pep8-naming",
    "pre-commit",
    "pytest",
    "pytest-cov",
    "pytest-html",
    "pytest-metadata",
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
    "examples": examples,
    "process": process,
    "openmoc": openmoc,
    "openmc": openmc,
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
        "console_scripts": ["bluemira=BLUEPRINT.blueprint_cli:cli"],
        "numba_extensions": ["init=numba_scipy:_init_extension"],
    },
)
