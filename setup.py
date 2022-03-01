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

install_requires = [  # PYL = Version limited by python version
    "asteval",  # 0.9.25
    "click",  # 8.0.3
    "CoolProp",  # 6.4.1
    "fortranformat",  # 1.1.1
    "imageio",  # 2.13.5
    "ipykernel",  # 6.6.1
    "matplotlib",  # 3.3.4   PYL
    "natsort",  # 8.0.2
    "neutronics-material-maker==0.1.11",  # Crash on upgrade
    "nlopt",  # 2.7.0
    "numba",  # 0.53.1
    "numba-scipy",  # 0.3.0
    "numpy",  # 1.19.5   PYL
    "pandas",  # 1.3.5    PYL
    "pyclipper",  # 1.2.1
    "pypet",  # 0.6.0
    "pyquaternion",  # 0.9.9
    "scikit-learn",  # 1.0.2
    "seaborn",  # 0.11.2
    "sectionproperties",  # 1.0.8 (with scipy <= 1.5)
    "Shapely",  # 1.8.0
    "tables",  # 3.7.0
    "tabulate",  # 0.8.9
    "trimesh",  # 3.9.42
    "scipy",  # 1.4.1 Last OK version
    "wrapt",  # 1.13.3
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
    "black",  # 21.12b0
    "flake8",  # 4.0.1
    "flake8-bandit",  # 2.1.2
    "flake8-docstrings",  # 1.6.0
    "flake8-absolute-import",
    "pep8-naming",  # 0.12.1
    "pre-commit",  # 2.16.0
    "pytest",  # 6.2.5
    "pytest-cov",  # 3.0.0
    "pytest-html",  # 3.1.1
    "pytest-metadata",  # 1.11.0
    "sphinx",  # 4.3.2
    "sphinx-autoapi",  # 1.8.4
    "sphinx-rtd-theme",  # 1.0.0
    "versioneer",  # 0.21
]

examples = [
    "notebook",  # 6.4.6
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
