# This script gets the msh2xdmf repository so that it can be loaded as a module when
# converting bluemira meshs for use in dolfin. Note that the msh2xdmf project does
# not have a setup.py or pyproject.toml, so cannot be installed as a normal python
# project.

set -e

if [[ $(basename $PWD) == *"bluemira"* ]]; then
  cd ..
fi

if [ ! -d msh2xdmf ]; then
  git clone https://github.com/floiseau/msh2xdmf.git
  cd msh2xdmf
  git checkout b562903
  cd ..
fi
