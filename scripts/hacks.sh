set -e

pip install -i https://test.pypi.org/simple/ CoolProp==6.4.2.dev0
pip install neutronics-material-maker==0.1.11
pip install -U git+https://github.com/numba/numba-scipy.git
