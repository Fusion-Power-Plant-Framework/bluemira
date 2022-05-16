set -e

cd
pip install -i https://test.pypi.org/simple/ CoolProp==6.4.2.dev0

git clone https://github.com/numba/numba-scipy.git
cd numba-scipy
sed -i s/1.7.1/1.7.3/g setup.py
pip install -U .
cd ..

pip install -U git+https://github.com/numba/numba.git@2109064da3

pip install -U numpy

pip install -U nlopt

cd bluemira
# git checkout new_displayer
sed -i s/"\"neutronics-material-maker==0.1.11\""/"#\"neutronics-material-maker==0.1.11\""/g setup.py
sed -i s/"\"numpy<=1.21.5\""/"#\"numpy<=1.21.5\""/g setup.py
sed -i s/"\"scipy<=1.5.3\""/"#\"scipy<=1.5.3\""/g setup.py
