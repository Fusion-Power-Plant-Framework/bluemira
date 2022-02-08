# Install without conda

This is a rough WIP procedure to try to get an install without conda. It currently will
only work on Linux (probably specifically Ubuntu 20.04):

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Update and install dependencies available through pip
python -m pip install --upgrade pip setuptools
python -m pip install --upgrade wheel
python -m pip install -r requirements.txt
python -m pip install -r requirements-develop.txt

# Build and install Qt5 (5.14.2)
sudo bash scripts/qt5/install-qt5-deps.sh
bash scripts/qt5/build-qt5.sh
sudo bash scripts/qt5/install-qt5.sh

# Build and install PySide2 and shiboken (5.14.2)
bash scripts/pyside2/install-pyside2.sh

# Build and install coin (4.0.0)
bash scripts/coin/build-coin.sh
sudo bash scripts/coin/install-coin.sh

# Build and install pivy (0.6.6)
bash scripts/pivy/install-pivy.sh

# Build and install freecad (0.19.3)
sudo bash scripts/freecad/install-freecad-deps.sh
bash scripts/freecad/install-freecad.sh

# [Optional] Build and install pythonocc (approx 7.5.2)
sudo bash scripts/occ/install-occ-deps.sh
bash scripts/occ/install-occ.sh
```
