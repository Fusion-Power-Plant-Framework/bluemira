set -e

if [ "$1" ]
  then
    REQ_BRANCH="$1"
else
    REQ_BRANCH="develop"
fi

# make sure bluemira is in the current environment
BM_INSTALLED=$(pip list | grep bluemira | awk 'NR==1 {print $1}')
[ "$BM_INSTALLED" = "bluemira" ] || { echo "bluemira not found in the current python environment, exiting" && exit 1; }

# check we are in the bluemira git repo
[ $(git rev-parse --is-inside-work-tree) = "true" ] ||  { echo "Not in bluemira git repo, exiting" && exit 1; }

CURRENT_REPO=$(git remote get-url origin)
REPO_GIT_URL="git@github.com:Fusion-Power-Plant-Framework/bluemira.git"
REPO_HTTPS_URL="https://github.com/Fusion-Power-Plant-Framework/bluemira.git"
[ "$CURRENT_REPO" = "$REPO_GIT_URL" ] || [ "$CURRENT_REPO" = "$REPO_HTTPS_URL" ] || { echo "Not in bluemira git repo, exiting" && exit 1; }

# check that the requirements files havent changed
git diff --exit-code requirements.txt
git diff --exit-code requirements-develop.txt

# checkout dependencies from develop
git checkout $REQ_BRANCH requirements.txt requirements-develop.txt

# install dependencies
pip install -r requirements.txt || { echo "pip update failed" && exit 1; }
pip install -r requirements-develop.txt || { echo "pip update failed" && exit 1; }

# revert checkout of dependicies files
git checkout $(git rev-parse --abbrev-ref HEAD) requirements.txt requirements-develop.txt
