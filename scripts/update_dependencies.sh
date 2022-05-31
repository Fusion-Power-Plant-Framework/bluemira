# Pip dependency update script
# Usage:
# bash <path to script> [Optional branch name]
#
# If the branch name is specified install the requirements from that branch
# otherwise install from develop
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

readonly BLUEMIRA_ROOT="$(realpath "$(dirname "$0")"/..)"
echo "Bluemira directory is $BLUEMIRA_ROOT"

GIT_CMD="git -C ""$BLUEMIRA_ROOT"

# check that the requirements files havent changed
$GIT_CMD diff --exit-code requirements.txt \
     || { echo "requirements.txt modified on this branch, exiting" && exit 1; }
$GIT_CMD diff --exit-code requirements-develop.txt \
     || { echo "requirements-develop.txt modified on this branch, exiting" && exit 1; }

# checkout dependencies from develop
$GIT_CMD checkout $REQ_BRANCH requirements.txt requirements-develop.txt

# install dependencies
pip install -r "$BLUEMIRA_ROOT/requirements.txt" || { echo "pip update failed" && exit 1; }
pip install -r "$BLUEMIRA_ROOT/requirements-develop.txt" || { echo "pip update failed" && exit 1; }

CURR_BRANCH=$($GIT_CMD rev-parse --abbrev-ref HEAD)
# revert checkout of dependencies files
$GIT_CMD checkout $CURR_BRANCH requirements.txt requirements-develop.txt
