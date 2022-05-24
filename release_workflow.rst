Create PR to merge develop into main
Merge develop into main if all tests pass (without squashing)
Tag last commit on main with the below

    git tag -a v<semantic_version> -m Release v<semantic_version>

Create PR to merge develop_dependencies into develop
