"""
Outputs the commands used to uptate the process variables.

Delete before merging PR #3636.

Usage
-----
1. Find the process variable name change PR.
    Open the markdown source text, save only part of the source related to variable
    deletion and renames.
    - verify that all variables are surrounded by ``
    - verify that all name-changes lines are denoted as `BEFORE` -> `AFTER`.
2. Save this text into /tmp/var_change.txt. `cd bluemira-process-develop/`
3. Run this file by
    - python scripts/update_process_variable.py /tmp/var.txt
4. Carry out the manual instructions at the bottom of the
5. `cd ../bluemira-private-data/`. Re-run this same file by
    - python ../bluemira-process-develop/scripts/update_process_variable.py /tmp/var.txt
"""

import sys
from subprocess import getstatusoutput  # noqa: S404

template_str = r"grep -rlw {v_before} | grep -v pycache | grep -vi process"


def populate_template(v_before, *, exclude_process=False, only_process=False):
    """Fill the template string."""  # noqa: DOC201, DOC501
    if exclude_process and only_process:
        raise ValueError(
            "The result of 'grep process' and 'grep -v process' would be null."
        )
    filled_str = template_str.format(v_before=v_before)
    if exclude_process:
        return filled_str + r" | grep -vi process"
    if only_process:
        return filled_str + r" | grep -i process"
    return filled_str


def safe_variable_rename(v_before, v_after):
    """Bash instructions to rename a variable.
    Safely renames as it only renames if it has 'process' in the file name.

    Example
    -------
    sed -i "s/\bBEFORE\b/AFTER/g" $(grep -rlw BEFORE | grep -v pycache | grep -i process)
    """  # noqa: DOC201
    return (
        r'sed -i "s/\b'
        + v_before
        + r"\b/"
        + v_after
        + r'/g" $('
        + populate_template(v_before, only_process=True)
        + ")"
    )


def grep(v_before, *, exclude_process=False):
    """Runs, using subprocess, grep.

    Examples
    --------
    output of `grep -r `
    """  # noqa: DOC201
    return getstatusoutput(populate_template(v_before, exclude_process=exclude_process))  # noqa: S605


if __name__ == "__main__":
    unsafe_variables = {}
    pull_request_text = sys.argv[1]
    with open(pull_request_text) as pr:
        for line in pr:
            if r"`" not in line:
                continue
            if "->" in line:
                delete_var = False
                _, var_before, _arrow, var_after, _ = line.split(r"`")
                var_before, var_after = var_before.strip(), var_after.strip()
                print(
                    f"Safely renaming all instances of the full word '{var_before}' into"
                    f" '{var_after}' in all files with 'process' in its path."
                )
                getstatusoutput(safe_variable_rename(var_before, var_after))  # noqa: S605
            else:
                delete_var = True
                _, var_before, _ = line.split(r"`")
                var_before = var_before.strip()
                print(
                    f"Manual deletion of all instances of '{var_before}' (if any) "
                    "required. See summary printed below."
                )
                # find the variable

            out_status, grep_matched_files = grep(
                var_before, exclude_process=not delete_var
            )
            if out_status == 0:
                if delete_var:
                    key_str = (
                        populate_template(var_before, exclude_process=False)
                        + f"\nDelete '{var_before}' from:"
                    )
                else:
                    key_str = (
                        populate_template(var_before, exclude_process=True)
                        + f"\nRename '{var_before}' into '{var_after}' in:"
                    )
                unsafe_variables[key_str] = grep_matched_files

    print("Check the following files manually:")
    for command, locations in unsafe_variables.items():
        print()
        print(command)
        for loc in locations.split("\n"):
            print(" " * 4 + loc)
