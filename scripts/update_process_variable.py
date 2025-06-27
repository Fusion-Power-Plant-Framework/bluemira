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
2. Save this text into /tmp/var.txt.
3. Run this file by
    - cd bluemira-process-develop/
    - python scripts/update_process_variable.py /tmp/var.txt
    3.1 Carry out the manual action specified in the summary printed at the end.
4. Re-run this same file in bluemira private data by
    - cd ../bluemira-private-data/
    - python ../bluemira-process-develop/scripts/update_process_variable.py /tmp/var.txt
    4.1 Carry out the manual action specified in the summary printed at the end.
"""

import os
import sys
from subprocess import getstatusoutput  # noqa: S404

template_str = r"grep -rlw {v_before} | grep -v pycache"


def populate_template(v_before, *, exclude_process=False, only_process=False):
    """Fill the template grep string.

    Parameters
    ----------
    v_before:
        the variable to be grepping for.
    exclude_process:
        Whether to exclude file paths with the word 'process' in its path.
    only_process:
        If true, only output files with the word 'process' in its path.

    Returns
    -------
    :
        A string that, when inputted into the terminal, greps for exact matches of
        v_before, and outputs the full file paths where at least one v_before is present
        in the file.

    Raises
    ------
    ValueError
        Prevents both exclude_process and only_process to be simultaneously true.

    Example
    -------
    1. grep -rlw v_before | grep -v pycache
    2. grep -rlw v_before | grep -v pycache | grep -vi process
    3. grep -rlw v_before | grep -v pycache | grep -i process
    """
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

    Parameters
    ----------
    v_before:
        The old variable name.
    v_after:
        The new variable name.

    Returns
    -------
    :
        The sed command that can be used to perform this precise renaming.

    Example
    -------
    sed -i "s/\bBEFORE\b/AFTER/g" $(grep -rlw BEFORE | grep -v pycache | grep -i process)
    """
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

    Parameters
    ----------
    v_before:
        the variable to be grepped.
    exclude_process:
        Whether to exclude files with the word 'process' in its path from the the grepped
        results.

    Returns
    -------
    status:
        Integer. The subprocess returned status. 0 for match(es) found, 1 for no matches
        found, any other integers (e.g. 255) means the command has failed, likely due to
        malformed syntax or any other interruptions.
    output:
        A single string of all file paths, separated by newline characters.

    Example
    -------
    output of `grep -r `
    """
    return getstatusoutput(populate_template(v_before, exclude_process=exclude_process))  # noqa: S605


if __name__ == "__main__":
    unsafe_variables = {}
    if len(sys.argv) == 1 or sys.argv[1].upper() == "-H":
        print(
            """Usage
-----
1. Find the process variable name change PR.
    Open the markdown source text, save only part of the source related to variable
    deletion and renames.
    - verify that all variables are surrounded by ``
    - verify that all name-changes lines are denoted as `BEFORE` -> `AFTER`.
2. Save this text into /tmp/var.txt.
3. Run this file by
    - cd bluemira-process-develop/
    - python scripts/update_process_variable.py /tmp/var.txt
    3.1 Carry out the manual action specified in the summary printed at the end.
4. Re-run this same file in bluemira private data by
    - cd ../bluemira-private-data/
    - python ../bluemira-process-develop/scripts/update_process_variable.py /tmp/var.txt
    4.1 Carry out the manual action specified in the summary printed at the end.
"""
        )
        sys.exit()
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

            out_status, grepped_files = grep(var_before, exclude_process=not delete_var)
            if out_status == 0:
                if delete_var:
                    unsafe_variables[f"Delete '{var_before}' from:"] = grepped_files
                else:
                    unsafe_variables[
                        r'sed -i "s/\b' + var_before + r"\b/" + f'{var_after}/g"'
                    ] = grepped_files

    print("█" * os.get_terminal_size().columns)
    if unsafe_variables:
        print("Please perform the following recommended actions manually.")
        print("Recommended actions:")
        for command, locations in unsafe_variables.items():
            print(command)
            for loc in locations.split("\n"):
                print(" " * 4 + loc)
    else:
        print("No manual actions is required.")
