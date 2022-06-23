#!/usr/bin/env python3

import argparse
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Set, Tuple

INDENT_SIZE = 2
PROG = "format_warning_report"
WARNINGS_KEY = "warnings"


@dataclass(frozen=True, eq=True)
class Warning:
    message: str
    category: str
    when: str
    filename: str
    lineno: int

    def __str__(self) -> str:
        return f"{self.filename}:{self.lineno}: {self.category}: {self.message}"


def parse_args(sys_args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(PROG)
    parser.add_argument("report_file", help="path to .report.json to parse")
    parser.add_argument(
        "--compare", default=None, help="path to .report.json to compare warnings to"
    )
    return parser.parse_args(sys_args)


def load_warnings(file_path: str) -> Set[Warning]:
    with open(file_path, "r") as f:
        data = json.load(f)
    return set([Warning(**warning) for warning in data[WARNINGS_KEY]])


def format_warnings_markdown(warnings: Iterable[Warning]):
    whens: Dict[str, List[Warning]] = {}
    for warning in warnings:
        try:
            whens[warning.when].append(warning)
        except KeyError:
            whens[warning.when] = [warning]
    markdown = ""
    for when, warns in whens.items():
        markdown += f"## On {when}\n\n"
        for warn in warns:
            markdown += f"- `{warn}`\n"
        markdown += "\n"
    return markdown[:-1]


def make_collapsable_md(markdown: str, summary: str):
    lines = markdown.split("\n")
    indented_lines = [f"{line}" for line in lines]
    new_lines = (
        [
            "<details>\n",
            f"{' '*INDENT_SIZE}<summary>{summary}</summary>\n",
        ]
        + indented_lines
        + ["</details>"]
    )
    return "\n".join(new_lines)


def elements_not_in(head: Iterable[Any], ref: Iterable[Any]) -> List:
    not_in = []
    for head_el in head:
        if head_el not in ref:
            not_in.append(head_el)
    return not_in


def compare_warnings(
    head_warnings: Set[Warning], ref_warnings: Set[Warning]
) -> Tuple[List[Warning], List[Warning]]:
    new_warnings = elements_not_in(head_warnings, ref_warnings)
    fixed_warnings = elements_not_in(ref_warnings, head_warnings)
    return new_warnings, fixed_warnings


def format_warning_report(sys_args: List[str]) -> str:
    inputs = parse_args(sys_args)
    warnings = load_warnings(inputs.report_file)
    report = ["# ⚠️ Warning Report\n"]
    warning_list_md = format_warnings_markdown(warnings)
    if inputs.compare is not None:
        ref_warnings = load_warnings(inputs.compare)
        new, fixed = compare_warnings(warnings, ref_warnings)
        report.append(f"Found {len(new)} new warnings, {len(fixed)} fixed warnings.\n")
        if new:
            warning_list = format_warnings_markdown(new)
            report.append(make_collapsable_md(warning_list, "New warnings"))
    else:
        report.append(f"Found {len(warnings)} warnings.\n")
    if warnings:
        report.append(
            make_collapsable_md(warning_list_md, f"All warnings ({len(warnings)})")
        )
    return "\n".join(report)


if __name__ == "__main__":
    import sys

    print(format_warning_report(sys.argv[1:]))
