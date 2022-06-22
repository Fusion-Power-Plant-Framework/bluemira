#!/usr/bin/env python3

import argparse
import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set

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
        return (
            f"On {self.when}:\n"
            f"  {self.filename}:{self.lineno}: {self.category}: {self.message}"
        )


def parse_args(sys_args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(PROG)
    parser.add_argument("report_file", help="path to .report.json to parse")
    return parser.parse_args(sys_args)


def load_json(file_path: str) -> Dict:
    with open(file_path, "r") as f:
        return json.load(f)


def parse_warning_list(warning_data: Iterable[Dict]) -> Set[Warning]:
    # Convert to set to remove duplicates
    return set([Warning(**warning) for warning in warning_data])


def display_warnings(warnings: Set[Warning]):
    print(f"{PROG}: found {len(warnings)} warnings.")
    for warning in warnings:
        print(warning)


def format_warning_report(sys_args: List[str]) -> int:
    inputs = parse_args(sys_args)
    report = load_json(inputs.report_file)
    warnings = parse_warning_list(report[WARNINGS_KEY])
    display_warnings(warnings)
    return 0


if __name__ == "__main__":
    import sys

    format_warning_report(sys.argv[1:])
