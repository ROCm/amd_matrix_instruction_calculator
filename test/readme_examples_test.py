#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""AMD Matrix Instruction Calculator README Examples Test
Runs every example command in the top-level README.md (lines starting with
"$ ./matrix_calculator.py") and checks that the calculator's current output matches the output
shown in the README. Unlike the delta test, these outputs are hand-checked reference answers
kept in the repository, so a mismatch means either a regression or a stale README.

Exits with 0 if every example matches, and -1 otherwise.
"""

import difflib
import shlex
import sys
from os import path
from subprocess import PIPE, run

PROMPT = "$ ./matrix_calculator.py"


def get_examples(readme_lines):
    """Returns a list of (line_number, command, expected_output_lines) for every example in the
    README. An example's expected output runs from the line after its command until the end
    of its code block or the next command.
    """
    examples = []
    i = 0
    while i < len(readme_lines):
        if not readme_lines[i].startswith(PROMPT):
            i += 1
            continue
        end = i + 1
        while (
            end < len(readme_lines)
            and not readme_lines[end].startswith("```")
            and not readme_lines[end].startswith("$ ")
        ):
            end += 1
        expected = [line.rstrip() for line in readme_lines[i + 1 : end]]
        while expected and expected[-1] == "":
            expected.pop()
        examples.append((i + 1, readme_lines[i][2:], expected))
        i = end
    return examples


def run_example(repo_root, command):
    """Runs one README example command from the repository root and returns its output lines."""
    proc = run(
        shlex.split(command),
        cwd=repo_root,
        stdout=PIPE,
        stderr=PIPE,
        universal_newlines=True,
        check=False,
    )
    output = (proc.stdout + proc.stderr).rstrip("\n")
    return [line.rstrip() for line in output.split("\n")]


def main():
    """Checks every README example and prints a diff for each one that does not match."""
    repo_root = path.realpath(path.join(path.dirname(__file__), ".."))
    with open(path.join(repo_root, "README.md"), "r", encoding="utf-8") as readme:
        readme_lines = readme.read().split("\n")
    examples = get_examples(readme_lines)
    if not examples:
        print(f"ERROR: No examples starting with '{PROMPT}' found in README.md.")
        sys.exit(-1)

    failures = 0
    for line_number, command, expected in examples:
        actual = run_example(repo_root, command)
        if actual != expected:
            failures += 1
            print(f"ERROR: README.md line {line_number} does not match the tool's output:")
            print(f"       {command}")
            diff = difflib.unified_diff(expected, actual, "README.md", "actual", lineterm="")
            for line in diff:
                print(f"       {line}")

    if failures > 0:
        print(f"{failures} of {len(examples)} README examples do not match.")
        sys.exit(-1)
    print(f"All {len(examples)} README examples match.")


if __name__ == '__main__':
    main()
