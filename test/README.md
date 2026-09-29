AMD Matrix Instruction Calculator Delta Test Tool
=====================================================================================
This directory contains a tool for running the AMD Matrix Instruction Calculator over many command-line options in order to test application code paths. In an effort to keep test execution time low, it runs tests in parallel and saves their output to temporary files. After all tests have run, it concatenates the outputs of those tests into a single user-defined file.

This allows a few types of application-level tests:
 * Test to see that the application exits properly and prints the correct error output when a user passes invalid input.
 * Test to see that the application does not crash when passing in input that is expected to be correct.
 * Create the actual generated output for good inputs, which can be compared against previous runs of this test script using tools like `diff` to see if code refactoring has resulted in unexpected output changes.

We do not ship a "known good" set of tool outputs, because such a file would be very large in comparison to the rest of the AMD Matrix Instruction Calculator repository: on the order of 10s to 100s of megabytes.
Therefore, this tool is meant to be run by developers before and after changes to check for unexpected 'deltas'.

If any command succeeds when it was expected to fail (or vice versa), the tool prints an `ERROR:` message for it and exits with a non-zero status after writing the output file.

Prerequisites
-------------------------------------------------------------------------------------
This tool requires the following:
* Python3
* The calculator's own prerequisites (see the top-level `README.md`)
* The Python packages in `requirements-test.txt` (`joblib` for this tool, plus the `pylint` development tool):
    * From the repository root, execute: `pip install -r requirements.txt -r requirements-test.txt`
* Installing prerequisites itself may require you to install `pip`

Delta Test Tool Usage
-------------------------------------------------------------------------------------
This section details the command-line parameters for this Delta Test Tool.

#### Required Arguments
This tool has a single required argument.
The last argument passed to the tool's command line should be a filename (optionally with directory information) to store the textual output of running the tests on the AMD Matrix Instruction Calculator.

By default, the tool will not overwrite existing files.
To force the tool to overwrite a file that already exists, use the `--overwrite` command-line option that is described below.

#### General-purpose Configuration Parameters
The following are general-purpose configuration parameters for this tool.

* `--version` (or `-v`): Print the version number of the tool.
* `--help` (or `-h`): Print out help information for the tool.
* `--overwrite` (or `-o`): By default, this Delta Test Tool will not overwrite or replace files that already exist. To force the tool's output to overwrite an existing file, pass this flag.
* `--cores {#}` (or `-c {#}`): By default, this tool will attempt to use all of the processing cores on the system to run tests in parallel. To limit the number of parallel tasks, pass the desired number of parallel tasks using this option.

Example of Using the Delta Test Tool
-------------------------------------------------------------------------------------
The following is an example of using this Delta Test Tool to execute a series of tests on the AMD Matrix Instruction Calculator.
After executing this command, the text output of every test will be contained in the `new_tests.txt` file within the user's current working directory.
```
$ ./delta_test.py new_tests.txt
Tests completed.
```

If the user attempted to run the above command a second time, the tool would fail to run because the `new_tests.txt` file already exists.
```
$ ./delta_test.py new_tests.txt
ERROR: new_tests.txt already exists, and --overwrite option was not passed.
To prevent files from being accidentally overwritten, this tool will exit.
```

To force the Delta Test Tool to overwrite the old `new_tests.txt`, pass the `--overwrite` option when running the tool.
```
$ ./delta_test.py --overwrite new_tests.txt
Tests completed.
```

Comparing Against a Previous Version
-------------------------------------------------------------------------------------
The `delta_diff.sh` script automates the before-and-after workflow.
It checks out a base git ref into a temporary git worktree, runs the Delta Test Tool on both that ref and the current working tree (including uncommitted changes), and diffs the two outputs.
The current `delta_test.py` is used for both runs, so the outputs are comparable even if the tester itself changed.

```
$ ./delta_diff.sh              # compare against the merge-base of HEAD and main
$ ./delta_diff.sh some_branch  # compare against any git ref
```

* `-c {#}`: Number of parallel test jobs, passed through to `delta_test.py`.
* `-k`: Keep the output directory even when there are no differences.

The script exits with 0 if the outputs are identical, 1 if they differ (the path to the full diff is printed), and 2 on error.

Checking the README Examples
-------------------------------------------------------------------------------------
The `readme_examples_test.py` script runs every example command in the top-level `README.md` (the lines starting with `$ ./matrix_calculator.py`) and checks that the calculator's current output matches the output shown in the README.
Unlike the Delta Test Tool, these are reference outputs kept in the repository, so it needs no previous version to compare against and runs in about a second.
A mismatch means either a regression or a README that needs updating; the script prints a diff for each mismatching example and exits with a non-zero status.

```
$ ./readme_examples_test.py
All 20 README examples match.
```

Trademark Attribution
-------------------------------------------------------------------------------------
&copy; 2022-2026 Advanced Micro Devices, Inc. All rights reserved. AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. in the United States and/or other jurisdictions. Other names are for informational purposes only and may be trademarks of their respective owners.
