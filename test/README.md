AMD Matrix Instruction Calculator Delta Test Tool
=====================================================================================
This repository contains a tool for running the AMD Matrix Instruction Calculator over many command-line options in order to test application code paths. In an effort to keep test execution time low, it runs tests in parallel and saves their output to temporary files. After all tests have completed successfully, it concatenates the outputs of those tests into a single user-defined file.

This allows a few types of application-level tests:
 * Test to see that the application exits properly and prints the correct error output when a user passes invalid input.
 * Test to see that the application does not crash when passing in input that is expected to be correct.
 * Create the actual generated output for good inputs, which can be compared against previous runs of this test script using tools like `diff` to see if code refactoring has resulted in unexpected output changes.

We do not ship a "known good" set of tool outputs, because such a file would be very large in comparison to the rest of the AMD Matrix Instruction Calculator repository: on the order of 10s to 100s of megabytes.
Therefore, this tool is meant to be run by developers before and after changes to check for unexpected 'deltas'.

Prerequisites
-------------------------------------------------------------------------------------
This tool requires the following:
* Python3
* The Python package `joblib`:
    * To install this package system wide, execute: `sudo pip install joblib`
    * To install this package for the local user, execute: `pip install joblib --user`
* Installing prerequisites itself may require you to install `pip`

Delta Test Tool Usage
-------------------------------------------------------------------------------------
This section details the command-line parameters for this Delta Test Tool.

#### Required Arguments
This tool has a single required argument.
The last argument passed to the tool's command line should be a filename (optionally with directory information) to store the textual output of running the tests on the AMD Matrix Instructions Calculator.

By default, the tool will not overwrite existing files.
To force the tool to overwrite a file that already exists, use the `--overwrite` command-line option that is described below.

#### General-purpose Configuration Parameters
The following are general-purpose tool configuration parameters that allow users to choose a desired instruction on a target processor.
Command line parameters are case sensitive, but inputs for the command-line parameters are case insensitive.

* `--version` (or `-v`): Print the version number of the tool.
* `--help` (or `-h`): Print out help information for the tool.
* `--overwrite` (or `-o`): By default, this Delta Test Tool will not overwrite or replace files that already exist. To force the tool's output to overwrite an existing file, pass this flag.
* `--cores {#}`(or `-c {#}`): By default, this tool will attempt to use all of the processing cores on the system to run tests in parallel. To limit the number of parallel tasks, pass the desired number of parallel tasks using this option.

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

Trademark Attribution
-------------------------------------------------------------------------------------
&copy; 2022-2023 Advanced Micro Devices, Inc. All rights reserved. AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. in the United States and/or other jurisdictions. Other names are for informational purposes only and may be trademarks of their respective owners.
