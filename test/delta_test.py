#!/usr/bin/env python3
# coding=utf-8
#
# Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
""" AMD Matrix Instruction Calculator Delta Test Tool
This tool will run the AMD Matrix Instruction Calculator over many command-line options
in order to test application code paths. In an effort to keep test execution time low,
it runs tests in parallel and saves their output to temporary files. After all tests
have completed successfully, it concatenates the outputs of those tests into a single
user-defined file.

This allows a few types of application-level tests:
 - Test to see that the application exits properly and prints the correct error output
   when a user passes invalid input.
 - Test to see that the application does not crash when passing in input that is expected
   to be correct.
 - Create the actual generated output for good inputs, which can be compared against
   previous runs of this test script using tools like `diff` to see if code refactoring
   has resulted in unexpected output changes.

We do not ship a "known good" set of tool outputs, because such a file would be very
large in comparison to the rest of the AMD MAtrix Instruction Calculator repository:
on the order of 10s to 100s of megabytes. Therefore this tool is meant to be run by
developers before and after changes to check for unexpected 'deltas'.
"""
import sys
import argparse
import math
from subprocess import Popen, PIPE
from tempfile import mkstemp, TemporaryDirectory
from shutil import copyfileobj
from textwrap import wrap
from pathlib import Path
from os import path
from re import findall, search, MULTILINE
from joblib import Parallel, delayed

VERSION = "1.0"

class TestRunner:
    """ Class to run the application under test with a chosen command line, redirect the output
        to a chosen file, and print error messages if the application fails when it should
        succeed, or succeeds when it should fail.
    """
    def __init__(self, test_app, output_file, expected_success=True):
        self.test_app = test_app
        self.path = str(output_file)
        self.output_file = open(output_file, "w+", encoding="utf-8")
        self.expected_success = expected_success

    def run(self, args_string):
        """ Function used to actually run the test and compare outputs """
        to_return = True
        if bool(search(r" ", str(self.test_app))):
            app_str = f"'{self.test_app}'"
        else:
            app_str = f"{self.test_app}"
        run_str = f"{app_str} {args_string}"
        run_str = list(filter(None, run_str.strip().split(' ')))
        print(str(' '.join(run_str)), file=self.output_file, flush=True)
        args_list = list(filter(None, args_string.strip().split(' ')))
        to_run = [str(self.test_app),] + args_list
        ret_val = Popen(to_run, stdout=self.output_file, stderr=self.output_file).wait()
        if (self.expected_success and ret_val != 0):
            print(f"ERROR: Expected this to succeed: {' '.join(run_str)}")
            print(f"       But it failed with a return value of {ret_val}")
            print(f"       Command array: {to_run}")
            print(f"       Output file at {self.path}")
            to_return = False
        elif (not self.expected_success and ret_val == 0):
            print(f"ERROR: Expected this to fail: {' '.join(run_str)}")
            print(f"       But it succeeded with a return value of {ret_val}")
            print(f"       Command array: {to_run}")
            print(f"       Output file at {self.path}")
            to_return = False
        return to_return

    def run_internal(self, args_string):
        """ Function used for running the matrix instruction calculator internally for this
            test script. Used to get things like detailed instruction info for calculations.
            Do not test the output success -- it should work or this script is broken.
            But return the output so that we can use it for other calculations.
        """
        if bool(search(r" ", str(self.test_app))):
            app_str = f"'{self.test_app}'"
        else:
            app_str = f"{self.test_app}"
        run_str = f"{app_str} {args_string}"
        run_str = list(filter(None, run_str.strip().split(' ')))
        args_list = list(filter(None, args_string.strip().split(' ')))
        to_run = [str(self.test_app),] + args_list
        proc = Popen(to_run, stdout=PIPE, universal_newlines=True)
        ret_str = proc.communicate()[0]
        if proc.returncode != 0:
            print(f"ERROR: Expected this internal command to succeed: {' '.join(run_str)}")
            print(f"       But it failed with a return value of {proc.returncode}")
            print(f"       Command array: {to_run}")
            print(f"       Output file at {self.path}")
            sys.exit(-1)
        return str(ret_str)

def run_error_tests(test_app, temp_dir):
    """ Run a variety of tests that should fail, because they are passing in bad values to the
        tool. The tool should catch these bad values and the runner expects to receive a
        failure.
    """
    temp_file = mkstemp(suffix='.txt', prefix='error_tests_', dir=temp_dir, text=True)
    r = TestRunner(test_app, temp_file[1], False)
    r.run("")
    r.run("-a")
    r.run("-a bad_arch")
    r.run("-a cdna1 -i")
    r.run("-a cdna1 -i bad_inst")
    for arg in ("I-coordinate", "J-coordinate", "K-coordinate", "block", "register", "lane",
                "cbsz", "abid", "blgp", "wavefront", "neg"):
        r.run(f"-a cdna1 -i v_mfma_f32_32x32x1f32 --{arg}")
        r.run(f"-a cdna1 -i v_mfma_f32_32x32x1f32 --{arg} 0xdeadbeef")
    r.run("-a cdna1 -i v_mfma_f32_32x32x1f32")
    r.run("-a cdna1 -i v_mfma_f32_32x32x1f32 --get-register --matrix-entry")
    r.run("-a cdna1 -i v_mfma_f32_32x32x1f32 --get-register --matrix-entry --register-layout "\
          "--matrix-layout")
    for arg in ("get-register", "matrix-entry", "register-layout", "matrix-layout"):
        r.run(f"-a cdna1 -i v_mfma_f32_32x32x1f32 --{arg} -A -B")
    r.run("-a cdna1 -i v_mfma_f32_32x32x1f32 --register-layout -C --output-calculation")
    r.run("-a cdna1 -i v_mfma_f32_32x32x1f32 --register-layout -B --cbsz 2")
    r.run("-a cdna1 -i v_mfma_f32_32x32x1f32 --register-layout -A --cbsz 4")
    r.run("-a cdna1 -i v_mfma_f32_32x32x1f32 --register-layout -B --abid 1")
    r.run("-a cdna1 -i v_mfma_f32_32x32x1f32 --register-layout -A --cbsz 0 --abid 1")
    r.run("-a cdna1 -i v_mfma_f32_32x32x1f32 --register-layout -A --cbsz 1 --abid 3")
    r.run("-a cdna1 -i v_mfma_f32_32x32x1f32 --register-layout -A --blgp 1")
    r.run("-a cdna1 -i v_mfma_f32_32x32x1f32 --register-layout -B --blgp 8")
    r.run("-a cdna1 -i v_mfma_f32_32x32x1f32 --register-layout -k")
    r.run("-a cdna1 -i v_mfma_f32_32x32x1f32 --register-layout -A -w 32")
    r.run("-a cdna1 -i v_mfma_f32_32x32x1f32 --register-layout -A -w 33")
    r.run("-a cdna1 -i v_mfma_f32_32x32x1f32 --register-layout -A --opsel 1")
    r.run("-a rdna3 -i v_wmma_f32_16x16x16_f16 --register-layout -A -w 33")
    r.run("-a rdna3 -i v_wmma_f32_16x16x16_f16 --register-layout -D --opsel 4")
    r.run("-a rdna3 -i v_wmma_f16_16x16x16_f16 --register-layout -D --opsel 3")
    r.run("-a rdna3 -i v_wmma_f16_16x16x16_f16 --register-layout -D --opsel 3")
    r.run("-a rdna3 -i v_wmma_f16_16x16x16_f16 --register-layout -D --opsel 3")
    r.run("-a rdna3 -i v_wmma_f16_16x16x16_f16 --register-layout -D --opsel 3")
    r.run("-a rdna3 -i v_wmma_f16_16x16x16_f16 --register-layout -B --opsel 4")
    r.run("-a rdna3 -i v_wmma_f16_16x16x16_f16 --register-layout -D --neg 1")
    r.run("-a rdna3 -i v_wmma_f16_16x16x16_f16 --register-layout -A --neg 42")
    r.run("-a rdna3 -i v_wmma_f16_16x16x16_f16 --register-layout -D --neg_hi 1")
    r.run("-a rdna3 -i v_wmma_f16_16x16x16_f16 --register-layout -A --neg_hi 42")
    r.run("-a rdna3 -i v_wmma_i32_16x16x16_iu8 -A --register-layout --neg 7")
    r.run("-a rdna3 -i v_wmma_i32_16x16x16_iu4 -A --register-layout --neg_hi 7")
    r.run("-a cdna1 -i v_mfma_f32_32x32x1f32 --register-layout -A --neg 1")
    r.run("-a cdna1 -i v_mfma_f32_32x32x1f32 --register-layout -A --neg_hi 1")
    r.run("-a cdna1 -i v_mfma_f32_32x32x1f32 --register-layout -A --csv --markdown")

    # Test bad coordinates for single register
    for coord in ("I-coordinate", "J-coordinate", "K-coordinate", "block"):
        r.run(f"-a cdna1 -i v_mfma_f32_32x32x1f32 --get-register -A --{coord} 256")

    # Test bad register/lanes for matrix-entry
    for arg in ("register", "lane"):
        r.run(f"-a cdna1 -i v_mfma_f32_32x32x1f32 --matrix-entry -A --{arg} 256")

    r.run("-a cdna1 -i v_mfma_f32_32x32x1f32 --matrix-entry -B --register 0 --lane 0 --blgp 2")
    return temp_file

def run_good_help_and_version(test_app, temp_dir):
    """ Run some very simple tests that check for help printout and tool version number.
        All of these tests should succeed.
    """
    temp_file = mkstemp(suffix='.txt', prefix='help_ver_', dir=temp_dir, text=True)
    r = TestRunner(test_app, temp_file[1])
    r.run("-h")
    r.run("--help")
    r.run("-v")
    r.run("--version")
    return temp_file

def get_architectures(r):
    """ Helper function to get the available architectures that can be used as primary inputs
        to the tool. This test function may need to be updated as the output string from
        the help function are updated, since it is doing checking against the output string.
    """
    outp = r.run_internal("--help").strip()
    found_lines = ''.join(findall(r"following architectures.+\n", outp)).strip()
    split_out = found_lines.split(": ")
    return split_out[1].split(", ")

def get_alt_architectures(r):
    """ Helper function to get the available alternative names for each of the main architectures.
        This is useful when we want to test that all of these alternative names are actually
        accepted by the tool. This may need to be updated as the string output of the tool's
        help function is updated, as it expects each line containing alternative names to start
        with "Alternately:"
    """
    outp = r.run_internal("--help")
    found_lines = ''.join(findall(r"Alternately:.+\n", outp))
    to_ret = str(found_lines).strip().replace(" ", "")
    to_ret = to_ret.replace("\n", "").replace("Alternately:", ",")
    return list(filter(None, to_ret.split(",")))

def run_instruction_list(test_app, temp_dir):
    """ Runs a basic set of tests of the --list-instructions and -L options, which should both
        do the same thing. Tests both to make sure the short form is properly accepted.
        It will run this against all avaialble architectures and alternate architecture
        names.
        Saves the output of running all of these tests into a file contained in the
        directory pointed to by temp_dir. Returns the (handle, file_name) of this
        output to the calling function.
    """
    temp_file = mkstemp(suffix='.txt', prefix='inst_list_', dir=temp_dir, text=True)
    r = TestRunner(test_app, temp_file[1])
    for arch in get_architectures(r):
        r.run(f"--architecture {arch} --list-instructions")
    for arch in get_alt_architectures(r):
        r.run(f"-a {arch} -L")
    return temp_file

def get_instructions(r, arch):
    """ Returns a list of all of the instructions avaialble in a target architecture. """
    outp = r.run_internal(f'--architecture {arch} --list-instructions')
    found_lines = ''.join(findall(r"^ .+\n", outp, flags=MULTILINE))
    to_ret = str(found_lines).strip().replace(" ", "")
    return list(filter(None, to_ret.split("\n")))

def run_detailed_instructions(test_app, temp_dir):
    """ Runs a test that prints out the detailed instruction information for all of the
        instructions on all avaialble architectures that the tool supports. Does not
        test against alternate architecture names.
        Switches between the long arguments (--instruction and --detail-instruction) and
        the short arguments (-i and -d) to ensure both work.
        Saves the output of running all of these tests into a file contained in the
        directory pointed to by temp_dir. Returns the (handle, file_name) of this
        output to the calling function.
    """
    temp_file = mkstemp(suffix='.txt', prefix='inst_list_', dir=temp_dir, text=True)
    r = TestRunner(test_app, temp_file[1])
    num_tests = 0
    for arch in get_architectures(r):
        for inst in get_instructions(r, arch):
            if num_tests % 2 == 0:
                r.run(f"--architecture {arch} --instruction {inst} --detail-instruction")
            else:
                r.run(f"-a {arch} -i {inst} -d")
            num_tests += 1
    return temp_file

def get_num(r, arch, inst, find_this, which_to_check=0):
    """ Search for a number output from the tool's detailed instruction print-out.
        Most of these numbers are of the form 'Thing to find: number\n'. As such,
        this function takes in the 'Thing to find' string and returns the number in
        integer form.
    """
    cmd_to_run = f'--architecture {arch} --instruction {inst} -d'
    outp = r.run_internal(cmd_to_run)
    to_find = fr"{find_this}:.+\n"
    found_lines = findall(to_find, outp)
    if len(found_lines) > 0:
        to_ret = found_lines[which_to_check].split(": ")[1].strip()
    else:
        to_ret = "1"
    if not to_ret.isdigit():
        print("ERROR: Tester's get_num function did not receive an integer.", file=sys.stderr)
        print(f"    Ran with arguments: {cmd_to_run}", file=sys.stderr)
        print(f"    Searched for: {to_find}", file=sys.stderr)
        print(f"    Returned: {to_ret}", file=sys.stderr)
        to_ret = "-1"
    return int(to_ret)

def get_matrix_regs(r, arch, inst, wave, matrix):
    """ Search for a number of registers needed for a particular matrix, from the tool's
        detailed instruction print-out. Matrix name is passed in as a string. Supports
        'A', 'B', 'C', and 'D'.
    """
    if (len(get_supported_wave_sizes(arch)) == 1 or wave == 32):
        check = 0
    else:
        check = 1
    return get_num(r, arch, inst, fr"GPRs required for {matrix}", check)

def get_supports(r, arch, inst, find_this):
    """ Search for whether this architecture and instruction supports a particular feature.
        For instance, BLGP or CBSZ. Uses the tool's detailed instruction print-out to
        get this info, which usually uses the form 'Thing to find: True/False'. This
        function therefore takes in a from of 'Thing to find' and returns a bool as to
        whether the tool says this instruction and architecture supports that thing.
    """
    outp = r.run_internal(f'--architecture {arch} --instruction {inst} -d')
    found_lines = findall(fr"{find_this}.+\n", outp)
    if len(found_lines) > 0:
        to_ret = (found_lines[0].split(": ")[1].strip() == 'True')
    else:
        to_ret = False
    return bool(to_ret)

def get_supported_wave_sizes(arch):
    """ Return a tuple of the supported wavefront sizes on the target architecture. """
    if arch.upper() == "RDNA3":
        ret_this = (32, 64)
    else:
        ret_this = (64,)
    return ret_this

def get_in_bits(r, arch, inst):
    """ For a particular architecture and instruction, return the number of bits required
        for the input entries. Uses the detailed instruction print-out from the tool to
        query this info.
    """
    outp = r.run_internal(f'--architecture {arch} --instruction {inst} -d')
    found_lines = findall(r"Instruction:.+\n", outp)
    to_query = found_lines[0].split(": ")
    is_64 = bool(search(r"64\n", to_query[1]))
    is_32 = bool(search(r"32\n", to_query[1]))
    is_16 = bool(search(r"16\n", to_query[1]))
    is_odd_16 = bool(search(r"16_1k\n", to_query[1]))
    is_8 = bool(search(r"8\n", to_query[1]))
    if is_64:
        to_ret = int(64)
    elif is_32:
        to_ret = int(32)
    elif (is_16 or is_odd_16):
        to_ret = int(16)
    elif is_8:
        to_ret = int(8)
    else:
        print(f"ERROR: Unknown size for instruction {to_query[1]}")
        sys.exit(-1)
    return to_ret

def run_get_register(runner, matrix, M, N, K, B, test_string):
    """ Runs the --get-register test over a series of I, J, K, and block values for the desired
        matrix. The test_string is used to pass in most of the line that will run, so it should
        contain the architecture and instruction at a minimum. This test will take in the M, N,
        K, and block sizes supported by this instruction, and walk over some of them for the
        target matrix when running the tests.
    """
    if matrix == 'A':
        row_letter = "-I"
        max_row = M
        col_letter = "-K"
        max_col = K
    elif matrix == 'B':
        row_letter = "-K"
        max_row = K
        col_letter = "-J"
        max_col = N
    else:
        row_letter = "-I"
        max_row = M
        col_letter = "-J"
        max_col = N
    if matrix == 'D':
        out_calc = "--output-calculation"
    else:
        out_calc = ""
    # Test up to 2 locations in each row, column, and block instead of the whole range, to reduce
    # the amount of test time taken. Test the first and last location within each range. This
    # should very that we can test 0 and non-zero in the get-register calculations. If the
    # actual "where does something live" calculations are bad, we will likely catch that problem
    # in the register-layout test.
    if max_row > 2:
        rows = (0, max_row-1)
    else:
        rows = range(max_row)
    if max_col > 2:
        cols = (0, max_col-1)
    else:
        cols = range(max_col)
    if B > 2:
        blocks = (0, B-1)
    else:
        blocks = range(B)
    for row in rows:
        for col in cols:
            for block in blocks:
                x = f"{test_string} {row_letter} {row} {col_letter} {col} -b {block} {out_calc}"
                runner.run(x)

def run_matrix_entry(runner, matrix, a_regs, b_regs, cd_regs, wave_size, blgp, test_string):
    """ Runs the --matrix-entry test over a series of registers and lanes on a particular matrix.
        The test_string is used to pass in most of the line that will run, so it should contain
        the architecture and instruction at a minimum. However, this test will take the A, B, and
        C/D maximum registers and walk over some of them for the target matrix. Same thing for
        the available lanes.
    """
    if matrix == 'A':
        max_reg_to_use = a_regs
    elif matrix == 'B':
        max_reg_to_use = b_regs
    else:
        max_reg_to_use = cd_regs
    # Test only 2 registers instead of the whole range, to reduce the amount of test time taken.
    # Test the first and last registers. This should verify that we can test 0 and non-zero in the
    # matrix-entry calculations. If the actual "where does something live" calculations are bad,
    # we will likely catch that problem in the matrix-layout test.
    if max_reg_to_use > 2:
        regs_to_use = (0, max_reg_to_use-1)
    else:
        regs_to_use = range(max_reg_to_use)

    # Test only 6 lanes instead of 64 or 32, to reduce the amount of test time taken.
    # Test the first and last lanes, some prime-number lanes, and one on the edge
    # of 16. This should hopefully catch many possible erroneous situtations.
    if wave_size == 64:
        if blgp in (0, 3):
            lanes_to_use = (0, 7, 16, 31, 43, 63)
        elif blgp == 1:
            lanes_to_use = (0, 7, 16, 31)
        elif blgp == 2:
            lanes_to_use = (43, 63)
        elif blgp == 4:
            lanes_to_use = (0, 7)
        elif blgp == 5:
            lanes_to_use = (16, 31)
        elif blgp == 6:
            lanes_to_use = (43,)
        else: # blgp == 7
            lanes_to_use = (63,)
    else:
        lanes_to_use = (0, 7, 15, 16, 27, 31)

    for reg in regs_to_use:
        # Test only 6 lanes instead of 64, to reduce the amount of test time taken.
        # Test the first and last lanes, some prime-number lanes, and one on the edge
        # of 16. This should hopefully catch many possible erroneous situtations.
        for lane in lanes_to_use:
            runner.run(f"{test_string} -r {reg} -l {lane}")

def run_parallel_matrix_test(test_app, test_name, arch, inst, temp_file_name):
    """ Function that will execute a test on all of the options available for  matrix and
        test type.
        Available options for the test name, which is the string name of the test to run,
        are the same as the main tool:
          - register-layout
          - matrix-layout
          - get-register
          - matrix-entry
        The final three arguments are the string of the architecture to pass to the tool,
        the name of the instruction to pass to the tool, and a filename where the output
        of this test should be stored.
    """
    # Open tester which will save out all our test output to the desired file
    r = TestRunner(test_app, temp_file_name)

    # Get parameters we may need for the variety of tests we want to run
    M = get_num(r, arch, inst, "M")
    N = get_num(r, arch, inst, "N")
    K = get_num(r, arch, inst, "K")
    B = get_num(r, arch, inst, "blocks")
    if get_supports(r, arch, inst, "CBSZ"):
        max_cbsz = int(math.log(B, 2))
    else:
        max_cbsz = 0
    if get_supports(r, arch, inst, "BLGP"):
        max_blgp = 7
    else:
        max_blgp = 0
    if get_supports(r, arch, inst, r"OPSEL\[2\]"):
        opsel_vals = (0, 4)
    else:
        opsel_vals = (0,)
    if get_supports(r, arch, inst, "NEG") and not bool(search("_i32_", inst)):
        max_neg = 7
    else:
        max_neg = 0

    matrices = ('A', 'B', 'D', 'C')

    wave_sizes = get_supported_wave_sizes(arch)

    num_done = 0

    for wave in wave_sizes:
        a_regs = get_matrix_regs(r, arch, inst, wave, "A")
        b_regs = get_matrix_regs(r, arch, inst, wave, "B")
        cd_regs = get_matrix_regs(r, arch, inst, wave, "D")
        for matrix in matrices:
            if matrix == 'A':
                for neg in range(max_neg+1):
                    for cbsz in range(max_cbsz+1):
                        max_abid = int(math.pow(2, int(cbsz)) - 1)
                        for abid in range(max_abid+1):
                            test_string = f"-a {arch} -i {inst} -{matrix} --{test_name} "
                            test_string += f"--cbsz {cbsz} --abid {abid} --neg {neg} "
                            test_string += f"--neg_hi {neg} -w {wave} "
                            if test_name in ("register-layout", "matrix-layout"):
                                if num_done % 3 == 1:
                                    test_string += "--csv"
                                elif num_done % 3 == 2:
                                    test_string += "--markdown"
                                if num_done % 2 == 1:
                                    test_string += " --transpose"
                                r.run(test_string)
                            elif test_name == "get-register":
                                run_get_register(r, matrix, M, N, K, B, test_string)
                            elif test_name == "matrix-entry":
                                run_matrix_entry(r, matrix, a_regs, b_regs, cd_regs, wave, 0,
                                                 test_string)
                            else:
                                print(f"Unknown test name {test_name}")
                                sys.exit(-1)
                            num_done += 1
            elif matrix == 'B':
                for neg in range(max_neg+1):
                    for blgp in range(max_blgp+1):
                        test_string = f"-a {arch} -i {inst} -{matrix} --{test_name} --blgp {blgp} "
                        test_string += f"--neg {neg} --neg_hi {neg} -w {wave} "
                        if test_name in ("register-layout", "matrix-layout"):
                            if num_done % 3 == 1:
                                test_string += "--csv"
                            elif num_done % 3 == 2:
                                test_string += "--markdown"
                            if num_done % 2 == 1:
                                test_string += " --transpose"
                            r.run(test_string)
                        elif test_name == "get-register":
                            run_get_register(r, matrix, M, N, K, B, test_string)
                        elif test_name == "matrix-entry":
                            run_matrix_entry(r, matrix, a_regs, b_regs, cd_regs, wave, blgp,
                                             test_string)
                        else:
                            print(f"Unknown test name {test_name}")
                            sys.exit(-1)
                        num_done += 1
            else:
                if matrix == 'D':
                    max_neg = 0
                for neg in range(max_neg+1):
                    for opsel in opsel_vals:
                        test_string = f"-a {arch} -i {inst} -{matrix} --{test_name} -w {wave} "
                        test_string += f"--opsel {opsel} --neg {max_neg} --neg_hi {max_neg} "
                        if test_name in ("register-layout", "matrix-layout"):
                            if num_done % 3 == 1:
                                test_string += "--csv"
                            elif num_done % 3 == 2:
                                test_string += "--markdown"
                            if num_done % 2 == 1:
                                test_string += " --transpose"
                            r.run(test_string)
                        elif test_name == "get-register":
                            run_get_register(r, matrix, M, N, K, B, test_string)
                        elif test_name == "matrix-entry":
                            run_matrix_entry(r, matrix, a_regs, b_regs, cd_regs, wave, 0,
                                             test_string)
                        else:
                            print(f"Unknown test name {test_name}")
                            sys.exit(-1)
                        num_done += 1

def run_matrix_tests(test_app, temp_dir, test_name, num_jobs):
    """ Function that will launch off parallel worker threads in order to execute one of the
        four matrix tests performed by the AMD Matrix Instruction Calculator.
        Each parallel test will write to its own out put file. This function will pass back
        a list of those filenames for another function to concatenate them together in
        order.
        Takes as arguments:
          - The filename of the application to test
          - The directory to put the temporary per-test output files in
          - The string name of the test to run. Options are the same as the main tool:
              - register-layout
              - matrix-layout
              - get-register
              - matrix-entry
          - The maximum number of parallel jobs to run.
        Saves the output of running all of these tests into files contained in the
        directory pointed to by temp_dir. Returns a list of the (handle, file_name) of
        these outputs to the calling function.
    """
    arch_inst_groups = []
    temp_files = []
    dummy_temp = mkstemp(suffix='.txt', prefix='dummy_', dir=temp_dir, text=True)
    r = TestRunner(test_app, dummy_temp[1])
    for arch in get_architectures(r):
        for inst in get_instructions(r, arch):
            temp_file = mkstemp(suffix='.txt', prefix='inst_list_', dir=temp_dir, text=True)
            arch_inst_groups.append((arch, inst, temp_file[1]))
            temp_files.append(temp_file)
    task = (delayed(run_parallel_matrix_test)(test_app, test_name, arch, inst, temp_file)
            for arch, inst, temp_file in arch_inst_groups)
    Parallel(n_jobs=num_jobs, backend="threading")(task)
    return temp_files

def parse_and_run():
    """ Parse the arguments for this script, then run the requested functions.
        Print help about the application if requested, or if wrong
        arguments are passed in. Check to make sure the arguments are valid
        with respect to one another and give recommendations if they are not.
    """
    parser = argparse.ArgumentParser(
                description='\n'.join(
                    wrap("This tool will run the AMD Matrix Instruction Calculator over many "
                         "command-line options in order to test application code paths. In an "
                         "effort to keep test execution time low, it runs tests in parallel "
                         "and saves their output to temporary files. After all tests have "
                         "completed successfully, it concatenates the outputs of those tests into "
                         "a single user-defined file.", 80)),
                formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-v', '--version', action='store_true',
                        dest='print_version',
                        help='Print the version of this tool')
    parser.add_argument('-o', '--overwrite', action='store_true',
                        dest='overwrite',
                        help='Overwrite output file if it already exists.')
    parser.add_argument('-c', '--cores', action='store',
                        dest='cores', default='-1', nargs='?',
                        help="Number of cores on which to run parallel tests. Default: all cores.")
    parser.add_argument('output_file', metavar='output_file_name', type=str, nargs='?',
                        default="",
                        help='Name of the file to used to hold the final tool outputs.')
    args = parser.parse_args()

    if args.print_version:
        print(f"AMD Matrix Instruction Calculator Delta Tester Version {VERSION}")
        sys.exit(0)

    if args.output_file == "":
        print("ERROR: Output file name is required as a command line argument to this tool.\n")
        parser.print_help()
        sys.exit(-1)

    output_file_path = Path(args.output_file)
    if (output_file_path.is_file() and not args.overwrite):
        print(f"ERROR: {args.output_file} already exists, and --overwrite option was not passed.")
        print("To prevent files from being accidentally overwritten, this tool will exit.")
        sys.exit(-1)

    if args.cores is None:
        parser.error('"--cores" argument required.')
    try:
        cores = int(args.cores)
    except ValueError:
        parser.error('"--cores" argument must be an integer >= 0, or -1 for "all cores".')
    if cores < -1 or cores == 0:
        parser.error('"--cores" argument must be an integer >= 0, or -1 for "all cores".')

    test_app_location = f"{path.dirname(__file__)}/../matrix_calculator.py"
    test_app_location = path.realpath(test_app_location)
    test_script_path = Path(test_app_location)
    if not test_script_path.is_file():
        print(f"ERROR: Cannot find tool to test at {test_app_location}")
        sys.exit(-1)

    temp_dir = TemporaryDirectory()

    # files to concatenate at the end of the run
    files = []
    files.append(run_error_tests(test_script_path, temp_dir.name))
    files.append(run_good_help_and_version(test_script_path, temp_dir.name))
    files.append(run_instruction_list(test_script_path, temp_dir.name))
    files.append(run_detailed_instructions(test_script_path, temp_dir.name))
    _ = [files.append(x) for x in run_matrix_tests(test_script_path, temp_dir.name,
                                                   "register-layout", cores)]
    _ = [files.append(x) for x in run_matrix_tests(test_script_path, temp_dir.name,
                                                   "matrix-layout", cores)]
    _ = [files.append(x) for x in run_matrix_tests(test_script_path, temp_dir.name,
                                                   "get-register", cores)]
    _ = [files.append(x) for x in run_matrix_tests(test_script_path, temp_dir.name,
                                                   "matrix-entry", cores)]

    # Concatenate output files together
    with open(output_file_path, 'w+', encoding="utf-8") as fout:
        for in_file in files:
            copyfileobj(open(in_file[1], "r", encoding="utf-8"), fout)
        fout.close()

    print("Tests completed.")

if __name__ == '__main__':
    parse_and_run()
