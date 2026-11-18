# SPDX-License-Identifier: MIT
"""Check the accumulator term in the calculator's public output."""
import subprocess
import sys
import unittest
from pathlib import Path


CALCULATOR = Path(__file__).resolve().parents[1] / 'matrix_calculator.py'


class SparseAccumulatorTest(unittest.TestCase):
    def test_output_calculation(self):
        for arch, instruction in [('cdna3', 'v_smfmac_f32_16x16x32_f16'),
                                  ('rdna4', 'v_swmmac_f32_16x16x32_f16')]:
            for mode in ('--get-register', '--matrix-entry'):
                with self.subTest(arch=arch, mode=mode):
                    output = subprocess.check_output(
                        [sys.executable, str(CALCULATOR), '-a', arch, '-i', instruction,
                         mode, '-D', '--output-calculation'], universal_newlines=True)
                    equation = output.splitlines()[-1]
                    _, destination, expression = equation.split(' = ')
                    self.assertTrue(expression.endswith(' + ' + destination),
                                    msg=equation)
