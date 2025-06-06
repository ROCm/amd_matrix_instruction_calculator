#!/usr/bin/env python3
# coding=utf-8
#
# Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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
""" AMD Matrix Instruction Calculator
This tool allows users to generate information about the register layout
for matrix multiplication instructions on AMD accelerators.

This includes the Matrix Fused Multiply Add (MFMA) and Sparse Matrix Fused
Multiply Accumulate (SMFMAC) instructions that drive the Matrix Cores in
AMD Instinct(tm) accelerators, including:
 - AMD CDNA(tm) 1 architecture accelerators, such as AMD Instinct MI100
 - AMD CDNA 2 architecture accelerators, such as AMD Instinct MI200
 - AMD CDNA 3 architecture accelerators, such as AMD Instinct MI300

This also includes the Wave Matrix Multiply Accumulate (WMMA) and Sparse
Wave Matrix Multiply Accumulate (SWMMAC) instructions that drive the AI
Accelerators in AMD Radeon(tm) GPUs, including:
 - AMD RDNA(tm) 3 architecture GPUs
 - AMD RDNA 4 architecture GPUs

There are five options for each matrix multiplication instruction:
 - Print general information about the instruction, such as its
   number of registers, computational throughput, and co-execution
   capabilities (--detail-instruction).
 - Print the register and lane for a user-chosen A[], B[], C[], or D[]
   matrix entry (--get-register).
 - Print the A[], B[], C[], or D[] matrix entry for a chosen combination
   of register and lane (--matrix-entry).
 - Print the register and lane combinations for an entire A[], B[], C[],
   or D[] matrix (--register-layout).
 - Print the A[], B[], C[], or D[] matrix entries for all of the
   instructions' registers and lanes (--matrix-layout).

SMFMAC and SWMMAC instructions work on 4:2-sparse matrices, and have a "K[]"
matrix. This is the compression index input, which is put into SrcC in place of
the C[] matrix.

This register is legal for the last four options (--get-register,
--matrix-entry, --register-layout, --matrix-layout), but only for instructions
that are sparse.
"""

import argparse
import math
import re
import sys
from abc import ABCMeta, abstractmethod
from textwrap import fill, dedent, wrap, TextWrapper
from traceback import extract_stack
from typing import Dict, List, Optional, TextIO, Tuple
try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict
from tabulate import tabulate

VERSION = "1.3.2"

# Dictionary of possible names for the various supported architectures
dict_isas = {
    'cdna1'            : 'cdna1',
    'cdna'             : 'cdna1',
    'gfx908'           : 'cdna1',
    'mi100'            : 'cdna1',
    'arcturus'         : 'cdna1',
    'cdna2'            : 'cdna2',
    'gfx90a'           : 'cdna2',
    'mi200'            : 'cdna2',
    'mi210'            : 'cdna2',
    'mi250'            : 'cdna2',
    'mi250x'           : 'cdna2',
    'aldebaran'        : 'cdna2',
    'cdna3'            : 'cdna3',
    'gfx940'           : 'cdna3',
    'gfx941'           : 'cdna3',
    'gfx942'           : 'cdna3',
    'mi300'            : 'cdna3',
    'mi300a'           : 'cdna3',
    'mi300x'           : 'cdna3',
    'mi325x'           : 'cdna3',
    'aqua_vanjaram'    : 'cdna3',
    'rdna3'            : 'rdna3',
    'gfx1100'          : 'rdna3',
    'gfx1101'          : 'rdna3',
    'gfx1102'          : 'rdna3',
    'gfx1103'          : 'rdna3',
    'gfx1150'          : 'rdna3',
    'gfx1151'          : 'rdna3',
    'gfx1152'          : 'rdna3',
    'gfx1153'          : 'rdna3',
    'rdna4'            : 'rdna4',
    'gfx1200'          : 'rdna4',
    'gfx1201'          : 'rdna4'
}

class MatrixNumericalType(TypedDict):
    """ Typed dictionary that defines numerical types used in matrix multiply instructions

    A typed dictionary container for the numerical types used in matrix multiply instructions.
    Data types have short-string identifiers, sizes in bits, and some strings describing the
    type. This container helps us go from the encoding to that data.

    Attributes:
        size: size of the type, in bits
        description: a string describing this number when used as a matrix input or output
    """
    size: int
    description: str

# Dictionary for calculating the size of particular data types
dict_math_types: Dict[str, MatrixNumericalType] = {
    'fp64': {
        'size': 64,
        'description': 'FP64 (IEEE binary64 floating point)',
    },
    'fp32': {
        'size': 32,
        'description': 'FP32 (IEEE binary32 floating point)',
    },
    'xf32': {
        'size': 32,
        'description': 'FP32 (IEEE binary32 floating point)',
    },
    'fp16': {
        'size': 16,
        'description': 'FP16 (IEEE binary16 floating point)',
    },
    'bf16': {
        'size': 16,
        'description': 'BF16 (Brain floating point)',
    },
    'int32': {
        'size': 32,
        'description': 'int32 (Signed 32-bit integer)',
    },
    'int8': {
        'size': 8,
        'description': 'int8 (Signed 8-bit integer)',
    },
    'amd_fp8': {
        'size': 8,
        'description': 'FP8 (AMD 4-bit exponent, 3-bit mantissa floating point)',
    },
    'amd_bf8': {
        'size': 8,
        'description': 'BF8 (AMD 5-bit exponent, 2-bit mantissa floating point)',
    },
    'fp8': {
        'size': 8,
        'description': 'FP8 (OCP 4-bit exponent, 3-bit mantissa floating point)',
    },
    'bf8': {
        'size': 8,
        'description': 'BF8 (OCP 5-bit exponent, 2-bit mantissa floating point)',
    },
    'iu8': {
        'size': 8,
        'description': 'IU8 (Signed/unsigned 8-bit integer)',
    },
    'iu4': {
        'size': 4,
        'description': 'IU4 (Signed/unsigned 4-bit integer)',
    }
}


class MatrixInstruction(TypedDict):
    """ Typed dictionary that defines a matrix multiply instruction

    A typed dictionary container for the data that defines a matrix multiplication instruction in
    this tool. The details of how the matrix multiplication instruction, the matrix it works on,
    the available modifiers, and the performance are contained in a series of fields in this
    container.

    Attributes:
        arch: a string to identify the accelerator architecture which can execute this instruction
        opcode: an integer opcode of this instruction within the VOP3P format
        in_type: a string that defines the data type of the matrix entry in the Src0 register,
            used as a key for the dict_math_types dictionary
        in_type_src1: a string that defines the data type of the matrix entry in the Src1 register,
            used as a key for the dict_math_types dictionary
        out_type: a string that defines each matrix entry's output data type, used as a key for
            the dict_math_types dictionary
        m: an integer that defines the M dimension of the matrix multiplication
        n: an integer that defines the N dimension of the matrix multiplication
        k: an integer that defines the K dimension of the matrix multiplication
        blocks: an integer holding the number of matrix multiplication blocks this instruction does
        cycles: an integer that calculate the number of cycles needed to execute this instruction
        integer: a boolean that is True if this instruction is operating on integer data types
        c_d_arch: a boolean that is True if this instruction can hold C & D matrices in archVGPRs
        gpr_byte_align: an integer for the byte alignment of matrix data held in registers
        blgp: a boolean that is True if the instruction can use the BLGP modifier
        cbsz_abid: a boolean that is True if the instruction can use the CBSZ and/or ABID modifiers
        sparse: a boolean that is True if the instruction is a structured sparse matrix multiply
        cd_opsel: a boolean that is True if OPSEL can choose the bits used for the C/D matrices
        neg: a boolean that is True if the NEG and/or NEG_HI fields can be used
        coexec: a boolean that is True if this instruction can co-execute with VALU instructions
        coexec_delay: an integer holding the number of cycles after issuing a matrix op before
            any co-executing VALU instructions can start issuing.
    """
    arch: str
    opcode: int
    in_type: str
    in_type_src1: str
    out_type: str
    m: int
    n: int
    k: int
    blocks: int
    cycles: int
    integer: bool
    c_d_arch: bool
    gpr_byte_align: int
    blgp: bool
    cbsz_abid: bool
    sparse: bool
    cd_opsel: bool
    neg: bool
    coexec: bool
    coexec_delay: int


# Dictionary of matrix math operators and their various parameters
# Outer dictionary key is the accelerator's architecture name.
# The values for that are another dictionary.
# The inner dictionary's keys are the actual instruction mnemonics,
# with values being a MatrixInstruction object holding all the
# necessary information about this instruction on this arch.
dict_insts: Dict[str, Dict[str, MatrixInstruction]] = {
    'cdna1': {
        'v_mfma_f32_32x32x1f32': {
            'arch': 'cdna1',
            'opcode': 64,
            'in_type': 'fp32',
            'in_type_src1': 'fp32',
            'out_type': 'fp32',
            'm': 32,
            'n': 32,
            'k': 1,
            'blocks': 2,
            'cycles': 64,
            'integer': False,
            'c_d_arch': False,
            'gpr_byte_align': 4,
            'blgp': True,
            'cbsz_abid': True,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 8
        },
        'v_mfma_f32_16x16x1f32': {
            'arch': 'cdna1',
            'opcode': 65,
            'in_type': 'fp32',
            'in_type_src1': 'fp32',
            'out_type': 'fp32',
            'm': 16,
            'n': 16,
            'k': 1,
            'blocks': 4,
            'cycles': 32,
            'integer': False,
            'c_d_arch': False,
            'gpr_byte_align': 4,
            'blgp': True,
            'cbsz_abid': True,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 8
        },
        'v_mfma_f32_4x4x1f32': {
            'arch': 'cdna1',
            'opcode': 66,
            'in_type': 'fp32',
            'in_type_src1': 'fp32',
            'out_type': 'fp32',
            'm': 4,
            'n': 4,
            'k': 1,
            'blocks': 16,
            'cycles': 8,
            'integer': False,
            'c_d_arch': False,
            'gpr_byte_align': 4,
            'blgp': True,
            'cbsz_abid': True,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 8
        },
        'v_mfma_f32_32x32x2f32': {
            'arch': 'cdna1',
            'opcode': 68,
            'in_type': 'fp32',
            'in_type_src1': 'fp32',
            'out_type': 'fp32',
            'm': 32,
            'n': 32,
            'k': 2,
            'blocks': 1,
            'cycles': 64,
            'integer': False,
            'c_d_arch': False,
            'gpr_byte_align': 4,
            'blgp': True,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 8
        },
        'v_mfma_f32_16x16x4f32': {
            'arch': 'cdna1',
            'opcode': 69,
            'in_type': 'fp32',
            'in_type_src1': 'fp32',
            'out_type': 'fp32',
            'm': 16,
            'n': 16,
            'k': 4,
            'blocks': 1,
            'cycles': 32,
            'integer': False,
            'c_d_arch': False,
            'gpr_byte_align': 4,
            'blgp': True,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 8
        },
        'v_mfma_f32_32x32x4f16': {
            'arch': 'cdna1',
            'opcode': 72,
            'in_type': 'fp16',
            'in_type_src1': 'fp16',
            'out_type': 'fp32',
            'm': 32,
            'n': 32,
            'k': 4,
            'blocks': 2,
            'cycles': 64,
            'integer': False,
            'c_d_arch': False,
            'gpr_byte_align': 4,
            'blgp': True,
            'cbsz_abid': True,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 8
        },
        'v_mfma_f32_16x16x4f16': {
            'arch': 'cdna1',
            'opcode': 73,
            'in_type': 'fp16',
            'in_type_src1': 'fp16',
            'out_type': 'fp32',
            'm': 16,
            'n': 16,
            'k': 4,
            'blocks': 4,
            'cycles': 32,
            'integer': False,
            'c_d_arch': False,
            'gpr_byte_align': 4,
            'blgp': True,
            'cbsz_abid': True,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 8
        },
        'v_mfma_f32_4x4x4f16': {
            'arch': 'cdna1',
            'opcode': 74,
            'in_type': 'fp16',
            'in_type_src1': 'fp16',
            'out_type': 'fp32',
            'm': 4,
            'n': 4,
            'k': 4,
            'blocks': 16,
            'cycles': 8,
            'integer': False,
            'c_d_arch': False,
            'gpr_byte_align': 4,
            'blgp': True,
            'cbsz_abid': True,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 8
        },
        'v_mfma_f32_32x32x8f16': {
            'arch': 'cdna1',
            'opcode': 76,
            'in_type': 'fp16',
            'in_type_src1': 'fp16',
            'out_type': 'fp32',
            'm': 32,
            'n': 32,
            'k': 8,
            'blocks': 1,
            'cycles': 64,
            'integer': False,
            'c_d_arch': False,
            'gpr_byte_align': 4,
            'blgp': True,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 8
        },
        'v_mfma_f32_16x16x16f16': {
            'arch': 'cdna1',
            'opcode': 77,
            'in_type': 'fp16',
            'in_type_src1': 'fp16',
            'out_type': 'fp32',
            'm': 16,
            'n': 16,
            'k': 16,
            'blocks': 1,
            'cycles': 32,
            'integer': False,
            'c_d_arch': False,
            'gpr_byte_align': 4,
            'blgp': True,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 8
        },
        'v_mfma_i32_32x32x4i8': {
            'arch': 'cdna1',
            'opcode': 80,
            'in_type': 'int8',
            'in_type_src1': 'int8',
            'out_type': 'int32',
            'm': 32,
            'n': 32,
            'k': 4,
            'blocks': 2,
            'cycles': 64,
            'integer': True,
            'c_d_arch': False,
            'gpr_byte_align': 4,
            'blgp': True,
            'cbsz_abid': True,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 8
        },
        'v_mfma_i32_16x16x4i8': {
            'arch': 'cdna1',
            'opcode': 81,
            'in_type': 'int8',
            'in_type_src1': 'int8',
            'out_type': 'int32',
            'm': 16,
            'n': 16,
            'k': 4,
            'blocks': 4,
            'cycles': 32,
            'integer': True,
            'c_d_arch': False,
            'gpr_byte_align': 4,
            'blgp': True,
            'cbsz_abid': True,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 8
        },
        'v_mfma_i32_4x4x4i8': {
            'arch': 'cdna1',
            'opcode': 82,
            'in_type': 'int8',
            'in_type_src1': 'int8',
            'out_type': 'int32',
            'm': 4,
            'n': 4,
            'k': 4,
            'blocks': 16,
            'cycles': 8,
            'integer': True,
            'c_d_arch': False,
            'gpr_byte_align': 4,
            'blgp': True,
            'cbsz_abid': True,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 8
        },
        'v_mfma_i32_32x32x8i8': {
            'arch': 'cdna1',
            'opcode': 84,
            'in_type': 'int8',
            'in_type_src1': 'int8',
            'out_type': 'int32',
            'm': 32,
            'n': 32,
            'k': 8,
            'blocks': 1,
            'cycles': 64,
            'integer': True,
            'c_d_arch': False,
            'gpr_byte_align': 4,
            'blgp': True,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 8
        },
        'v_mfma_i32_16x16x16i8': {
            'arch': 'cdna1',
            'opcode': 84,
            'in_type': 'int8',
            'in_type_src1': 'int8',
            'out_type': 'int32',
            'm': 16,
            'n': 16,
            'k': 16,
            'blocks': 1,
            'cycles': 32,
            'integer': True,
            'c_d_arch': False,
            'gpr_byte_align': 4,
            'blgp': True,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 8
        },
        'v_mfma_f32_32x32x2bf16': {
            'arch': 'cdna1',
            'opcode': 104,
            'in_type': 'bf16',
            'in_type_src1': 'bf16',
            'out_type': 'fp32',
            'm': 32,
            'n': 32,
            'k': 2,
            'blocks': 2,
            'cycles': 64,
            'integer': False,
            'c_d_arch': False,
            'gpr_byte_align': 4,
            'blgp': True,
            'cbsz_abid': True,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 8
        },
        'v_mfma_f32_16x16x2bf16': {
            'arch': 'cdna1',
            'opcode': 105,
            'in_type': 'bf16',
            'in_type_src1': 'bf16',
            'out_type': 'fp32',
            'm': 16,
            'n': 16,
            'k': 2,
            'blocks': 4,
            'cycles': 32,
            'integer': False,
            'c_d_arch': False,
            'gpr_byte_align': 4,
            'blgp': True,
            'cbsz_abid': True,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 8
        },
        'v_mfma_f32_4x4x2bf16': {
            'arch': 'cdna1',
            'opcode': 107,
            'in_type': 'bf16',
            'in_type_src1': 'bf16',
            'out_type': 'fp32',
            'm': 4,
            'n': 4,
            'k': 2,
            'blocks': 16,
            'cycles': 8,
            'integer': False,
            'c_d_arch': False,
            'gpr_byte_align': 4,
            'blgp': True,
            'cbsz_abid': True,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 8
        },
        'v_mfma_f32_32x32x4bf16': {
            'arch': 'cdna1',
            'opcode': 108,
            'in_type': 'bf16',
            'in_type_src1': 'bf16',
            'out_type': 'fp32',
            'm': 32,
            'n': 32,
            'k': 4,
            'blocks': 1,
            'cycles': 64,
            'integer': False,
            'c_d_arch': False,
            'gpr_byte_align': 4,
            'blgp': True,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 8
        },
        'v_mfma_f32_16x16x8bf16': {
            'arch': 'cdna1',
            'opcode': 109,
            'in_type': 'bf16',
            'in_type_src1': 'bf16',
            'out_type': 'fp32',
            'm': 16,
            'n': 16,
            'k': 8,
            'blocks': 1,
            'cycles': 32,
            'integer': False,
            'c_d_arch': False,
            'gpr_byte_align': 4,
            'blgp': True,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 8
        }
    },
    'cdna2': {
        'v_mfma_f32_32x32x1f32': {
            'arch': 'cdna2',
            'opcode': 64,
            'in_type': 'fp32',
            'in_type_src1': 'fp32',
            'out_type': 'fp32',
            'm': 32,
            'n': 32,
            'k': 1,
            'blocks': 2,
            'cycles': 64,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': True,
            'cbsz_abid': True,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_mfma_f32_16x16x1f32': {
            'arch': 'cdna2',
            'opcode': 65,
            'in_type': 'fp32',
            'in_type_src1': 'fp32',
            'out_type': 'fp32',
            'm': 16,
            'n': 16,
            'k': 1,
            'blocks': 4,
            'cycles': 32,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': True,
            'cbsz_abid': True,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_mfma_f32_4x4x1f32': {
            'arch': 'cdna2',
            'opcode': 66,
            'in_type': 'fp32',
            'in_type_src1': 'fp32',
            'out_type': 'fp32',
            'm': 4,
            'n': 4,
            'k': 1,
            'blocks': 16,
            'cycles': 8,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': True,
            'cbsz_abid': True,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_mfma_f32_32x32x2f32': {
            'arch': 'cdna2',
            'opcode': 68,
            'in_type': 'fp32',
            'in_type_src1': 'fp32',
            'out_type': 'fp32',
            'm': 32,
            'n': 32,
            'k': 2,
            'blocks': 1,
            'cycles': 64,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': True,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_mfma_f32_16x16x4f32': {
            'arch': 'cdna2',
            'opcode': 69,
            'in_type': 'fp32',
            'in_type_src1': 'fp32',
            'out_type': 'fp32',
            'm': 16,
            'n': 16,
            'k': 4,
            'blocks': 1,
            'cycles': 32,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': True,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_mfma_f32_32x32x4f16': {
            'arch': 'cdna2',
            'opcode': 72,
            'in_type': 'fp16',
            'in_type_src1': 'fp16',
            'out_type': 'fp32',
            'm': 32,
            'n': 32,
            'k': 4,
            'blocks': 2,
            'cycles': 64,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': True,
            'cbsz_abid': True,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_mfma_f32_16x16x4f16': {
            'arch': 'cdna2',
            'opcode': 73,
            'in_type': 'fp16',
            'in_type_src1': 'fp16',
            'out_type': 'fp32',
            'm': 16,
            'n': 16,
            'k': 4,
            'blocks': 4,
            'cycles': 32,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': True,
            'cbsz_abid': True,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_mfma_f32_4x4x4f16': {
            'arch': 'cdna2',
            'opcode': 74,
            'in_type': 'fp16',
            'in_type_src1': 'fp16',
            'out_type': 'fp32',
            'm': 4,
            'n': 4,
            'k': 4,
            'blocks': 16,
            'cycles': 8,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': True,
            'cbsz_abid': True,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_mfma_f32_32x32x8f16': {
            'arch': 'cdna2',
            'opcode': 76,
            'in_type': 'fp16',
            'in_type_src1': 'fp16',
            'out_type': 'fp32',
            'm': 32,
            'n': 32,
            'k': 8,
            'blocks': 1,
            'cycles': 64,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': True,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_mfma_f32_16x16x16f16': {
            'arch': 'cdna2',
            'opcode': 77,
            'in_type': 'fp16',
            'in_type_src1': 'fp16',
            'out_type': 'fp32',
            'm': 16,
            'n': 16,
            'k': 16,
            'blocks': 1,
            'cycles': 32,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': True,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_mfma_i32_32x32x4i8': {
            'arch': 'cdna2',
            'opcode': 80,
            'in_type': 'int8',
            'in_type_src1': 'int8',
            'out_type': 'int32',
            'm': 32,
            'n': 32,
            'k': 4,
            'blocks': 2,
            'cycles': 64,
            'integer': True,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': True,
            'cbsz_abid': True,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_mfma_i32_16x16x4i8': {
            'arch': 'cdna2',
            'opcode': 81,
            'in_type': 'int8',
            'in_type_src1': 'int8',
            'out_type': 'int32',
            'm': 16,
            'n': 16,
            'k': 4,
            'blocks': 4,
            'cycles': 32,
            'integer': True,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': True,
            'cbsz_abid': True,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_mfma_i32_4x4x4i8': {
            'arch': 'cdna2',
            'opcode': 82,
            'in_type': 'int8',
            'in_type_src1': 'int8',
            'out_type': 'int32',
            'm': 4,
            'n': 4,
            'k': 4,
            'blocks': 16,
            'cycles': 8,
            'integer': True,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': True,
            'cbsz_abid': True,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_mfma_i32_32x32x8i8': {
            'arch': 'cdna2',
            'opcode': 84,
            'in_type': 'int8',
            'in_type_src1': 'int8',
            'out_type': 'int32',
            'm': 32,
            'n': 32,
            'k': 8,
            'blocks': 1,
            'cycles': 64,
            'integer': True,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': True,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_mfma_i32_16x16x16i8': {
            'arch': 'cdna2',
            'opcode': 84,
            'in_type': 'int8',
            'in_type_src1': 'int8',
            'out_type': 'int32',
            'm': 16,
            'n': 16,
            'k': 16,
            'blocks': 1,
            'cycles': 32,
            'integer': True,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': True,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_mfma_f32_32x32x4bf16_1k': {
            'arch': 'cdna2',
            'opcode': 99,
            'in_type': 'bf16',
            'in_type_src1': 'bf16',
            'out_type': 'fp32',
            'm': 32,
            'n': 32,
            'k': 4,
            'blocks': 2,
            'cycles': 64,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': True,
            'cbsz_abid': True,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_mfma_f32_16x16x4bf16_1k': {
            'arch': 'cdna2',
            'opcode': 100,
            'in_type': 'bf16',
            'in_type_src1': 'bf16',
            'out_type': 'fp32',
            'm': 16,
            'n': 16,
            'k': 4,
            'blocks': 4,
            'cycles': 32,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': True,
            'cbsz_abid': True,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_mfma_f32_4x4x4bf16_1k': {
            'arch': 'cdna2',
            'opcode': 101,
            'in_type': 'bf16',
            'in_type_src1': 'bf16',
            'out_type': 'fp32',
            'm': 4,
            'n': 4,
            'k': 4,
            'blocks': 16,
            'cycles': 8,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': True,
            'cbsz_abid': True,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_mfma_f32_32x32x8bf16_1k': {
            'arch': 'cdna2',
            'opcode': 102,
            'in_type': 'bf16',
            'in_type_src1': 'bf16',
            'out_type': 'fp32',
            'm': 32,
            'n': 32,
            'k': 8,
            'blocks': 1,
            'cycles': 64,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': True,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_mfma_f32_16x16x16bf16_1k': {
            'arch': 'cdna2',
            'opcode': 103,
            'in_type': 'bf16',
            'in_type_src1': 'bf16',
            'out_type': 'fp32',
            'm': 16,
            'n': 16,
            'k': 16,
            'blocks': 1,
            'cycles': 32,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': True,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_mfma_f32_32x32x2bf16': {
            'arch': 'cdna2',
            'opcode': 104,
            'in_type': 'bf16',
            'in_type_src1': 'bf16',
            'out_type': 'fp32',
            'm': 32,
            'n': 32,
            'k': 2,
            'blocks': 2,
            'cycles': 64,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': True,
            'cbsz_abid': True,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_mfma_f32_16x16x2bf16': {
            'arch': 'cdna2',
            'opcode': 105,
            'in_type': 'bf16',
            'in_type_src1': 'bf16',
            'out_type': 'fp32',
            'm': 16,
            'n': 16,
            'k': 2,
            'blocks': 4,
            'cycles': 32,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': True,
            'cbsz_abid': True,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_mfma_f32_4x4x2bf16': {
            'arch': 'cdna2',
            'opcode': 107,
            'in_type': 'bf16',
            'in_type_src1': 'bf16',
            'out_type': 'fp32',
            'm': 4,
            'n': 4,
            'k': 2,
            'blocks': 16,
            'cycles': 8,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': True,
            'cbsz_abid': True,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_mfma_f32_32x32x4bf16': {
            'arch': 'cdna2',
            'opcode': 108,
            'in_type': 'bf16',
            'in_type_src1': 'bf16',
            'out_type': 'fp32',
            'm': 32,
            'n': 32,
            'k': 4,
            'blocks': 1,
            'cycles': 64,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': True,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_mfma_f32_16x16x8bf16': {
            'arch': 'cdna2',
            'opcode': 109,
            'in_type': 'bf16',
            'in_type_src1': 'bf16',
            'out_type': 'fp32',
            'm': 16,
            'n': 16,
            'k': 8,
            'blocks': 1,
            'cycles': 32,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': True,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_mfma_f64_16x16x4f64': {
            'arch': 'cdna2',
            'opcode': 110,
            'in_type': 'fp64',
            'in_type_src1': 'fp64',
            'out_type': 'fp64',
            'm': 16,
            'n': 16,
            'k': 4,
            'blocks': 1,
            'cycles': 32,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': False,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': False,
            'coexec_delay': -1
        },
        'v_mfma_f64_4x4x4f64': {
            'arch': 'cdna2',
            'opcode': 111,
            'in_type': 'fp64',
            'in_type_src1': 'fp64',
            'out_type': 'fp64',
            'm': 4,
            'n': 4,
            'k': 4,
            'blocks': 4,
            'cycles': 16,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': False,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': False,
            'coexec_delay': -1
        }
    },
    'cdna3': {
        'v_mfma_f32_16x16x8_xf32': {
            'arch': 'cdna3',
            'opcode': 62,
            'in_type': 'xf32',
            'in_type_src1': 'xf32',
            'out_type': 'fp32',
            'm': 16,
            'n': 16,
            'k': 8,
            'blocks': 1,
            'cycles': 16,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': False,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_mfma_f32_32x32x4_xf32': {
            'arch': 'cdna3',
            'opcode': 63,
            'in_type': 'xf32',
            'in_type_src1': 'xf32',
            'out_type': 'fp32',
            'm': 32,
            'n': 32,
            'k': 4,
            'blocks': 1,
            'cycles': 32,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': False,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_mfma_f32_32x32x1_2b_f32': {
            'arch': 'cdna3',
            'opcode': 64,
            'in_type': 'fp32',
            'in_type_src1': 'fp32',
            'out_type': 'fp32',
            'm': 32,
            'n': 32,
            'k': 1,
            'blocks': 2,
            'cycles': 64,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': True,
            'cbsz_abid': True,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': False,
            'coexec_delay': -1
        },
        'v_mfma_f32_16x16x1_4b_f32': {
            'arch': 'cdna3',
            'opcode': 65,
            'in_type': 'fp32',
            'in_type_src1': 'fp32',
            'out_type': 'fp32',
            'm': 16,
            'n': 16,
            'k': 1,
            'blocks': 4,
            'cycles': 32,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': True,
            'cbsz_abid': True,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': False,
            'coexec_delay': -1
        },
        'v_mfma_f32_4x4x1_16b_f32': {
            'arch': 'cdna3',
            'opcode': 66,
            'in_type': 'fp32',
            'in_type_src1': 'fp32',
            'out_type': 'fp32',
            'm': 4,
            'n': 4,
            'k': 1,
            'blocks': 16,
            'cycles': 8,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': True,
            'cbsz_abid': True,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': False,
            'coexec_delay': -1
        },
        'v_mfma_f32_32x32x2_f32': {
            'arch': 'cdna3',
            'opcode': 68,
            'in_type': 'fp32',
            'in_type_src1': 'fp32',
            'out_type': 'fp32',
            'm': 32,
            'n': 32,
            'k': 2,
            'blocks': 1,
            'cycles': 64,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': True,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': False,
            'coexec_delay': -1
        },
        'v_mfma_f32_16x16x4_f32': {
            'arch': 'cdna3',
            'opcode': 69,
            'in_type': 'fp32',
            'in_type_src1': 'fp32',
            'out_type': 'fp32',
            'm': 16,
            'n': 16,
            'k': 4,
            'blocks': 1,
            'cycles': 32,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': True,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': False,
            'coexec_delay': -1
        },
        'v_mfma_f32_32x32x4_2b_f16': {
            'arch': 'cdna3',
            'opcode': 72,
            'in_type': 'fp16',
            'in_type_src1': 'fp16',
            'out_type': 'fp32',
            'm': 32,
            'n': 32,
            'k': 4,
            'blocks': 2,
            'cycles': 64,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': True,
            'cbsz_abid': True,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_mfma_f32_16x16x4_4b_f16': {
            'arch': 'cdna3',
            'opcode': 73,
            'in_type': 'fp16',
            'in_type_src1': 'fp16',
            'out_type': 'fp32',
            'm': 16,
            'n': 16,
            'k': 4,
            'blocks': 4,
            'cycles': 32,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': True,
            'cbsz_abid': True,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_mfma_f32_4x4x4_16b_f16': {
            'arch': 'cdna3',
            'opcode': 74,
            'in_type': 'fp16',
            'in_type_src1': 'fp16',
            'out_type': 'fp32',
            'm': 4,
            'n': 4,
            'k': 4,
            'blocks': 16,
            'cycles': 8,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': True,
            'cbsz_abid': True,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_mfma_f32_32x32x8_f16': {
            'arch': 'cdna3',
            'opcode': 76,
            'in_type': 'fp16',
            'in_type_src1': 'fp16',
            'out_type': 'fp32',
            'm': 32,
            'n': 32,
            'k': 8,
            'blocks': 1,
            'cycles': 32,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': False,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_mfma_f32_16x16x16_f16': {
            'arch': 'cdna3',
            'opcode': 77,
            'in_type': 'fp16',
            'in_type_src1': 'fp16',
            'out_type': 'fp32',
            'm': 16,
            'n': 16,
            'k': 16,
            'blocks': 1,
            'cycles': 16,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': False,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_mfma_i32_32x32x4_2b_i8': {
            'arch': 'cdna3',
            'opcode': 80,
            'in_type': 'int8',
            'in_type_src1': 'int8',
            'out_type': 'int32',
            'm': 32,
            'n': 32,
            'k': 4,
            'blocks': 2,
            'cycles': 64,
            'integer': True,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': True,
            'cbsz_abid': True,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_mfma_i32_16x16x4_4b_i8': {
            'arch': 'cdna3',
            'opcode': 81,
            'in_type': 'int8',
            'in_type_src1': 'int8',
            'out_type': 'int32',
            'm': 16,
            'n': 16,
            'k': 4,
            'blocks': 4,
            'cycles': 32,
            'integer': True,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': True,
            'cbsz_abid': True,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_mfma_i32_4x4x4_16b_i8': {
            'arch': 'cdna3',
            'opcode': 82,
            'in_type': 'int8',
            'in_type_src1': 'int8',
            'out_type': 'int32',
            'm': 4,
            'n': 4,
            'k': 4,
            'blocks': 16,
            'cycles': 8,
            'integer': True,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': True,
            'cbsz_abid': True,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_mfma_i32_32x32x16_i8': {
            'arch': 'cdna3',
            'opcode': 84,
            'in_type': 'int8',
            'in_type_src1': 'int8',
            'out_type': 'int32',
            'm': 32,
            'n': 32,
            'k': 16,
            'blocks': 1,
            'cycles': 32,
            'integer': True,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': False,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_mfma_i32_16x16x32_i8': {
            'arch': 'cdna3',
            'opcode': 84,
            'in_type': 'int8',
            'in_type_src1': 'int8',
            'out_type': 'int32',
            'm': 16,
            'n': 16,
            'k': 32,
            'blocks': 1,
            'cycles': 16,
            'integer': True,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': False,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_mfma_f32_32x32x4_2b_bf16': {
            'arch': 'cdna3',
            'opcode': 93,
            'in_type': 'bf16',
            'in_type_src1': 'bf16',
            'out_type': 'fp32',
            'm': 32,
            'n': 32,
            'k': 4,
            'blocks': 2,
            'cycles': 64,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': True,
            'cbsz_abid': True,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_mfma_f32_16x16x4_4b_bf16': {
            'arch': 'cdna3',
            'opcode': 94,
            'in_type': 'bf16',
            'in_type_src1': 'bf16',
            'out_type': 'fp32',
            'm': 16,
            'n': 16,
            'k': 4,
            'blocks': 4,
            'cycles': 32,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': True,
            'cbsz_abid': True,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_mfma_f32_4x4x4_16b_bf16': {
            'arch': 'cdna3',
            'opcode': 95,
            'in_type': 'bf16',
            'in_type_src1': 'bf16',
            'out_type': 'fp32',
            'm': 4,
            'n': 4,
            'k': 4,
            'blocks': 16,
            'cycles': 8,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': True,
            'cbsz_abid': True,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_mfma_f32_32x32x8_bf16': {
            'arch': 'cdna3',
            'opcode': 96,
            'in_type': 'bf16',
            'in_type_src1': 'bf16',
            'out_type': 'fp32',
            'm': 32,
            'n': 32,
            'k': 8,
            'blocks': 1,
            'cycles': 32,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': False,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_mfma_f32_16x16x16_bf16': {
            'arch': 'cdna3',
            'opcode': 97,
            'in_type': 'bf16',
            'in_type_src1': 'bf16',
            'out_type': 'fp32',
            'm': 16,
            'n': 16,
            'k': 16,
            'blocks': 1,
            'cycles': 16,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': False,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_smfmac_f32_16x16x32_f16': {
            'arch': 'cdna3',
            'opcode': 98,
            'in_type': 'fp16',
            'in_type_src1': 'fp16',
            'out_type': 'fp32',
            'm': 16,
            'n': 16,
            'k': 32,
            'blocks': 1,
            'cycles': 16,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': False,
            'cbsz_abid': True,
            'sparse': True,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 8
        },
        'v_smfmac_f32_32x32x16_f16': {
            'arch': 'cdna3',
            'opcode': 100,
            'in_type': 'fp16',
            'in_type_src1': 'fp16',
            'out_type': 'fp32',
            'm': 32,
            'n': 32,
            'k': 16,
            'blocks': 1,
            'cycles': 32,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': False,
            'cbsz_abid': True,
            'sparse': True,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 8
        },
        'v_smfmac_f32_16x16x32_bf16': {
            'arch': 'cdna3',
            'opcode': 102,
            'in_type': 'bf16',
            'in_type_src1': 'bf16',
            'out_type': 'fp32',
            'm': 16,
            'n': 16,
            'k': 32,
            'blocks': 1,
            'cycles': 16,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': False,
            'cbsz_abid': True,
            'sparse': True,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 8
        },
        'v_smfmac_f32_32x32x16_bf16': {
            'arch': 'cdna3',
            'opcode': 104,
            'in_type': 'bf16',
            'in_type_src1': 'bf16',
            'out_type': 'fp32',
            'm': 32,
            'n': 32,
            'k': 16,
            'blocks': 1,
            'cycles': 32,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': False,
            'cbsz_abid': True,
            'sparse': True,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 8
        },
        'v_smfmac_i32_16x16x64_i8': {
            'arch': 'cdna3',
            'opcode': 106,
            'in_type': 'int8',
            'in_type_src1': 'int8',
            'out_type': 'int32',
            'm': 16,
            'n': 16,
            'k': 64,
            'blocks': 1,
            'cycles': 16,
            'integer': True,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': False,
            'cbsz_abid': True,
            'sparse': True,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 8
        },
        'v_smfmac_i32_32x32x32_i8': {
            'arch': 'cdna3',
            'opcode': 108,
            'in_type': 'int8',
            'in_type_src1': 'int8',
            'out_type': 'int32',
            'm': 32,
            'n': 32,
            'k': 32,
            'blocks': 1,
            'cycles': 32,
            'integer': True,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': False,
            'cbsz_abid': True,
            'sparse': True,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 8
        },
        'v_mfma_f64_16x16x4_f64': {
            'arch': 'cdna3',
            'opcode': 110,
            'in_type': 'fp64',
            'in_type_src1': 'fp64',
            'out_type': 'fp64',
            'm': 16,
            'n': 16,
            'k': 4,
            'blocks': 1,
            'cycles': 32,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': True,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': False,
            'coexec_delay': -1
        },
        'v_mfma_f64_4x4x4_4b_f64': {
            'arch': 'cdna3',
            'opcode': 111,
            'in_type': 'fp64',
            'in_type_src1': 'fp64',
            'out_type': 'fp64',
            'm': 4,
            'n': 4,
            'k': 4,
            'blocks': 4,
            'cycles': 16,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': True,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': False,
            'coexec_delay': -1
        },
        'v_mfma_f32_16x16x32_bf8_bf8': {
            'arch': 'cdna3',
            'opcode': 112,
            'in_type': 'amd_bf8',
            'in_type_src1': 'amd_bf8',
            'out_type': 'fp32',
            'm': 16,
            'n': 16,
            'k': 32,
            'blocks': 1,
            'cycles': 16,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': False,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_mfma_f32_16x16x32_bf8_fp8': {
            'arch': 'cdna3',
            'opcode': 113,
            'in_type': 'amd_bf8',
            'in_type_src1': 'amd_fp8',
            'out_type': 'fp32',
            'm': 16,
            'n': 16,
            'k': 32,
            'blocks': 1,
            'cycles': 16,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': False,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_mfma_f32_16x16x32_fp8_bf8': {
            'arch': 'cdna3',
            'opcode': 114,
            'in_type': 'amd_fp8',
            'in_type_src1': 'amd_bf8',
            'out_type': 'fp32',
            'm': 16,
            'n': 16,
            'k': 32,
            'blocks': 1,
            'cycles': 16,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': False,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_mfma_f32_16x16x32_fp8_fp8': {
            'arch': 'cdna3',
            'opcode': 115,
            'in_type': 'amd_fp8',
            'in_type_src1': 'amd_fp8',
            'out_type': 'fp32',
            'm': 16,
            'n': 16,
            'k': 32,
            'blocks': 1,
            'cycles': 16,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': False,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_mfma_f32_32x32x16_bf8_bf8': {
            'arch': 'cdna3',
            'opcode': 116,
            'in_type': 'amd_bf8',
            'in_type_src1': 'amd_bf8',
            'out_type': 'fp32',
            'm': 32,
            'n': 32,
            'k': 16,
            'blocks': 1,
            'cycles': 32,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': False,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_mfma_f32_32x32x16_bf8_fp8': {
            'arch': 'cdna3',
            'opcode': 117,
            'in_type': 'amd_bf8',
            'in_type_src1': 'amd_fp8',
            'out_type': 'fp32',
            'm': 32,
            'n': 32,
            'k': 16,
            'blocks': 1,
            'cycles': 32,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': False,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_mfma_f32_32x32x16_fp8_bf8': {
            'arch': 'cdna3',
            'opcode': 118,
            'in_type': 'amd_fp8',
            'in_type_src1': 'amd_bf8',
            'out_type': 'fp32',
            'm': 32,
            'n': 32,
            'k': 16,
            'blocks': 1,
            'cycles': 32,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': False,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_mfma_f32_32x32x16_fp8_fp8': {
            'arch': 'cdna3',
            'opcode': 119,
            'in_type': 'amd_fp8',
            'in_type_src1': 'amd_fp8',
            'out_type': 'fp32',
            'm': 32,
            'n': 32,
            'k': 16,
            'blocks': 1,
            'cycles': 32,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': False,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 4
        },
        'v_smfmac_f32_16x16x64_bf8_bf8': {
            'arch': 'cdna3',
            'opcode': 120,
            'in_type': 'amd_bf8',
            'in_type_src1': 'amd_bf8',
            'out_type': 'fp32',
            'm': 16,
            'n': 16,
            'k': 64,
            'blocks': 1,
            'cycles': 16,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': False,
            'cbsz_abid': True,
            'sparse': True,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 8
        },
        'v_smfmac_f32_16x16x64_bf8_fp8': {
            'arch': 'cdna3',
            'opcode': 121,
            'in_type': 'amd_bf8',
            'in_type_src1': 'amd_fp8',
            'out_type': 'fp32',
            'm': 16,
            'n': 16,
            'k': 64,
            'blocks': 1,
            'cycles': 16,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': False,
            'cbsz_abid': True,
            'sparse': True,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 8
        },
        'v_smfmac_f32_16x16x64_fp8_bf8': {
            'arch': 'cdna3',
            'opcode': 122,
            'in_type': 'amd_fp8',
            'in_type_src1': 'amd_bf8',
            'out_type': 'fp32',
            'm': 16,
            'n': 16,
            'k': 64,
            'blocks': 1,
            'cycles': 16,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': False,
            'cbsz_abid': True,
            'sparse': True,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 8
        },
        'v_smfmac_f32_16x16x64_fp8_fp8': {
            'arch': 'cdna3',
            'opcode': 123,
            'in_type': 'amd_fp8',
            'in_type_src1': 'amd_fp8',
            'out_type': 'fp32',
            'm': 16,
            'n': 16,
            'k': 64,
            'blocks': 1,
            'cycles': 16,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': False,
            'cbsz_abid': True,
            'sparse': True,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 8
        },
        'v_smfmac_f32_32x32x32_bf8_bf8': {
            'arch': 'cdna3',
            'opcode': 124,
            'in_type': 'amd_bf8',
            'in_type_src1': 'amd_bf8',
            'out_type': 'fp32',
            'm': 32,
            'n': 32,
            'k': 32,
            'blocks': 1,
            'cycles': 32,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': False,
            'cbsz_abid': True,
            'sparse': True,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 8
        },
        'v_smfmac_f32_32x32x32_bf8_fp8': {
            'arch': 'cdna3',
            'opcode': 125,
            'in_type': 'amd_bf8',
            'in_type_src1': 'amd_fp8',
            'out_type': 'fp32',
            'm': 32,
            'n': 32,
            'k': 32,
            'blocks': 1,
            'cycles': 32,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': False,
            'cbsz_abid': True,
            'sparse': True,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 8
        },
        'v_smfmac_f32_32x32x32_fp8_bf8': {
            'arch': 'cdna3',
            'opcode': 126,
            'in_type': 'amd_fp8',
            'in_type_src1': 'amd_bf8',
            'out_type': 'fp32',
            'm': 32,
            'n': 32,
            'k': 32,
            'blocks': 1,
            'cycles': 32,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': False,
            'cbsz_abid': True,
            'sparse': True,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 8
        },
        'v_smfmac_f32_32x32x32_fp8_fp8': {
            'arch': 'cdna3',
            'opcode': 127,
            'in_type': 'amd_fp8',
            'in_type_src1': 'amd_fp8',
            'out_type': 'fp32',
            'm': 32,
            'n': 32,
            'k': 32,
            'blocks': 1,
            'cycles': 32,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 8,
            'blgp': False,
            'cbsz_abid': True,
            'sparse': True,
            'cd_opsel': False,
            'neg': False,
            'coexec': True,
            'coexec_delay': 8
        }
    },
    'rdna3': {
        'v_wmma_f32_16x16x16_f16': {
            'arch': 'rdna3',
            'opcode': 64,
            'in_type': 'fp16',
            'in_type_src1': 'fp16',
            'out_type': 'fp32',
            'm': 16,
            'n': 16,
            'k': 16,
            'blocks': 1,
            'cycles': 32,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 4,
            'blgp': False,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': True,
            'coexec': False,
            'coexec_delay': -1
        },
        'v_wmma_f32_16x16x16_bf16': {
            'arch': 'rdna3',
            'opcode': 65,
            'in_type': 'bf16',
            'in_type_src1': 'bf16',
            'out_type': 'fp32',
            'm': 16,
            'n': 16,
            'k': 16,
            'blocks': 1,
            'cycles': 32,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 4,
            'blgp': False,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': True,
            'coexec': False,
            'coexec_delay': -1
        },
        'v_wmma_f16_16x16x16_f16': {
            'arch': 'rdna3',
            'opcode': 66,
            'in_type': 'fp16',
            'in_type_src1': 'fp16',
            'out_type': 'fp16',
            'm': 16,
            'n': 16,
            'k': 16,
            'blocks': 1,
            'cycles': 32,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 4,
            'blgp': False,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': True,
            'neg': True,
            'coexec': False,
            'coexec_delay': -1
        },
        'v_wmma_bf16_16x16x16_bf16': {
            'arch': 'rdna3',
            'opcode': 67,
            'in_type': 'bf16',
            'in_type_src1': 'bf16',
            'out_type': 'bf16',
            'm': 16,
            'n': 16,
            'k': 16,
            'blocks': 1,
            'cycles': 32,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 4,
            'blgp': False,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': True,
            'neg': True,
            'coexec': False,
            'coexec_delay': -1
        },
        'v_wmma_i32_16x16x16_iu8': {
            'arch': 'rdna3',
            'opcode': 68,
            'in_type': 'iu8',
            'in_type_src1': 'iu8',
            'out_type': 'int32',
            'm': 16,
            'n': 16,
            'k': 16,
            'blocks': 1,
            'cycles': 32,
            'integer': True,
            'c_d_arch': True,
            'gpr_byte_align': 4,
            'blgp': False,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': True,
            'coexec': False,
            'coexec_delay': -1
        },
        'v_wmma_i32_16x16x16_iu4': {
            'arch': 'rdna3',
            'opcode': 69,
            'in_type': 'iu4',
            'in_type_src1': 'iu4',
            'out_type': 'int32',
            'm': 16,
            'n': 16,
            'k': 16,
            'blocks': 1,
            'cycles': 16,
            'integer': True,
            'c_d_arch': True,
            'gpr_byte_align': 4,
            'blgp': False,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': True,
            'coexec': False,
            'coexec_delay': -1
        }
    },
    'rdna4': {
        'v_wmma_f32_16x16x16_f16': {
            'arch': 'rdna4',
            'opcode': 64,
            'in_type': 'fp16',
            'in_type_src1': 'fp16',
            'out_type': 'fp32',
            'm': 16,
            'n': 16,
            'k': 16,
            'blocks': 1,
            'cycles': 16,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 4,
            'blgp': False,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': True,
            'coexec': False,
            'coexec_delay': -1
        },
        'v_wmma_f32_16x16x16_bf16': {
            'arch': 'rdna4',
            'opcode': 65,
            'in_type': 'bf16',
            'in_type_src1': 'bf16',
            'out_type': 'fp32',
            'm': 16,
            'n': 16,
            'k': 16,
            'blocks': 1,
            'cycles': 16,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 4,
            'blgp': False,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': True,
            'coexec': False,
            'coexec_delay': -1
        },
        'v_wmma_f16_16x16x16_f16': {
            'arch': 'rdna4',
            'opcode': 66,
            'in_type': 'fp16',
            'in_type_src1': 'fp16',
            'out_type': 'fp16',
            'm': 16,
            'n': 16,
            'k': 16,
            'blocks': 1,
            'cycles': 16,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 4,
            'blgp': False,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': True,
            'coexec': False,
            'coexec_delay': -1
        },
        'v_wmma_bf16_16x16x16_bf16': {
            'arch': 'rdna4',
            'opcode': 67,
            'in_type': 'bf16',
            'in_type_src1': 'bf16',
            'out_type': 'bf16',
            'm': 16,
            'n': 16,
            'k': 16,
            'blocks': 1,
            'cycles': 16,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 4,
            'blgp': False,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': True,
            'coexec': False,
            'coexec_delay': -1
        },
        'v_wmma_i32_16x16x16_iu8': {
            'arch': 'rdna4',
            'opcode': 68,
            'in_type': 'iu8',
            'in_type_src1': 'iu8',
            'out_type': 'int32',
            'm': 16,
            'n': 16,
            'k': 16,
            'blocks': 1,
            'cycles': 8,
            'integer': True,
            'c_d_arch': True,
            'gpr_byte_align': 4,
            'blgp': False,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': True,
            'coexec': False,
            'coexec_delay': -1
        },
        'v_wmma_i32_16x16x16_iu4': {
            'arch': 'rdna4',
            'opcode': 69,
            'in_type': 'iu4',
            'in_type_src1': 'iu4',
            'out_type': 'int32',
            'm': 16,
            'n': 16,
            'k': 16,
            'blocks': 1,
            'cycles': 8,
            'integer': True,
            'c_d_arch': True,
            'gpr_byte_align': 4,
            'blgp': False,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': True,
            'coexec': False,
            'coexec_delay': -1
        },
        'v_wmma_i32_16x16x32_iu4': {
            'arch': 'rdna4',
            'opcode': 74,
            'in_type': 'iu4',
            'in_type_src1': 'iu4',
            'out_type': 'int32',
            'm': 16,
            'n': 16,
            'k': 32,
            'blocks': 1,
            'cycles': 8,
            'integer': True,
            'c_d_arch': True,
            'gpr_byte_align': 4,
            'blgp': False,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': True,
            'coexec': False,
            'coexec_delay': -1
        },
        'v_wmma_f32_16x16x16_fp8_fp8': {
            'arch': 'rdna4',
            'opcode': 70,
            'in_type': 'fp8',
            'in_type_src1': 'fp8',
            'out_type': 'fp32',
            'm': 16,
            'n': 16,
            'k': 16,
            'blocks': 1,
            'cycles': 8,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 4,
            'blgp': False,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': False,
            'coexec_delay': -1
        },
        'v_wmma_f32_16x16x16_fp8_bf8': {
            'arch': 'rdna4',
            'opcode': 71,
            'in_type': 'fp8',
            'in_type_src1': 'bf8',
            'out_type': 'fp32',
            'm': 16,
            'n': 16,
            'k': 16,
            'blocks': 1,
            'cycles': 8,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 4,
            'blgp': False,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': False,
            'coexec_delay': -1
        },
        'v_wmma_f32_16x16x16_bf8_fp8': {
            'arch': 'rdna4',
            'opcode': 72,
            'in_type': 'bf8',
            'in_type_src1': 'fp8',
            'out_type': 'fp32',
            'm': 16,
            'n': 16,
            'k': 16,
            'blocks': 1,
            'cycles': 8,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 4,
            'blgp': False,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': False,
            'coexec_delay': -1
        },
        'v_wmma_f32_16x16x16_bf8_bf8': {
            'arch': 'rdna4',
            'opcode': 73,
            'in_type': 'bf8',
            'in_type_src1': 'bf8',
            'out_type': 'fp32',
            'm': 16,
            'n': 16,
            'k': 16,
            'blocks': 1,
            'cycles': 8,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 4,
            'blgp': False,
            'cbsz_abid': False,
            'sparse': False,
            'cd_opsel': False,
            'neg': False,
            'coexec': False,
            'coexec_delay': -1
        },
        'v_swmmac_f32_16x16x32_f16': {
            'arch': 'rdna4',
            'opcode': 80,
            'in_type': 'fp16',
            'in_type_src1': 'fp16',
            'out_type': 'fp32',
            'm': 16,
            'n': 16,
            'k': 32,
            'blocks': 1,
            'cycles': 16,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 4,
            'blgp': False,
            'cbsz_abid': False,
            'sparse': True,
            'cd_opsel': False,
            'neg': True,
            'coexec': False,
            'coexec_delay': -1
        },
        'v_swmmac_f32_16x16x32_bf16': {
            'arch': 'rdna4',
            'opcode': 81,
            'in_type': 'bf16',
            'in_type_src1': 'bf16',
            'out_type': 'fp32',
            'm': 16,
            'n': 16,
            'k': 32,
            'blocks': 1,
            'cycles': 16,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 4,
            'blgp': False,
            'cbsz_abid': False,
            'sparse': True,
            'cd_opsel': False,
            'neg': True,
            'coexec': False,
            'coexec_delay': -1
        },
        'v_swmmac_f16_16x16x32_f16': {
            'arch': 'rdna4',
            'opcode': 82,
            'in_type': 'fp16',
            'in_type_src1': 'fp16',
            'out_type': 'fp16',
            'm': 16,
            'n': 16,
            'k': 32,
            'blocks': 1,
            'cycles': 16,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 4,
            'blgp': False,
            'cbsz_abid': False,
            'sparse': True,
            'cd_opsel': False,
            'neg': True,
            'coexec': False,
            'coexec_delay': -1
        },
        'v_swmmac_bf16_16x16x32_bf16': {
            'arch': 'rdna4',
            'opcode': 83,
            'in_type': 'bf16',
            'in_type_src1': 'bf16',
            'out_type': 'bf16',
            'm': 16,
            'n': 16,
            'k': 32,
            'blocks': 1,
            'cycles': 16,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 4,
            'blgp': False,
            'cbsz_abid': False,
            'sparse': True,
            'cd_opsel': False,
            'neg': True,
            'coexec': False,
            'coexec_delay': -1
        },
        'v_swmmac_i32_16x16x32_iu8': {
            'arch': 'rdna4',
            'opcode': 84,
            'in_type': 'iu8',
            'in_type_src1': 'iu8',
            'out_type': 'int32',
            'm': 16,
            'n': 16,
            'k': 32,
            'blocks': 1,
            'cycles': 8,
            'integer': True,
            'c_d_arch': True,
            'gpr_byte_align': 4,
            'blgp': False,
            'cbsz_abid': False,
            'sparse': True,
            'cd_opsel': False,
            'neg': True,
            'coexec': False,
            'coexec_delay': -1
        },
        'v_swmmac_i32_16x16x32_iu4': {
            'arch': 'rdna4',
            'opcode': 85,
            'in_type': 'iu4',
            'in_type_src1': 'iu4',
            'out_type': 'int32',
            'm': 16,
            'n': 16,
            'k': 32,
            'blocks': 1,
            'cycles': 8,
            'integer': True,
            'c_d_arch': True,
            'gpr_byte_align': 4,
            'blgp': False,
            'cbsz_abid': False,
            'sparse': True,
            'cd_opsel': False,
            'neg': True,
            'coexec': False,
            'coexec_delay': -1
        },
        'v_swmmac_i32_16x16x64_iu4': {
            'arch': 'rdna4',
            'opcode': 86,
            'in_type': 'iu4',
            'in_type_src1': 'iu4',
            'out_type': 'int32',
            'm': 16,
            'n': 16,
            'k': 64,
            'blocks': 1,
            'cycles': 8,
            'integer': True,
            'c_d_arch': True,
            'gpr_byte_align': 4,
            'blgp': False,
            'cbsz_abid': False,
            'sparse': True,
            'cd_opsel': False,
            'neg': True,
            'coexec': False,
            'coexec_delay': -1
        },
        'v_swmmac_f32_16x16x32_fp8_fp8': {
            'arch': 'rdna4',
            'opcode': 87,
            'in_type': 'fp8',
            'in_type_src1': 'fp8',
            'out_type': 'fp32',
            'm': 16,
            'n': 16,
            'k': 32,
            'blocks': 1,
            'cycles': 8,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 4,
            'blgp': False,
            'cbsz_abid': False,
            'sparse': True,
            'cd_opsel': False,
            'neg': False,
            'coexec': False,
            'coexec_delay': -1
        },
        'v_swmmac_f32_16x16x32_fp8_bf8': {
            'arch': 'rdna4',
            'opcode': 88,
            'in_type': 'fp8',
            'in_type_src1': 'bf8',
            'out_type': 'fp32',
            'm': 16,
            'n': 16,
            'k': 32,
            'blocks': 1,
            'cycles': 8,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 4,
            'blgp': False,
            'cbsz_abid': False,
            'sparse': True,
            'cd_opsel': False,
            'neg': False,
            'coexec': False,
            'coexec_delay': -1
        },
        'v_swmmac_f32_16x16x32_bf8_fp8': {
            'arch': 'rdna4',
            'opcode': 89,
            'in_type': 'bf8',
            'in_type_src1': 'fp8',
            'out_type': 'fp32',
            'm': 16,
            'n': 16,
            'k': 32,
            'blocks': 1,
            'cycles': 8,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 4,
            'blgp': False,
            'cbsz_abid': False,
            'sparse': True,
            'cd_opsel': False,
            'neg': False,
            'coexec': False,
            'coexec_delay': -1
        },
        'v_swmmac_f32_16x16x32_bf8_bf8': {
            'arch': 'rdna4',
            'opcode': 90,
            'in_type': 'bf8',
            'in_type_src1': 'bf8',
            'out_type': 'fp32',
            'm': 16,
            'n': 16,
            'k': 32,
            'blocks': 1,
            'cycles': 8,
            'integer': False,
            'c_d_arch': True,
            'gpr_byte_align': 4,
            'blgp': False,
            'cbsz_abid': False,
            'sparse': True,
            'cd_opsel': False,
            'neg': False,
            'coexec': False,
            'coexec_delay': -1
        }
    }
}

def is_gfx9_arch(inst_info: MatrixInstruction) -> bool:
    """ Queries if an instruction is in the gfx9 architecture.

    Args:
        inst_info: MatrixInstruction that holds the details of a matrix
            multiplication instruction.

    Returns:
        A boolean that is True when the architecture for the input MatrixInstruction is from
        a gfx9 architecture (i.e., CDNA1, CDNA2, or CDNA3).
        The boolean will be False if the MatrixInstruction is from a non-gfx9 architecture.
    """
    return inst_info['arch'] in ('cdna1', 'cdna2', 'cdna3')

def is_gfx11_arch(inst_info: MatrixInstruction) -> bool:
    """ Queries if an instruction is in the gfx11 architecture.

    Args:
        inst_info: MatrixInstruction that holds the details of a matrix
            multiplication instruction.

    Returns:
        A boolean that is True when the architecture for the input MatrixInstruction is from
        a gfx11 architecture (i.e., RDNA3).
        The boolean will be False if the MatrixInstruction is from a non-gfx11 architecture.
    """
    return inst_info['arch'] == 'rdna3'

def is_gfx12_arch(inst_info: MatrixInstruction) -> bool:
    """ Queries if an instruction is in the gfx12 architecture.

    Args:
        inst_info: MatrixInstruction that holds the details of a matrix
            multiplication instruction.

    Returns:
        A boolean that is True when the architecture for the input MatrixInstruction is from
        a gfx12 architecture (i.e., RDNA4).
        The boolean will be False if the MatrixInstruction is from a non-gfx12 architecture.
    """
    return inst_info['arch'] == 'rdna4'

def print_instructions(arch: str, instructions: Dict[str, MatrixInstruction],
                       to_print: TextIO) -> None:
    """ Prints all of the instructions available for the chosen architecture.

    Args:
        arch: string that contains the accelerator architecture's name to print
        instructions: dictionary of [mnemonic, instruction_details] pairs for
            the instructions available in this architecture. All of the mnemonics
            will be printed by this function.
        to_print: file to print instruction to.

    Returns:
        None
    """
    print(f"Available instructions in the {arch.upper()} architecture:", file=to_print)
    for inst in instructions:
        print("    " + inst, file=to_print)

def print_arch_inst(arch: str, inst_name: str) -> None:
    """ Prints the architecture and instruction strings in upper-case.

    Args:
        arch: string that contains the accelerator architecture's name to print
        inst_name: string that contains the instruction's name to print

    Returns:
        None
    """
    print(f"Architecture: {arch.upper()}")
    print(f"Instruction: {inst_name.upper()}", flush=True)

def get_data_size(data_type: str) -> int:
    """ Returns the size of a data type, in bits.

    Args:
        data_type: string that contains the data type, from the dict_math_types dictionary keys

    Returns:
        An integer value that describes the size of one unit of the data type, in bits
    """
    return dict_math_types[data_type]['size']

def get_type_desc(data_type: str) -> str:
    """ Returns the pretty print string that describes a data type.

    Args:
        data_type: string that contains the data type, from the dict_math_types dictionary keys

    Returns:
        A string that describes the data type in a form that can be printed to the screen
    """
    return dict_math_types[data_type]['description']

def check_matrix_support(matrix: str, supported_list: Tuple[str, ...], func_name: str) -> None:
    """ Raises a ValueError if the requested matrix is not in the supported list """
    if matrix.lower() not in supported_list:
        err_str = f"Input matrix format {matrix.lower()} is not supported by "
        err_str += f"{func_name}.{extract_stack(limit=2)[-2][2]}()"
        raise ValueError(err_str)

# Disabling the check for too many returns in this function, because we have a large
# number of possible times we want to return on error. Rather than refactor this into
# a huge number of single-use functions all for testing individual arguments, just
# allow us to return on error. That yields many returns.
#pylint: disable=too-many-return-statements
def parse_and_run() -> int:
    """ Parses command line arguments and runs matrix calculations.

    Parses the command-line arguments for this script, then run the requested functions.
    Prints help about the application if requested, or if bad arguments are passed in.
    Checks to make sure the arguments are valid  respect to one another and gives
    recommendations if they are not.

    Returns:
        An integer to pass back to the command-line caller of this application
        0 means that things passed successfully
        -1 means there was an internal error in the application
        -2 means there was a bad command line parameter passed to the application
    """
    invert_isa: Dict[str, List[str]] = {}
    for key, value in dict_isas.items():
        invert_isa.setdefault(value, []).append(key)
    list_of_archs = ""
    for arch in dict_insts:
        list_of_archs += arch.upper()
        alt_name = "    Alternately: "
        alt_list = ", ".join([x.upper() for x in invert_isa[arch] if x != arch])
        wrapped_alt_list = wrap(alt_list, 50)
        list_of_archs += '\n' + alt_name + f'\n{alt_name}'.join(wrapped_alt_list) + "\n"
    parser = argparse.ArgumentParser(
                description='\n'.join(
                    wrap('This tool will generate information about the register layout for '
                         'matrix multiplication instructions on AMD accelerators, '
                         'including those built from the following architectures: '
                         + ", ".join(dict_insts.keys()).upper(),
                         80)) +
                '\n\n' +
                '\n'.join(wrap('There are five options for each matrix multiplication '
                               'instruction:')) + '\n' +
                '\n'.join(wrap('- Print general information about the instruction, '
                               'such as its number of registers, computational '
                               'throughput, and co-execution capabilities '
                               '(--detail-instruction)')) + '\n' +
                '\n'.join(wrap('- Print the register and lane for a '
                               'user-chosen A[], B[], C[], or D[] matrix '
                               'entry (--get-register)')) + '\n' +
                '\n'.join(wrap('- Print the A[], B[], C[], or D[] matrix '
                               'entry for a chosen combination of register '
                               'and lane (--matrix-entry)')) + '\n' +
                '\n'.join(wrap('- Print the register and lane '
                               'combinations for an entire A[], B[], C[], '
                               'or D[] matrix (--register-layout)')) + '\n' +
                '\n'.join(wrap('- Print the A[], B[], C[], or D[] matrix entries '
                               "for all of the instructions' registers and lanes "
                               '(--matrix-layout)')) +
                '\n\n' +
                '\n'.join(wrap('4:2 Sparse matrix instructions (SMFMAC) also have a K[] matrix. '
                               'This is the compression index input, which is put into SrcC in '
                               'place of the C[] matrix. This register is legal for the last four '
                               'options (--get-register, --matrix-entry, --register-layout, '
                               '--matrix-layout), but only for instructions that are sparse.')),
                formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-v', '--version', action='store_true', dest='print_version',
                        help='Print the version of this tool')
    parser.add_argument('-a', '--architecture', action='store',
                        dest='arch', default=None, nargs='?',
                        help='\n'.join(
                            wrap('AMD accelerator architecture or chip against which to query ' +
                                 'the instructions, registers, and matrix layouts. ' +
                                 'Valid options are:')) + '\n' + list_of_archs)
    parser.add_argument('-i', '--instruction', action='store',
                        dest='instruction', default=None, nargs='?',
                        help='\n'.join(
                            wrap('Opcode to query. See the --list-instructions parameter to ' +
                                 'show the legal instructions for the chosen architecture')))
    parser.add_argument('-L', '--list-instructions', '--list_instructions', action='store_true',
                        dest='list_instructions',
                        help='Print available instructions in desired architecture')
    parser.add_argument('-d', '--detail-instruction', '--detail_instruction', action='store_true',
                        dest='detail_instruction',
                        help='Print detailed information about the chosen instruction')
    parser.add_argument('-A', '--A-matrix', '--A_matrix', action='store_true', dest='A_matrix',
                        help='Query information about the A[] matrix')
    parser.add_argument('-B', '--B-matrix', '--B_matrix', action='store_true', dest='B_matrix',
                        help='Query information about the B[] matrix')
    parser.add_argument('-C', '--C-matrix', '--C_matrix', action='store_true', dest='C_matrix',
                        help='Query information about the C[] matrix')
    parser.add_argument('-D', '--D-matrix', '--D_matrix', action='store_true',
                        dest='D_matrix',
                        help='Query information about the D[] matrix')
    parser.add_argument('-k', '--compression', '--compression', action='store_true',
                        dest='compression',
                        help="Query information about spasrse instructions' compression indices")
    parser.add_argument('--cbsz', action='store', metavar="#", dest='cbsz', default='0', nargs='?',
                        help='When querying the A matrix, set the CBSZ control field')
    parser.add_argument('--abid', action='store', metavar="#", dest='abid', default='0', nargs='?',
                        help='When querying the A matrix, set the ABID broadcast field')
    parser.add_argument('--blgp', action='store', metavar="#", dest='blgp', default='0', nargs='?',
                        help='When querying the B matrix, set the BLGP broadcast field')
    parser.add_argument('--opsel', action='store', metavar="#", dest='opsel', default='0',
                        nargs='?',
                        help='When querying the C or D matrix, set the OPSEL field')
    parser.add_argument('--neg', action='store', metavar="#", dest='neg', default='0', nargs='?',
                        help='When querying the A, B, or C matrices, set the NEG field')
    parser.add_argument('--neg_hi', action='store', metavar="#", dest='neg_hi', default='0',
                        nargs='?',
                        help='When querying the A, B, or C matrices, set the NEG_HI field')
    parser.add_argument('-w', '--wavefront', action='store', metavar="32/64",
                        dest='wavefront', default='0', nargs='?',
                        help="Set the wavefront width on architectures that allow multiple widths")
    parser.add_argument('-g', '--get-register', '--get_register', action='store_true',
                        dest='get_register',
                        help='Print a register and lane for a particular matrix entry')
    parser.add_argument('-I', '--I-coordinate', '--I_coordinate', action='store', metavar="#",
                        dest="I_coordinate", default='0', nargs='?',
                        help='When printing a single register, the I coordinate within a row')
    parser.add_argument('-J', '--J-coordinate', '--J_coordinate', action='store', metavar="#",
                        dest="J_coordinate", default='0', nargs='?',
                        help='When printing a single register, the J coordinate within a column')
    parser.add_argument('-K', '--K-coordinate', '--K_coordinate', action='store', metavar="#",
                        dest="K_coordinate", default='0', nargs='?',
                        help='When printing a single register, the K coordinate')
    parser.add_argument('-b', '--block', action='store', metavar="#", dest='block', default='0',
                        nargs='?',
                        help='When printing a single register, the block')
    parser.add_argument('-m', '--matrix-entry', '--matrix_entry', action='store_true',
                        dest='matrix_entry',
                        help='Print the block and I/J/K coordinates for a lane/register')
    parser.add_argument('-r', '--register', action='store', metavar="#", dest='register',
                        default='0', nargs='?',
                        help='When printing the matrix location, the register to query')
    parser.add_argument('-l', '--lane', action='store', metavar="#", dest='lane', default='0',
                        nargs='?',
                        help='When printing the matrix location, the wavefront lane to query')
    parser.add_argument('-o', '--output-calculation', '--output_calculation', action='store_true',
                        dest='output_calc',
                        help='\n'.join(
                            wrap('When printing the matrix element or register information for ' +
                                 'the D output matrix, also print the matrix location or the ' +
                                 'register information for the input matrices that are used ' +
                                 'to calculate this output.')))
    parser.add_argument('-R', '--register-layout', '--register_layout', action='store_true',
                        dest='register_layout',
                        help='Print the register/lane needed for the entire matrix')
    parser.add_argument('-M', '--matrix-layout', '--matrix_layout', action='store_true',
                        dest='matrix_layout',
                        help='Print the matrix entries stored in all registers/lanes')
    parser.add_argument('-c', '--csv', action='store_true', dest='csv',
                        help='Print register usage or matrix layout as a CSV instead of a table')
    parser.add_argument('--markdown', action='store_true', dest='md',
                        help='Print register usage or matrix layout as a Markdown table')
    parser.add_argument('--asciidoc', action='store_true', dest='ad',
                        help='Print register usage or matrix layout as an AsciiDoc table')
    parser.add_argument('--transpose', action='store_true', dest='transpose',
                        help='When displaying a register or matrix layout, transpose the output')
    args = parser.parse_args()

    if args.print_version:
        print(f"AMD Matrix Instruction Calculator Version {VERSION}")
        return 0

    if args.arch is None:
        parser.error('"--architecture" argument required. Please choose between: ' +
                     ', '.join(dict_insts.keys()).upper())

    arch_to_use = str(args.arch).lower()
    if arch_to_use not in dict_isas:
        print(f"Desired architecture, '{args.arch}', is not supported.", file=sys.stderr)
        print("Please choose between: ", end="", file=sys.stderr)
        print(f"{', '.join(dict_insts.keys()).upper()} ", end="", file=sys.stderr)
        print("or their alternative names.", file=sys.stderr)
        return -2
    arch_to_use = dict_isas[arch_to_use]

    if args.list_instructions:
        print_instructions(arch_to_use, dict_insts[arch_to_use], sys.stdout)
        return 0

    if args.instruction is None:
        parser.error('"--instruction" argument required.')
    inst_to_use = str(args.instruction).lower()
    if inst_to_use not in dict_insts[arch_to_use]:
        print(f"Instruction '{args.instruction}' is not supported ", end="", file=sys.stderr)
        print("in the requested architecture.", file=sys.stderr)
        print_instructions(arch_to_use, dict_insts[arch_to_use], sys.stderr)
        return -2

    inst_info = dict_insts[arch_to_use][inst_to_use]

    if args.wavefront is None:
        parser.error('"--wavefront" argument required.')
    if not args.wavefront.isdigit():
        parser.error('"--wavefront" argument must be a positive integer >= 0.')
    # Set default wavefront width
    if int(args.wavefront) == 0:
        if is_gfx9_arch(inst_info):
            args.wavefront = 64
        else:
            args.wavefront = 32
    if (is_gfx9_arch(inst_info) and int(args.wavefront) != 64):
        parser.error(f'"--wavefront" may only be set to 64 on {arch_to_use.upper()}.')
    elif not is_gfx9_arch(inst_info):
        if (int(args.wavefront) != 32 and int(args.wavefront) != 64):
            parser.error(f'"--wavefront" may only be set to 32 or 64 on {arch_to_use.upper()}.')

    calc: InstCalc
    if is_gfx9_arch(inst_info):
        calc = InstCalcGfx9(inst_to_use, inst_info, int(args.wavefront))
    elif is_gfx11_arch(inst_info):
        calc = InstCalcGfx11(inst_to_use, inst_info, int(args.wavefront))
    else:
        calc = InstCalcGfx12(inst_to_use, inst_info, int(args.wavefront))

    if args.detail_instruction:
        calc.print_instruction_information()
        return 0

    if args.I_coordinate is None:
        parser.error('"--I-coordinate" argument required.')
    if not args.I_coordinate.isdigit():
        parser.error('"--I-coordinate" argument must be a positive integer >= 0.')
    if args.J_coordinate is None:
        parser.error('"--J-coordinate" argument required.')
    if not args.J_coordinate.isdigit():
        parser.error('"--J-coordinate" argument must be a positive integer >= 0.')
    if args.K_coordinate is None:
        parser.error('"--K-coordinate" argument required.')
    if not args.K_coordinate.isdigit():
        parser.error('"--K-coordinate" argument must be a positive integer >= 0.')
    if args.block is None:
        parser.error('"--block" argument required.')
    if not args.block.isdigit():
        parser.error('"--block" argument must be a positive integer >= 0.')

    if args.register is None:
        parser.error('"--register" argument required.')
    if not args.register.isdigit():
        parser.error('"--register" argument must be a positive integer >= 0.')
    if args.lane is None:
        parser.error('"--lane" argument required.')
    if not args.lane.isdigit():
        parser.error('"--lane" argument must be a positive integer >= 0.')

    if args.cbsz is None:
        parser.error('"--cbsz" argument required.')
    if not args.cbsz.isdigit():
        parser.error('"--cbsz" argument must be a positive integer >= 0.')
    if args.abid is None:
        parser.error('"--abid" argument required.')
    if not args.abid.isdigit():
        parser.error('"--abid" argument must be a positive integer >= 0.')
    if args.blgp is None:
        parser.error('"--blgp" argument required.')
    if not args.blgp.isdigit():
        parser.error('"--blgp" argument must be a positive integer >= 0.')
    if args.opsel is None:
        parser.error('"--opsel" argument required.')
    if not args.opsel.isdigit():
        parser.error('"--opsel" argument must be a positive integer >= 0.')
    if args.neg is None:
        parser.error('"--neg" argument required.')
    if not args.neg.isdigit():
        parser.error('"--neg" argument must be a positive integer >= 0.')
    if args.neg_hi is None:
        parser.error('"--neg_hi" argument required.')
    if not args.neg_hi.isdigit():
        parser.error('"--neg_hi" argument must be a positive integer >= 0.')

    options = [args.get_register, args.matrix_entry, args.register_layout, args.matrix_layout]
    if options.count(True) != 1:
        print("Please choose " + ("only " if options.count(True) > 1 else "") +
              "one of: '--get-register', '--matrix-entry', " +
              "'--register-layout', '--matrix-layout', or '--detail-instruction'", file=sys.stderr)
        return -2

    mats = [args.A_matrix, args.B_matrix, args.C_matrix, args.D_matrix, args.compression]

    mat_names = ['a', 'b', 'c', 'd', 'k']
    matrix_to_use = None
    for (which, name) in zip(mats, mat_names):
        if which:
            matrix_to_use = name

    if matrix_to_use is None or mats.count(True) != 1:
        print("For the chosen option, '", end="", file=sys.stderr)
        if args.get_register:
            print("--get-register", end="", file=sys.stderr)
        elif args.matrix_entry:
            print("--matrix-entry", end="", file=sys.stderr)
        elif args.register_layout:
            print("--register-layout", end="", file=sys.stderr)
        else:
            print("--matrix-layout", end="", file=sys.stderr)
        print("', please choose " + ("only " if mats.count(True) > 1 else "") +
              "one of: '--A-matrix', '--B-matrix', '--C-matrix', '--D-matrix', " +
              "or '--compression'", file=sys.stderr)
        return -2

    if (args.output_calc and matrix_to_use != 'd'):
        print("The option '--output-calculation' is only possible for the D matrix.",
              file=sys.stderr)
        return -2

    # For gfx9 sparse matrices, a zero CBSZ[1:0] causes ABID to choose the compression index
    # set. Technically CBSZ[2] is undefined -- for safety, let us say setting it is illegal.
    # For non-sparse matrices, CBSZ chooses 2^(how many blocks to broadcast into).
    # After that, ABID chooses which block within the 2^(CBSZ) to broadcast to the others.
    if int(args.cbsz) > 0:
        if not inst_info['cbsz_abid']:
            print(f"The chosen instruction, {inst_to_use.upper()}, in ", end="", file=sys.stderr)
            print(f"the {arch_to_use.upper()} architecture, does not ", end="", file=sys.stderr)
            print("support the CBSZ modifier.", file=sys.stderr)
            return -2
        if matrix_to_use not in ('a', 'k'):
            if not (matrix_to_use == 'd' and args.output_calc):
                print("The CBSZ modifier may only be used on the A ", end="", file=sys.stderr)
                print("input matrix, the K compression index register, ", end="", file=sys.stderr)
                print("or the D output matrix when ", end="", file=sys.stderr)
                print("'--output-calculation' is set.", file=sys.stderr)
                return -2
        if inst_info['sparse']:
            max_cbsz = 3
        else:
            max_cbsz = int(math.log(inst_info['blocks'], 2))
            max_cbsz = min(4, max_cbsz)
        if int(args.cbsz) > max_cbsz:
            print("The CBSZ modifier for the instruction ", end="", file=sys.stderr)
            print(f"{inst_to_use.upper()}", end="", file=sys.stderr)
            print(f", in the {arch_to_use.upper()} architecture,", end="", file=sys.stderr)
            print(f" may only contain values between 0 - {max_cbsz}, inclusive.", file=sys.stderr)
            return -2
    # ABID is complicated, because its legal values depend both on whether we are working
    # on a sparse matrix, and on the legal values of CBSZ
    # For sparse matrices on 16b values, a non-zero CBSZ[1:0] causes ABID[1:0] to choose the
    # compression index set. For 8b sparse matrices, a non-zero CBSZ[1:0] causes ABID[0] to
    # choose the compression index set.
    # For non-sparse matrices, the ABID chooses which lane gets broadcast to its aligned
    # 2^(CBSZ) neighbors.
    if int(args.abid) > 0:
        if not inst_info['cbsz_abid']:
            print(f"The chosen instruction, {inst_to_use.upper()}, in ", end="", file=sys.stderr)
            print(f"the {arch_to_use.upper()} architecture, does not ", end="", file=sys.stderr)
            print("support the ABID modifier.", file=sys.stderr)
            return -2
        if matrix_to_use not in ('a', 'k'):
            if not (matrix_to_use == 'd' and args.output_calc):
                print("The ABID modifier may only be used on the A ", end="", file=sys.stderr)
                print("input matrix, the K compression index register, ", end="", file=sys.stderr)
                print("or the D output matrix when '--output-calculation' is set.", file=sys.stderr)
                return -2
        if inst_info['sparse']:
            max_abid = calc.get_num_compression_sets() - 1
        else:
            max_abid = int(math.pow(2, int(args.cbsz))-1)
        if int(args.abid) > max_abid:
            print("The ABID modifier for the instruction ", end="", file=sys.stderr)
            print(f"{inst_to_use.upper()}, in the ", end="", file=sys.stderr)
            print(f"{arch_to_use.upper()} architecture, ", end="", file=sys.stderr)
            print(f"with the CBSZ modifier {args.cbsz},", end="", file=sys.stderr)
            if max_abid != 0:
                print(" may only contain values between ", end="", file=sys.stderr)
                print(f"0 - {max_abid}, inclusive.", file=sys.stderr)
            else:
                print(" may only be set to zero.", file=sys.stderr)
            return -2
    # BLGP chooses between 8 different B matrix lane swizzles or broadcasts. 0 is default.
    # In CDNA3, the three BLGP bits are used to automatically negate the values in
    # matrices A, B, and C, respectively.
    if int(args.blgp) > 0:
        if not inst_info['blgp']:
            print(f"The chosen instruction, {inst_to_use.upper()}, ", end="", file=sys.stderr)
            print(f"in the {arch_to_use.upper()} architecture, does ", end="", file=sys.stderr)
            print("not support the BLGP modifier.", file=sys.stderr)
            return -2
        if not (matrix_to_use == 'd' and args.output_calc):
            if (inst_info['in_type'] != 'fp64' and matrix_to_use != 'b'):
                print("The BLGP modifier may only be used on the B ", end="", file=sys.stderr)
                print("input matrix for the instruction ", end="", file=sys.stderr)
                print(f"{inst_to_use.upper()}, or with the D matrix when ", end="", file=sys.stderr)
                print("'--output-calculation' is set.", file=sys.stderr)
                return -2
            if (inst_info['in_type'] == 'fp64' and matrix_to_use in ('d', 'k')):
                print("The BLGP modifier may only be used on matrices ", end="", file=sys.stderr)
                print("A, B, and C for the instruction ", end="", file=sys.stderr)
                print(f"{inst_to_use.upper()}, or with the D matrix when ", end="", file=sys.stderr)
                print("'--output-calculation' is set.", file=sys.stderr)
                return -2
        if int(args.blgp) > 7:
            print("The BLGP modifier may only contain values between ", end="", file=sys.stderr)
            print("0 - 7, inclusive.", file=sys.stderr)
            return -2
    if int(args.opsel) > 0:
        if is_gfx9_arch(inst_info):
            print(f"The chosen architecture, {arch_to_use.upper()}, ", end="", file=sys.stderr)
            print("does not support the OPSEL modifier.", file=sys.stderr)
            return -2
        if is_gfx11_arch(inst_info):
            # In gfx11 matrix operations with 16b outputs, OPSEL[2] is used to decide whether to
            # use the lower or upper half of each register in C[] and D[]. As such, the only valid
            # values are 0 (OPSEL[2] = 0) and 4 (OPSEL[2] = 1). Also only available on a subset of
            # instructions and architectures.
            if not inst_info['cd_opsel']:
                print(f"The chosen instruction, {inst_to_use.upper()}, ", end="", file=sys.stderr)
                print(f"in the {arch_to_use.upper()} architecture, does ", end="", file=sys.stderr)
                print("not support the OPSEL modifier.", file=sys.stderr)
                return -2
            if matrix_to_use not in ('c', 'd'):
                print("The OPSEL modifier may only be used on matrices C ", end="", file=sys.stderr)
                print(f"and D for the instruction {inst_to_use.upper()} ", end="", file=sys.stderr)
                print(f"on the {arch_to_use.upper()} architecture.", file=sys.stderr)
                return -2
            if int(args.opsel) != 4:
                print(f"The chosen instruction, {inst_to_use.upper()}, ", end="", file=sys.stderr)
                print(f"in the {arch_to_use.upper()} architecture, ", end="", file=sys.stderr)
                print("only supports the OPSEL values 0 and 4.", file=sys.stderr)
                return -2
        if is_gfx12_arch(inst_info):
            # In gfx12 sparse operations, OPSEL is used to pick the set within the compression
            # index register. For wave32, OPSEL[0] picks between two sets. For wave64, OPSEL[0] and
            # OPSEL[1] are used to pick between four sets.
            if not inst_info['sparse']:
                print(f"The chosen instruction, {inst_to_use.upper()}, ", end="", file=sys.stderr)
                print(f"in the {arch_to_use.upper()} architecture, does ", end="", file=sys.stderr)
                print("not support the OPSEL modifier.", file=sys.stderr)
                return -2
            if matrix_to_use != 'k':
                print("The OPSEL modifier may only be used to choose ", end="", file=sys.stderr)
                print("the set within the sparse compression index ", end="", file=sys.stderr)
                print(f"register on the {arch_to_use.upper()} architecture.", file=sys.stderr)
                return -2
            max_opsel = calc.get_num_compression_sets() - 1
            if int(args.opsel) > max_opsel:
                print("The OPSEL modifier may only be ", end="", file=sys.stderr)
                if max_opsel == 0:
                    print("the value 0 when used ", end="", file=sys.stderr)
                else:
                    print(f"values between 0 and {max_opsel} when used ", end="", file=sys.stderr)
                print(f"with the instruction {inst_to_use.upper()} and ", end="", file=sys.stderr)
                print(f"{int(args.wavefront)}-wide wavefronts ", end="", file=sys.stderr)
                print(f"on the {arch_to_use.upper()} architecture.", file=sys.stderr)
                return -2

    # On RDNA3 and RDNA4 architectures, the NEG field is used for two things: for FP operations,
    # its three bits are used to negate the values of the A, B, and C matrices, respectively.
    # For FP, NEG[0] and NEG[1] affect the low 16 bits of each A and B register, respectively.
    # To fully negate an input matrix, the NEG_HI field in the instruction should also be set.
    # For floating-point inputs, NEG_HI[0] and NEG_HI[1] negate the high 16 bits of each
    # A and B register, respectively.
    # For floating-point outputs, NEG[2] causes the instruction to subtract (instead of add)
    # the C matrix, while NEG_HI[2] causes the addition/subtraction of the absolute value of
    # the C matrix.
    # For integer instructions, the first two bits are used to set the A and B matrices to
    # signed/unsigned respectively and the third bit must-be-zero. NEG_HI must be 0 for integers.
    if (int(args.neg) > 0 or int(args.neg_hi) > 0):
        if is_gfx9_arch(inst_info):
            print(f"The chosen architecture, {arch_to_use.upper()}, ", end="", file=sys.stderr)
            print("does not support using the NEG or NEG_HI modifiers.", file=sys.stderr)
            return -2
        if not inst_info['neg']:
            print(f"The chosen instruction, {inst_to_use.upper()}, ", end="", file=sys.stderr)
            print(f"in the {arch_to_use.upper()} architecture, does ", end="", file=sys.stderr)
            print("not support the NEG or NEG_HI modifiers.", file=sys.stderr)
            return -2
        if matrix_to_use == 'd':
            if not (args.output_calc and (args.matrix_entry or args.get_register)):
                print("The NEG and NEG_HI modifiers may only be used ", end="", file=sys.stderr)
                print("on matrices A, B, or C for the instruction ", end="", file=sys.stderr)
                print(f"{inst_to_use.upper()} on the ", end="", file=sys.stderr)
                print(f"{arch_to_use.upper()} architecture.", file=sys.stderr)
                print("When using the D matrix, the NEG modifier may ", end="", file=sys.stderr)
                print("only be used when asking for the --output-calculation.", file=sys.stderr)
                return -2
        max_neg_val = 7
        max_neg_hi_val = 7
        if inst_info['integer']:
            max_neg_val = 3
            max_neg_hi_val = 0
        if int(args.neg) > max_neg_val:
            print(f"For the instruction {inst_to_use.upper()} in the ", end="", file=sys.stderr)
            print(f"{arch_to_use.upper()} architecture, ", end="", file=sys.stderr)
            print("the NEG modifier may only contain values between ", end="", file=sys.stderr)
            print(f"0 - {max_neg_val}, inclusive.", file=sys.stderr)
            return -2
        if int(args.neg_hi) > max_neg_hi_val:
            print(f"For the instruction {inst_to_use.upper()} in the ", end="", file=sys.stderr)
            print(f"{arch_to_use.upper()} architecture, ", end="", file=sys.stderr)
            print("the NEG_HI modifier may only contain values ", end="", file=sys.stderr)
            if max_neg_hi_val == 0:
                print("of 0.", file=sys.stderr)
            else:
                print(f"between 0 - {max_neg_hi_val}, inclusive.", file=sys.stderr)
            return -2

    negate = {'a': False, 'a_lo': False, 'a_hi': False, 'b': False, 'b_lo': False, 'b_hi': False,
              'c': False, 'c_abs': False, 'c_lo': False, 'c_hi': False, 'd': False, 'd_lo': False,
              'd_hi': False, 'k': False, 'k_lo': False, 'k_hi': False}
    if (is_gfx9_arch(inst_info) and inst_info['in_type'] == 'fp64' and int(args.blgp) > 0):
        # CDNA3 only supports negation on FP64 matrix multiplications. The BLGP field of
        # VOP3P-MAI is instead used for negation.
        # CDNA2 does not support BLGP field on FP64 matrix multiplications, so we would not reach
        # this point -- we would catch lack of BLGP support in a check up above.
        neg_this = int(args.blgp)
        negate['a'] = bool(neg_this & 0x1)
        negate['b'] = bool(neg_this & 0x2)
        negate['c'] = bool(neg_this & 0x4)
        args.blgp = '0'
    elif not is_gfx9_arch(inst_info):
        # RDNA3 and RDNA4 use the NEG and NEG_HI fields for negation decisions. But NEG[2] and
        # NEG_HI must be zero for integer instructions.
        # For floating-point NEG_HI[2] is actually used to set absolute value on the C matrix.
        neg = int(args.neg)
        neg_hi = int(args.neg_hi)
        if (neg > 0 or neg_hi > 0):
            if inst_info['integer']:
                if neg & 0x4 != 0:
                    print("The chosen instruction, ", end="", file=sys.stderr)
                    print(f"{inst_to_use.upper()}, cannot have NEG[2] set.", file=sys.stderr)
                    return -2
                if neg_hi != 0:
                    print("The chosen instruction, ", end="", file=sys.stderr)
                    print(f"{inst_to_use.upper()}, cannot have NEG_HI set.", file=sys.stderr)
                    return -2
            negate['a_lo'] = bool(neg & 0x1)
            negate['a_hi'] = bool(neg_hi & 0x1)
            negate['b_lo'] = bool(neg & 0x2)
            negate['b_hi'] = bool(neg_hi & 0x2)
            negate['c'] = bool(neg & 0x4)
            negate['c_abs'] = bool(neg_hi & 0x4)


    if is_gfx9_arch(inst_info):
        sparse_inst_name = "MFMAC"
    else:
        sparse_inst_name = "WMMAC"
    if (inst_info['sparse'] and matrix_to_use == 'c'):
        print(f"The chosen instruction, {inst_to_use.upper()}, ", end="", file=sys.stderr)
        print(f"is a sparse {sparse_inst_name} op and performs D += A*B.", file=sys.stderr)
        print("This instruction does not support the C matrix as an input.", file=sys.stderr)
        return -2
    if ((not inst_info['sparse']) and matrix_to_use == 'k'):
        print(f"The chosen instruction, {inst_to_use.upper()}, ", end="", file=sys.stderr)
        print(f"is not a sparse {sparse_inst_name} op.", file=sys.stderr)
        print("This instruction does not support the sparse ", end="", file=sys.stderr)
        print("index register as an input.", file=sys.stderr)
        return -2

    if [args.csv, args.md, args.ad].count(True) > 1:
        print('Can only use one of "--csv", "--markdown", and ', end="", file=sys.stderr)
        print('"--asciidoc" at the same time.', file=sys.stderr)
        return -2

    if args.csv:
        requested_output = "csv"
    elif args.md:
        requested_output = "markdown"
    elif args.ad:
        requested_output = "asciidoc"
    else:
        requested_output = "grid"

    print_arch_inst(arch_to_use, inst_to_use)
    try:
        if args.get_register:
            calc.calculate_get_register(matrix_to_use, args.output_calc, negate,
                                        int(args.I_coordinate), int(args.J_coordinate),
                                        int(args.K_coordinate), int(args.block),
                                        int(args.cbsz), int(args.abid), int(args.blgp),
                                        int(args.opsel))
        elif args.matrix_entry:
            calc.calculate_single_location(matrix_to_use, args.output_calc, negate,
                                           int(args.register), int(args.lane),
                                           int(args.cbsz), int(args.abid), int(args.blgp),
                                           int(args.opsel))
        elif args.register_layout:
            calc.calculate_register_layout(matrix_to_use, requested_output, negate,
                                           int(args.cbsz), int(args.abid), int(args.blgp),
                                           int(args.opsel), bool(args.transpose))
        elif args.matrix_layout:
            calc.calculate_matrix_layout(matrix_to_use, requested_output, negate,
                                         int(args.cbsz), int(args.abid), int(args.blgp),
                                         int(args.opsel), bool(args.transpose))
        else:
            print("No action requested. This should not be possible!", file=sys.stderr)
            return -1
    except ValueError as err_msg:
        print(err_msg, file=sys.stderr)
        return -2
    return 0


class InstCalc(metaclass=ABCMeta):
    """ Calculator for matrix multiplication instruction details.

    Different accelerator architectures require different calculations for many of the instruction
    details and capabilities. This abstract class holds shared functions that are common between
    architectures, and abstract methods which will be over-ridden by per-architecture child classes.

    Attributes:
        arch_name: string that holds the accelerator architecture's name
        inst_name: string holding the mnemonic of the instruction that will be used in calculations
        inst_info: MatrixInstruction holding the details of the instruction used in calculations
        wave_width: Some architectures allow wavefronts of various widths, and these widths can
            affect the resulting calculations. This integer holds the width that will be used for
            further calculations.
    """
    def __init__(self, inst: str, inst_info: MatrixInstruction, wave_width: int) -> None:
        """ Initializes InstCalc attributes """
        self.arch_name = inst_info['arch']
        self.inst_name = inst
        self.inst_info = inst_info
        self.wave_width = wave_width

    def __repr__(self) -> str:
        ret_str = f"Matrix instruction calculator for {self.inst_name} on"
        ret_str += f"{self.inst_info['arch']} with wave width {self.wave_width}"
        return ret_str

    @abstractmethod
    def check_valid_reg_lane(self, matrix: str, register: int, lane: int) -> bool:
        """ Checks if the register and lane being used for calculation are legal

        Checks whether the matrix and lane being requested are legal for indexing into the
        requested matrix. Callers should use this to ensure that the register is within the
        bounds of the matrix. In addition, some architectures do not allow all lanes in a
        wave to hold data in some matrices.

        Args:
            matrix: String indicating the matrix to query: 'a', 'b', 'c', 'd', or 'k'
                for the compression index matrix in sparse matrix ops
            register: Integer of the register number being requested, starting from 0
            lane: Integer of the lane being requested, starting from 0

        Returns:
            True if the register and lane contain data for this matrix on the current instruction
            False if the register and lane do not contain data for this matrix in this instruction
        """

    # Disabling check here because this function can be over-ridden by child classes that need
    # this to be a method and have access to self variables.
    #pylint: disable=no-self-use
    def get_num_compression_sets(self) -> int:
        """ Returns the number of compression index sets that an instruction has

        Returns:
            An integer that contains the number of compression index sets that an
                instruction allows for the given configuration
        """
        # Abstract class always returns 0, only architectures where this matter needs to override
        return 0

    # Disabling check here because this function can be over-ridden by child classes that need
    # this to be a method and have access to self variables.
    #pylint: disable=no-self-use
    def _get_num_compression_bit_offset(self, opsel: int) -> int:
        """ Returns the bit offset of the first compression index, based on instruction modifiers

        Args:
            opsel: The OPSEL modifier for the matrix instruction

        Returns:
            An integer that contains the starting bit of this set of compression indices
        """
        # Abstract class always returns 0, only architectures where this matter needs to override
        del opsel
        return 0

    @staticmethod
    def _get_reg_name(data_size: int, sparse: bool, compression_index: bool, k_cbsz: int,
                      k_abid: int, k_opsel_offset: int, regno: int) -> str:
        """ Calculates the register name and formats it as Va.c.

        Calculates the register name (but not lane) based on a regno. Formats it in Va.c
        format. The regno argument not a VGPR, but the storage location based on the
        data size. For instance, for 8-byte values, regno=2 will be V[5:4], because
        each VGPR is 4 bytes so each 8-byte storage location is 2 VGPRs.
        For 1-byte values, regno=2 is V0.[23:16], because this storage location is still
        within the first 4-byte register.

        Args:
            data_size: integer holding the size of this regno's data, in bits.
            sparse: True if this register is working on a structured sparsity input register
            compression_index: True if this register holds the compression index for a
                structured sparsity instruction
            k_cbsz: When working on compression indices in gfx9, a non-zero CBSZ value will cause
                different register locations to be used for the compression index. As such,
                when passing in compression_index=True, this integer field holds the
                instruction's CBSZ modifier.
            k_abid: When working on compression indices in gfx9, a non-zero CBSZ value will cause
                different register locations to be used for the compression index. The ABID
                field chooses which register location to use. As such, when passing in
                compression_index=True and k_cbsz!=0, this integer field holds the
                instruction's ABID modifier.
            k_opsel_offset: When working on compression indices in gfx12, the OPSEL field chooses
                which register location to use. As such, when passing in compression_index=true,
                this field holds the number of VGPR bits this instruction's OPSEL modifier shifted
                the output.
            regno: integer that holds the 'regno' (register storage location) that this
                function will transform into a VGPR number and name.

        Returns:
            A string containing the register name in the Va.c format.
            'V' is a constant character that shows this is a vector register
            'a' is the actual register number. For example, if an instruction has 8 VGPRs, then
                this value would be between 0-7. When this is a 64b register, this will combine
                two contiguous numbers together, and so may be e.g. "1:0" to show that this
                single regno is using both VGPR1 and VGPR0.
            '.c' is an optional extension that, when a value accesses a subset of a VGPR.
                e.g., when accessing the bottom half of a register, it would be .[15:0].
        """
        this_str = "v"
        if not compression_index:
            if data_size == 32:
                this_str += (str(regno))
            elif data_size == 64:
                this_str += "[" + (str(regno * 2 + 1) + ":")
                this_str += (str(regno * 2) + "]")
            elif data_size == 16:
                this_str += (str(int(regno / 2)))
                # If we are in a sparse matrix, each of the 2 .[15:0]/.[31:16] entries are combined
                # to hold 4 values (2 of which are 0). Which of the 4 exist is dynamically chosen
                # by data in another register. As such, in this static tool, we cannot tell
                # you if a value exists in the .[15:0]/.[31:16] pair, nor which one it might be in.
                # As such, we tell users that these entries just live in the 4-byte register
                # by skipping the .[15:0]/.[31:16] distinction in the name.
                if not sparse:
                    bitno = regno % 2
                    this_str += f".[{16 * bitno + 15}:{16 * bitno}]"
            elif data_size == 8:
                this_str += (str(int(regno / 4)))
                # See the above comment about sparse matrices. The concept is similar here
                # in 1B values. .[7:0]/.[15:8] holding 4 values means we just report it as
                # .[15:0] holding 4 matrix entries, etc.
                if not sparse:
                    bitno = regno % 4
                    this_str += f".[{8 * bitno + 7}:{8 * bitno}]"
                else:
                    bitno = int(regno/2) % 2
                    this_str += f".[{16 * bitno + 15}:{16 * bitno}]"
            elif data_size == 4:
                this_str += (str(int(regno / 8)))
                # See the above comment about sparse matrices. The concept is similar here
                # in 4b values. .[3:0]/.[7:4] holding 4 values means we just report it as
                # .[7:0] holding 4 matrix entries, etc.
                if not sparse:
                    bitno = regno % 8
                    this_str += f".[{4 * bitno + 3}:{4 * bitno}]"
                else:
                    bitno = int(regno/2) % 4
                    this_str += f".[{8 * bitno + 7}:{8 * bitno}]"
        else:
            # In the 4:2 compression case, the compression index requires 2 bits for each
            # actually-stored entry in the matrix (which of the four values is stored in the
            # first location, and which of the four values is stored in the second location).
            # As such, every entry in the compression array has 2 bits.
            # The way this is stored in the compression matrix is in "sets" that are indexed by
            # ABID register (on gfx9). The first starts at VGPR0 bits 1:0, and extends for
            # however many storage locations are contained in a lane of the A matrix's input VGPR.
            # The A matrix holds 2 VGPRs, that are each 4B.
            # With a data size of N bytes, this yields (8 / N) locations to store into.
            #this_str += str(int(regno * data_size / 64))
            # However, on all existing architectures, the actual VGPR number is always 0
            # This is a hack that will need to be fixed if some architecture in the future adds
            # more
            this_str += "0"
            # Move the register up by (8 / N) register slots for every value of ABID
            if k_cbsz == 0:
                regno += int((64 * k_abid) / data_size)
            # Do a floor(regno / 2) to get the base register bits that will hold this register,
            # then multiply by 4 to offset the bits.
            index = int(regno / 2) * 4
            # On some architectures, OPSEL will shift these starting locations instead of the
            # ABID shift.
            this_str += f".[{index + 3 + k_opsel_offset}:{index + k_opsel_offset}]"
        return this_str

    @staticmethod
    def __format_reg_lane(reg: str, lane: int) -> str:
        """ Calculates a register+lane name from a register str and lane number.

        Takes a partially complete register name of the form V#.[bits] and inserts
        {lane_number} between the register number and the trailing half-identifier.

        Args:
            reg: string holding a register number, of the form Va.c.
                'V' is a constant character that shows this is a vector register.
                'a' is the actual register number. For example, if an instruction has 8 VGPRs,
                    then this value would be between 0-7. When this is a 64b register, this
                    will combine two contiguous numbers together, and so may be e.g. "1:0" to
                    show that this single regno is using both VGPR1 and VGPR0.
                '.c' is an optional extension that, when a value accesses a subset of a VGPR.
                    e.g, when accessing the bottom half of a register, it would be .[15:0].
            lane: integer holding the lane number to combine with the above reg

        Returns:
            A string of the form Va{lane#}.c
        """
        reg_halves = reg.split('.')
        full_name = reg_halves[0]
        full_name += f"{{{lane}}}"
        if len(reg_halves) > 1:
            full_name += '.' + reg_halves[1]
        return full_name

    @staticmethod
    def _get_elements_per_gpr(data_size: int, sparse_register: bool) -> float:
        """ Calculates how many elements of a matrix value fit in each register.

        Returns how many elements of the matrix (each element of size data_size)
        fit into each GPR value. All current registers are 4B in size, so the
        question is "how many elements will fit into a 4B register."

        Args:
            data_size: integer holding the size of the requested data, in bits.
            sparse_register: boolean that is True if this query is about a register that
                holds a 4:2 structured-sparsity data, so twice as many values will
                fit into a 4B register.

        Returns:
            A floating-point value that holds how many matrix values will fit into a 4B VGPR.
            Because values with data_size >4 may need multiple VGPRs, the returned value can
            be a fraction <1, which indicates multiple registers are needed to hold a single
            value.
        """
        reg_size = 32
        # 1B values fit four units of data per register (0.25 registers per data)
        # 2B values fit two units of data per register (0.5 registers per data)
        # 4B values need one register per unit of data
        # 8B values need two registers per unit of data
        elements = reg_size / data_size
        # Values in a sparse matrix effectively need half as much storage, so
        # we say they fit twice as many elements per gpr.
        if sparse_register:
            elements *= 2
        return elements

    @staticmethod
    def _get_cbsz_abid_transformed_block(block: int, cbsz: int, abid: int) -> int:
        """ Calculates the new block used for an input after CBSZ and ABID transformation.

        CBSZ and ABID combine for some instructions to replace some input data with
        values from other blocks. For instance, normally an instruction will calculate
        on the A matrix input values from block X. After CBSZ and ABID values are taken
        into account, the same calculations would instead pull their input data from
        input block Y.

        CBSZ says "broadcast one block to 2^(CBSZ) other blocks".
        ABID says "which block will be broadcast".
        The function takes as an input 'block X', the CBSZ, and ABID values, and
        calculates 'block Y'.

        Args:
            block: an integer representing the original block of A that would be used
                in a calculation
            cbsz: an integer that contains the instruction's CBSZ value
            abid: an integer that contains the instruction's ABID value

        Returns:
            An integer that shows which block of A would be used for the calculation,
            after taking CBSZ and ABID into account.
        """
        num_receiving_blocks = math.pow(2, cbsz)
        base_block = int(int(block / num_receiving_blocks) * num_receiving_blocks)
        return base_block + abid

    @staticmethod
    def _get_blgp_transformed_lane(lane: int, blgp: int) -> int:
        """ Calculates the new lane used for an input after BLGP transformation.

        For some instructions, BLGP will cause the B matrix input into the math functions
        to be pulled from different lanes than normal. This function calculate how BLGP
        transforms a B matrix lane.

        Calculations are:
        BLGP=0 : Default pattern
        BLGP=1 : Broadcast the first 32 lanes to all lanes
        BLGP=2 : Broadcast the 2nd 32 lanes to all lanes
        BLGP=3 : Rotate all lanes down by 16
        BLGP=4 : Broadcast the first 16 lanes
        BLGP=5 : Broadcast the 2nd 16 lanes
        BLGP=6 : Broadcast the 3rd 16 lanes
        BLGP=7 : Broadcast the 4th 16 lanes

        Args:
            lane: an integer representing the original lane of B that would be used
                in a calculation
            blgp: an integer that contains the instruction's BLGP value

        Returns:
            An integer that shows which lane's value of B would be used for the calculation,
            after taking BLGP into account.
        """
        lane_to_ret = lane
        if blgp == 1:
            lane_to_ret = lane % 32
        elif blgp == 2 and lane < 32:
            lane_to_ret = lane + 32
        elif blgp == 3:
            lane_to_ret = lane + 16
            lane_to_ret %= 64
        elif blgp == 4:
            lane_to_ret %= 16
        elif blgp == 5:
            if lane < 16:
                lane_to_ret = lane + 16
            elif lane >= 48:
                lane_to_ret = lane - 32
            elif lane >= 32:
                lane_to_ret = lane - 16
        elif blgp == 6:
            if lane >= 48:
                lane_to_ret = lane - 16
            elif lane < 16:
                lane_to_ret = lane + 32
            elif lane < 32:
                lane_to_ret = lane + 16
        elif blgp == 7:
            mul_factor = lane / 16
            mul_factor = 3 - int(mul_factor)
            lane_to_ret = lane + mul_factor * 16
        return lane_to_ret

    @abstractmethod
    def _get_reg_lanes(self, matrix: str, i: int, j: int, k: int, block: int, cbsz: int,
                       abid: int, blgp: int, opsel: int) -> Tuple[str, str, List[int]]:
        """ Calculates a matrix's register and lane number based on coordinates.

        This is an abstract method, and should be filled in by any child class to
        actually calculate this data for the target architecture.

        For the target architecture and the instruction set up in this class's init
        function, this function calculates the register and lane that hold a requested
        matrix entry in a requested matrix. This location information can vary based on
        per-instruction modifiers, so those are all input arguments.

        Args:
            matrix: String indicating the matrix to query: 'a', 'b', 'c', 'd', or 'k'
                for the compression index matrix in sparse matrix ops
            i: integer coordinate for the query of the matrix row for A, C, D, & K matrices
            j: integer coordinate for the query of the matrix column for the B, C, & D matrices
            k: integer coordinate for the query of the A & K column or B row
            block: integer coordinate for the block to query
            cbsz: integer value of the instruction's CBSZ modifier
            abid: integer value of the instruction's ABID modifier
            blgp: integer value of the instruction's BLGP modifier
            opsel: integer value of the instruction's OPSEL modifier

        Returns:
            Based on the matrix and requested coordinates, return three things in a tuple:
            1. String requested element name, in the format A[i][k], B[k][j], or C[i][j]
            2. String register name that holds the element, in the format V#.[bits]
            3. List of integers containing the lane numbers within that register that
                hold the element
            Tuple: (matrix entry, register holding that entry, lanes within that register)
        """

    def _find_matching_b_lane(self, a_lane: int, b_lanes: List[int]) -> int:
        """ Finds the lane in a list of B matrix lanes that match the A matrix lane.

        In some architectures, matrix values can exist simultaneously in multiple
        lanes. Or, more specifically, multiple lanes must store the same value from
        the matrix. If a value was stored in both lanes 0 and lane 16, when printing
        out "lane 0 of A is multiplied by lane X of B", this function will do that
        matching. It takes as an argument a list of lanes from B, and the requested
        lane of A. Returns the lane of B that is multiplied by the requested lane of A.

        Args:
            a_lane: integer for the lane of A that we want to match
            b_lanes: list of integers containing all the lanes of B to query

        Returns:
            Integer from the available lanes of B that match the requested lane of A
        """
        del a_lane # Unused in architectures that do not have multiple B lanes
        return b_lanes[0]

    @staticmethod
    def __neg_abs_name(reg: str, mat_val: str, matrix: str, negate: Dict[str, bool]) -> str:
        """ Negates a requested name based on the matrix and negate list.

        When preparing to print a matrix value, fix up the string to add negation and
        absolute values, depending on the input matrix and the 'negate' structure.

        Args:
            reg: string that lists the register that holds this matrix entry
                This string should contain [15:0] or [31:16] if the entry is in the lower
                or upper halves, respectively.
            mat_val: string that contains the matrix entry value, which this function
                may add "-" to negate, and "|mat_val|" to indicate absolute value
            matrix: string name of the matrix this entry comes from
            negate: dictionary of matrix names to bools that indicate whether to
                negate and absolute-val entries from this matrix

        Returns:
            String that may just be mat_val (if no negation or abs-val), -mat_val
            (negation), |mat_val| (absolute value), or -|mat_val| (negation and absolute value)
        """
        mat_lo = f"{matrix}_lo"
        mat_hi = f"{matrix}_hi"
        if matrix.lower() == 'c' and negate['c_abs']:
            mat_val = f"|{mat_val}|"
        if (negate[matrix] or
                (negate[mat_lo] and re.search("15:0", reg)) or
                (negate[mat_hi] and re.search("31:16", reg))):
            mat_val = f"-{mat_val}"
        return mat_val

    def __calculate_source_string(self, d_matrix_entry: str, find_element: bool,
                                  negate: Dict[str, bool], cbsz: int, abid: int, blgp: int,
                                  opsel: int) -> str:
        """ Calculates the input values that went into calculating a particular D matrix entry.

        For a particular output matrix entry, this function will calculate the input
        register-lane values that went into calculating it. Alternately, or for a register-lane
        that held a particular output matrix entry, calculate the input matrix entries that
        went into calculating it. This split is controlled by the find_element argument.

        Args:
            d_matrix_entry: string that contains the D[i][j] matrix entry to query against
            find_element: True causes this function to calculate the the A[], B[], and C[]
                matrix elements and their indices that went into calculating the D matrix.
                False causes the string to contain the VGPRs and lanes that make up the
                A[], B[], and C[] matrices that go into the calculation.
            negate: dictionary of matrix names to bools that indicate whether to
                negate and absolute-val entries from this matrix
            cbsz: integer containing this instruction's CBSZ modifier
            abid: integer containing this instruction's ABID modifier
            blgp: integer containing this instruction's BLGP modifier
            opsel: integer containing this instruction's OPSEL modifier

        Returns:
            A string containing a mathematical formula of the matrix entries or register-lanes
            that went into calculating the requested D matrix entry
        """
        inst_info = self.inst_info
        # Matrix being sent in will look like D[i][j].Bb, where the ".Bb" is optional.
        row_col_block = re.findall('[0-9]+', d_matrix_entry)
        i = int(row_col_block[0])
        j = int(row_col_block[1])
        if len(row_col_block) > 2:
            block = int(row_col_block[2])
        else:
            block = 0
        to_ret = []
        # When BLGP, CBSZ, and/or ABID are set, we can get the correct register{lane} back,
        # but if we want to get the original matrix entry associated with that post-
        # transformed register, so that folks know what data the hardware will actually
        # use. The following create maps back to "original matrix data" without the
        # CBSZ/ABID/BLGP modifiers
        if find_element:
            a_reg_dict = self.__create_register_dict('a', 0, 0, 0, opsel)
            b_reg_dict = self.__create_register_dict('b', 0, 0, 0, opsel)
        for k in range(inst_info['k']):
            # Find the register/lane that will actually be pulled from after the modifiers
            (_, a_reg, a_lanes) = self._get_reg_lanes('a', i, j, k, block, cbsz, abid, 0, opsel)
            (_, b_reg, b_lanes) = self._get_reg_lanes('b', i, j, k, block, 0, 0, blgp, opsel)
            # Both gfx9 and gfx11 only have a single lane per A matrix entry
            a_lane = a_lanes[0]
            b_lane = self._find_matching_b_lane(a_lane, b_lanes)
            if find_element:
                a_lane_name = self.__format_reg_lane(a_reg, a_lane)
                b_lane_name = self.__format_reg_lane(b_reg, b_lane)
                # Look up the original matrix entries that would have been in those registers
                a_entries = a_reg_dict[a_lane_name]
                (b_entry,) = b_reg_dict[b_lane_name]
                for a_entry in a_entries:
                    a_coords = re.findall('[0-9]+', a_entry)
                    if int(a_coords[1]) != k:
                        continue
                    a_entry = self.__neg_abs_name(a_reg, a_entry, 'a', negate)
                    b_entry = self.__neg_abs_name(b_reg, b_entry, 'b', negate)
                    to_ret.append(f"{a_entry}*{b_entry}")
            else:
                a_reg_lane = self.__format_reg_lane(a_reg, a_lane)
                a_name = self.__neg_abs_name(a_reg, f"Src0_{a_reg_lane}", 'a', negate)
                b_reg_lane = self.__format_reg_lane(b_reg, b_lane)
                b_name = self.__neg_abs_name(b_reg, f"Src1_{b_reg_lane}", 'b', negate)
                to_ret.append(f"{a_name}*{b_name}")
        ret_string = " + ".join(to_ret)
        if not inst_info['sparse']:
            (c_ele, c_reg, c_lanes) = self._get_reg_lanes('c', i, j, k, block, 0, 0, 0, opsel)
            c_lane = c_lanes[0]
            if negate['c']:
                ret_string += " - "
            else:
                ret_string += " + "
            abs_str = ""
            if negate['c_abs']:
                abs_str = "|"
            if find_element:
                ret_string += f"{abs_str}{c_ele}{abs_str}"
            else:
                c_reg_lane = self.__format_reg_lane(c_reg, c_lane)
                ret_string += f"{abs_str}Src2_{c_reg_lane}{abs_str}"
        return ret_string

    def calculate_get_register(self, matrix: str, out_calc: bool, negate: Dict[str, bool],
                               i: int, j: int, k: int, block: int, cbsz: int, abid: int,
                               blgp: int, opsel: int) -> None:
        """ Prints the register location and wavefront lane for the chosen matrix.

        For the class's architecture and instruction, this function calculates the
        register and lane for a desired matrix entry, after all of the modifiers that
        can change up the matrix location.
        The registers are displayed as: Va{b}.c:
            - a is the register number
            - b is the lane within that register
            - c is an optional identifier for sub-Dword parts of the 32 bit register:
                    [15:0]: the lower 16b of a 32b register
                    [31:16]: the upper 16b of a 32b register
                    [7:0]: the least significant 8b of a 32b register
                    [15:8]: the second-lowest 8b of a 32b register
                    [23:16]: the second-highest 8b of a 32b register
                    [31:24]: the most significant 8b of a 32b register

        Args:
            matrix: string that contains the name of the matrix
                Legal values are a, b, c, d, and k (for the compression index of sparse matrices).
            out_calc: True if, when printing the register location for the D
                output matrix entry, you desire to also print the register
                locations of the A, B, and C matrices that went into the
                calculation of the output
            negate: dictionary of matrix names to bools that indicate whether to
                negate and absolute-val entries from this matrix
            i: integer coordinate for the query of the matrix row for A, C, D, & K matrices
            j: integer coordinate for the query of the matrix column for the B, C, & D matrices
            k: integer coordinate for the query of the A & K column or B row
            block: integer coordinate for the block to query
            cbsz: integer value of the instruction's CBSZ modifier
            abid: integer value of the instruction's ABID modifier
            blgp: integer value of the instruction's BLGP modifier
            opsel: integer value of the instruction's OPSEL modifier

        Raises:
            ValueError: An i/j/k/block coordinate was not valid for this instruction
                or an unsupported matrix type was requested.
        """
        inst_info = self.inst_info
        inst = self.inst_name
        # Input validation
        if i < 0:
            raise ValueError(f"Input value for 'i', {i}, must not be less than zero.")
        if j < 0:
            raise ValueError(f"Input value for 'j', {j}, must not be less than zero.")
        if k < 0:
            raise ValueError(f"Input value for 'k', {k}, must not be less than zero.")
        if block < 0:
            raise ValueError(f"Input value for 'block', {block}, must not be less than zero.")
        if i >= inst_info['m']:
            err_line = fill(dedent(f"""Input value for 'i', {i}, is too large.
                                   Maximum value of row for {inst} is {inst_info['m'] - 1}."""))
            raise ValueError(err_line)
        if j >= inst_info['n']:
            err_line = fill(dedent(f"""Input value for 'j', {j}, is too large.
                                   Maximum value of column for {inst} is {inst_info['n'] - 1}."""))
            raise ValueError(err_line)
        if k >= inst_info['k']:
            if matrix.lower() == 'b':
                val_name = "row"
            else:
                val_name = "column"
            err_line = fill(dedent(f"""Input value for 'k', {k}, is too large.
                                   Maximum value of {val_name} for {inst} is """
                                   f"""{inst_info['k'] - 1}."""))
            raise ValueError(err_line)
        if block >= inst_info['blocks']:
            err_line = fill(dedent(f"""Input value for 'block', {block}, is too large.
                                   Maximum value of block for {inst} is """
                                   f"""{inst_info['blocks'] - 1}."""))
            raise ValueError(err_line)

        check_matrix_support(matrix, ('a', 'b', 'c', 'd', 'k'), self.__class__.__name__)

        # Calculate register and lane based on matrix layout
        (element_name, reg, lanes) = self._get_reg_lanes(matrix, i, j, k, block,
                                                         cbsz, abid, blgp, opsel)
        if (matrix.lower() == 'd' and out_calc):
            source_string = self.__calculate_source_string(element_name, False, negate, cbsz, abid,
                                                           blgp, opsel)
            for lane in lanes:
                print(f"{element_name} = Vdst_{self.__format_reg_lane(reg, lane)} ", end="")
                print(f"= {source_string}")
        else:
            for lane in lanes:
                print(f"{element_name} = {self.__format_reg_lane(reg, lane)}")

    def __create_register_dict(self, matrix: str, cbsz: int, abid: int, blgp: int,
                               opsel: int) -> Dict[str, List[str]]:
        """ Creates a dictionary that maps vector registers to matrix elements.

        For the class's instruction and the input matrix, create a dictionary that maps
        all of the vector register entries in this matrix, such as 'V0{17}.[23:16]' to
        the matrix element that is contained in that entry.
        So V0{17}.[23:16] -> A[i][k] for whichever values of i and k.

        Args:
            matrix: string that contains the name of the matrix
                Legal values are a, b, c, d, and k (for the compression index of sparse matrices).
            cbsz: integer value of the instruction's CBSZ modifier
            abid: integer value of the instruction's ABID modifier
            blgp: integer value of the instruction's BLGP modifier
            opsel: integer value of the instruction's OPSEL modifier

        Returns:
            Dictionary of register-lane string keys that map to lists of matrix entry strings
            If the VGPR is used for multiple matrix entries (e.g., due to modifiers pushing the
            VGPR's data to multiple matrix inputs), the list of strings may be >1 entry.

        Raises:
            ValueError: An unsupported matrix was requested.
        """
        check_matrix_support(matrix, ('a', 'b', 'c', 'd', 'k'), self.__class__.__name__)
        inst_info = self.inst_info
        B = inst_info['blocks']
        M = inst_info['m']
        N = inst_info['n']
        K = inst_info['k']

        register_dict: Dict[str, List[str]] = {}
        if matrix.lower() in ('a', 'k'):
            N = 1
        elif matrix.lower() == 'b':
            M = 1
        else: # 'c' or 'd'
            K = 1
        for b in range(B):
            for i in range(M):
                for j in range(N):
                    for k in range(K):
                        (mat_val, reg, lanes) = self._get_reg_lanes(matrix, i, j, k, b,
                                                                    cbsz, abid, blgp, opsel)
                        for lane in lanes:
                            reg_key = self.__format_reg_lane(reg, lane)
                            # With 4:2 sparsity, we can end up with multiple matrix entries in the
                            # same register, since they are compressed out when they are zero.
                            # This is also true with BLGP set.
                            if reg_key in register_dict:
                                old_list = register_dict[reg_key]
                                old_list.append(mat_val)
                                register_dict[reg_key] = old_list
                            else:
                                register_dict[reg_key] = [mat_val]
        return register_dict

    # Disabling check here because this function can be over-ridden by child classes that need
    # this to be a method and have access to self variables.
    #pylint: disable=no-self-use
    def _calculate_initial_regno_offset(self, matrix: str, opsel: int) -> int:
        """ Calculates an offset into a register slot based on OPSEL.

        On some architectures, partial registers (such as a 16b output in a 32b register)
        aren't tightly packed. For example, "lower" or "upper halves may be skipped
        instead of storing two contiguous values, one in the lower and one in the upper.
        In such architectures and for certain matrices, the OPSEL value controls whether
        to use the upper or lower halves.

        This function returns an offset, so that these slots (which we call regnos) can
        be skipped.

        On most architectures, register printing takes place on a regno boundary.
        As such, child classes that need to do this offsetting should override this
        function.

        Args:
            matrix: string that contains the name of the matrix
                Legal values are a, b, c, d, and k (for the compression index of sparse matrices).
            opsel: integer value of the instruction's OPSEL modifier

        Returns:
            Integer which indicates the regno offset for this matrix+OPSEL pair
        """
        del matrix, opsel # unused in base instruction class
        return 0

    # Disabling check here because this function can be over-ridden by child classes that need
    # this to be a method and have access to self variables.
    #pylint: disable=no-self-use
    def _calculate_num_regnos_to_print(self, matrix: str, gpr_ratio: float) -> int:
        """ Calculates the number of register slots to print for this matrix & instruction.

        On some architectures, partial registers (such as a 16b output in a 32b register)
        aren't tightly packed. For example, "lower" or "upper halves may be skipped
        instead of storing two contiguous values, one in the lower and one in the upper.
        In such architectures and for  certain matrices, the OPSEL value controls whether
        to use the upper or lower halves.

        This function returns the number of slots to print in each register; the rest can
        be skipped.

        Most architectures print out all registers. GPR ratio is the number of regnos
        in each GPR. As such, child classes that need to do this offsetting should override this
        function.

        Args:
            matrix: string that contains the name of the matrix
                Legal values are a, b, c, d, and k (for the compression index of sparse matrices).
            gpr_ratio: the number of regnos in each GPR

        Returns:
            Integer indicating how many of the regnos in each GPR to print

        Raises:
            ValueError: An unsupported matrix was requested.
        """
        check_matrix_support(matrix, ('a', 'b', 'c', 'd', 'k'), self.__class__.__name__)
        return math.ceil(gpr_ratio)

    def calculate_single_location(self, matrix: str, out_calc: bool, negate: Dict[str, bool],
                                  reg: int, lane: int, cbsz: int, abid: int, blgp: int,
                                  opsel: int) -> None:
        """ Prints the matrix entries associated for a register and lane combination.

        Calculates and displays the matrix elements for all of the sub-elements of the
        requested register and lane combination. The resulting entry shows the
        register entry (in V#{lane}.[bits] format) and the matrix entry that lives in
        that storage, e.g. A[i][k].

        If a single VGPR holds more than one matrix entry (e.g. a 32b VGPR entry
        holds 2 FP16 values, one in 'lo' and the other in 'hi'), this function prints
        all of the matrix entries, one line at a time.

        If a single matrix entry can't fit in a VGPR (e.g. a 64b matrix entry requires
        2 32b VGPR entries), we print a single entry with V[n+1:n]{lane}, regardless
        of whether the requested register was Vn+1 or Vn.

        Args:
            matrix: string that contains the name of the matrix
                Legal values are a, b, c, d, and k (for the compression index of sparse matrices).
            out_calc: True if, when printing the matrix entry for the D output
                location, you desire to also print the matrix entries of the A, B,
                and C matrices that went into the calculation of the output
            negate: dictionary of matrix names to bools that indicate whether to
                negate and absolute-val entries from this matrix
            reg: integer value of the register number to request
            lane: integer value of the lane to request
            cbsz: integer value of the instruction's CBSZ modifier
            abid: integer value of the instruction's ABID modifier
            blgp: integer value of the instruction's BLGP modifier
            opsel: integer value of the instruction's OPSEL modifier

        Raises:
            ValueError: An i/j/k/block coordinate was not valid for this instruction
                or an unsupported matrix was requested.
        """
        inst_info = self.inst_info
        in_type = inst_info['in_type']
        out_type = inst_info['out_type']
        sparse = inst_info['sparse'] and (matrix.lower() in ('a', 'k'))

        orig_cbsz = cbsz
        orig_abid = abid
        orig_blgp = blgp
        if out_calc:
            cbsz = 0
            abid = 0
            blgp = 0

        # Basic input validation
        if reg < 0:
            raise ValueError(f"Input value for 'register', {reg}, must not be less than 0.")
        if lane < 0:
            raise ValueError(f"Input value for 'lane', {lane}, must not be less than 0.")
        if lane >= self.wave_width:
            raise ValueError(fill(dedent(f"""Input value for 'lane', {lane}, is too large.
                                         Maximum value of lane for any instruction must not be """
                                         "greater than or equal to the wave size of "
                                         f"{self.wave_width-1}.")))
        if lane != self._get_blgp_transformed_lane(lane, blgp):
            transforms = []
            for to_try in range(self.wave_width):
                transforms.append(self._get_blgp_transformed_lane(to_try, blgp))
            if lane not in transforms:
                raise ValueError(f"BLGP input of {blgp} means that lane {lane} "
                                 "will not be used by this instruction.")

        check_matrix_support(matrix, ('a', 'b', 'c', 'd', 'k'), self.__class__.__name__)

        # First, each register is held in a storage location that may be a
        # VGPR (32b values), part of a VGPR (<32b values) or multiple VGPRs
        # (64b values).  So calculating the total number of VGPRs required
        # by a matrix requires knowing how many elements of the matrix we
        # can fit in a VGPR.
        if matrix.lower() in ('a', 'b', 'k'):
            data_size = get_data_size(in_type)
            if matrix.lower() == 'a':
                gpr_ratio = self._get_elements_per_gpr(data_size, sparse)
                total_gprs = self._get_instruction_num_gprs(matrix)
            elif matrix.lower() == 'k':
                gpr_ratio = self._get_elements_per_gpr(data_size, sparse)
                total_gprs = 1
            else:
                gpr_ratio = self._get_elements_per_gpr(data_size, False)
                total_gprs = self._get_instruction_num_gprs(matrix)
        else:
            data_size = get_data_size(out_type)
            gpr_ratio = self._get_elements_per_gpr(data_size, False)
            total_gprs = self._get_instruction_num_gprs(matrix)

        if reg >= total_gprs:
            err_line = fill(dedent(f"""Input value for 'register', {reg}, is too large.
                                   Maximum value of register for {self.inst_name} using """
                                   f"input matrix {matrix.upper()} is {total_gprs - 1}."),
                            width=80)
            raise ValueError(err_line)
        if not self.check_valid_reg_lane(matrix, reg, lane):
            err_line = fill(dedent(f"Input register {reg} and lane {lane} pair is not valid for "
                                   f"the matrix {matrix.upper()} in the instruction "
                                   f"{self.inst_name.upper()} on the {self.arch_name.upper()} "
                                   "architecture."),
                            width=80)
            raise ValueError(err_line)

        # Take the lazy way out for mapping VGPR -> matrix element.
        # Instead of coming up with a calculation, just fill in the element->VGPR table
        # and query it going the other direction.
        register_dict = self.__create_register_dict(matrix, cbsz, abid, blgp, opsel)

        offset = float(self._calculate_initial_regno_offset(matrix, opsel))
        num_regnos_to_print = self._calculate_num_regnos_to_print(matrix, gpr_ratio)

        # We need to address later functions based on the 'storage location',
        # (not the VGPR). See the above comment about multiple values per GPR
        # or multiple GPRs per value. So we calculate the 'regno' based on the
        # user-requested VGPR number and the storage size
        if sparse:
            # Reduce the storage size by half due to being in a 4:2 compression matrix
            # Otherwise, when we try to calculate all the regno below, we will talk
            # past the end of the actual registers available in this instruction
            gpr_ratio /= 2

        num_printed = 0
        opsel_offset = self._get_num_compression_bit_offset(opsel)
        while num_printed < num_regnos_to_print:
            regno = int(reg * gpr_ratio) + int(offset)
            is_k = matrix.lower() == 'k'
            base_gpr_name = self._get_reg_name(data_size, sparse, is_k, cbsz, abid, opsel_offset,
                                               regno)
            gpr_lane_name = self.__format_reg_lane(base_gpr_name, lane)
            if gpr_lane_name not in register_dict:
                if ((matrix.lower() in ('a', 'k')) and cbsz != 0):
                    print(f"Due to instruction modifiers CBSZ and ABID, lane {lane} ", end="")
                elif (matrix.lower() == 'b' and blgp != 0):
                    print(f"Due to instruction modifier BLGP, lane {lane} ", end="")
                else:
                    raise ValueError("An attempt to print too many registers has failed in an "
                                     "unknown way.")
                print("is not used for this instruction.")
                return
            entry_list = register_dict[gpr_lane_name]
            for entry in entry_list:
                print(gpr_lane_name + " = ", end="")
                if (matrix.lower() == 'd' and out_calc):
                    source_string = self.__calculate_source_string(entry, True, negate, orig_cbsz,
                                                                   orig_abid, orig_blgp, opsel)
                    print(f"{entry} = {source_string}")
                else:
                    print(self.__neg_abs_name(base_gpr_name, entry, matrix, negate))
                num_printed += 1
                offset += self.__entries_per_regno(matrix)

    def __entries_per_regno(self, matrix: str) -> float:
        """ Calculates the number of entries per register slot.

        When printing out the matrix entries are specific register/lane combinations, we
        sometimes have more than one register at that location. For instance, SMFMAC
        instructions have 4 entries in a register that would only hold two values -- two of the
        entries are compressed out. For the compression indices, we have the similar type of
        compression. As such, we need to move the "regno" (the actual register{lane}.subset
        combination) ahead by a certain amount for every matrix entry we print.

        This function calculates this scaling value based on the matrix and type of instruction.

        Args:
            matrix: string that contains the name of the matrix
                Legal values are a, b, c, d, and k (for the compression index of sparse matrices).

        Returns:
            Floating point scaling calculation, e.g. a value of 0.5 indicates that there are
            two entries in each "regno" slot, so only move the regno value by 1 only after
            printing twice

        Raises:
            ValueError: An unsupported matrix was requested.
        """
        check_matrix_support(matrix, ('a', 'b', 'c', 'd', 'k'), self.__class__.__name__)
        if ((self.inst_info['sparse'] and matrix.lower() == 'a') or matrix.lower() == 'k'):
            ret_this = 0.5
        else:
            ret_this = 1
        return ret_this

    @staticmethod
    def __get_join_char(output_type: str) -> str:
        """ Returns a character to separate matrix entries in the same table entry.

        Different output tables need different splits for multiple items ending up in the same
        table entry. For normal table printouts, we use newlines. For CSV files, however,
        we do not want to use newlines within double quotes -- while that matches the CSV
        specification, many CSV viewers do not properly handle it. Instead, we use a semi-colon
        that we replace in the format output function.
        For Markdown, we put an HTML line-break because many MD viewers handle that as a
        newline within a table entry.

        Args:
            output_type: string that indicates the type of output, from the list of
                csv, markdown, asciidoc, or grid
        """
        if str(output_type.lower().strip()) == str("csv"):
            join_char = ";"
        elif output_type.lower() == "markdown":
            join_char = "<br />"
        else:
            join_char = "\n"
        return join_char

    @staticmethod
    def __format_output_table(table_to_print: List[List[str]], output_type: str) -> str:
        """ Format the output table as requested.

        Takes an output table, passed as a list of list of strings, and passes it into the
        tabulate tool in a way that generates the requested output type.

        Args:
            table_to_print: List of list of strings, which makes up a 2D table to print
            output_type: string that indicates the type of output, from the list of
                csv, markdown, asciidoc, or grid

        Returns:
            String returned from tabulate, ready to print
        """
        if output_type.lower() == "csv":
            table = tabulate(table_to_print, headers='firstrow', tablefmt='tsv')
            # Convert TSV to CSV
            table = re.sub("\t", ",", table)
            # Remove whitespace characters left over from tabulate's TSV
            table = re.sub(" ", "", table)
            # Replace semi-colons with the proper multi-spacer
            table = re.sub(";", " ", table)
        elif output_type.lower() == "markdown":
            table = tabulate(table_to_print, headers='firstrow', tablefmt='github')
        else:
            table = tabulate(table_to_print, headers='firstrow', tablefmt=output_type.lower())
        return table

    def calculate_register_layout(self, matrix: str, requested_output: str,
                                  negate: Dict[str, bool], cbsz: int, abid: int, blgp: int,
                                  opsel: int, transpose: bool, print_blocks: bool = True) -> None:
        """ Displays the registers+lanes for an entire matrix.

        Calculate and display the registers and lanes for an entire input or
        output matrix. Displays the matrix formatted as its rows and columns,
        with the registers and their lanes displayed in a tabular format.
        Can optionally print this tabular format as a CSV, markdown, or asciidoc
        for other processing. Can also choose to transpose the matrix to visually
        show rows as columns and vice versa.
        The registers are displayed as: Va{b}.c:
            - a is the register number
            - b is the lane within that register
            - c is an optional identifier for sub-Dword parts of the 32 bit register:
                    [15:0]: the lower 16b of a 32b register
                    [31:16]: the upper 16b of a 32b register
                    [7:0]: the least significant 8b of a 32b register
                    [15:8]: the second-lowest 8b of a 32b register
                    [23:16]: the second-highest 8b of a 32b register
                    [31:24]: the most significant 8b of a 32b register

        Args:
            matrix: string that contains the name of the matrix
                Legal values are a, b, c, d, and k (for the compression index of sparse matrices).
            requested_output: string that indicates the type of output, from the list of
                csv, markdown, asciidoc, or grid
            negate: dictionary of matrix names to bools that indicate whether to
                negate and absolute-val entries from this matrix.
            cbsz: integer value of the instruction's CBSZ modifier
            abid: integer value of the instruction's ABID modifier
            blgp: integer value of the instruction's BLGP modifier
            opsel: integer value of the instruction's OPSEL modifier
            transpose: boolean set to true to cause the matrix to be printed transposed
            print_blocks: boolean set to true if this architecture and instruction
                should print the word "Block #" above each block of the matrix

        Raises:
            ValueError: An unsupported matrix was requested.
        """
        check_matrix_support(matrix, ('a', 'b', 'c', 'd', 'k'), self.__class__.__name__)
        inst_info = self.inst_info
        M = inst_info['m']
        N = inst_info['n']
        K = inst_info['k']
        B = inst_info['blocks']
        join_char = self.__get_join_char(requested_output)

        for b in range(B):
            if matrix.lower() in ('a', 'k'):
                if print_blocks:
                    # By setting CBSZ and ABID, it is possible to have multiple blocks
                    # of the matrix math stored in a single register. So we want to
                    # print them as a single group.
                    if matrix.lower() == 'a':
                        if b % math.pow(2, cbsz) != 0:
                            continue
                        block_list = []
                        for new_block in range(b, b + int(math.pow(2, cbsz))):
                            block_list.append(str(new_block))
                        if len(block_list) > 1:
                            print("Blocks ", end="")
                        else:
                            print("Block ", end="")
                        print(", ".join(block_list))
                    else:
                        print(f"Block {b}")

                if not transpose:
                    header = [f"{matrix.upper()}[M][K]"]
                else:
                    header = [f"{matrix.upper()}[K][M]"]
                n = 0
                for k in range(K):
                    header.append(str(k))
                table_to_print = [header]
                for m in range(M):
                    row_tab = [str(m)]
                    for k in range(K):
                        (_, reg, lanes) = self._get_reg_lanes(matrix, m, n, k, b,
                                                              cbsz, abid, blgp, opsel)
                        to_append = []
                        for lane in lanes:
                            reglane = self.__format_reg_lane(reg, lane)
                            to_append.append(self.__neg_abs_name(reg, reglane, matrix, negate))
                        row_tab.append(join_char.join(to_append))
                    table_to_print.append(row_tab)
            elif matrix.lower() == 'b':
                if print_blocks:
                    print(f"Block {b}")
                if not transpose:
                    header = [f"{matrix.upper()}[K][N]"]
                else:
                    header = [f"{matrix.upper()}[N][K]"]
                m = 0
                for n in range(N):
                    header.append(str(n))
                table_to_print = [header]
                for k in range(K):
                    row_tab = [str(k)]
                    for n in range(N):
                        (_, reg, lanes) = self._get_reg_lanes(matrix, m, n, k, b,
                                                              cbsz, abid, blgp, opsel)
                        to_append = []
                        for lane in lanes:
                            reglane = self.__format_reg_lane(reg, lane)
                            to_append.append(self.__neg_abs_name(reg, reglane, matrix, negate))
                        row_tab.append(join_char.join(to_append))
                    table_to_print.append(row_tab)
            else: # matrix.lower() == 'c' or matrix.lower() == 'd'
                if print_blocks:
                    print(f"Block {b}")
                if not transpose:
                    header = [f"{matrix.upper()}[M][N]"]
                else:
                    header = [f"{matrix.upper()}[N][M]"]
                k = 0
                for n in range(N):
                    header.append(str(n))
                table_to_print = [header]
                for m in range(M):
                    row_tab = [str(m)]
                    for n in range(N):
                        (_, reg, lanes) = self._get_reg_lanes(matrix, m, n, k, b,
                                                              cbsz, abid, blgp, opsel)
                        to_append = []
                        for lane in lanes:
                            reglane = self.__format_reg_lane(reg, lane)
                            to_append.append(self.__neg_abs_name(reg, reglane, matrix, negate))
                        row_tab.append(join_char.join(to_append))
                    table_to_print.append(row_tab)
            if transpose:
                table_to_print = list(map(list, zip(*table_to_print)))
            print(self.__format_output_table(table_to_print, requested_output))

    def calculate_matrix_layout(self, matrix: str, requested_output: str, negate: Dict[str, bool],
                                cbsz: int, abid: int, blgp: int, opsel: int, transpose: bool,
                                contig_values: int = 64) -> None:
        """ Displays the matrix entries for all of the registers+lanes used by an instruction.

        Calculate and display the matrix elements for all register entries and
        lanes used by the requesting instruction.
        Displays the registers formatted by register entry (X axis) and wavefront lane
        (Y axis). The resulting table then shows the MatrixName[col][row] held in
        that lane's register entry.
        Can optionally print this tabular format as a CSV, markdown, or asciidoc
        for other processing. Can also choose to transpose the matrix to visually
        show rows as columns and vice versa.

        Args:
            matrix: string that contains the name of the matrix
                Legal values are a, b, c, d, and k (for the compression index of sparse matrices).
            requested_output: string that indicates the type of output, from the list of
                csv, markdown, asciidoc, or grid
            negate: dictionary of matrix names to bools that indicate whether to
                negate and absolute-val entries from this matrix
            cbsz: integer value of the instruction's CBSZ modifier
            abid: integer value of the instruction's ABID modifier
            blgp: integer value of the instruction's BLGP modifier
            opsel: integer value of the instruction's OPSEL modifier
            transpose: boolean set to true to cause the matrix to be printed transposed
            contig_values: an integer that defines the number of contiguous values of a
                register that are used to hold unique values of a matrix

        Raises:
            ValueError: An unsupported matrix was requested.
        """
        check_matrix_support(matrix, ('a', 'b', 'c', 'd', 'k'), self.__class__.__name__)
        inst_info = self.inst_info
        M = inst_info['m']
        N = inst_info['n']
        K = inst_info['k']
        B = inst_info['blocks']
        in_type = inst_info['in_type']
        out_type = inst_info['out_type']
        sparse = inst_info['sparse'] and (matrix.lower() in ('a', 'k'))
        join_char = self.__get_join_char(requested_output)

        register_dict = self.__create_register_dict(matrix, cbsz, abid, blgp, opsel)

        if matrix.lower() in ('a', 'k'):
            data_size = get_data_size(in_type)
            # In sparse matrices, there are only half as many registers
            # due to the 4:2 compression.
            if sparse:
                K = int(K / 2)
            total_gpr_slots = int(M * K * B / contig_values)
        elif matrix.lower() == 'b':
            data_size = get_data_size(in_type)
            total_gpr_slots = int(N * K * B / contig_values)
        else:
            data_size = get_data_size(out_type)
            total_gpr_slots = int(M * N * B / contig_values)

        header = ["lane"]
        table_to_print = []
        found_regnos = []
        seen_regno = set()
        opsel_offset = self._get_num_compression_bit_offset(opsel)
        for lane in range(self.wave_width):
            lane = self._get_blgp_transformed_lane(lane, blgp)
            row_tab = [str(lane)]
            for regno in range(total_gpr_slots):
                base_gpr_name = self._get_reg_name(data_size, sparse, matrix.lower() == 'k',
                                                   cbsz, abid, opsel_offset, regno)
                gpr_lane_name = self.__format_reg_lane(base_gpr_name, lane)
                # If we have CBSZ and ABID set, some lanes may not exist in this
                # table, so skip over putting them in the list to print.
                if gpr_lane_name not in register_dict:
                    continue
                matrix_element = register_dict[gpr_lane_name]
                elements_to_print = []
                for to_print in matrix_element:
                    to_print = self.__neg_abs_name(base_gpr_name, to_print, matrix, negate)
                    elements_to_print.append(to_print)
                if not sparse:
                    row_tab.append(join_char.join(elements_to_print))
                elif regno % 2 == 0:
                    row_tab.append(join_char.join(elements_to_print))
                if regno not in seen_regno:
                    seen_regno.add(regno)
                    found_regnos.append(regno)
            # If we skipped over lanes due to CBSZ and ABID, then the only thing
            # we will have in this row is the lane number. Don't put it
            # in the table to print.
            if len(row_tab) > 1:
                table_to_print.append(row_tab)

        for regno in found_regnos:
            # In sparse matrices, the 4:2 compression puts 4 values into
            # 2 register slots. For instance, the lower slot could have
            # value 3 and the upper slot could have 3. As such, we need
            # to print both slots containing all 4 values. As such, we
            # only print 1/4 of the table entries, and put more info into
            # each of the entries.
            # We already cut K in half above to create half as many
            # register slots. Now skip every other one because we will
            # fill them with 4 matrix entries.
            if (not sparse or regno % 2 == 0):
                header.append(self._get_reg_name(data_size, sparse, matrix.lower() == 'k',
                                                 cbsz, abid, opsel_offset, regno))

        deduplicated = []
        for x in table_to_print:
            if x not in deduplicated:
                deduplicated.append(x)
        deduplicated.sort(key=lambda x: int(x[0]))
        deduplicated.insert(0, header)
        table_to_print = deduplicated
        if transpose:
            table_to_print = list(map(list, zip(*table_to_print)))
        print(self.__format_output_table(table_to_print, requested_output))

    def _get_instruction_num_gprs(self, matrix: str, in_lanes: Optional[int] = None,
                                  out_size: Optional[int] = None) -> int:
        """ Calculates the number of GPRs needed to hold a matrix.

        Args:
            matrix: string that contains the name of the matrix
                Legal values are a, b, c, d, or k (for the compression index of sparse matrices).
            in_lanes: an integer that defines the number of contiguous lanes that are used to hold
                values of a matrix. If this is not passed in, the argument is matched to the
                wavefront size of the target architecture.
            out_size: The number of bits used to hold output values for this instruction
                For instance: some devices may store 16b values into either the low or high
                half of a 32b output register. These architectures would need out_type=32.
                If this is not passed in, the argument is matched to the actual instruction's
                output data size (e.g. 16b in this example case).

        Returns:
            An integer that defines the number of GPRs needed to hold the requested matrix

        Raises:
            ValueError: An unsupported matrix was requested.
        """
        check_matrix_support(matrix, ('a', 'b', 'c', 'd', 'k'), self.__class__.__name__)
        inst_info = self.inst_info
        if in_lanes is None:
            in_lanes = self.wave_width
        sparse_op = (matrix.lower() == 'a' and inst_info['sparse'])
        if matrix.lower() in ('a', 'b'):
            lanes_used = int(in_lanes)
            gpr_ratio = self._get_elements_per_gpr(get_data_size(inst_info['in_type']), sparse_op)
        else:
            if out_size is None:
                out_size = get_data_size(inst_info['out_type'])
            lanes_used = int(self.wave_width)
            gpr_ratio = self._get_elements_per_gpr(out_size, False)
        if matrix.lower() in ('a', 'c', 'd'):
            rows = inst_info['m']
        else:
            rows = inst_info['k']
        if matrix.lower() in ('b', 'c', 'd'):
            cols = inst_info['n']
        else:
            cols = inst_info['k']
        return int(math.ceil(rows * cols * inst_info['blocks'] / (lanes_used * gpr_ratio)))

    @abstractmethod
    def _coord_to_input_reg_eqn(self, matrix: str, wave_size: Optional[int] = None) ->str:
        """ Returns formula for mapping a matrix coordinate to its input register number.

        Takes the instruction info and matrix, return a string with an equation that lets a user
        calculate the input register that holds a particular entry in the matrix from its
        i/j/k/block coordinates.

        This is an abstract method, because every architecture needs a different formula.
        Should be filled in by the architecture child classes.

        Args:
            matrix: string that contains the name of the matrix
                Legal values are a, b, or k (for the compression index of sparse matrices).
            wave_size: integer that holds the wave size for the requested conversion function
                Some architectures change register equations for different wave sizes.

        Returns:
            String that contains the simple formula mapping coordinates to input registers
        """

    @abstractmethod
    def _coord_to_output_reg_eqn(self, wave_size: Optional[int] = None) -> str:
        """ Returns formula for mapping a matrix coordinate to its output register number.

        Takes the instruction info and matrix, return a string with an equation that lets a user
        calculate the output register that holds a particular entry in the matrix from its
        i/j/k/block coordinates.

        This is an abstract method, because every architecture needs a different formula.
        Should be filled in by the architecture child classes.

        Args:
            wave_size: integer that holds the wave size for the requested conversion function
                Some architectures change register equations for different wave sizes.

        Returns:
            String that contains the simple formula mapping coordinates to output registers
        """

    def __coord_to_reg_eqn(self, matrix: str, wave_size: Optional[int] = None) -> str:
        """ Returns formula for mapping a matrix coordinate to its register number.

        Takes the instruction info and matrix, return a string with an equation that lets a user
        calculate the register that holds a particular entry in the matrix from its i/j/k/block
        coordinates.

        Args:
            matrix: string that contains the name of the matrix
                Legal values are a, b, c, d, or k (for the compression index of sparse matrices).
            wave_size: integer that holds the wave size for the requested conversion
                function. Some architectures change register equations for different wave sizes.

        Returns:
            String that contains the simple formula mapping coordinates to registers

        Raises:
            ValueError: An unsupported matrix was requested.
        """
        check_matrix_support(matrix, ('a', 'b', 'c', 'd', 'k'), self.__class__.__name__)
        if matrix.lower() in ('a', 'b', 'k'):
            ret_str = self._coord_to_input_reg_eqn(matrix, wave_size)
        else: # C/D matrices
            ret_str = self._coord_to_output_reg_eqn(wave_size)
        return ret_str

    @abstractmethod
    def _coord_to_lane_eqn(self, matrix: str, wave_size: Optional[int] = None) -> str:
        """ Returns formula for mapping a matrix coordinate to its wavefront lane.

        Takes the instruction info and matrix, return a string with an equation that lets a user
        calculate the lane that holds a particular entry in the matrix from its i/j/k/block
        coordinates.

        This is an abstract method, because every architecture needs a different formula.
        Should be filled in by the architecture child classes.

        Args:
            matrix: string that contains the name of the matrix
                Legal values are a, b, c, d, or k (for the compression index of sparse matrices).
            wave_size: integer that holds the wave size for the requested conversion
                function. Some architectures change lane equations for different wave sizes.

        Returns:
            String that contains the simple formula mapping coordinates to lanes
        """

    def _print_element_to_register_eqn(self, block: str = "Unknown",
                                       wave_size: Optional[int] = None) -> None:
        """ Prints formula for matrix entry to GPR and lane mapping.

        Prints out the simple formulae to calculate the the mapping of a matrix element to its
        register and lane.

        Will always print A[], B[], and D[]. If the instruction allows C[] matrices, then it will
        print D[] as "C or D[i][j]". For sparse instructions, will also print the compression
        matrix.

        Args:
            block: string that contains text to place in the matrix entry map for blocks
                For instance, pass ".block" if an architecture should print an entry as
                A[i][k].block. Pass "" to just print A[i][k].
            wave_size: integer that contains the requested wave size to use for printing the
                formula. Ignore this argument to avoid printing the wave size.
        """
        if wave_size is None:
            print("    Matrix element to register mapping with no modifiers:")
        else:
            print(f"    Wave{wave_size} matrix element to register mapping with no modifiers:")
        print(f"        A[i][k]{block} GPR: {self.__coord_to_reg_eqn('a', wave_size)}")
        print(f"        A[i][k]{block} Lane: {self._coord_to_lane_eqn('a', wave_size)}")
        if not self.inst_info['sparse']:
            cd_str = "C or D"
        else:
            cd_str = "D"
            self.__print_long_element_eqn("compression[i][k] GPR",
                                          self.__coord_to_reg_eqn('k', wave_size))
            self.__print_long_element_eqn("compression[i][k] Lane",
                                          self._coord_to_lane_eqn('k', wave_size))
        print(f"        B[k][j]{block} GPR: {self.__coord_to_reg_eqn('b', wave_size)}")
        print(f"        B[k][j]{block} Lane: {self._coord_to_lane_eqn('b', wave_size)}")
        print(f"        {cd_str}[i][j]{block} GPR: {self.__coord_to_reg_eqn('d', wave_size)}")
        print(f"        {cd_str}[i][j]{block} Lane: {self._coord_to_lane_eqn('d', wave_size)}")

    def __reg_lane_to_input_ij_coord_eqn(self) -> str:
        """ Returns equation to map register+lane to i or j index for input matrices.

        Takes instruction info and returns a string containing an equation which lets users
        calculate the A matrix's i coordinate or the B matrix's j coordinate based on the
        register and lane.

        Returns:
            A string which contains the equation to calculate the i or j coordinate for
            an input matrix held by a particular register and lane combination
        """
        return f"(lane % {self.inst_info['m']})"

    def _reg_lane_to_i_coord_eqn(self, matrix: str,
                                 wave_size: Optional[int] = None) -> str:
        """ Returns equation to map register+lane to i index.

        Takes instruction info and returns a string containing an equation which lets users
        calculate the i coordinate for the A, C, D, or compression index matrices.

        c and d matrix calculations are machine-specific, and should be handled by
        child classes overloading this function.

        Args:
            matrix: string that contains the name of the matrix
                Legal values are a, c, d, or k (for the compression index of sparse matrices).
            wave_size: integer that holds the wave size for the requested conversion function
                Legal values are 32 and 64.

        Returns:
            A string which contains the equation to calculate the i coordinate for
            an input matrix held by a particular register and lane combination

        Raises:
            ValueError: An unsupported matrix was requested.
        """
        check_matrix_support(matrix, ('a', 'c', 'd', 'k'), self.__class__.__name__)
        del wave_size # Unused by base class
        ret_string = "Unknown" # c and d matrix must be handled by child classes
        if matrix.lower() == 'a':
            ret_string = self.__reg_lane_to_input_ij_coord_eqn()
        elif matrix.lower() == 'k':
            ret_string = f"(lane % {self.inst_info['m']})"
        return ret_string

    def __reg_lane_to_output_j_coord_eqn(self) -> str:
        """ Returns equation to map register+lane to j index for an output register.

        Takes instruction info and a target matrix, and returns a string containing an
        equation which lets users calculate the j coordinate based on the register and lane.
        Targets a particular instruction, and does not handle modifiers.

        Returns:
            A string which contains the equation to calculate the j coordinate held by a
            particular output register and lane combination
        """
        return f"(lane % {self.inst_info['m']})"

    def __reg_lane_to_j_coord_eqn(self, matrix: str) -> str:
        """ Returns equation to map register+lane to j index.

        Takes instruction info and a target matrix, and returns a string containing an
        equation which lets users calculate the j coordinate based on the register and lane.
        Targets a particular instruction, and does not handle modifiers.

        Args:
            matrix: string that contains the name of the matrix
                Legal values are b, c, and d.

        Returns:
            A string which contains the equation to calculate the j coordinate held by a
            particular register and lane combination

        Raises:
            ValueError: An unsupported matrix was requested.
        """
        check_matrix_support(matrix, ('b', 'c', 'd'), self.__class__.__name__)
        if matrix.lower() == 'b':
            ret_string = self.__reg_lane_to_input_ij_coord_eqn()
        else: # c or d
            ret_string = self.__reg_lane_to_output_j_coord_eqn()
        return ret_string

    @abstractmethod
    def _reg_lane_to_k_coord_eqn(self, matrix: str,
                                 wave_size: Optional[int] = None) -> str:
        """ Returns equation to map register+lane to k index.

        Takes instruction info and a target matrix, and returns a string containing an
        equation which lets users calculate the k coordinate based on the register and lane.
        Targets a particular instruction, and does not handle modifiers.

        Because every architecture will treat this equation differently, this is an abstract
        method and must be filled in by child classes.

        Args:
            matrix: string that contains the name of the matrix
                Legal values are a, b, and k (for the compression index of sparse matrices).
            wave_size: integer that holds the wave size for the requested conversion function
                Legal values are 32 and 64.

        Returns:
            A string which contains the equation to calculate the k coordinate held by a
            particular register and lane combination
        """

    @abstractmethod
    def _reg_lane_to_block_eqn(self, matrix: str) -> str:
        """ Returns equation to map register+lane to block.

        Return a string that can be used to quickly calculate how to go from a register and
        lane to the block of the input or output matrix. Targets a particular instruction,
        and does not handle modifiers.

        Because every architecture will treat this equation differently, this is an abstract
        method and must be filled in by child classes.

        Args:
            matrix: string that contains the name of the matrix
                Legal values are a, b, c, d, and k (for the compression index of sparse matrices).

        Returns:
            A string which contains the equation to calculate the block held by a particular
            register and lane combination
        """

    @staticmethod
    def __print_long_element_eqn(lead_line: str, to_print: str) -> None:
        """ Print long equation lines formatted so newlines align with leading label.

        Split long equations, such as for ranges of sparse matrices, that have manually
        put newlines in. Format the horizontal locations so that the newlines all line
        up with a leading label.

        Args:
            lead_line: string that holds the line (such as the foo in Foo: Bar_Equation)
            to_print: string that holds the long equation to print (such as the Bar_Equation)
        """
        lead_line = f"        {lead_line}: "
        next_line = f"{' '*len(lead_line)}"
        first_wrapper = TextWrapper(initial_indent=lead_line, width=100,
                                    subsequent_indent=" "*len(lead_line))
        second_wrapper = TextWrapper(initial_indent=next_line, width=100,
                                     subsequent_indent=" "*len(next_line))
        first_time = False
        for x in to_print.splitlines():
            if not first_time:
                first_time = True
                print(first_wrapper.fill(x))
            else:
                print(second_wrapper.fill(x))

    def _print_register_to_element_eqn(self, print_block: bool = False,
                                       wave_size: Optional[int] = None,
                                       max_a_lane: Optional[int] = None,
                                       max_b_lane: Optional[int] = None) -> None:
        """ Prints equation to map register+lane to matrix element.

        Print out simple equations for mapping a register and its lane to the element in the
        matrix and block which they hold. These can be used by developers that do not want
        to set modifiers like CBSZ/ABID or BLGP to quickly unpack values from registers to
        put them into output matrix locations.

        Args:
            print_block: True if this equation should print information about blocks
            wave_size: integer that holds the wave size for the requested conversion function
                Legal values are 32, 64, or passing nothing.
            max_a_lane: optional integer that holds the maximum lane that can be used in this
                instruction for A and K matrices.
            max_b_lane: optional integer that holds the maximum lane that can be used in this
                instruction for B matrices.
        """
        inst_info = self.inst_info
        sparse_op = inst_info['sparse']

        if wave_size is None:
            print("    Register to matrix element mapping with no modifiers:")
        else:
            print(f"    Wave{wave_size} register to matrix element mapping with no modifiers:")
        print(f"        A i: {self._reg_lane_to_i_coord_eqn('a', wave_size)}")
        if not sparse_op:
            print(f"        A k: {self._reg_lane_to_k_coord_eqn('a', wave_size)}")
        else:
            self.__print_long_element_eqn("A k", self._reg_lane_to_k_coord_eqn('a', wave_size))
        if print_block:
            print(f"        A block: {self._reg_lane_to_block_eqn('a')}")
        if max_a_lane is not None:
            print(f"            All lanes > {max_a_lane} are ignored for A matrix")
        if not inst_info['sparse']:
            cd_str = "C or D"
        else:
            cd_str = "D"
            print(f"        compression i: {self._reg_lane_to_i_coord_eqn('k', wave_size)}")
            self.__print_long_element_eqn("compression k",
                                          self._reg_lane_to_k_coord_eqn('k', wave_size))
            if max_a_lane is not None:
                print(f"            All lanes > {max_a_lane} are ignored for compression indices")
        print(f"        B j: {self.__reg_lane_to_j_coord_eqn('b')}")
        print(f"        B k: {self._reg_lane_to_k_coord_eqn('b', wave_size)}")
        if print_block:
            print(f"        B block: {self._reg_lane_to_block_eqn('b')}")
        if max_b_lane is not None:
            print(f"            All lanes > {max_b_lane} are ignored for B matrix")
        print(f"        {cd_str} i: {self._reg_lane_to_i_coord_eqn('d', wave_size)}")
        print(f"        {cd_str} j: {self.__reg_lane_to_j_coord_eqn('d')}")
        if print_block:
            print(f"        {cd_str} block: {self._reg_lane_to_block_eqn('d')}")

    def _print_opcode(self, encoding_name: str = "Unknown") -> None:
        """ Prints encoding name and VOP3P opcode for an instruction.

        Args:
            encoding_name: String containing the name of the encoding format for the
                current architecture
                Defaults to "Unknown", and should either be filled in by child class
                specializations of this function or directly by callers.
        """
        print(f"    Encoding: {encoding_name}")
        print(f"    VOP3P Opcode: {self.inst_info['opcode']:#02x}")

    def _print_matrix_dims(self) -> None:
        """ Prints the matrix dimensions for a matrix multiplication instruction. """
        inst_info = self.inst_info
        M = inst_info['m']
        N = inst_info['n']
        K = inst_info['k']

        print("    Matrix Dimensions:")
        print(f"        M: {M}")
        print(f"        N: {N}")
        print(f"        K: {K}")

    def _print_execution_statistics(self, cu_name: str = "Unknown") -> None:
        """ Prints execution statistics for a matrix multiplication instruction.

        Prints the execution statistics, such as computational throughput and co-execution
        information, for a matrix multiplication instruction on a target architecture.
        The instruction and architecture are part of the class.

        Args:
            cu_name: string that contains the name of the compute unit on the target arch
                Different architectures may use e.g., CU, or WGP, or some other name.
                Defaults to "Unknown", and should either be filled in by child class
                specializations of this function or directly by callers.
        """
        inst_info = self.inst_info
        M = inst_info['m']
        N = inst_info['n']
        K = inst_info['k']
        B = inst_info['blocks']
        cycles = inst_info['cycles']

        ops = B * M * N * K * 2
        ops_per_cycle = int(ops/cycles)
        ops_per_cu_per_cycle = int(ops_per_cycle * 4)

        op_name = "Ops" if inst_info['integer'] else "FLOPs"

        can_valu_coexec = inst_info['coexec']
        if can_valu_coexec:
            possible_coexec_cycles = cycles - inst_info['coexec_delay']
            if possible_coexec_cycles <= 0:
                can_valu_coexec = False
        print("    Execution statistics:")
        print(f"        {op_name}: {ops}")
        print(f"        Execution cycles: {cycles}")
        print(f"        {op_name}/{cu_name}/cycle: {ops_per_cu_per_cycle}")
        print(f"        Can co-execute with VALU: {can_valu_coexec}")
        if can_valu_coexec:
            print(f"        VALU co-execution cycles possible: {possible_coexec_cycles}")

    def _print_register_usage(self, wave_sizes: Tuple[int, ...] = ()) -> None:
        """ Prints the register count for each input and output matrix.

        Prints the number of registers used by each matrix for a matrix multiplication
        instruction on a target architecture. Some architectures support more than one
        wavefront size. In that case, print out the register usage for each wavefront size.
        This is needed because different wavefront sizes can cause different GPR usage for
        the same matrix size.

        Args:
            wave_sizes: a tuple of wavefront sizes to use to print this info
                Defaults to an empty tuple, and should either be filled in by child class
                specializations of this function or directly by callers.
        """
        inst_info = self.inst_info
        for size in wave_sizes:
            self.wave_width = size
            total_in_a_gprs = self._get_instruction_num_gprs('a', in_lanes=size)
            total_in_b_gprs = self._get_instruction_num_gprs('b', in_lanes=size)
            total_out_gprs = self._get_instruction_num_gprs('d')
            if len(wave_sizes) == 1:
                print("    Register usage:")
            else:
                print(f"    Wave{size} register usage:")
            print(f"        GPRs required for A: {total_in_a_gprs}")
            print(f"        GPRs required for B: {total_in_b_gprs}")
            # Sparse matrix instructions are D += A * B, so there is no C matrix at all.
            # Therefore, skip printing information about the C matrix.
            if not inst_info['sparse']:
                print(f"        GPRs required for C: {total_out_gprs}")
            print(f"        GPRs required for D: {total_out_gprs}")
            print(f"        GPR alignment requirement: {inst_info['gpr_byte_align']} bytes")

    def _print_register_types(self) -> None:
        """ Prints the data type for the registers used in this matrix instruction.

        Prints the type that this instruction will use to interpret the data held in
        each of the registers used by this instruction.
        """
        print("    Register data types:")
        print(f"        Src0: {get_type_desc(self.inst_info['in_type'])}")
        print(f"        Src1: {get_type_desc(self.inst_info['in_type_src1'])}")
        if not self.inst_info['sparse']:
            print(f"        Src2: {get_type_desc(self.inst_info['out_type'])}")
        else:
            print("        Src2: A matrix compression indices")
        print(f"        Vdst: {get_type_desc(self.inst_info['out_type'])}")

    def _print_register_info(self, encoding_name: str = "Unknown") -> None:
        """ Prints the encoding and register information for a matrix instruction.

        Prints the encoding and modifier information for the registers used by a
        matrix multiplication instruction on a particular architecture.
        Architecture matters, because different architectures support different
        encodings and matrix modifier fields.

        Args:
            encoding_name: String containing the name of the encoding format for the
                current architecture
                Defaults to "Unknown", and should either be filled in by child class
                specializations of this function or directly by callers.
        """
        print(f"    {encoding_name} register encoding:")
        print("        A matrix source field: Src0")
        print("        B matrix source field: Src1")
        if not self.inst_info['sparse']:
            print("        C matrix source field: Src2")
        else:
            print("        Compression index field: Src2")
        print("        D matrix source field: Vdst")
        self._print_register_types()

    def print_instruction_information(self) -> None:
        """ Prints full information about the desired instruction on the target architecture. """
        print_arch_inst(self.arch_name, self.inst_name)
        self._print_opcode()
        self._print_matrix_dims()
        self._print_execution_statistics()
        self._print_register_usage()
        self._print_register_info()
        self._print_element_to_register_eqn()
        self._print_register_to_element_eqn()


class InstCalcGfx9(InstCalc):
    """ Calculator for matrix multiplication instruction details on gfx9 architecture.

    This is a child class of the InstCalc class, because gfx9 architectures require certain
    different calculations that other architectures. CDNA1, CDNA2, and CDNA3 all use the same
    rough calculations for placing matrix values into registers, though the instructions, their
    dimensions, and throughputs change in each generation.

    Attributes:
        arch_name: string that holds the accelerator architecture's name
        inst_name: string holding the mnemonic of the instruction that will be used in calculations
        inst_info: MatrixInstruction holding the details of the instruction used in calculations
        wave_width: Some architectures allow wavefronts of various widths, and these widths can
            affect the resulting calculations. This integer holds the width that will be used for
            further calculations.
    """

    def check_valid_reg_lane(self, matrix: str, register: int, lane: int) -> bool:
        """ Checks if the register and lane being used for calculation are legal

        Checks whether the matrix and lane being requested are legal for indexing into the
        requested matrix. Callers should use this to ensure that the register is within the
        bounds of the matrix. In addition, gfx9 only allows lanes 0-63, inclusive.

        Args:
            matrix: String indicating the matrix to query: 'a', 'b', 'c', 'd', or 'k'
                for the compression index matrix in sparse matrix ops
            register: Integer of the register number being requested, starting from 0
            lane: Integer of the lane being requested, starting from 0

        Returns:
            True if the register and lane contain data for this matrix on the current instruction
            False if the register and lane do not contain data for this matrix in this instruction
        """
        if (lane < 0 or lane > 63):
            return False
        num_gprs = self._get_instruction_num_gprs(matrix)
        if (register < 0 or register > num_gprs):
            return False
        return True

    def get_num_compression_sets(self) -> int:
        """ Returns the number of compression index sets that an instruction has

        Returns:
            An integer that contains the number of compression index sets that an
                instruction allows for the given configuration
        """
        ret_val = 0
        if self.arch_name.lower() == 'cdna3':
            if get_data_size(self.inst_info['in_type']) == 16:
                ret_val = 4
            elif get_data_size(self.inst_info['in_type']) == 8:
                ret_val = 2
        return ret_val

    def __get_input_reg_lanes(self, M: int, K: int, B: int, i: int, k: int, b: int,
                              data_size: int, sparse: bool, compression_index: bool,
                              k_cbsz: int, k_abid: int, blgp: int) -> Tuple[str, List[int]]:
        """ Calculates a matrix's input register and lane number based on coordinates.

        For gfx9, calculates the input register and the lane within that register for an
        instruction based on its parameters. The algorithm for calculating these is
        described in the comments within the function.

        Args:
            M: integer "outer" dimension of the input matrix, in matrix entries
                For A and K matrices, this is the height. For B matrices, this is the width
            K: integer "inner" dimension of the input matrix, in matrix entries
                For A and K matrices, this is the width. For B matrices, this is the height
            B: integer number of blocks in the input matrix
            i: integer location within the input matrix's "outer" dimension
                For A and K matrices, this is the desired row
                For B matrices this is the desired column
            k: integer location within the input matrix's "inner" dimension
                For A and K matrices, this is the desired column
                For B matrices, this is the desired row
            b: integer block number within the input matrix
            data_size: integer size of the input data, in bits
            sparse: True if this register is working on a structured sparsity input register
            compression_index: True if this register holds the compression index for a
                structured sparsity instruction
            k_cbsz: When working on compression indices, a non-zero CBSZ value will cause
                different register locations to be used for the compression index. As such,
                when passing in compression_index=True, this integer field holds the
                instruction's CBSZ modifier.
            k_abid: When working on compression indices, a non-zero CBSZ value will cause
                different register locations to be used for the compression index. The ABID
                field chooses which register location to use. As such, when passing in
                compression_index=True and k_cbsz!=0, this integer field holds the
                instruction's ABID modifier.
            blgp: integer value of the instruction's BLGP modifier

        Returns:
            Based on the matrix and requested coordinates, return two things in a tuple:
            1. String register name that holds the element, in the format V#.[bits]
            2. List of integers, containing the lane numbers within that register that
                hold the element
            Tuple: (register holding the matrix entry, lanes within that register)
        """
        # Start by calculating how many elements of the matrix there are in the
        # neighboring pair of VGPRs. For instance, for F64 inputs, a lane of
        # neighboring VGPRs 1:0 will hold 1 element.
        # While for F16 inputs, a lane of the same two registers will hold
        # 4 elements.
        # F32 MFMA can be different, because only one VGPR holds the F32 input
        # so we use the term "contiguous gprs" instead of VGPR pair here.
        # The MFMA instructions let you calculate this from the height,
        # width, and number of blocks used for the input matrix.
        elements_in_contiguous_gprs = int(64 / (M * B))
        elements_in_contiguous_gprs = int(K / elements_in_contiguous_gprs)

        # The storage (VGPR pair, 32b VGPR, sub-32b section of a VGPR)
        # that holds the matrix element can be calculated by taking the
        # k dimension, which walks between register elements, and making
        # sure we take into account that some types (like some FP32 instructions)
        # will not move to different registers as we walk over k.
        local_element = k % elements_in_contiguous_gprs
        register_name = self._get_reg_name(data_size, sparse, compression_index, k_cbsz, k_abid,
                                           0, local_element)

        # The lane within the chosen register has three parts:
        # First: every block will walk over an entire column of the input (if
        # we are talking about the A matrix), so if working on later blocks,
        # offset over the lanes that hold previous blocks
        lane = b * M
        # Walking across the A matrix's columns moves us through storage
        # locations, but should not change lanes until we have gone through
        # a whole contiguous-GPR.
        # At that point, the next column is stored after the same columns
        # for all other blocks, so offset past all of them.
        lane += int(k / elements_in_contiguous_gprs) * M * B
        # Finally, index into this based on the row within the column (if A matrix)
        lane += i

        # Perform BLGP transformation. This is only legal for B matrices, so no one
        # should pass in blgp!=0 into this function for other matrices.
        lane = self._get_blgp_transformed_lane(lane, blgp)

        return (register_name, [lane])

    def __get_output_reg_lanes(self, M: int, N: int, i: int, j: int, b: int,
                               data_size: int) -> Tuple[str, List[int]]:
        """ Calculates a matrix's output register and lane number based on coordinates.

        For gfx9, calculates the output register and the lane within that register for
        an instruction based on its parameters. The algorithm for calculating these
        is described in the comments within the function.

        Args:
            M: integer height of the output matrix, in matrix entries
            N: integer width of the output matrix, in matrix entries
            i: integer location within the output matrix's rows
            j: integer location within the output matrix's columns
            b: integer block number within the output matrix
            data_size: integer size of the output data, in bits

        Returns:
            Based on the matrix and requested coordinates, return two things in a tuple:
            1. String register name that holds the element, in the format V#.[bits]
            2. List of integers, containing the lane numbers within that register that
                hold the element
            Tuple: (register holding the matrix entry, lanes within that register)
        """
        # The 64b output layout and 32b output layout are different.
        # 32b outputs are written out in multi-row segments that are N columns (lanes) wide and
        # 4 rows (registers) tall.
        # After completing one of these 4xN multi-rows, we move to the next set of lanes in this
        # register.
        # When this register is full, move down 4 registers and start again.
        # 64b outputs are a similar algorithm, but the output is only 1 row (register-pair) tall.
        multirows_per_register = int(64/N)
        if data_size == 64:
            multirow_height = 1
        else:
            multirow_height = 4
        # This is the same as B_I:
        blocks_per_register = math.ceil(64 / int((N * M) / multirow_height))

        # Find which register (or register-pair) this matrix element is in
        # Start by calculating where the requested block will live.
        # Each register is 64 entries wide, and a block will take up M*N of them.
        local_element = b * int((M * N) / 64)
        # Within the block, move to the next starting register after going through
        # 'multirows_per_register' multi-row heights. That starting register is
        # a multirow height further into the register space.
        local_element += int(i / (multirow_height * multirows_per_register)) * multirow_height
        # Finally, choose the register for the row within the multi-row
        local_element += int(i % multirow_height)
        register_name = self._get_reg_name(data_size, False, False, 0, 0, 0, local_element)

        # Logic to find the lane within the register found above
        # Set the initial offset
        # If there are multiple blocks in this register, offset by a row for each one
        lane = (b % blocks_per_register) * N
        # Further offset into the chunk; if there is more than one multi-row in this
        # register, then after every 'multirow_height' rows, move the starting point
        # over N lane per block in this row
        lane += (int(i / multirow_height) % multirows_per_register) * blocks_per_register * N
        # Finally, directly move based on the column
        lane += j
        return (register_name, [lane])

    def _get_reg_lanes(self, matrix: str, i: int, j: int, k: int, block: int, cbsz: int,
                       abid: int, blgp: int, opsel: int) -> Tuple[str, str, List[int]]:
        """ Calculates a matrix's register and lane number based on coordinates.

        For the target architecture and the instruction set up in this class's init
        function, this function calculates the register and lane that hold a requested
        matrix entry in a requested matrix. This location information can vary based on
        per-instruction modifiers, so those are all input arguments.

        opsel is ignored by this function, because gfx9 does not use this instruction modifier.
        Legal matrix values are 'a', 'b', 'c', 'd', and 'k'.

        Args:
            matrix: String indicating the matrix to query: 'a', 'b', 'c', 'd', or 'k'
                for the compression index matrix in sparse matrix ops
            i: integer coordinate for the query of the matrix row for A, C, D, & K matrices
            j: integer coordinate for the query of the matrix column for the B, C, & D matrices
            k: integer coordinate for the query of the A & K column or B row
            block: integer coordinate for the block to query
            cbsz: integer value of the instruction's CBSZ modifier
            abid: integer value of the instruction's ABID modifier
            blgp: integer value of the instruction's BLGP modifier
            opsel: integer value of the instruction's OPSEL modifier

        Returns:
            Based on the matrix and requested coordinates, return three things in a tuple:
            1. String requested element name, in the format A[i][k], B[k][j], or C[i][j]
            2. String register name that holds the element, in the format V#.[bits]
            3. List of integers, containing the lane numbers within that register that
                hold the element
            Tuple: (matrix entry, register holding that entry, lanes within that register)
        """
        del opsel # Unused in gfx9

        inst_info = self.inst_info
        M = inst_info['m']
        N = inst_info['n']
        K = inst_info['k']
        B = inst_info['blocks']
        # Leave these as false out here, in case we are checking against matrix B
        sparse = False
        compress_index = False
        if matrix.lower() in ('a', 'b', 'k'):
            size = get_data_size(inst_info['in_type'])
        else:
            size = get_data_size(inst_info['out_type'])
        if matrix.lower() in ('a', 'k'):
            # For 4:2 structural sparsity, on the A matrix the k dimension fits
            # 2 values in what would have traditionally been 4 storage locations
            # We thus cut K in half for calculating the register and lane, because
            # we don't need half of the registers a normal M*K calculation would
            # give you. We also need to cut k in half because the user requested
            # a k based on that original value, so we need to scale it too.
            # However, we use 'k_to_calc' because we need the original 'k' for
            # string that is returned to the screen.
            if inst_info['sparse']:
                K = int(K / 2)
                k_to_calc = int(k / 2)
                post_cbsz_abid_block = block
                sparse = True
            else:
                post_cbsz_abid_block = self._get_cbsz_abid_transformed_block(block, cbsz, abid)
                k_to_calc = k
            if matrix.lower() == 'k':
                compress_index = True
                (reg, lanes) = self.__get_input_reg_lanes(M, K, B, i, k_to_calc,
                                                          post_cbsz_abid_block, size, sparse,
                                                          compress_index, cbsz, abid, blgp)
            else:
                (reg, lanes) = self.__get_input_reg_lanes(M, K, B, i, k_to_calc,
                                                          post_cbsz_abid_block, size, sparse,
                                                          compress_index, 0, 0, blgp)
            element_name = f"{matrix.upper()}[{i}][{k}]"
        elif matrix.lower() == 'b':
            (reg, lanes) = self.__get_input_reg_lanes(N, K, B, j, k, block, size, sparse,
                                                      compress_index, 0, 0, blgp)
            element_name = f"{matrix.upper()}[{k}][{j}]"
        else: # (matrix.lower() == 'c' or matrix.lower() == 'd'):
            (reg, lanes) = self.__get_output_reg_lanes(M, N, i, j, block, size)
            element_name = f"{matrix.upper()}[{i}][{j}]"
        if B > 1:
            element_name += f".B{block}"
        return (element_name, reg, lanes)

    def _get_instruction_num_gprs(self, matrix: str, in_lanes: Optional[int] = 64,
                                  out_size: Optional[int] = None) -> int:
        """ Calculates the number of GPRs needed to hold a matrix.

        Args:
            matrix: string that contains the name of the matrix
                Legal values are a, b, c, d, or k (for the compression index of sparse matrices).
            in_lanes: an integer that defines the number of contiguous lanes that are used to hold
                values of a matrix
                In gfx9, this is always 64, so the default of this argument is 64.
            out_size: The number of bits used to hold output values for this instruction
                For instance: some devices may store 16b values into either the low or high
                half of a 32b output register. Some gfx9 architectures allow 64b outputs (e.g,
                for F64 calculations). If this argument is not passed in, the function
                will initialize the value to the instruction's out_type attribute.

        Returns:
            An integer that defines the number of GPRs needed to hold the requested matrix

        Raises:
            ValueError: An unsupported matrix was requested.
        """
        check_matrix_support(matrix, ('a', 'b', 'c', 'd', 'k'), self.__class__.__name__)
        # In gfx9, we always use all 64 lanes for input calculations, but output sizes
        # can change for 64b outputs
        if out_size is None:
            out_size = get_data_size(self.inst_info['out_type'])
        return super()._get_instruction_num_gprs(matrix, in_lanes, out_size)

    def _calculate_num_regnos_to_print(self, matrix: str, gpr_ratio: float) -> int:
        """ Calculates the number of register slots to print for this matrix & instruction.

        On some architectures, partial registers (such as a 16b output in a 32b register)
        aren't tightly packed. For example, "lower" or "upper halves may be skipped
        instead of storing two contiguous values, one in the lower and one in the upper.
        In such architectures and for  certain matrices, the OPSEL value controls whether
        to use the upper or lower halves.

        This function returns the number of slots to print in each register; the rest can
        be skipped.

        In gfx9, sparse matrix values need to print twice as many "regnos" because the
        register printing function needing to print twice as much data per register

        Args:
            matrix: string that contains the name of the matrix
                Legal values are a, b, c, and d.
            gpr_ratio: the number of regnos in each GPR

        Returns:
            Integer indicating how many of the regnos in each GPR to print

        Raises:
            ValueError: An unsupported matrix was requested.
        """
        check_matrix_support(matrix, ('a', 'b', 'c', 'd', 'k'), self.__class__.__name__)
        num_regnos_to_print = super()._calculate_num_regnos_to_print(matrix, gpr_ratio)
        if matrix.lower() == 'k':
            num_regnos_to_print *= 2
        return num_regnos_to_print

    def _coord_to_input_reg_eqn(self, matrix: str, wave_size: Optional[int] = 64) -> str:
        """ Returns formula for mapping a matrix coordinate to its input register number.

        Takes the instruction info and matrix, return a string with an equation that lets a user
        calculate the input register that holds a particular entry in the matrix from its
        i/j/k/block coordinates.

        Args:
            matrix: string that contains the name of the matrix
                Legal values are a, b, or k (for the compression index of sparse matrices).
            wave_size: integer that holds the wave size for the requested conversion
                function. In gfx9, we only support wave64, so the default of this argument is 64.

        Returns:
            String that contains the simple formula mapping coordinates to input registers

        Raises:
            ValueError: An unsupported matrix was requested.
        """
        check_matrix_support(matrix, ('a', 'b', 'k'), self.__class__.__name__)
        del wave_size # Unused in gfx9
        inst_info = self.inst_info
        in_size = get_data_size(inst_info['in_type'])
        K = inst_info['k']
        num_gprs = self._get_instruction_num_gprs(matrix)

        ret_string = "Unknown"
        normal_a = (matrix.lower() == 'a' and not inst_info['sparse'])
        if (normal_a or matrix.lower() == 'b'):
            # We do not need a sub-register, so no reason to print anything but reg 0
            if (in_size == 32 and num_gprs == 1):
                ret_string = '0'
            elif in_size == 64:
                ret_string = '[1:0]'
            elif (in_size == 32 and num_gprs == 2):
                ret_string = '(k % 2)'
            elif num_gprs == 1:
                if (in_size == 16 and K == 2):
                    ret_string = '0.[16*k+15 : 16*k]'
                elif (in_size == 16 and K >= 4):
                    ret_string = '0.[16*(k % 2)+15 : 16*(k % 2)]'
                elif (in_size == 8 and K <= 4):
                    ret_string = '0.[8*k+7 : 8*k]'
                elif (in_size == 8 and K > 4):
                    ret_string = '0.[8*(k % 4)+7 : 8*(k % 4)]'
            elif num_gprs == 2:
                if (in_size == 16 and K <= 4):
                    ret_string = 'floor(k / 2).[16*(k % 2)+15 : 16*(k % 2)]'
                elif (in_size == 16 and K <= 16):
                    ret_string = '(floor(k / 2) % 2).[16*(k % 2)+15 : 16*(k % 2)]'
                elif in_size == 8:
                    ret_string = '(floor(k / 4) % 2).[8*(k % 4)+7 : 8*(k % 4)]'
            else:
                if in_size == 16:
                    ret_string = '(k % 4).[16*(k % 2)+15 : 16*(k % 2)]'
                elif in_size == 8:
                    ret_string = '(floor(k / 4) % 4).[8*(k % 4)+7 : 8*(k % 4)]'
        elif matrix.lower() == 'k':
            contig_vals = int(32 / in_size)
            ret_string = f"0.[4*(floor(k / 4) % {contig_vals})+3 : "
            ret_string += f"4*(floor(k / 4) % {contig_vals})]"
        else: # sparse a
            if in_size == 16:
                ret_string = '(floor(k / 4) % 2)'
            else:
                ret_string = '(floor(k / 8) % 2).[16*(floor(k / 4) % 2)+15 : 16*(floor(k / 4) % 2)]'
        return ret_string

    def _coord_to_output_reg_eqn(self, wave_size: Optional[int] = 64) -> str:
        """ Returns formula for mapping a matrix coordinate to its output register number.

        Takes the instruction info and matrix, return a string with an equation that lets a user
        calculate the output register that holds a particular entry in the matrix from its
        i/j/k/block coordinates.

        Args:
            wave_size: integer that holds the wave size for the requested conversion
                function. In gfx9, we only support wave64, so the default of this argument is 64.

        Returns:
            String that contains the simple formula mapping coordinates to output registers
        """
        del wave_size # Unused in gfx9
        inst_info = self.inst_info
        out_type = inst_info['out_type']
        M = inst_info['m']
        N = inst_info['n']
        blocks = inst_info['blocks']

        ret_string = "Unknown"
        if out_type == 'fp64':
            if blocks == 1:
                ret_string = '[2*floor(i / 4)+1 : 2*floor(i / 4)]'
            elif blocks == 4:
                ret_string = '[1:0]'
        elif (M == 4 and N == 4):
            if blocks == 16:
                ret_string = 'i'
        elif (M == 16 and N == 16):
            if blocks == 4:
                ret_string = '4 * block + (i % 4)'
            elif blocks == 1:
                ret_string = '(i % 4)'
        elif (M == 32 and N == 32):
            if blocks == 2:
                ret_string = '16 * block + 4 * floor(i / 8) + (i % 4)'
            elif blocks == 1:
                ret_string = '4 * floor(i / 8) + (i % 4)'
        return ret_string

    def _coord_to_lane_eqn(self, matrix: str, wave_size: Optional[int] = 64) -> str:
        """ Returns formula for mapping a matrix coordinate to its wavefront lane.

        Takes the instruction info and matrix, return a string with an equation that lets a user
        calculate the lane that holds a particular entry in the matrix from its i/j/k/block
        coordinates.

        Args:
            matrix: string that contains the name of the matrix
                Legal values are a, b, c, d, or k (for the compression index of sparse matrices).
            wave_size: integer that holds the wave size for the requested conversion
                function. In gfx9, we only support wave64, so the default of this argument is 64.

        Returns:
            String that contains the simple formula mapping coordinates to lanes

        Raises:
            ValueError: An unsupported matrix was requested.
        """
        check_matrix_support(matrix, ('a', 'b', 'c', 'd', 'k'), self.__class__.__name__)
        inst_info = self.inst_info
        in_size = get_data_size(inst_info['in_type'])
        out_type = inst_info['out_type']
        M = inst_info['m']
        N = inst_info['n']
        K = inst_info['k']
        blocks = inst_info['blocks']

        ret_string = "Unknown"
        if matrix.lower() in ('a', 'b'):
            if blocks > 1:
                ret_string = f"{M} * block + "
            else:
                ret_string = ""
            if M * blocks < 64:
                div_val = int((M * K * blocks) / 64)
                ret_string += str(M * blocks)
                if div_val == 1:
                    ret_string += " * k + "
                else:
                    ret_string += f" * floor(k / {div_val}) + "
            if matrix.lower() == 'a':
                ret_string += 'i'
            else:
                ret_string += 'j'
        elif matrix.lower() == 'k':
            contig_vals = int(32 / in_size) * 4
            ret_string = f"{M} * floor(k / {contig_vals}) + i"
        else: # c or d
            ret_string = ""
            if out_type != 'fp64':
                if int((N * M) / 4) > 64:
                    ret_string += f"({N} * floor(i / 4)) % 64 + "
                elif int((N * M) / 4) == 64:
                    ret_string += f"{N} * floor(i / 4) + "
            else:
                ret_string = "16 * (i % 4) + "
            if M * N < 64:
                ret_string += f"{N} * block + "
            ret_string += 'j'
        return ret_string

    def _print_element_to_register_eqn(self, block: str = ".block",
                                       wave_size: Optional[int] = None) -> None:
        """ Prints formula for matrix entry to GPR and lane mapping.

        Prints out the simple formulae to calculate the the mapping of a matrix element to its
        register and lane.

        Will always print A[], B[], and D[]. If the instruction allows C[] matrices, then it will
        print D[] as "C or D[i][j]". For sparse instructions, will also print the compression
        matrix.

        Args:
            block: string that contains text to place in the matrix entry map for blocks
                For instance, gfx9 uses blocks, so by default this function passes ".block"
                to print an entry as A[i][k].block.
            wave_size: integer that contains the requested wave size to use for printing the
                formula. gfx9 only supports one wave size, so this option is passed along silently
        """
        super()._print_element_to_register_eqn(block, wave_size)

    def _reg_lane_to_i_coord_eqn(self, matrix: str,
                                       wave_size: Optional[int] = None) -> str:
        """ Returns equation to map register+lane to i index.

        Takes instruction info and returns a string containing an equation which lets users
        calculate the i coordinate for the A, C, D, or compression index matrices.

        Args:
            matrix: string that contains the name of the matrix
                Legal values are a, c, d, or k (for the compression index of sparse matrices).
            wave_size: integer that holds the wave size for the requested conversion
                function. gfx9 only supports wave64.

        Returns:
            A string which contains the equation to calculate the i coordinate for
            an input matrix held by a particular register and lane combination

        Raises:
            ValueError: An unsupported matrix was requested.
        """
        check_matrix_support(matrix, ('a', 'c', 'd', 'k'), self.__class__.__name__)
        ret_string = super()._reg_lane_to_i_coord_eqn(matrix, wave_size)
        if matrix not in ('a', 'b', 'k'):
            inst_info = self.inst_info
            M = inst_info['m']
            out_type = inst_info['out_type']
            ret_string = ""
            if out_type != 'fp64':
                if M > 16:
                    ret_string += "(8 * floor(GPR_num / 4) % 32) + "
                if M != 4:
                    ret_string += f"4 * floor(lane / {M}) + "
                ret_string += "(GPR_num % 4)"
            else:
                num_i_per_block = int(4 / inst_info['blocks'])
                if num_i_per_block > 1:
                    ret_string += f"{num_i_per_block} * "
                    ret_string += "floor(GPR_num / 2)"
                else:
                    ret_string += "floor(lane / 16)"
        return ret_string

    def _reg_lane_to_k_coord_eqn(self, matrix: str,
                                 wave_size: Optional[int] = None) -> str:
        """ Returns equation to map register+lane to k index.

        Takes instruction info and a target matrix, and returns a string containing an
        equation which lets users calculate the k coordinate based on the register and lane.
        Targets a particular instruction, and does not handle modifiers.

        Args:
            matrix: string that contains the name of the matrix
                Legal values are a, b, and k (for the compression index of sparse matrices).
            wave_size: integer that holds the wave size for the requested conversion
                function. gfx9 only supports wave64.

        Returns:
            A string which contains the equation to calculate the k coordinate held by a
            particular register and lane combination

        Raises:
            ValueError: An unsupported matrix was requested.
        """
        check_matrix_support(matrix, ('a', 'b', 'k'), self.__class__.__name__)
        inst_info = self.inst_info
        in_gprs = self._get_instruction_num_gprs(matrix)
        data_type = inst_info['in_type']
        data_size = get_data_size(data_type)
        sparse = inst_info['sparse'] and matrix.lower() in ('a', 'k')
        ret_string = ""
        k_per_register = int(self._get_elements_per_gpr(data_size, sparse))
        if not sparse:
            if inst_info['k'] == 1:
                ret_string = "0"
            elif data_type == 'fp64':
                ret_string = 'floor(lane / 16)'
            else:
                k_per_lane_skip = k_per_register * in_gprs
                if k_per_lane_skip != 1 and k_per_lane_skip < inst_info['k']:
                    ret_string = f"{k_per_lane_skip} * "
                if k_per_lane_skip < inst_info['k']:
                    ret_string += f"floor(lane / {inst_info['m']})"
                if k_per_register != 1:
                    if ret_string != "":
                        ret_string += " + "
                    if in_gprs > 1:
                        ret_string += f"{k_per_register} * GPR_num + "
                    ret_string += f"floor(GPR_bits / {data_size})"
                elif in_gprs > 1:
                    ret_string += " + GPR_num"
        else:
            if matrix.lower() == 'a':
                if data_size != 8:
                    start_point = f"{k_per_register} * GPR_num"
                    end_point = f"({start_point} + {k_per_register-1})"
                    ret_string = f"{end_point} through {start_point}"
                else:
                    start_point = f"16 * floor(lane / {inst_info['m']}"
                    start_point += ") + (8 * GPR_num) + (4 * floor(GPR_bits / 16))"
                    end_point = f"{start_point} + 3"
                    ret_string = f"{end_point}\nthrough\n{start_point}"
            else: # matrix.lower() == 'k'
                M = inst_info['m']
                contig_vals = int(128 / data_size)
                start_point = f"{contig_vals} * floor(lane / {M}) + 4 * floor(GPR_bits / 4)"
                end_point = f"{start_point} + 3"
                ret_string = f"{end_point}\nthrough\n{start_point}"
        return ret_string

    def _print_opcode(self, encoding_name="VOP3P-MAI"):
        """ Prints encoding name and VOP3P opcode for an instruction.

        Args:
            encoding_name: String containing the name of the encoding format for the
                current architecture; defaults to "VOP3P-MAI" in gfx9.
        """
        super()._print_opcode(encoding_name)
        print(f"    {encoding_name} Opcode: {self.inst_info['opcode'] & 0x3f:#02x}")

    def _print_matrix_dims(self) -> None:
        """ Prints the dimensions of matrices used by an instruction on a target architecture.

            In CDNA/gfx9, we also have the concept of "blocks", so this function first prints
            out the generic matrix dimensions, then adds information about the blocks afterwards.
        """
        super()._print_matrix_dims()
        print(f"        blocks: {self.inst_info['blocks']}")

    def _print_register_usage(self, wave_sizes: Tuple[int, ...] = (64,)) -> None:
        """ Prints the register count for each input and output matrix.

        Prints the number of registers used by each matrix for a matrix multiplication
        instruction on a target architecture. Some architectures support more than one
        wavefront size. In that case, print out the register usage for each wavefront size.
        This is needed because different wavefront sizes can cause different GPR usage for
        the same matrix size.

        Args:
            wave_sizes: a tuple of wavefront sizes to use to print this info
                Defaults to a single tuple entry of integer 64 in gfx9, because gfx9
                only supports wave64.
        """
        super()._print_register_usage(wave_sizes)

    def _print_register_info(self, encoding_name: str = "VOP3P-MAI") -> None:
        """ Prints the encoding and register information for a matrix instruction.

        Prints the encoding and modifier information for the registers used by a
        matrix multiplication instruction on a particular architecture.
        Architecture matters, because different architectures support different
        encodings and matrix modifier fields.

        Args:
            encoding_name: String containing the name of the encoding format for the
                current architecture; gfx9 defaults to "VOP3P-MAI".
        """
        super()._print_register_info(encoding_name)
        print("    Register capabilities:")
        print(f"        A matrix can use ArchVGPRs: {True}")
        print(f"        A matrix can use AccVGPRs: {True}")
        print(f"        B matrix can use ArchVGPRs: {True}")
        print(f"        B matrix can use AccVGPRs: {True}")
        # Skip printing information about the C matrix if in a sparse matrix.
        if not self.inst_info['sparse']:
            print(f"        C and D matrix can use ArchVGPRs: {self.inst_info['c_d_arch']}")
            print(f"        C and D matrix can use AccVGPRs: {True}")
        else:
            print(f"        D matrix can use ArchVGPRs: {self.inst_info['c_d_arch']}")
            print(f"        D matrix can use AccVGPRs: {True}")
        print("    Register modifiers:")
        print(f"        Sparse A matrix: {self.inst_info['sparse']}")
        print(f"        CBSZ and ABID bits supported: {self.inst_info['cbsz_abid']}")
        print(f"        BLGP bits supported: {self.inst_info['blgp']}")

    def __reg_lane_to_input_block_eqn(self) -> str:
        """ Returns equation to map register+lane to an input matrix block.

        Return a string that can be used to quickly calculate how to go from a register and
        lane to the block of the input matrix. Targets a particular instruction,
        and does not handle modifiers.

        Returns:
            A string which contains the equation to calculate the block held by a particular
            register and lane combination
        """
        inst_info = self.inst_info
        ret_string = ""
        if inst_info['blocks'] == 1:
            ret_string = "0"
        else:
            ret_string = f"floor(lane / {inst_info['m']})"
            if inst_info['in_type'] == 'fp64':
                ret_string = f"({ret_string} % 4)"
        return ret_string

    def __reg_lane_to_output_block_eqn(self) -> str:
        """ Returns equation to map register+lane to an output matrix block.

        Return a string that can be used to quickly calculate how to go from a register and
        lane to the block of the output matrix. Targets a particular instruction,
        and does not handle modifiers.

        Returns:
            A string which contains the equation to calculate the block held by a particular
            register and lane combination
        """
        inst_info = self.inst_info
        ret_string = ""
        if inst_info['blocks'] == 1:
            ret_string = "0"
        else:
            out_gprs = self._get_instruction_num_gprs('d')
            gpr_per_block = int(out_gprs / inst_info['blocks'])
            if gpr_per_block == 0:
                ret_string = f"floor(lane / {inst_info['m']})"
            else:
                ret_string = f"floor(GPR_num / {gpr_per_block})"
            if inst_info['out_type'] == 'fp64':
                ret_string = f"({ret_string} % 4)"
        return ret_string

    def _reg_lane_to_block_eqn(self, matrix: str) -> str:
        """ Returns equation to map register+lane to block.

        Return a string that can be used to quickly calculate how to go from a register and
        lane to the block of the input or output matrix. Targets a particular instruction,
        and does not handle modifiers.

        Args:
            matrix: string that contains the name of the matrix
                Legal values are a, b, c, d, and k (for the compression index of sparse matrices).

        Returns:
            A string which contains the equation to calculate the block held by a particular
            register and lane combination

        Raises:
            ValueError: An unsupported matrix was requested.
        """
        check_matrix_support(matrix, ('a', 'b', 'c', 'd', 'k'), self.__class__.__name__)
        if matrix.lower() in ('a', 'b'):
            ret_string = self.__reg_lane_to_input_block_eqn()
        else:
            ret_string = self.__reg_lane_to_output_block_eqn()
        return ret_string

    def _print_execution_statistics(self, cu_name: str = "CU") -> None:
        """ Prints execution statistics for a matrix multiplication instruction.

        Prints the execution statistics, such as computational throughput and co-execution
        information, for a matrix multiplication instruction on a target architecture.
        The instruction and architecture are part of the class.

        Args:
            cu_name: string that contains the name of the compute unit on the target arch
                Different architectures may use e.g., CU, or WGP, or some other name.
                Defaults to "CU" in gfx9.
        """
        super()._print_execution_statistics(cu_name)

    def _print_register_to_element_eqn(self, print_block: bool = True,
                                       wave_size: Optional[int] = None,
                                       max_a_lane: Optional[int] = None,
                                       max_b_lane: Optional[int] = None) -> None:
        """ Prints equation to map register+lane to matrix element.

        Print out simple equations for mapping a register and its lane to the element in the
        matrix and block which they hold. These can be used by developers that do not want to
        set modifiers like CBSZ/ABID or BLGP to quickly unpack values from registers to
        put them into output matrix locations.

        In gfx9, we print should print block information because gfx9 supports matrix blocks.
        This child specialization of the function is meant specifically to set a default
        value of print_block=True.

        Args:
            print_block: True if this equation should print information about blocks
            wave_size: integer that holds the wave size for the requested conversion
                function. gfx9 only supports wave64.
            max_a_lane: optional integer that holds the maximum lane that can be used in this
                instruction for A and K matrices.
            max_b_lane: optional integer that holds the maximum lane that can be used in this
                instruction for B matrices.
        """
        super()._print_register_to_element_eqn(print_block)


class InstCalcGfx11(InstCalc):
    """ Calculator for matrix multiplication instruction details in gfx11 architecture.

    This is a child class of the InstCalc class, because gfx11/RDNA3 requires different
    calculations that other architectures.

    Attributes:
        arch_name: string that holds the accelerator architecture's name
        inst_name: string holding the mnemonic of the instruction that will be used in calculations
        inst_info: MatrixInstruction holding the details of the instruction used in calculations
        wave_width: Some architectures allow wavefronts of various widths, and these widths can
            affect the resulting calculations. This integer holds the width that will be used for
            further calculations.
    """

    def check_valid_reg_lane(self, matrix: str, register: int, lane: int) -> bool:
        """ Checks if the register and lane being used for calculation are legal

        Checks whether the matrix and lane being requested are legal for indexing into the
        requested matrix. Callers should use this to ensure that the register is within the
        bounds of the matrix. In addition, ensure that the requested lane is within the
        current wave_width settings for this instruction + gfx 11 architecture

        Args:
            matrix: String indicating the matrix to query: 'a', 'b', 'c', 'd', or 'k'
                for the compression index matrix in sparse matrix ops
            register: Integer of the register number being requested, starting from 0
            lane: Integer of the lane being requested, starting from 0

        Returns:
            True if the register and lane contain data for this matrix on the current instruction
            False if the register and lane do not contain data for this matrix in this instruction
        """
        if (lane < 0 or lane >= (self.wave_width)):
            return False
        num_gprs = self._get_instruction_num_gprs(matrix)
        if (register < 0 or register > num_gprs):
            return False
        return True

    def _find_matching_b_lane(self, a_lane: int, b_lanes: List[int]) -> int:
        """ Finds the lane in a list of B matrix lanes that match the A matrix lane.

        In some architectures, matrix values can exist simultaneously in multiple
        lanes. Or, more specifically, multiple lanes must store the same value from
        the matrix. If a value was stored in both lanes 0 and lane 16, when printing
        out "lane 0 of A is multiplied by lane X of B", this function will do that
        matching. It takes as an argument a list of lanes from B, and the requested
        lane of A. Returns the lane of B that is multiplied by the requested lane of A.

        Args:
            a_lane: integer for the lane of A that we want to match
            b_lanes: list of integers containing all the lanes of B to query

        Returns:
            Integer from the available lanes of B that match the requested lane of A
            In gfx11, the A and B matrix entries are lane-matched. As such, this
            returns the same lane as A.
        """
        del b_lanes # Unused in gfx11
        return a_lane

    def __get_input_reg_lanes(self, i: int, k: int, data_size: int) -> Tuple[str, List[int]]:
        """ Calculates a matrix's input register and lane number based on coordinates.

        For gfx11, calculates the input register and the lanes within that register
        for an instruction based on its parameters.

        Args:
            i: integer location within the input matrix's "outer" dimension
                For A matrices, this is the desired row
                For B matrices this is the desired column
            k: integer location within the input matrix's "inner" dimension
                For A matrices, this is the desired column
                For B matrices, this is the desired row
            data_size: integer size of the input data, in bits

        Returns:
            Based on the matrix and requested coordinates, return two things in a tuple:
            1. String register name that holds the element, in the format V#.[bits]
            2. List of integers, containing the lane numbers within that register that
                hold the element
            Tuple: (register holding the matrix entry, lanes within that register)
        """
        reg = self._get_reg_name(data_size, False, False, 0, 0, 0, k)

        # Odd columns are actually stored 16 lanes later
        lanes_to_ret = []

        if self.wave_width == 32:
            copies_to_return = 2
        else:
            copies_to_return = 4
        for _ in range(copies_to_return):
            lanes_to_ret.append(i)
            i += 16

        return (reg, lanes_to_ret)

    def __get_output_reg_lanes(self, N: int, i: int, j: int, data_size: int,
                               opsel: int) -> Tuple[str, List[int]]:
        """ Calculates a matrix's output register and lane number based on coordinates.

        For gfx11, calculates the output register and the lane within that register for
        an instruction based on its parameters. The algorithm for calculating these
        is described in the comments within the function.

        Args:
            N: integer width of the output matrix, in matrix entries
            i: integer location within the output matrix's rows
            j: integer location within the output matrix's columns
            data_size: integer size of the output data, in bits
            opsel: integer value of the instruction's OPSEL modifier

        Returns:
            Based on the matrix and requested coordinates, return two things in a tuple:
            1. String register name that holds the element, in the format V#.[bits]
            2. List of integers, containing the lane numbers within that register that
                hold the element
            Tuple: (register holding the matrix entry, lanes within that register)
        """
        # When the output is 16b, we only write into the lower or upper half of
        # a register, so we need to "skip" the other half of the register slots
        rows_per_reg_slot = self.wave_width / 16
        skip_half = 2 if data_size == 16 else 1
        regno = int(skip_half * int(i / rows_per_reg_slot)) + (opsel>>2)
        reg = self._get_reg_name(data_size, False, False, 0, 0, 0, regno)

        # Output lanes are 16 elements wide, and depending on the wave size,
        # this ends up meaning we walk over a different number of lanes before
        # we move on to the next register as we go over the rows
        rows_per_vgpr = int((self.wave_width * 16) / N)
        lane = (N * (i % rows_per_vgpr) + j) % self.wave_width
        return (reg, [lane])

    def _get_reg_lanes(self, matrix: str, i: int, j: int, k: int, block: int, cbsz: int,
                       abid: int, blgp: int, opsel: int) -> Tuple[str, str, List[int]]:
        """ Calculates a matrix's register and lane number based on coordinates.

        For the target architecture and the instruction set up in this class's init
        function, this function calculates the register and lane that hold a requested
        matrix entry in a requested matrix. This location information can vary based on
        per-instruction modifiers, so those are all input arguments.

        block, cbsz, abid, and blgp are ignored by this function, because gfx11 does not use
        these instruction modifiers.

        Args:
            matrix: String indicating the matrix to query: 'a', 'b', 'c', or 'd'
            i: integer coordinate for the query of the matrix row for A, C, D, & K matrices
            j: integer coordinate for the query of the matrix column for the B, C, & D matrices
            k: integer coordinate for the query of the A & K column or B row
            block: integer coordinate for the block to query
            cbsz: integer value of the instruction's CBSZ modifier
            abid: integer value of the instruction's ABID modifier
            blgp: integer value of the instruction's BLGP modifier
            opsel: integer value of the instruction's OPSEL modifier

        Returns:
            Based on the matrix and requested coordinates, return three things in a tuple:
            1. String requested element name, in the format A[i][k], B[k][j], or C[i][j]
            2. String register name that holds the element, in the format V#.[bits]
            3. List of integers, containing the lane numbers within that register that
                hold the element
            Tuple: (matrix entry, register holding that entry, lanes within that register)
        """
        del block, cbsz, abid, blgp # Unused in gfx11
        inst_info = self.inst_info
        N = inst_info['n']

        if matrix.lower() in ('a', 'b'):
            size = get_data_size(inst_info['in_type'])
        else:
            size = get_data_size(inst_info['out_type'])
        if matrix.lower() == 'a':
            (reg, lanes) = self.__get_input_reg_lanes(i, k, size)
            element_name = f"{matrix.upper()}[{i}][{k}]"
        elif matrix.lower() == 'b':
            (reg, lanes) = self.__get_input_reg_lanes(j, k, size)
            element_name = f"{matrix.upper()}[{k}][{j}]"
        else: # (matrix.lower() == 'c' or matrix.lower() == 'd'):
            (reg, lanes) = self.__get_output_reg_lanes(N, i, j, size, opsel)
            element_name = f"{matrix.upper()}[{i}][{j}]"
        return (element_name, reg, lanes)

    def _calculate_initial_regno_offset(self, matrix: str, opsel: int) -> int:
        """ Calculates an offset into a register slot based on OPSEL.

        On some architectures, partial registers (such as a 16b output in a 32b register)
        aren't tightly packed. For example, "lower" or "upper halves may be skipped
        instead of storing two contiguous values, one in the lower and one in the upper.
        In such architectures and for certain matrices, the OPSEL value controls whether
        to use the upper or lower halves.

        This function returns an offset, so that these slots (which we call regnos) can
        be skipped.

        In gfx11, 16b outputs only use the top or bottom half of the registers, not both at
        the same time. Opsel=0 selects to store into the bottom half, Opsel=4 selects to store
        into the upper half. As such, when we are in gfx11, looking at the C/D matrix,
        and the output is 4, we only actually print half as many output registers. And
        further, when opsel == 4, we bump forward one regno to hit the top half of
        the register.

        Args:
            matrix: string that contains the name of the matrix
                Legal values are a, b, c, and d.
            opsel: integer value of the instruction's OPSEL modifier

        Returns:
            Integer which indicates the regno offset for this matrix+OPSEL pair

        Raises:
            ValueError: An unsupported matrix was requested.
        """
        check_matrix_support(matrix, ('a', 'b', 'c', 'd'), self.__class__.__name__)
        if (matrix.lower() in ('c', 'd') and get_data_size(self.inst_info['out_type']) == 16 and
                opsel == 4):
            offset = 1
        else:
            offset = 0
        return offset

    def _calculate_num_regnos_to_print(self, matrix: str, gpr_ratio: float) -> int:
        """ Calculates the number of register slots to print for this matrix & instruction.

        On some architectures, partial registers (such as a 16b output in a 32b register)
        aren't tightly packed. For example, "lower" or "upper halves may be skipped
        instead of storing two contiguous values, one in the lower and one in the upper.
        In such architectures and for  certain matrices, the OPSEL value controls whether
        to use the upper or lower halves.

        This function returns the number of slots to print in each register; the rest can
        be skipped.

        In gfx11, 16b outputs only use the top or bottom half of the registers, not both at
        the same time. As such, when we are in gfx11, looking at the C/D matrix,
        and the output is 4, we only actually print half as many output registers.
        Leave "gpr_ratio" the same, however, so that we skip over both halves when we
        are iterating over things.

        Args:
            matrix: string that contains the name of the matrix
                Legal values are a, b, c, and d.
            gpr_ratio: the number of regnos in each GPR

        Returns:
            Integer indicating how many of the regnos in each GPR to print

        Raises:
            ValueError: An unsupported matrix was requested.
        """
        check_matrix_support(matrix, ('a', 'b', 'c', 'd'), self.__class__.__name__)
        if (matrix.lower() in ('c', 'd') and get_data_size(self.inst_info['out_type']) == 16):
            num_regnos_to_print = math.ceil(gpr_ratio/2)
        else:
            num_regnos_to_print = math.ceil(gpr_ratio)
        return num_regnos_to_print

    def calculate_register_layout(self, matrix: str, requested_output: str,
                                  negate: Dict[str, bool], cbsz: int, abid: int, blgp: int,
                                  opsel: int, transpose: bool, print_blocks: bool = False) -> None:
        """ Displays the registers+lanes for an entire matrix.

        Calculate and display the registers and lanes for an entire input or
        output matrix. Displays the matrix formatted as its rows and columns,
        with the registers and their lanes displayed in a tabular format.
        Can optionally print this tabular format as a CSV, markdown, or asciidoc
        for other processing. Can also choose to transpose the matrix to visually
        show rows as columns and vice versa.
        The registers are displayed as: Va{b}.c:
            - a is the register number
            - b is the lane within that register
            - c is an optional identifier for sub-Dword parts of the 32 bit register:
                    [15:0]: the lower 16b of a 32b register
                    [31:16]: the upper 16b of a 32b register
                    [7:0]: the least significant 8b of a 32b register
                    [15:8]: the second-lowest 8b of a 32b register
                    [23:16]: the second-highest 8b of a 32b register
                    [31:24]: the most significant 8b of a 32b register

        Args:
            matrix: string that contains the name of the matrix
                Legal values are a, b, c, or d.
            requested_output: string that indicates the type of output, from the list of
                csv, markdown, asciidoc, or grid
            negate: dictionary of matrix names to bools that indicate whether to
                negate and absolute-val entries from this matrix.
            cbsz: integer value of the instruction's CBSZ modifier
            abid: integer value of the instruction's ABID modifier
            blgp: integer value of the instruction's BLGP modifier
            opsel: integer value of the instruction's OPSEL modifier
            transpose: boolean set to true to cause the matrix to be printed transposed
            print_blocks: boolean set to true if this architecture and instruction
                should print the word "Block #" above each block of the matrix
                gfx11 does not do this by default.
        """
        super().calculate_register_layout(matrix, requested_output, negate, cbsz, abid, blgp,
                                          opsel, transpose, print_blocks)

    def calculate_matrix_layout(self, matrix: str, requested_output: str, negate: Dict[str, bool],
                                cbsz: int, abid: int, blgp: int, opsel: int, transpose: bool,
                                contig_values: int = 16) -> None:
        """ Displays the matrix entries for all of the registers+lanes used by an instruction.

        Calculate and display the matrix elements for all register entries and
        lanes used by the requesting instruction.
        Displays the registers formatted by register entry (X axis) and wavefront lane
        (Y axis). The resulting table then shows the MatrixName[col][row] held in
        that lane's register entry.
        Can optionally print this tabular format as a CSV, markdown, or asciidoc
        for other processing. Can also choose to transpose the matrix to visually
        show rows as columns and vice versa.

        Args:
            matrix: string that contains the name of the matrix
                Legal values are a, b, c, or d.
            requested_output: string that indicates the type of output, from the list of
                csv, markdown, asciidoc, or grid
            negate: dictionary of matrix names to bools that indicate whether to
                negate and absolute-val entries from this matrix.
            cbsz: integer value of the instruction's CBSZ modifier
            abid: integer value of the instruction's ABID modifier
            blgp: integer value of the instruction's BLGP modifier
            opsel: integer value of the instruction's OPSEL modifier
            transpose: boolean set to true to cause the matrix to be printed transposed
            contig_values: an integer that defines the number of contiguous values of a
                register that are used to hold unique values of a matrix
                gfx11 uses 16 lanes by default.
        """
        super().calculate_matrix_layout(matrix, requested_output, negate, cbsz, abid, blgp, opsel,
                                        transpose, contig_values)

    def _get_instruction_num_gprs(self, matrix: str, in_lanes: Optional[int] = 16,
                                  out_size: Optional[int] = 32) -> int:
        """ Calculates the number of GPRs needed to hold a matrix.

        Args:
            matrix: string that contains the name of the matrix
                Legal values are a, b, c, or d.
            in_lanes: an integer that defines the number of contiguous lanes that are used to hold
                values of a matrix; in gfx11, this is always 16, so the default of this
                argument is 16.
            out_size: The number of bits used to hold output values for this instruction
                For instance: some devices may store 16b values into either the low or high
                half of a 32b output register. If this argument is not passed in, the function
                will initialize the value to 32.

        Returns:
            An integer that defines the number of GPRs needed to hold the requested matrix
        """
        # gfx11 uses 16 lanes for its inputs, and all outputs (even 2B values) are stored
        # into 4B locations.
        del in_lanes
        return super()._get_instruction_num_gprs(matrix, 16, out_size)

    def _coord_to_input_reg_eqn(self, matrix: str, wave_size: Optional[int] = 32) -> str:
        """ Returns formula for mapping a matrix coordinate to its input register number.

        Takes the instruction info and matrix, return a string with an equation that lets a user
        calculate the input register that holds a particular entry in the matrix from its
        i/j/k/block coordinates.

        Args:
            matrix: string that contains the name of the matrix
                Legal values are a or b.
            wave_size: integer that holds the wave size for the requested conversion
                function. gfx11 supports wave64 and wave32, but it does not matter
                for this equation, so we default this to 32.

        Returns:
            String that contains the simple formula mapping coordinates to input registers

        Raises:
            ValueError: An unsupported matrix was requested.
        """
        check_matrix_support(matrix, ('a', 'b'), self.__class__.__name__)
        del wave_size # Unused in gfx11
        data_size = get_data_size(self.inst_info['in_type'])
        if data_size == 16:
            ret_string = 'floor(k / 2).[16*(k % 2)+15 : 16*(k % 2)]'
        elif data_size == 8:
            ret_string = 'floor(k / 4).[8*(k % 4)+7 : 8*(k % 4)]'
        else: # data_size == 4:
            ret_string = 'floor(k / 8).[4*(k % 8)+3 : 4*(k % 8)]'
        return ret_string

    def _coord_to_output_reg_eqn(self, wave_size: Optional[int] = 32) -> str:
        """ Returns formula for mapping a matrix coordinate to its output register number.

        Takes the instruction info and matrix, return a string with an equation that lets a user
        calculate the output register that holds a particular entry in the matrix from its
        i/j/k/block coordinates.

        Args:
            wave_size: integer that holds the wave size for the requested conversion
                function. gfx11 supports wave64 and wave32, but it does not matter
                for this equation, so we default this to 32.

        Returns:
            String that contains the simple formula mapping coordinates to output registers
        """
        ret_string = "Unknown"
        if wave_size is not None:
            ret_string = f'floor(i / {int(wave_size / 16)})'
        if get_data_size(self.inst_info['out_type']) == 16:
            ret_string = f"({ret_string}).[15:0]"
        return ret_string

    def _coord_to_lane_eqn(self, matrix: str, wave_size: Optional[int] = 32) -> str:
        """ Returns formula for mapping a matrix coordinate to its wavefront lane.

        Takes the instruction info and matrix, return a string with an equation that lets a user
        calculate the lane that holds a particular entry in the matrix from its i/j/k/block
        coordinates.

        Args:
            matrix: string that contains the name of the matrix
                Legal values are a, b, c, or d.
            wave_size: integer that holds the wave size for the requested conversion
                function. gfx11 supports wave64 and wave32.

        Returns:
            String that contains the simple formula mapping coordinates to lanes

        Raises:
            ValueError: An unsupported matrix was requested.
        """
        check_matrix_support(matrix, ('a', 'b', 'c', 'd'), self.__class__.__name__)
        if matrix.lower() == 'a':
            if wave_size == 32:
                ret_this = 'i and i+16'
            else:
                ret_this = 'i, i+16, i+32, and i+48'
        elif matrix.lower() == 'b':
            if wave_size == 32:
                ret_this = 'j and j+16'
            else:
                ret_this = 'j, j+16, j+32, and j+48'
        else: # C, D
            ret_this = f'((16 * i) % {wave_size}) + j'
        return ret_this

    def _reg_lane_to_block_eqn(self, matrix: str) -> str:
        """ Returns equation to map register+lane to block.

        Return a string that can be used to quickly calculate how to go from a register and
        lane to the block of the input or output matrix. Targets a particular instruction,
        and does not handle modifiers.

        Args:
            matrix: string that contains the name of the matrix

        Returns:
            A blank string, because there are no blocks in gfx11
        """
        del matrix # Unused in gfx11
        return ""

    def _print_element_to_register_eqn(self, block: str = "",
                                       wave_size: Optional[int] = None) -> None:
        """ Prints formula for matrix entry to GPR and lane mapping.

        Prints out the simple formulae to calculate the the mapping of a matrix element to its
        register and lane.

        Will always print A[], B[], and D[]. If the instruction allows C[] matrices, then it will
        print D[] as "C or D[i][j]".

        Args:
            block: string that contains text to place in the matrix entry map for blocks
                For instance, gfx11 does not uses blocks, so by default this function passes an
                empty string, to print an entry as A[i][k]
            wave_size: integer that contains the requested wave size to use for printing the
                formula. gfx11 supports two different wave sizes, and if this optional value is
                not passed in, we will print both
        """
        if wave_size is None:
            super()._print_element_to_register_eqn(block, 32)
            super()._print_element_to_register_eqn(block, 64)
        else:
            super()._print_element_to_register_eqn(block, wave_size)

    def _reg_lane_to_i_coord_eqn(self, matrix: str,
                                       wave_size: Optional[int] = None) -> str:
        """ Returns equation to map register+lane to i index.

        Takes instruction info and returns a string containing an equation which lets users
        calculate the i coordinate for the A, C, or D matrices.

        Args:
            matrix: string that contains the name of the matrix
                Legal values are a, c, or d.
            wave_size: integer that holds the wave size for the requested conversion
                function. gfx11 supports wave32 and wave64.

        Returns:
            A string which contains the equation to calculate the i coordinate for
            an input matrix held by a particular register and lane combination

        Raises:
            ValueError: An unsupported matrix was requested.
        """
        check_matrix_support(matrix, ('a', 'c', 'd'), self.__class__.__name__)
        ret_string = super()._reg_lane_to_i_coord_eqn(matrix, wave_size)
        if matrix not in ('a', 'b', 'k') and wave_size is not None:
            ret_string = f"{int(wave_size / 16)} * GPR_num + floor(lane / 16)"
            if get_data_size(self.inst_info['out_type']) == 16:
                ret_string = f"({ret_string}).[15:0]"
        return ret_string

    def _reg_lane_to_k_coord_eqn(self, matrix: str,
                                 wave_size: Optional[int] = None) -> str:
        """ Returns equation to map register+lane to k index.

        Takes instruction info and a target matrix, and returns a string containing an
        equation which lets users calculate the k coordinate based on the register and lane.
        Targets a particular instruction, and does not handle modifiers.

        Args:
            matrix: string that contains the name of the matrix
                Legal values are a or b.
            wave_size: integer that holds the wave size for the requested conversion
                function. gfx11 supports wave32 and wave64.

        Returns:
            A string which contains the equation to calculate the k coordinate held by a
            particular register and lane combination

        Raises:
            ValueError: An unsupported matrix was requested.
        """
        check_matrix_support(matrix, ('a', 'b'), self.__class__.__name__)
        data_size = get_data_size(self.inst_info['in_type'])
        if data_size == 16:
            ret_string = "2 * GPR_num + floor(GPR_bits / 16)"
        elif data_size == 8:
            ret_string = "4 * GPR_num + floor(GPR_bits / 8)"
        else:
            ret_string = "8 * GPR_num + floor(GPR_bits / 4)"
        return ret_string

    def _print_opcode(self, encoding_name: str = "VOP3P") -> None:
        """ Prints encoding name and VOP3P opcode for an instruction.

        Args:
            encoding_name: String containing the name of the encoding format for the
                current architecture; defaults to "VOP3P" in gfx11.
        """
        super()._print_opcode(encoding_name)

    def _print_register_info(self, encoding_name: str = "VOP3P") -> None:
        """ Prints the encoding and register information for a matrix instruction.

        Prints the encoding and modifier information for the registers used by a
        matrix multiplication instruction on a particular architecture.
        Architecture matters, because different architectures support different
        encodings and matrix modifier fields.

        Args:
            encoding_name: String containing the name of the encoding format for the
                current architecture; gfx11 defaults to "VOP3P".
        """
        super()._print_register_info(encoding_name)
        print("    Register modifiers:")
        print("        OPSEL[1:0] supported: False")
        print(f"        OPSEL[2] supported: {self.inst_info['cd_opsel']}")
        print(f"        NEG bits supported: {self.inst_info['neg']}")

    def _print_execution_statistics(self, cu_name: str = "WGP") -> None:
        """ Prints execution statistics for a matrix multiplication instruction.

        Prints the execution statistics, such as computational throughput and co-execution
        information, for a matrix multiplication instruction on a target architecture.
        The instruction and architecture are part of the class.

        Args:
            cu_name: string that contains the name of the compute unit on the target arch
                Different architectures may use e.g., CU, or WGP, or some other name.
                Defaults to "WGP" in gfx11.
        """
        super()._print_execution_statistics(cu_name)

    def _print_register_usage(self, wave_sizes: Tuple[int, ...] = (32, 64)) -> None:
        """ Prints the register count for each input and output matrix.

        Prints the number of registers used by each matrix for a matrix multiplication
        instruction on a target architecture. Some architectures support more than one
        wavefront size. In that case, print out the register usage for each wavefront size.
        This is needed because different wavefront sizes can cause different GPR usage for
        the same matrix size.

        Args:
            wave_sizes: a tuple of wavefront sizes to use to print this info
                Defaults to a tuple of integers 32 and 64 in gfx11, because gfx11
                supports both wave32 and wave64.
        """
        super()._print_register_usage(wave_sizes)

    def _print_register_to_element_eqn(self, print_block: bool = False,
                                       wave_size: Optional[int] = None,
                                       max_a_lane: Optional[int] = None,
                                       max_b_lane: Optional[int] = None) -> None:
        """ Prints equation to map register+lane to matrix element.

        Print out simple equations for mapping a register and its lane to the element in the
        matrix and block which they hold. These can be used by developers that do not want to
        set modifiers like CBSZ/ABID or BLGP to quickly unpack values from registers to
        put them into output matrix locations.

        Args:
            print_block: True if this equation should print information about blocks
            wave_size: integer that holds the wave size for the requested conversion
                function. gfx11 supports wave32 and wave64.
            max_a_lane: optional integer that holds the maximum lane that can be used in this
                instruction for A and K matrices.
            max_b_lane: optional integer that holds the maximum lane that can be used in this
                instruction for B matrices.
        """
        if wave_size is None:
            super()._print_register_to_element_eqn(print_block, 32)
            super()._print_register_to_element_eqn(print_block, 64)
        else:
            super()._print_register_to_element_eqn(print_block)

class InstCalcGfx12(InstCalc):
    """ Calculator for matrix multiplication instruction details on gfx12 architecture.

    This is a child class of the InstCalcGfx11 class, because gfx12/RDNA4 shares many, but not
    all, of its calculations with gfx11/RDNA3.

    Attributes:
        arch_name: string that holds the accelerator architecture's name
        inst_name: string holding the mnemonic of the instruction that will be used in calculations
        inst_info: MatrixInstruction holding the details of the instruction used in calculations
        wave_width: Some architectures allow wavefronts of various widths, and these widths can
            affect the resulting calculations. This integer holds the width that will be used for
            further calculations.
    """
    def check_valid_reg_lane(self, matrix: str, register: int, lane: int) -> bool:
        """ Checks if the register and lane being used for calculation are legal

        Checks whether the matrix and lane being requested are legal for indexing into the
        requested matrix. Callers should use this to ensure that the register is within the
        bounds of the matrix. In addition, ensure that the requested lane is within the
        current wave_width settings for this instruction + gfx 11 architecture

        Args:
            matrix: String indicating the matrix to query: 'a', 'b', 'c', 'd', or 'k'
                for the compression index matrix in sparse matrix ops
            register: Integer of the register number being requested, starting from 0
            lane: Integer of the lane being requested, starting from 0

        Returns:
            True if the register and lane contain data for this matrix on the current instruction
            False if the register and lane do not contain data for this matrix in this instruction
        """
        if (lane < 0 or lane >= (self.wave_width)):
            return False
        num_gprs = self._get_instruction_num_gprs(matrix)
        if (register < 0 or register > num_gprs):
            return False
        # V_SWMMAC_I32_16X16X32_IU4 only uses 32 lanes for A matrix and compression
        # indices when in wave64 mode
        # V_SWMMAC_I32_16X16X32_IU4 only uses 32 lanes for A and B matrices when in
        # wave64 mode
        if (self.inst_name.upper() == 'V_SWMMAC_I32_16X16X32_IU4' and
                matrix in ('a', 'k') and self.wave_width == 64 and lane > 31):
            return False
        if (self.inst_name.upper() == 'V_WMMA_I32_16X16X16_IU4' and
                matrix in ('a', 'b') and self.wave_width == 64 and lane > 31):
            return False
        return True

    def __get_input_reg(self, matrix:str, k: int) -> str:
        """ Calculates a matrix's input register based on coordinates.

        For gfx12, calculates the input register for an instruction's input based on its
        parameters. Only works on the A and B matrices.

        Args:
            matrix: String indicating the matrix to query: 'a', 'b'
            k: integer location within the input matrix's "inner" dimension
                For A matrices, this is the desired column
                For B matrices, this is the desired row

        Returns:
            Based on the matrix and requested coordinates, return the register name that
            holds the element, in a string of the format V#.[bits]
        """
        check_matrix_support(matrix, ('a', 'b'), self.__class__.__name__)
        data_size = get_data_size(self.inst_info['in_type'])
        sparse = self.inst_info['sparse']
        register = 0
        base_bits = -1
        bits_size = data_size
        if sparse:
            if matrix.lower() == 'a': # Sparse matrix A
                if self.wave_width == 32:
                    if data_size == 16:
                        register = 2 * int(k / 16) + (int(k / 4) % 2)
                    elif data_size == 8:
                        register = int(k / 8) % 2
                    else: # data_size == 4
                        if self.inst_info['k'] == 64:
                            register = int(k / 16) % 2
                        # else k == 32 and register = 0
                else: # self.wave_width == 64
                    if data_size == 16:
                        register = int(k / 4) % 2
                    # else data_size == 8 or data_size == 4 and register = 0
                # 16-bit input with sparse A matrix uses whole reg for 4 inputs
                if data_size != 16:
                    base_bits = 2*data_size
                    base_bits *= (int(k / 4) % int(16/data_size))
                    bits_size *= 2
            else: # Dense matrix B for sparse instruction
                if self.wave_width == 32:
                    if data_size == 16:
                        register = 4 * int(k / 16) + (int(k / 2) % 4)
                    elif data_size == 8:
                        register = int(k / 4) % 4
                    else: # data_size == 4
                        if self.inst_info['k'] == 32:
                            register = int(k / 8) % 2
                        else: # Matrix K == 64
                            register = int(k / 8) % 4
                else: # self.wave_width == 64
                    if data_size == 16:
                        register = int(k / 2) % 4
                    elif data_size == 8:
                        register = int(k / 4) % 2
                    else: # data_size == 4
                        if self.inst_info['k'] != 32: # K == 64
                            register = int(k / 8) % 2
                        # else K == 32 and register = 0
                base_bits = data_size * (k % int(32/data_size))
        else: # non-sparse A or B
            if self.wave_width == 32:
                if data_size == 16:
                    register = 2 * int(k/8) + (int(k / 2) % 2)
                elif data_size == 8:
                    register = int(k / 4) % 2
                elif self.inst_info['k'] == 32: # data_size == 4 and K width is 32
                    register = int(k / 8) % 2
                else:
                    register = 0
            else: # self.wave_width == 64
                if data_size == 16:
                    register = int(k / 2) % 2
                # else data-size == 8 or data_size == 4 and register = 0
            base_bits = data_size * (k % int(32/data_size))
        ret_str = f'v{register}'
        if base_bits != -1:
            ret_str = f'{ret_str}.[{base_bits+bits_size-1}:{base_bits}]'
        return ret_str

    def get_num_compression_sets(self) -> int:
        """ Returns the number of compression index sets that an instruction has

        Returns:
            An integer that contains the number of compression index sets that an
                instruction allows for the given configuration
        """
        data_size = get_data_size(self.inst_info['in_type'])
        num_sets = 0
        if self.inst_info['sparse']:
            if self.wave_width == 32:
                if data_size == 4 and self.inst_info['k'] == 64:
                    num_sets = 1
                else:
                    num_sets = 2
            else:
                if data_size in (8, 16):
                    num_sets = 4
                else:
                    num_sets = 2
        return num_sets

    def _get_num_compression_bit_offset(self, opsel: int) -> int:
        """ Returns the bit offset of the first compression index, based on instruction modifiers

        Args:
            opsel: The OPSEL modifier for the matrix instruction

        Returns:
            An integer that contains the starting bit of this set of compression indices
        """
        num_sets = self.get_num_compression_sets()
        per_opsel_offset = 0
        if num_sets > 0:
            per_opsel_offset = int(32 / num_sets)
        return opsel * per_opsel_offset

    def __get_compression_bits(self, matrix:str, k: int, opsel: int) -> str:
        """ Calculates the bit locations for a compression entry based on coordinates.

        For gfx12, calculates the bits in the compression index VGPR based on the
        instruction and the input matrix coordinates.

        Args:
            matrix: String indicating the matrix to query: 'a', 'b', 'c', 'd', or 'k'
            k: integer location of the A matrix column for this compression index
            opsel: integer value of the instruction's OPSEL modifier

        Returns:
            Based on the requested coordinates, return the bits in the compression VGPR
            that hold the compression index, in a string of the format [bits]
        """
        check_matrix_support(matrix, ('k', ), self.__class__.__name__)
        data_size = get_data_size(self.inst_info['in_type'])
        # register is always 0
        if self.wave_width == 32:
            if data_size == 16:
                base_bits = 8 * int(k/16) + 4 * (int(k/4) % 2)
            elif data_size == 8 or self.inst_info['k'] == 32:
                # 8-bit type or 4-bit type with matrix K == 32
                base_bits = 4 * (int(k / 4) % 4)
            else:
                # data_size == 4 and matrix K == 64
                base_bits = 4 * (int(k / 4) % 8)
        else:
            if data_size in (8, 16):
                base_bits = 4 * (int(k / 4) % 2)
            else:
                # data_size == 4
                base_bits = 4 * (int(k / 4) % 4)
        base_bits += self._get_num_compression_bit_offset(opsel)
        return f'[{base_bits+3}:{base_bits}]'

    def __get_input_lane(self, matrix: str, i: int, k: int) -> int:
        """ Calculates the lane for an input matrix based on coordinates.

        For gfx12, calculates the lane for an instruction's matrix based on its parameters.
        Only works on the A and B matrices, or for the compression index K.

        Args:
            matrix: String indicating the matrix to query: 'a', 'b', 'c', 'd', or 'k'
            i: integer location within the input matrix's "outer" dimension
                For A and K matrices, this is the desired row
                For B matrices, this is the desired column
            k: integer location within the input matrix's "inner" dimension
                For A and K matrices, this is the desired column
                For B matrices, this is the desired row
            data_size: integer size of the input data, in bits

        Returns:
            Based on the matrix and requested coordinates, return the register name that
            holds the element, in a string of the format V#.[bits]

        Takes the instruction info and matrix, return a string with an equation that lets a user
        calculate the lane that holds a particular entry in the matrix from its i/j/k/block
        coordinates.

        Args:
            matrix: string that contains the name of the matrix
                Legal values are a, b, c, d, or k.
            wave_size: integer that holds the wave size for the requested conversion
                function. gfx12 supports wave64 and wave32.

        Returns:
            String that contains the simple formula mapping coordinates to lanes
        """
        check_matrix_support(matrix, ('a', 'b', 'k'), self.__class__.__name__)
        data_size = get_data_size(self.inst_info['in_type'])
        sparse = self.inst_info['sparse']
        ret_val = 0

        if sparse:
            if self.wave_width == 32:
                if data_size == 16:
                    ret_val = 16 * (int(k / 8) % 2) + i
                elif data_size == 8 or self.inst_info['k'] == 32:
                    # 8-bit type or 4-bit type with matrix K == 32
                    ret_val = 16 * int(k / 16) + i
                else:
                    # data_size == 4 and matrix K == 64
                    ret_val = 16 * int(k / 32) + i
            else: # self.wave_width == 64
                if data_size == 16:
                    ret_val = 16 * int(k / 8) + i
                elif data_size == 8 or self.inst_info['k'] == 32:
                    if data_size == 4 and matrix.lower() in ('a', 'k'):
                        # 4-bit K == 32 sparse A can only use 32 of 64 lanes
                        ret_val = 16 * int(k / 16) + i
                    else:
                        # 8-bit type or B matrix for 4-bit type with matrix K == 32
                        ret_val = 32 * (int(k / 8) % 2) + 16 * int(k / 16) + i
                else:
                    # data_size == 4 and matrix K == 64
                    ret_val = 32 * (int(k / 16) % 2) + 16 * int(k / 32) + i
        else:
            if self.wave_width == 32:
                if data_size == 16:
                    ret_val = 16 * (int(k / 4) % 2) + i
                elif data_size == 8 or self.inst_info['k'] == 16:
                    # 8-bit type, or 4-bit type with matrix K == 16
                    ret_val = 16 * int(k / 8) + i
                else: # data_size == 4 and matrix K == 32
                    ret_val = 16 * int(k / 16) + i
            else: # self.wave_width == 64
                if data_size == 16:
                    ret_val = 32 * (int(k / 8) % 2) + 16 * (int(k / 4) % 2) + i
                elif data_size == 8:
                    ret_val = 32 * (int(k / 4) % 2) + 16 * (int(k / 8) % 2) + i
                elif self.inst_info['k'] == 16:
                    # data_size == 4 and matrix K == 16
                    # Only half of the wave64 is enabled in this case
                    ret_val = 16 * int(k / 8) + i
                else: # data_size == 4 and matrix K == 32
                    ret_val = 32 * (int(k / 8) % 2) + 16 * (int(k / 16) % 2) + i
        return ret_val

    def __get_input_reg_lanes(self, matrix: str, i: int, k: int,
                              opsel: int) -> Tuple[str, List[int]]:
        """ Calculates a matrix's input register and lane number based on coordinates.

        For gfx12, calculates the input register and the lane within that register for an
        instruction based on its parameters.

        Args:
            matrix: String indicating the matrix to query: 'a', 'b', or 'k'
            i: integer location within the input matrix's "outer" dimension
                For A and K matrices, this is the desired row
                For B matrices, this is the desired column
            k: integer location within the input matrix's "inner" dimension
                For A and K matrices, this is the desired column
                For B matrices, this is the desired row
            opsel: integer value of the instruction's OPSEL modifier

        Returns:
            Based on the matrix and requested coordinates, return two things in a tuple:
            1. String register name that holds the element, in the format V#.[bits]
            2. List of integers, containing the lane numbers within that register that
                hold the element
            Tuple: (register holding the matrix entry, lanes within that register)
        """
        check_matrix_support(matrix, ('a', 'b', 'k'), self.__class__.__name__)
        reg_string = ""
        if matrix.lower() == 'k':
            reg_string = f'v0.{self.__get_compression_bits(matrix, k, opsel)}'
        else: # matrix.lower() == 'a' or 'b'
            reg_string = self.__get_input_reg(matrix, k)
        lane = self.__get_input_lane(matrix, i, k)
        return (reg_string, [lane])

    def __get_output_reg_lanes(self, i: int, j: int) -> Tuple[str, List[int]]:
        """ Calculates a matrix's output register and lane number based on coordinates.

        For gfx12, calculates the output register and the lane within that register for
        an instruction based on its parameters.

        Args:
            i: integer location within the output matrix's rows
            j: integer location within the output matrix's columns

        Returns:
            Based on the matrix and requested coordinates, return two things in a tuple:
            1. String register name that holds the element, in the format V#.[bits]
            2. List of integers, containing the lane numbers within that register that
                hold the element
            Tuple: (register holding the matrix entry, lanes within that register)
        """
        data_size = get_data_size(self.inst_info['out_type'])
        register = ""
        lane = 0
        if self.wave_width == 32:
            if data_size == 32:
                register = f'v{i % 8}'
            else: # data_size == 16
                base_bits = 16 * (i % 2)
                register = f'v{int(i / 2) % 4}.[{base_bits + 15}:{base_bits}]'
            lane = 16 * int(i / 8) + j
        else: # self.wave_width == 64
            if data_size == 32:
                register = f'v{i % 4}'
            else: # data_size == 16
                base_bits = 16 * (i % 2)
                register = f'v{int(i / 2) % 2}.[{base_bits + 15}:{base_bits}]'
            lane = 32 * (int(i / 4) % 2) + 16 * int(i / 8) + j
        return (register, [lane])

    def _get_reg_lanes(self, matrix: str, i: int, j: int, k: int, block: int, cbsz: int,
                       abid: int, blgp: int, opsel: int) -> Tuple[str, str, List[int]]:
        """ Calculates a matrix's register and lane number based on coordinates.

        For the target architecture and the instruction set up in this class's init
        function, this function calculates the register and lane that hold a requested
        matrix entry in a requested matrix. This location information can vary based on
        per-instruction modifiers, so those are all input arguments.

        block, cbsz, abid, and blgp are ignored by this function, because gfx12 does not use
        these instruction modifiers.

        Args:
            matrix: String indicating the matrix to query: 'a', 'b', 'c', 'd', or 'k'
            i: integer coordinate for the query of the matrix row for A, C, D, & K matrices
            j: integer coordinate for the query of the matrix column for the B, C, & D matrices
            k: integer coordinate for the query of the A & K column or B row
            block: integer coordinate for the block to query
            cbsz: integer value of the instruction's CBSZ modifier
            abid: integer value of the instruction's ABID modifier
            blgp: integer value of the instruction's BLGP modifier
            opsel: integer value of the instruction's OPSEL modifier

        Returns:
            Based on the matrix and requested coordinates, return three things in a tuple:
            1. String requested element name, in the format A[i][k], B[k][j], or C[i][j]
            2. String register name that holds the element, in the format V#.[bits]
            3. List of integers, containing the lane numbers within that register that
                hold the element
            Tuple: (matrix entry, register holding that entry, lanes within that register)
        """
        del block, cbsz, abid, blgp # Unused in gfx12
        check_matrix_support(matrix, ('a', 'b', 'c', 'd', 'k'), self.__class__.__name__)
        if matrix.lower() in ('a', 'k'):
            (reg, lanes) = self.__get_input_reg_lanes(matrix, i, k, opsel)
            element_name = f"{matrix.upper()}[{i}][{k}]"
        elif matrix.lower() == 'b':
            (reg, lanes) = self.__get_input_reg_lanes(matrix, j, k, opsel)
            element_name = f"{matrix.upper()}[{k}][{j}]"
        else: # (matrix.lower() == 'c' or matrix.lower() == 'd'):
            (reg, lanes) = self.__get_output_reg_lanes(i, j)
            element_name = f"{matrix.upper()}[{i}][{j}]"
        return (element_name, reg, lanes)

    def _calculate_num_regnos_to_print(self, matrix: str, gpr_ratio: float) -> int:
        """ Calculates the number of register slots to print for this matrix & instruction.

        On some architectures, partial registers (such as a 16b output in a 32b register)
        aren't tightly packed. For example, "lower" or "upper halves may be skipped
        instead of storing two contiguous values, one in the lower and one in the upper.
        In such architectures and for  certain matrices, the OPSEL value controls whether
        to use the upper or lower halves.

        This function returns the number of slots to print in each register; the rest can
        be skipped.

        In gfx12, sparse matrix values need to print twice as many "regnos" because the
        register printing function needing to print twice as much data per register
        However, for the compression index matrix 'k' in wave64, this is not needed
        because each compression index is spread over more lanes. So there are half as
        many regnos to print per lane.

        Args:
            matrix: string that contains the name of the matrix
                Legal values are a, b, c, and d.
            gpr_ratio: the number of regnos in each GPR

        Returns:
            Integer indicating how many of the regnos in each GPR to print

        Raises:
            ValueError: An unsupported matrix was requested.
        """
        check_matrix_support(matrix, ('a', 'b', 'c', 'd', 'k'), self.__class__.__name__)
        num_regnos_to_print = super()._calculate_num_regnos_to_print(matrix, gpr_ratio)
        if (matrix.lower() == 'k' and self.wave_width == 32 and self.inst_info['k'] == 64):
            num_regnos_to_print *= 2
        return num_regnos_to_print

    def calculate_register_layout(self, matrix: str, requested_output: str,
                                  negate: Dict[str, bool], cbsz: int, abid: int, blgp: int,
                                  opsel: int, transpose: bool, print_blocks: bool = False) -> None:
        """ Displays the registers+lanes for an entire matrix.

        Calculate and display the registers and lanes for an entire input or
        output matrix. Displays the matrix formatted as its rows and columns,
        with the registers and their lanes displayed in a tabular format.
        Can optionally print this tabular format as a CSV, markdown, or asciidoc
        for other processing. Can also choose to transpose the matrix to visually
        show rows as columns and vice versa.
        The registers are displayed as: Va{b}.c:
            - a is the register number
            - b is the lane within that register
            - c is an optional identifier for sub-Dword parts of the 32 bit register:
                    [15:0]: the lower 16b of a 32b register
                    [31:16]: the upper 16b of a 32b register
                    [7:0]: the least significant 8b of a 32b register
                    [15:8]: the second-lowest 8b of a 32b register
                    [23:16]: the second-highest 8b of a 32b register
                    [31:24]: the most significant 8b of a 32b register

        Args:
            matrix: string that contains the name of the matrix
                Legal values are a, b, c, or d.
            requested_output: string that indicates the type of output, from the list of
                csv, markdown, asciidoc, or grid
            negate: dictionary of matrix names to bools that indicate whether to
                negate and absolute-val entries from this matrix.
            cbsz: integer value of the instruction's CBSZ modifier
            abid: integer value of the instruction's ABID modifier
            blgp: integer value of the instruction's BLGP modifier
            opsel: integer value of the instruction's OPSEL modifier
            transpose: boolean set to true to cause the matrix to be printed transposed
            print_blocks: boolean set to true if this architecture and instruction
                should print the word "Block #" above each block of the matrix
                gfx12 does not do this by default.
        """
        super().calculate_register_layout(matrix, requested_output, negate, cbsz, abid, blgp,
                                          opsel, transpose, print_blocks)

    def calculate_matrix_layout(self, matrix: str, requested_output: str, negate: Dict[str, bool],
                                cbsz: int, abid: int, blgp: int, opsel: int, transpose: bool,
                                contig_values: int = 32) -> None:
        """ Displays the matrix entries for all of the registers+lanes used by an instruction.

        Calculate and display the matrix elements for all register entries and
        lanes used by the requesting instruction.
        Displays the registers formatted by register entry (X axis) and wavefront lane
        (Y axis). The resulting table then shows the MatrixName[col][row] held in
        that lane's register entry.
        Can optionally print this tabular format as a CSV, markdown, or asciidoc
        for other processing. Can also choose to transpose the matrix to visually
        show rows as columns and vice versa.

        Args:
            matrix: string that contains the name of the matrix
                Legal values are a, b, c, or d.
            requested_output: string that indicates the type of output, from the list of
                csv, markdown, asciidoc, or grid
            negate: dictionary of matrix names to bools that indicate whether to
                negate and absolute-val entries from this matrix.
            cbsz: integer value of the instruction's CBSZ modifier
            abid: integer value of the instruction's ABID modifier
            blgp: integer value of the instruction's BLGP modifier
            opsel: integer value of the instruction's OPSEL modifier
            transpose: boolean set to true to cause the matrix to be printed transposed
            contig_values: an integer that defines the number of contiguous values of a
                register that are used to hold unique values of a matrix
                gfx12's values change depending on the wavefront width in use
        """
        contig_values = self.wave_width
        data_size = get_data_size(self.inst_info['in_type'])
        K = self.inst_info['k']
        if data_size == 4 and K == 16:
            # 4-bit ops with K==16 can only use 32 lanes no matter what
            contig_values = 32
        if matrix.lower() in ('a', 'k') and data_size == 4 and K == 32:
            # k and SparseA for 4-bit ops with K==32 can only use 32 lanes
            contig_values = 32
        super().calculate_matrix_layout(matrix, requested_output, negate, cbsz, abid, blgp, opsel,
                                        transpose, contig_values)

    def _get_instruction_num_gprs(self, matrix: str, in_lanes: Optional[int] = None,
                                  out_size: Optional[int] = None) -> int:
        """ Calculates the number of GPRs needed to hold a matrix.

        Args:
            matrix: string that contains the name of the matrix
                Legal values are a, b, c, d, or k.
            in_lanes: an integer that defines the number of contiguous lanes that are used to hold
                values of a matrix. On gfx12, this ends up being the wave width.
            out_size: The number of bits used to hold output values for this instruction
                For instance: some devices may store 16b values into either the low or high
                half of a 32b output register. If this argument is not passed in, the function
                will initialize the value to 32.

        Returns:
            An integer that defines the number of GPRs needed to hold the requested matrix

        Raises:
            ValueError: An unsupported matrix was requested.
        """
        check_matrix_support(matrix, ('a', 'b', 'c', 'd', 'k'), self.__class__.__name__)
        del out_size # Unused in gfx12
        out_size = get_data_size(self.inst_info['out_type'])
        # Parent class, gfx11 will always override the in_lanes, and we want to actually pass
        # the number of lanes through for gfx12, so we want to call its parent directly
        return super()._get_instruction_num_gprs(matrix, in_lanes, out_size)

    def _print_register_info(self, encoding_name: str = "VOP3P") -> None:
        """ Prints the encoding and register information for a matrix instruction.

        Prints the encoding and modifier information for the registers used by a
        matrix multiplication instruction on a particular architecture.
        Architecture matters, because different architectures support different
        encodings and matrix modifier fields.

        Args:
            encoding_name: String containing the name of the encoding format for the
                current architecture
                gfx12 defaults to "VOP3P".
        """
        super()._print_register_info(encoding_name)
        print("    Register modifiers:")
        print(f"        OPSEL supported: {self.inst_info['sparse']}")
        print(f"        NEG bits supported: {self.inst_info['neg']}")

    def _coord_to_input_reg_eqn(self, matrix: str, wave_size: Optional[int] = 32) -> str:
        """ Returns formula for mapping a matrix coordinate to its input register number.

        Takes the instruction info and matrix, return a string with an equation that lets a user
        calculate the input register that holds a particular entry in the matrix from its
        i/j/k/block coordinates.

        Args:
            matrix: string that contains the name of the matrix
                Legal values are a, b, or k (for the compression index of sparse matrices).
            wave_size: integer that holds the wave size for the requested conversion
                function. gfx12 changes register equations for different wave sizes.

        Returns:
            String that contains the simple formula mapping coordinates to input registers

        Raises:
            ValueError: An unsupported matrix was requested.
        """
        check_matrix_support(matrix, ('a', 'b', 'k'), self.__class__.__name__)
        data_size = get_data_size(self.inst_info['in_type'])
        sparse = (matrix.lower() in ('a', 'b') and self.inst_info['sparse'])

        reg_string = "Unknown"
        bit_string = ""
        if sparse:
            if matrix.lower() == 'a': # Sparse matrix A
                if wave_size == 32:
                    if data_size == 16:
                        reg_string = '2 * floor(k / 16) + (floor(k / 4) % 2)'
                    elif data_size == 8:
                        reg_string = 'floor(k / 8) % 2'
                    else: # data_size == 4
                        if self.inst_info['k'] == 32:
                            reg_string = '0'
                        else: # Matrix K == 64
                            reg_string = 'floor(k / 16) % 2'
                else: # wave_size == 64
                    if data_size == 16:
                        reg_string = 'floor(k / 4) % 2'
                    else: #data_size == 8 or data_size == 4
                        reg_string = '0'
                if data_size != 16:
                    base_string = f'{2*data_size} * '
                    base_string += f'(floor(k / 4) % {int(16 / data_size)})'
                    bit_string = f'{base_string} + {2*data_size-1} : {base_string}'
            else: # Dense matrix B for sparse instruction
                if wave_size == 32:
                    if data_size == 16:
                        reg_string = '4 * floor(k / 16) + (floor(k / 2) % 4)'
                    elif data_size == 8:
                        reg_string = 'floor(k / 4) % 4'
                    else: # data_size == 4
                        if self.inst_info['k'] == 32:
                            reg_string = 'floor(k / 8) % 2'
                        else: # Matrix K == 64
                            reg_string = 'floor(k / 8) % 4'
                else: # wave_size == 64
                    if data_size == 16:
                        reg_string = 'floor(k / 2) % 4'
                    elif data_size == 8:
                        reg_string = 'floor(k / 4) % 2'
                    else: # data_size == 4
                        if self.inst_info['k'] == 32:
                            reg_string = '0'
                        else: # Matrix K == 64
                            reg_string = 'floor(k / 8) % 2'
                base_string = f'{data_size} * (k % {int(32/data_size)})'
                bit_string = f'{base_string} + {data_size-1} : {base_string}'
        elif matrix.lower() == 'k':
            reg_string = '0'
            if wave_size == 32:
                if data_size == 16:
                    base_string = '8 * floor(k / 16) + 4 * (floor(k / 4) % 2)'
                elif data_size == 8 or self.inst_info['k'] == 32:
                    # 8-bit type or 4-bit type with matrix K == 32
                    base_string = '4 * (floor(k / 4) % 4)'
                else: # data_size == 4 and matrix K == 64
                    base_string = '4 * (floor(k / 4) % 8)'
            else:
                if data_size in (8, 16):
                    base_string = '4 * (floor(k / 4) % 2)'
                else: # data_size == 4
                    base_string = '4 * (floor(k / 4) % 4)'
            bit_string = f'{base_string} + 3 : {base_string}'
        else: # non-sparse A or B
            if wave_size == 32:
                if data_size == 16:
                    reg_string = '2 * floor(k / 8) + (floor(k / 2) % 2)'
                elif data_size == 8:
                    reg_string = 'floor(k / 4) % 2'
                elif self.inst_info['k'] == 32: # data_size == 4 and K width is 32
                    reg_string = 'floor(k / 8) % 2'
                else:
                    reg_string = '0'
            else: # wave_size == 64
                if data_size == 16:
                    reg_string = 'floor(k / 2) % 2'
                else: # data_size == 8 or data_size == 4
                    reg_string = '0'
            base_string = f'{data_size} * (k % {int(32/data_size)})'
            bit_string = f'{base_string} + {data_size-1} : {base_string}'
        ret_string = reg_string
        if reg_string != '0':
            ret_string = '(' + reg_string + ')'
        if bit_string != '':
            ret_string += '.[' + bit_string + ']'
        return ret_string

    def _coord_to_output_reg_eqn(self, wave_size: Optional[int] = 32) -> str:
        """ Returns formula for mapping a matrix coordinate to its output register number.

        Takes the instruction info and matrix, return a string with an equation that lets a user
        calculate the output register that holds a particular entry in the matrix from its
        i/j/k/block coordinates.

        Args:
            wave_size: integer that holds the wave size for the requested conversion
                function. gfx12 changes register equations for different wave sizes.

        Returns:
            String that contains the simple formula mapping coordinates to output registers
        """
        data_size = get_data_size(self.inst_info['out_type'])

        ret_string = "Unknown"
        if wave_size == 32:
            if data_size == 32:
                ret_string = 'i % 8'
            else: # data_size == 16
                ret_string = '(floor(i / 2) % 4).[16 * (i % 2) + 15 : 16 * (i % 2)]'
        else: # wave_size == 64
            if data_size == 32:
                ret_string = 'i % 4'
            else: # data_size == 16
                ret_string = '(floor(i / 2) % 2).[16 * (i % 2) + 15 : 16 * (i % 2)]'
        return ret_string

    def _coord_to_lane_eqn(self, matrix: str, wave_size: Optional[int] = 32) -> str:
        """ Returns formula for mapping a matrix coordinate to its wavefront lane.

        Takes the instruction info and matrix, return a string with an equation that lets a user
        calculate the lane that holds a particular entry in the matrix from its i/j/k/block
        coordinates.

        Args:
            matrix: string that contains the name of the matrix
                Legal values are a, b, c, d, or k.
            wave_size: integer that holds the wave size for the requested conversion
                function. gfx12 supports wave64 and wave32.

        Returns:
            String that contains the simple formula mapping coordinates to lanes
        """
        check_matrix_support(matrix, ('a', 'b', 'c', 'd', 'k'), self.__class__.__name__)
        sparse = (matrix.lower() in ('a', 'b', 'k') and self.inst_info['sparse'])
        ret_string = "Unknown"
        if sparse:
            data_size = get_data_size(self.inst_info['in_type'])
            matrix_index = 'j' if matrix.lower() == 'b' else 'i'
            if wave_size == 32:
                if data_size == 16:
                    ret_string = f"16 * (floor(k / 8) % 2) + {matrix_index}"
                elif data_size == 8 or self.inst_info['k'] == 32:
                    # 8-bit type or 4-bit type with matrix K == 32
                    ret_string = f"16 * floor(k / 16) + {matrix_index}"
                else:
                    # data_size == 4 and matrix K == 64
                    ret_string = f"16 * floor(k / 32) + {matrix_index}"
            else: # wave_size == 64
                if data_size == 16:
                    ret_string = f"16 * floor(k / 8) + {matrix_index}"
                elif data_size == 8 or self.inst_info['k'] == 32:
                    if data_size == 4 and matrix.lower() in ('a', 'k'):
                        # 4-bit K == 32 sparse A can only use 32 of 64 lanes
                        ret_string = f"(16 * floor(k / 16) + {matrix_index}"
                    else:
                        # 8-bit type or B matrix for 4-bit type with matrix K == 32
                        ret_string = "32 * (floor(k / 8) % 2) + 16 * "
                        ret_string += f"floor(k / 16) + {matrix_index}"
                else:
                    # data_size == 4 and matrix K == 64
                    ret_string = f"32 * (floor(k / 16) % 2) + 16 * floor(k / 32) + {matrix_index}"
        else:
            if matrix.lower() in ('a', 'b'): # non-sparse A or B
                data_size = get_data_size(self.inst_info['in_type'])
                matrix_index = 'i' if matrix.lower() == 'a' else 'j'
                if wave_size == 32:
                    if data_size == 16:
                        ret_string = f"16 * (floor(k / 4) % 2) + {matrix_index}"
                    elif data_size == 8 or self.inst_info['k'] == 16:
                        # 8-bit type, or 4-bit type with matrix K == 16
                        ret_string = f"16 * floor(k / 8) + {matrix_index}"
                    else: # data_size == 4 and matrix K == 32
                        ret_string = f"16 * floor(k / 16) + {matrix_index}"
                else: # wave_size == 64
                    if data_size == 16:
                        ret_string = '32 * (floor(k / 8) % 2) + 16 * (floor(k / 4) % 2) + '
                        ret_string += f"{matrix_index}"
                    elif data_size == 8:
                        ret_string = '32 * (floor(k / 4) % 2) + 16 * (floor(k / 8) % 2) + '
                        ret_string += f"{matrix_index}"
                    elif self.inst_info['k'] == 16:
                        # data_size == 4 and matrix K == 16
                        # Only half of the wave64 is enabled in this case
                        ret_string = f"16 * floor(k / 8) + {matrix_index}"
                    else: # data_size == 4 and matrix K == 32
                        ret_string = '32 * (floor(k / 8) % 2) + 16 * (floor(k / 16) % 2) + '
                        ret_string += f"{matrix_index}"
            else: # C or D
                if wave_size == 32:
                    ret_string = '16 * floor(i / 8) + j'
                else: # wave_size == 64
                    ret_string = '32 * (floor(i / 4) % 2) + 16 * floor(i / 8) + j'
        return ret_string

    def _reg_lane_to_block_eqn(self, matrix: str) -> str:
        """ Returns equation to map register+lane to block.

        Return a string that can be used to quickly calculate how to go from a register and
        lane to the block of the input or output matrix. Targets a particular instruction,
        and does not handle modifiers.

        Args:
            matrix: string that contains the name of the matrix

        Returns:
            A blank string, because there are no blocks in gfx12
        """
        del matrix # Unused in gfx12
        return ""

    def _print_element_to_register_eqn(self, block: str = "",
                                       wave_size: Optional[int] = None) -> None:
        """ Prints formula for matrix entry to GPR and lane mapping.

        Prints out the simple formulae to calculate the the mapping of a matrix element to its
        register and lane.

        Will always print A[], B[], and D[]. If the instruction allows C[] matrices, then it will
        print D[] as "C or D[i][j]".

        Args:
            block: string that contains text to place in the matrix entry map for blocks
                For instance, gfx12 does not uses blocks, so by default this function passes an
                empty string, to print an entry as A[i][k]
            wave_size: integer that contains the requested wave size to use for printing the
                formula. gfx12 supports two different wave sizes, and if this optional value is
                not passed in, we will print both
        """
        if wave_size is None:
            super()._print_element_to_register_eqn(block, 32)
            super()._print_element_to_register_eqn(block, 64)
        else:
            super()._print_element_to_register_eqn(block, wave_size)

    def _reg_lane_to_i_coord_eqn(self, matrix: str,
                                       wave_size: Optional[int] = 32) -> str:
        """ Returns equation to map register+lane to i index.

        Takes instruction info and returns a string containing an equation which lets users
        calculate the i coordinate for the A, C, or D matrices.

        Args:
            matrix: string that contains the name of the matrix
                Legal values are a, c, d, or k (for the compression index of sparse matrices).
            wave_size: integer that holds the wave size for the requested conversion
                function. gfx12 supports wave64 and wave32.

        Returns:
            A string which contains the equation to calculate the i coordinate for
            an input matrix held by a particular register and lane combination

        Raises:
            ValueError: An unsupported matrix was requested.
        """
        check_matrix_support(matrix, ('a', 'c', 'd', 'k'), self.__class__.__name__)
        ret_string = "Unknown"
        data_size = get_data_size(self.inst_info['out_type'])

        if matrix.lower() in ('a', 'k'):
            ret_string = super()._reg_lane_to_i_coord_eqn(matrix, wave_size)
        else:
            # C or D
            if wave_size == 32:
                if data_size == 32:
                    ret_string = "8 * floor(lane / 16) + GPR_num"
                else: # data_size == 16
                    ret_string = "8 * floor(lane / 16) + 2 * GPR_num + floor(GPR_bits / 16)"
            else: # wave_size == 64
                if data_size == 32:
                    ret_string = "8 * (floor(lane / 16) % 2) + 4 * floor(lane / 32) + GPR_num"
                else: # data_size == 16
                    ret_string = "8 * (floor(lane / 16) % 2) + 4 * floor(lane / 32) + "
                    ret_string += "2 * GPR_num + floor(GPR_bits / 16)"
        return ret_string

    def _reg_lane_to_k_coord_eqn(self, matrix: str,
                                 wave_size: Optional[int] = 32) -> str:
        """ Returns equation to map register+lane to k index.

        Takes instruction info and a target matrix, and returns a string containing an
        equation which lets users calculate the k coordinate based on the register and lane.
        Targets a particular instruction, and does not handle modifiers.

        Args:
            matrix: string that contains the name of the matrix
                Legal values are a, b, and k (for the compression index of sparse matrices).
            wave_size: integer that holds the wave size for the requested conversion
                function. gfx12 supports wave64 and wave32.

        Returns:
            A string which contains the equation to calculate the k coordinate held by a
            particular register and lane combination

        Raises:
            ValueError: An unsupported matrix was requested.
        """
        check_matrix_support(matrix, ('a', 'b', 'k'), self.__class__.__name__)
        ret_string = "Unknown"
        data_size = get_data_size(self.inst_info['in_type'])

        if self.inst_info['sparse']:
            if matrix == 'a':
                if data_size == 16:
                    start_point = ""
                    if wave_size == 32:
                        start_point = "16 * floor(GPR_num / 2) + "
                    start_point += "8 * floor(lane / 16) + "
                    start_point += "4 * (GPR_num % 2)"
                elif data_size == 8 or self.inst_info['k'] == 32:
                    if wave_size == 32:
                        start_point = "16 * floor(lane / 16) + "
                        if data_size == 8:
                            start_point += "8 * GPR_num + "
                    else: # wave_size == 64
                        start_point = "16 * (floor(lane / 16) % 2) + "
                        if data_size == 8:
                            start_point += "8 * floor(lane / 32) + "
                    start_point += f"4 * floor(GPR_bits / {int(2*data_size)})"
                else: # data_size == 4
                    if wave_size == 32:
                        start_point = "32 * floor(lane / 16) + "
                        start_point += "16 * GPR_num + "
                    else: # wave_size == 64
                        start_point = "32 * (floor(lane / 16) % 2) + "
                        start_point += "16 * floor(lane / 32) + "
                    start_point += "4 * floor(GPR_bits / 8)"
                end_point = f"({start_point} + 3)"
                ret_string = f"{end_point}\nthrough\n{start_point}"
            elif matrix == 'k':
                if data_size == 16:
                    start_point = ""
                    if wave_size == 32:
                        start_point = "16 * (floor(GPR_bits / 8) % 2) + "
                    start_point += "8 * floor(lane / 16) + "
                    start_point += "4 * (floor(GPR_bits / 4) % 2)"
                elif data_size == 8:
                    if wave_size == 32:
                        start_point = "16 * floor(lane / 16) + "
                    else: # wave_size == 64:
                        start_point = "16 * (floor(lane / 16) % 2) + "
                        start_point += "8 * floor(lane / 32) + "
                    start_point += "4 * (floor(GPR_bits / 4) % 4)"
                elif self.inst_info['k'] == 32:
                    start_point = "16 * floor(lane / 16) + "
                    start_point += "4 * floor(GPR_bits / 4) % 4)"
                else: #data_size == 4 and matrix K == 64
                    if wave_size == 32:
                        start_point = "32 * floor(lane / 16) + "
                    else: # wave_size == 64:
                        start_point = "32 * (floor(lane / 16) % 2) + "
                        start_point += "16 * floor(lane / 32) + "
                    start_point += "4 * (floor(GPR_bits / 4) % 8)"
                prefix=""
                end_point = f"{start_point} + 3"
                if data_size == 4 and self.inst_info['k'] == 32 and wave_size == 64:
                    prefix = "(lanes 0-31 only)\n"
                ret_string = f"{prefix}{end_point}\nthrough\n{start_point}"
            else: # matrix is dense B for sparse instruction
                if data_size == 16:
                    if wave_size == 32:
                        ret_string = "16 * floor(GPR_num / 4) + 8 * floor(lane / 16) + "
                    else: # wave_size == 64
                        ret_string = "8 * floor(lane / 16) + "
                    ret_string += "2 * (GPR_num % 4) + floor(GPR_bits / 16)"
                else:
                    if data_size == 8:
                        if wave_size == 32:
                            ret_string = "16 * floor(lane / 16) + "
                        else: # wave_size == 64
                            ret_string = "16 * (floor(lane / 16) % 2) + "
                        if data_size == 8:
                            if wave_size == 64:
                                ret_string += "8 * floor(lane / 32) + "
                            ret_string += "4 * GPR_num + "
                    elif self.inst_info['k'] == 32:
                        if wave_size == 32:
                            ret_string = "16 * floor(lane / 16) + 8 * GPR_num + "
                        else: # wave_size == 64
                            ret_string = "16 * (floor(lane / 16) % 2) + 8 * floor(lane / 32) + "
                    else: # 4-bit with k == 64
                        if wave_size == 32:
                            ret_string = "32 * floor(lane / 16) + "
                        else: # wave_size == 64
                            ret_string = "32 * (floor(lane / 16) % 2) + 16 * floor(lane / 32) + "
                        ret_string += "8 * GPR_num + "
                    ret_string += f"floor(GPR_bits / {data_size})"
        else:
            if data_size == 16:
                if wave_size == 32:
                    ret_string = "8 * floor(GPR_num / 2) + 4 * floor(lane / 16) + "
                else: # wave_size == 64
                    ret_string = "4 * floor(lane / 16) + "
                ret_string += "2 * (GPR_num % 2) + floor(GPR_bits / 16)"
            else:
                if data_size == 8:
                    if wave_size == 32:
                        ret_string = "8 * floor(lane / 16) + 4 * GPR_num + "
                    else: # wave_size == 64
                        ret_string = "8 * (floor(lane / 16) % 2) + 4 * floor(lane / 32) + "
                elif (data_size == 4 and self.inst_info['k'] == 16):
                    # 4-bit with k == 16 will only use 32 lanes out of 64.
                    ret_string = "8 * floor(lane / 16) + "
                else: # 4-bit with k == 32
                    if wave_size == 32:
                        ret_string = "16 * floor(lane / 16) + "
                        ret_string += "8 * GPR_num + "
                    else:
                        ret_string = "16 * (floor(lane / 16) % 2) + "
                        ret_string += "8 * floor(lane / 32) + "
                ret_string += f"floor(GPR_bits / {data_size})"
        return ret_string

    def _print_opcode(self, encoding_name: str = "VOP3P") -> None:
        """ Prints encoding name and VOP3P opcode for an instruction.

        Args:
            encoding_name: String containing the name of the encoding format for the
                current architecture; defaults to "VOP3P" in gfx12.
        """
        super()._print_opcode(encoding_name)

    def _print_execution_statistics(self, cu_name: str = "WGP") -> None:
        """ Prints execution statistics for a matrix multiplication instruction.

        Prints the execution statistics, such as computational throughput and co-execution
        information, for a matrix multiplication instruction on a target architecture.
        The instruction and architecture are part of the class.

        Args:
            cu_name: string that contains the name of the compute unit on the target arch
                Different architectures may use e.g., CU, or WGP, or some other name.
                Defaults to "WGP" in gfx12.
        """
        super()._print_execution_statistics(cu_name)

    def _print_register_usage(self, wave_sizes: Tuple[int, ...] = (32, 64)) -> None:
        """ Prints the register count for each input and output matrix.

        Prints the number of registers used by each matrix for a matrix multiplication
        instruction on a target architecture. Some architectures support more than one
        wavefront size. In that case, print out the register usage for each wavefront size.
        This is needed because different wavefront sizes can cause different GPR usage for
        the same matrix size.

        Args:
            wave_sizes: a tuple of wavefront sizes to use to print this info
                Defaults to a tuple of integers 32 and 64 in gfx12, because gfx12
                supports both wave32 and wave64.
        """
        super()._print_register_usage(wave_sizes)

    def _print_register_to_element_eqn(self, print_block: bool = False,
                                       wave_size: Optional[int] = None,
                                       max_a_lane: Optional[int] = None,
                                       max_b_lane: Optional[int] = None) -> None:
        """ Prints equation to map register+lane to matrix element.

        Print out simple equations for mapping a register and its lane to the element in the
        matrix and block which they hold. These can be used by developers that do not want to
        set modifiers like CBSZ/ABID or BLGP to quickly unpack values from registers to
        put them into output matrix locations.

        Args:
            print_block: True if this equation should print information about blocks
            wave_size: integer that holds the wave size for the requested conversion
                function. gfx12 supports wave32 and wave64.
            max_a_lane: optional integer that holds the maximum lane that can be used in this
                instruction for A and K matrices.
            max_b_lane: optional integer that holds the maximum lane that can be used in this
                instruction for B matrices.
        """
        if wave_size is None:
            super()._print_register_to_element_eqn(print_block, 32)
            max_a_lane = None
            max_b_lane = None
            # V_SWMMAC_I32_16X16X32_IU4 only uses 32 lanes for A matrix and compression
            # indices when in wave64 mode
            # V_SWMMAC_I32_16X16X32_IU4 only uses 32 lanes for A and B matrices when in
            # wave64 mode
            if self.inst_name.upper() == 'V_SWMMAC_I32_16X16X32_IU4':
                max_a_lane = 32
            elif self.inst_name.upper() == 'V_WMMA_I32_16X16X16_IU4':
                max_a_lane = 32
                max_b_lane = 32
            super()._print_register_to_element_eqn(print_block, 64, max_a_lane, max_b_lane)
        else:
            super()._print_register_to_element_eqn(print_block)

if __name__ == '__main__':
    sys.exit(parse_and_run())
