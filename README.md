AMD Matrix Instruction Calculator
=======================================================================================================

This repository contains a tool for generating information about the matrix multiplication instructions in AMD Radeon&trade; and AMD Instinct&trade; accelerators.
This tool allows users to generate instruction-level information such as computational throughput and register usage.
It also allows users to generate mappings between matrix element and hardware registers.

This tool supports the Matrix Fused-Multiply Add (MFMA) and Sparse Matrix Fused Multiply Accumulate (SMFMAC) instructions that are the ISA-level interface to the Matrix Cores within the AMD Instinct MI100, MI200, and MI300 series processors.
It also supports the Wave Matrix Multiply Accumulate (WMMA) and Sparse Wave Matrix Multiply Accumulate (SWMMAC) instructions that are the ISA-level interface to the AI Accelerators within the AMD Radeon RDNA&trade; 3 and RDNA 4 processors.

This tool offers five options for each matrix multiplication instruction:
* Print general information about the instruction, such as its number of registers, computational throughput, and execution options (`--detail-instruction`).
* Print the register and lane for a user-chosen A\[\], B\[\], C\[\], or D\[\] matrix entry (`--get-register`)
* Print the A\[\], B\[\], C\[\], or D\[\] matrix entry for a chosen combination of register and lane (`--matrix-entry`)
* Print the register and lane combinations for an entire A\[\], B\[\], C\[\], or D\[\] matrix (`--register-layout`)
* Print the A\[\], B\[\], C\[\], or D\[\] matrix entries for all the instructions' registers and lanes (`--matrix-layout`)

The sparse matrix multiplication instructions included in the AMD CDNA3- and RDNA4-based accelerators also have a K\[\] matrix, which contains the compression information for the A\[\] matrix.
This K\[\] matrix takes the place of the C\[\] input data matrix.
This tool can also print the register and lane mappings of the K\[\] matrix using the last four options (`--get-register`, `--matrix-entry`, `--register-layout`, `--matrix-layout`), but only for instructions that are sparse.

Some matrix multiplication instructions may be modified by fields in the instructions: Control Broadcast Size (CBSZ), A-matrix Broadcaster Identifier (ABID), B-matrix Lane Group Pattern (BLGP), Operand Select (OPSEL), and Negate (NEG).
This tool allows setting these fields, which may modify the output of the other queries.

Table of Contents
-------------------------------------------------------------------------------------------------------

* [AMD Matrix Instruction Calculator](#amd-matrix-instruction-calculator)
    * [Prerequisites](#prerequisites)
    * [Calculator Usage](#calculator-usage)
        * [General-purpose Configuration Parameters](#general-purpose-configuration-parameters)
        * [Querying Matrix Multiplication Instruction Information](#querying-matrix-multiplication-instruction-information)
        * [Choosing the Matrix to Query](#choosing-the-matrix-to-query)
        * [Modifying the Queried Matrix](#modifying-the-queried-matrix)
        * [Querying Register and Lane Information for a Single Matrix Element](#querying-register-and-lane-information-for-a-single-matrix-element)
        * [Querying the Matrix Element Coordinates for a Register and Lane Combination](#querying-the-matrix-element-coordinates-for-a-register-and-lane-combination)
        * [Printing the Register and Lane Layout for an Entire Matrix](#printing-the-register-and-lane-layout-for-an-entire-matrix)
        * [Printing the Matrix Element Layout for all Registers and Lanes](#printing-the-matrix-element-layout-for-all-registers-and-lanes)
    * [Example of Querying the List of Instructions in a Chip](#example-of-querying-the-list-of-instructions-in-a-chip)
    * [Example of Querying Instruction Information](#example-of-querying-instruction-information)
    * [Example of Querying Register and Lane for a Single Matrix Element](#example-of-querying-register-and-lane-for-a-single-matrix-element)
        * [Example of Querying the Source Registers and Lanes for D Matrix Outputs](#example-of-querying-the-source-registers-and-lanes-for-d-matrix-outputs)
    * [Example of Querying Matrix Element Information for a Register and Lane](#example-of-querying-matrix-element-information-for-a-register-and-lane)
        * [Example of Querying the Source Matrix Entries for D Matrix Outputs](#example-of-querying-the-source-matrix-entries-for-d-matrix-outputs)
    * [Example of Printing the Registers and Lanes for an Entire Matrix](#example-of-printing-the-registers-and-lanes-for-an-entire-matrix)
    * [Example of Printing the Matrix Elements for all Registers and Lanes](#example-of-printing-the-matrix-elements-for-all-registers-and-lanes)
    * [Examples of Setting Modifiers in AMD CDNA™ Architectures](#examples-of-setting-modifiers-in-amd-cdna-architectures)
        * [Example of Using the CBSZ and ABID Modifiers to Change the A Matrix Layout](#example-of-using-the-cbsz-and-abid-modifiers-to-change-the-a-matrix-layout)
        * [Example of Using the BLGP Modifier to Change the B Matrix Layout](#example-of-using-the-blgp-modifier-to-change-the-b-matrix-layout)
            * [Example of Using the BLGP Modifier to Negate FP64 Input Matrices in the CDNA™ 3 Architecture](#example-of-using-the-blgp-modifier-to-negate-fp64-input-matrices-in-the-cdna-3-architecture)
        * [Example of Printing Compression Index Information for a Sparse Matrix Instruction](#example-of-printing-compression-index-information-for-a-sparse-matrix-instruction)
    * [Examples of Setting Modifiers in the AMD RDNA™ 3 and RDNA 4 Architectures](#examples-of-setting-modifiers-in-the-amd-rdna-3-and-rdna-4-architectures)
        * [Example of Using the OPSEL Modifier to Change the C and D Matrix Storage in the AMD RDNA™ 3 Architecture](#example-of-using-the-opsel-modifier-to-change-the-c-and-d-matrix-storage-in-the-amd-rdna-3-architecture)
        * [Example of Using the OPSEL Modifier to Change the Compression Index Set in the AMD RDNA™ 4 Architecture](#example-of-using-the-opsel-modifier-to-change-the-compression-index-set-in-the-amd-rdna-4-architecture)
        * [Example of Using the NEG Modifier to Negate Input Matrices](#example-of-using-the-neg-modifier-to-negate-input-matrices)
    * [Details of AMD Matrix Multiplication Instruction Encodings](#details-of-amd-matrix-multiplication-instruction-encodings)
        * [Details of the VOP3P-MAI Matrix Multiplication Instruction Encoding in the AMD CDNA™ 1 - CDNA 3 Architectures](#details-of-the-vop3p-mai-matrix-multiplication-instruction-encoding-in-the-amd-cdna-1---cdna-3-architectures)
        * [Details of the VOP3P Matrix Multiplication Instruction Encoding in the AMD RDNA™ 3 and RDNA 4 Architectures](#details-of-the-vop3p-matrix-multiplication-instruction-encoding-in-the-amd-rdna-3-and-rdna-4-architectures)
    * [Further Materials](#further-materials)
    * [Trademark Attribution](#trademark-attribution)

Prerequisites
-------------------------------------------------------------------------------------------------------
This tool requires the following:
* Python3
* The Python package `tabulate`:
    * To install this package system wide, execute: `sudo pip install tabulate`
    * To install this package for the local user, execute: `pip install tabulate --user`
* For Python3 versions below 3.8, the `typing_extensions` package:
    * To install this package system wide, execute: `sudo pip install typing_extensions`
    * To install this package for the local user, execute: `pip install typing_extensions --user`
* Installing prerequisites itself may require you to install `pip`
* Now users can use `pip install -r requirements.txt` to install dependent packages

Calculator Usage
-------------------------------------------------------------------------------------------------------
This tool offers many command line parameters to configure the desired information to print.
This section details what these parameters do and how users should configure them to achieve desired results.

### General-purpose Configuration Parameters
The following are general-purpose tool configuration parameters that allow users to choose a desired instruction on a target processor.
Command line parameters are case sensitive, but inputs for the command-line parameters are case insensitive.

* `--version` (or `-v`): Print the version number of the tool.
* `--help` (or `-h`): Print out help information for the tool.
* `--architecture {arch name}` (or `-a {arch name}`): This parameter chooses which AMD architecture to use for further calculations. This must be set for every other action in the application. Legal inputs are:
    * `CDNA`, `CDNA1`, `gfx908`, `arcturus`, or `MI100`: The AMD Instinct&trade; MI100 series of accelerators
    * `CDNA2`, `gfx90a`, `aldebaran`, `MI200`, `MI210`, `MI250`, or `MI250X`: The AMD Instinct MI200 series of accelerators, including AMD Instinct MI210, AMD Instinct MI250, and AMD Instinct MI250X
    * `CDNA3`, `gfx940`, `gfx941`, `gfx942`, `aqua_vanjaram`, `MI300`, `MI300A`, `MI300X`, or `MI325X`: The AMD Instinct MI300 series of accelerators
    * `RDNA3`, `gfx1100`, `gfx1101`, `gfx1102`, `gfx1103`, `gfx1150`, `gfx1151`, `gfx1152`, or `gfx1153`: The AMD Radeon&trade; RDNA&trade; 3 series of GPUs
    * `RDNA4`, `gfx1200` or `gfx1201`: The AMD Radeon RDNA 4 series of GPUs
* `--list-instructions` (or `-L`): This parameter will print the supported matrix multiplication instructions for the chosen architecture and exit the application.
* `--instruction {instruction mnemonic}` (or `-i {instruction mnemonic}`): This parameter chooses which instruction, from the list of legal matrix multiplication instructions in the chosen architecture, to use for the calculations in this tool.

### Querying Matrix Multiplication Instruction Information
The following option requires both the `--architecture` and `--instruction` parameters to be set.

* `--detail-instruction` (or `-d`): Print detailed information about the chosen matrix multiplication instruction, including its opcode, register usage, and computational throughput.

### Choosing the Matrix to Query
Most of the matrix multiplication instructions in AMD accelerators perform matrix multiplication of the form `D = A * B + C`, where A, B, and C are input matrices and D is an output matrix.
The remaining options for this tool allow users to query information about registers and matrix element layouts of the input and output matrices of the chosen instruction.
All the remaining options require users to set one (and only one) of the following options:

* `--A-matrix` (or `-A`): Query information about the A input matrix
* `--B-matrix` (or `-B`): Query information about the B input matrix
* `--C-matrix` (or `-C`): Query information about the C input matrix
* `--D-matrix` (or `-D`): Query information about the D output matrix
* `--compression` (or `-k`): Query information about the compression indices for the A input matrix of a sparse matrix instruction

### Modifying the Queried Matrix
Some matrix multiplication instructions in AMD accelerators have modifiers that change which input matrix entries are used by the Matrix Cores or AI Accelerators.
These modifiers allow a wider variety of matrix multiplication operations to perform matrix multiplication without spending cycles performing complex data swizzling operations.
This tool allows setting these modifiers on instructions that support them.
The modifiers will change the output of register and matrix element queries to match what the instruction would do with the modifier fields set.

* `--cbsz {#}`: Change the Control Broadcast Size field on instructions that support it. This causes some blocks of the A matrix to be broadcast to the Matrix Core in place of other blocks. This modifier configures how many blocks will receive that broadcast, the ABID modifier chooses which block to broadcast. This is only supported by some instructions and is only useful for queries of the A and K matrices. The legal values are between 0 and $`log_2(blocks)`$, inclusive.
* `--abid {#}`: Change the A-matrix Broadcast Identifier field on instructions that support it. In some architectures, this chooses the block of the dense A matrix to broadcast to the Matrix Core, in conjunction with the CBSZ modifier. This is only supported by some instructions and is only useful for queries of the A and K matrices. For these instructions, the legal values are between 0 and $`2^{CBSZ}-1`$, inclusive. In other architectures, this is used to pick between compression index sets.
* `--blgp {#}`: Change the B-matrix Lane Group Pattern field on instructions that support it. This causes some input lanes of the B matrix to be rotated or broadcast to replace the matrix multiplication inputs that would have originally come from a different lane. This is only supported by some instructions. For instructions where it is supported, valid values are between 0-7, inclusive.
* `--opsel {#}`: Change the Operand Select field on instructions that support it. In some architectures, this causes some 16-bit inputs and outputs to be stored in either the top or bottom half of the 32-bit registers, rather than packing two values into the register. For those instructions, this field chooses whether to read and write from the top of bottom half of those registers. In other architectures, this is used to pick between compression index sets.
* `--neg {#}`: Change the NEG (Negate) field on instructions that support it. The bits in this 3-bit field indicate whether to negate the values read from the A\[\], B\[\], or C\[\] matrices, respectively. For instructions where it is supported, valid values are between 0-7, inclusive.
* `--neg_hi {#}`: Change the NEG\_HI (Negate Hi Bits) field on instructions that support it. The bits in this 3-bit field indicate whether to negate the values read from the A\[\] or B\[\] matrices, or to take the absolute value from C\[\] matrices. For instructions where it is supported, valid values are between 0-7, inclusive.

### Querying Register and Lane Information for a Single Matrix Element
The `--get-register` parameter will cause the tool to print the vector register number, the lane within that vector register, and the bits within each vector register lane that holds the matrix element at a single coordinate within the desired matrix.

* `--get-register` (or `-g`): This parameter can be further configured using some of the following options. The options used depend on both the instruction and matrix being queried.
    * `--I-coordinate {#}` (or `-I {#}`): Chooses the i coordinate, the row for A, C, D, or K matrices. Ignored for B matrix. Default value when this option is not used is 0.
    * `--J-coordinate {#}` (or `-J {#}`): Chooses the j coordinate, the column for B, C, or D matrices. Ignored for A and K matrices. Default value when this option is not used is 0.
    * `--K-coordinate {#}` (or `-K {#}`): Chooses the k coordinate, the column for A and K matrices or row for B matrix. Ignored for C and D matrices. Default value when this option is not used is 0.
    * `--block` (or `-b {#}`): Chooses the block to query if the input instruction is a multi-block instruction. Default value when this option is not used is 0.
    * `--output-calculation` (or `-o`): This option is only valid with the `--D-matrix` parameter. With this option set, the tool will print the registers and lanes for both the D output matrix and all of the A, B, and C matrix entries that went into calculating it.

### Querying the Matrix Element Coordinates for a Register and Lane Combination
The `--matrix-entry` parameter will cause the tool to print the row, column, and block information for the matrix element (or elements) that are contained in the lane of a vector register.

* `--matrix-entry` (or `-m`): This parameter can be further configured using the following options.
    * `--register {#}` (or `-r {#}`): Chooses the register number to query. The number must be a whole number (&ge; 0). The maximum register value depends on the chosen instruction. Default value when this option is not used is 0.
    * `--lane {#}` (or `-l {#}`): Chooses the vector lane within the register to query. This number must be between 0 and the size of the wave minus 1, inclusive. Default value when this option is not used is 0.
    * `--output-calculation` (or `-o`): This option is only valid with the `--D-matrix` parameter. With this option set, the tool will print the row, column, and block information for both the D output matrix and all of the A, B, and C matrix entries that went into calculating it.

### Printing the Register and Lane Layout for an Entire Matrix
The `--register-layout` parameter will print a tabular layout of the chosen matrix.
For each element in the matrix, print out the vector register number that will be used and the lane within that register.
If a single 32b register holds multiple matrix elements, it will differentiate the bits within the register that holds the element.

* `--register-layout` (or `-R`): Print the registers and lanes for each element of the chosen matrix. This will print in an ASCII tabular format by default.
    * `--csv` (or `-c`): If this option is also passed, the tool will print the same information, but in a CSV (comma-separated value) format instead of an ASCII table.
    * `--markdown`: If this option is also passed, the tool will print the same information, but in a Markdown format instead of an ASCII table.
    * `--asciidoc`: If this option is also passed, the tool will print the same information, but in a AsciiDoc format instead of an ASCII table.
    * `--transpose`: If this option is also passed, the tool will print the register layout in a transposed format, swapping rows and columns.

### Printing the Matrix Element Layout for all Registers and Lanes
The `--matrix-layout` parameter will print a tabular layout of the chosen matrix.
For each register used by the matrix, and the lanes within those registers, print out the matrix element or elements that are stored in that register + lane combination.
If a single 32b register holds multiple matrix elements, the tool will print out the table split into bit ranges.

* `--matrix-layout` (or `-M`): Print the matrix elements for each register/lane combination. This will print in an ASCII tabular format by default.
    * `--csv` (or `-c`): If this option is also passed, the tool will print the same information, but in a CSV (comma-separated value) format instead of an ASCII table.
    * `--markdown`: If this option is also passed, the tool will print the same information, but in a Markdown format instead of an ASCII table.
    * `--asciidoc`: If this option is also passed, the tool will print the same information, but in a AsciiDoc format instead of an ASCII table.
    * `--transpose`: If this option is also passed, the tool will print the register layout in a transposed format, swapping rows and columns.

Example of Querying the List of Instructions in a Chip
-------------------------------------------------------------------------------------------------------
The `--list-instructions` flag allows users to query which matrix multiplication instructions are available in a chosen architecture.
These are the instructions that are supported by the `--instruction` flag.

The following is an example usage, where we query the list of instructions in the CDNA&trade; 2 architecture:
```
$ ./matrix_calculator.py --architecture cdna2 --list-instructions
Available instructions in the CDNA2 architecture:
    v_mfma_f32_32x32x1f32
    v_mfma_f32_16x16x1f32
    v_mfma_f32_4x4x1f32
    v_mfma_f32_32x32x2f32
    v_mfma_f32_16x16x4f32
    v_mfma_f32_32x32x4f16
    v_mfma_f32_16x16x4f16
    v_mfma_f32_4x4x4f16
    v_mfma_f32_32x32x8f16
    v_mfma_f32_16x16x16f16
    v_mfma_i32_32x32x4i8
    v_mfma_i32_16x16x4i8
    v_mfma_i32_4x4x4i8
    v_mfma_i32_32x32x8i8
    v_mfma_i32_16x16x16i8
    v_mfma_f32_32x32x4bf16_1k
    v_mfma_f32_16x16x4bf16_1k
    v_mfma_f32_4x4x4bf16_1k
    v_mfma_f32_32x32x8bf16_1k
    v_mfma_f32_16x16x16bf16_1k
    v_mfma_f32_32x32x2bf16
    v_mfma_f32_16x16x2bf16
    v_mfma_f32_4x4x2bf16
    v_mfma_f32_32x32x4bf16
    v_mfma_f32_16x16x8bf16
    v_mfma_f64_16x16x4f64
    v_mfma_f64_4x4x4f64
```

Example of Querying Instruction Information
-------------------------------------------------------------------------------------------------------
The `--detail-instruction` flag allows users to query information about the desired matrix multiplication instruction in a chosen architecture.
This information includes:
* The instruction encoding class and opcode
* The matrix dimensions and number of blocks used by the instruction
* The computational throughput and co-execution capabilities
* The number of general-purpose registers (GPRs) used by the instruction's matrices
* A description of the data types this instruction takes as input and produces as output
* Whether the matrices can be stored in architected vector registers (ArchVGPRs) and/or the accumulation vector registers (AccVGPRs)
* Support for matrix modifiers such as CBSZ, ABID, BLGP, OPSEL, and NEG
* Formulae for mapping between matrix entries and the registers and lanes that hold them, when no matrix modifier fields are set

This flag requires the `--architecture` and `--instruction` flags to be set to pick the chip and instruction to detail.

The following is an example usage, where we are querying information about the V\_MFMA\_F32\_4X4X1F32 instruction in the CDNA&trade; 2 architecture:
```
$ ./matrix_calculator.py --architecture cdna2 --instruction v_mfma_f32_4x4x1f32 --detail-instruction
Architecture: CDNA2
Instruction: V_MFMA_F32_4X4X1F32
    Encoding: VOP3P-MAI
    VOP3P Opcode: 0x42
    VOP3P-MAI Opcode: 0x2
    Matrix Dimensions:
        M: 4
        N: 4
        K: 1
        blocks: 16
    Execution statistics:
        FLOPs: 512
        Execution cycles: 8
        FLOPs/CU/cycle: 256
        Can co-execute with VALU: True
        VALU co-execution cycles possible: 4
    Register usage:
        GPRs required for A: 1
        GPRs required for B: 1
        GPRs required for C: 4
        GPRs required for D: 4
        GPR alignment requirement: 8 bytes
    VOP3P-MAI register encoding:
        A matrix source field: Src0
        B matrix source field: Src1
        C matrix source field: Src2
        D matrix source field: Vdst
    Register data types:
        Src0: FP32 (IEEE binary32 floating point)
        Src1: FP32 (IEEE binary32 floating point)
        Src2: FP32 (IEEE binary32 floating point)
        Vdst: FP32 (IEEE binary32 floating point)
    Register capabilities:
        A matrix can use ArchVGPRs: True
        A matrix can use AccVGPRs: True
        B matrix can use ArchVGPRs: True
        B matrix can use AccVGPRs: True
        C and D matrix can use ArchVGPRs: True
        C and D matrix can use AccVGPRs: True
    Register modifiers:
        Sparse A matrix: False
        CBSZ and ABID bits supported: True
        BLGP bits supported: True
    Matrix element to register mapping with no modifiers:
        A[i][k].block GPR: 0
        A[i][k].block Lane: 4 * block + i
        B[k][j].block GPR: 0
        B[k][j].block Lane: 4 * block + j
        C or D[i][j].block GPR: i
        C or D[i][j].block Lane: 4 * block + j
    Register to matrix element mapping with no modifiers:
        A i: (lane % 4)
        A k: 0
        A block: floor(lane / 4)
        B j: (lane % 4)
        B k: 0
        B block: floor(lane / 4)
        C or D i: (GPR_num % 4)
        C or D j: (lane % 4)
        C or D block: floor(lane / 4)
```

Example of Querying Register and Lane for a Single Matrix Element
-------------------------------------------------------------------------------------------------------
The `--get-register` flag allows users to query an element within a matrix for a desired instruction.
The tool will then translate this matrix entry into the vector register and lane number within that register that contain the value.

This flag requires the `--architecture` and `--instruction` flags to be set to pick the chip and instruction to query.
One of `--A-matrix`, `--B-matrix`, `--C-matrix`, `--D-matrix`, or `--compression` must be set to pick which matrix to query.

There are four optional parameters to this function, to pick the matrix element: `--I-coordinate`, `--J-coordinate`, `--K-coordinate`, and `--block`.
The `j` coordinate is ignored for the A matrix.
The `i` coordinate is ignored for the B matrix.
The `k` coordinate is ignored for the C and D matrices.
Any parameter that is not explicitly set will default to 0.

The output format is printed in the form: `Vx{y}.z`, where:
* `x` is the vector register offset from the value put into the matrix multiplication instruction's register field.
Registers holding 64-bit values are printed as a register pair, `[x+1:x]`.
* `y` is the lane within the vector register
* `.z` is an optional identifier for sub-register if the matrix value is less than 32 bits
    * `.[15:0]` is the least significant 16b of a 32b register
    * `.[31:16]` is the most significant 16b of a 32b register
    * `.[7:0]` is the least significant 8b of a 32b register
    * `.[15:8]` is the second least significant 8b of a 32b register
    * `.[23:16]` is the second most significant 8b of a 32b register
    * `.[31:24]` is the most significant 8b of a 32b register
    * `.[hi_bit:lo_bit]` is also used to show the bit-range for values smaller than 8b

The following is an example that requests the register which holds the 16-bit value in the 4th block of matrix A at coordinate `A[1][2]` of the V\_MFMA\_F32\_4X4X4F16 in the CDNA&trade; 2 architecture:
```
$ ./matrix_calculator.py --architecture cdna2 --instruction v_mfma_f32_4x4x4f16 --get-register --I-coordinate 1 --K-coordinate 2 --block 4 --A-matrix
Architecture: CDNA2
Instruction: V_MFMA_F32_4X4X4F16
A[1][2].B4 = v1{17}.[15:0]
```

This output shows that this matrix entry is contained in the low 16b of the 17th lane of register 1.

### Example of Querying the Source Registers and Lanes for D Matrix Outputs
The `--get-register` flag normally displays the matrix element, and which register it maps to, as described above.
The `--output-calculation` option allows printing further information for the D\[\] output matrix.
When this option is set, the tool will also print out the registers and lanes for the A\[\], B\[\], and C\[\] input matrix entries that were used to calculate this output matrix entry.

This can be useful when debugging because it allows developers to quickly ascertain which input values were used to calculate the answer seen in the D\[\] matrix.

Because there are now up to four registers printed to the screen (one for each of A\[\], B\[\], C\[\], and D\[\]), these registers are differentiated by prepending their opcode's register field.
For instance, the base register for A\[\] is held in the Src0 field, so all register values for A\[\] are prepended with `Src0_`.
The register name to matrix mapping can be found by querying the instruction using `--detail-instruction`.

This option is only valid when the `--D-matrix` parameter is used.

The following is an example that requests the output register and input registers used to calculate the value at coordinate `D[3][2]` of the 1st block for the instruction V\_MFMA\_F32\_4X4X4F16 in the CDNA&trade; 2 architecture:
```
$ ./matrix_calculator.py --architecture cdna2 --instruction v_mfma_f32_4x4x4f16 --get-register --I-coordinate 3 --J-coordinate 2 --block 1 --D-matrix --output-calculation
Architecture: CDNA2
Instruction: V_MFMA_F32_4X4X4F16
D[3][2].B1 = Vdst_v3{6} = Src0_v0{7}.[15:0]*Src1_v0{6}.[15:0] + Src0_v0{7}.[31:16]*Src1_v0{6}.[31:16] + Src0_v1{7}.[15:0]*Src1_v1{6}.[15:0] + Src0_v1{7}.[31:16]*Src1_v1{6}.[31:16] + Src2_v3{6}
```

This output shows the four multiplications and five additions that are used to calculate this output matrix entry, as well as which registers are used as input to these calculations.

Example of Querying Matrix Element Information for a Register and Lane
-------------------------------------------------------------------------------------------------------
The `--matrix-entry` flag allows users to query a register and lane combination for a desired instruction.
The tool will then translate this vector-lane pair into the matrix entry (or entries) that it holds.

This flag requires the `--architecture` and `--instruction` flags to be set to pick the chip and instruction to query.
One of `--A-matrix`, `--B-matrix`, `--C-matrix`, `--D-matrix`, or `--compression` must be set to pick which matrix to query.

There are two optional parameters to this function, to pick the vector-register pair: `--register` and `--lane`.
Any parameter that is not explicitly set will default to 0.

The output format is printed in the form: `Matrix[row][col].block`, where:
* `Matrix` is the name of the chosen matrix: A, B, C, or D
* `row` is the row of the matrix, such as the `i` input to the A matrix or the `k` input to the B matrix
* `col` is the column of the matrix, such as the `k` input to the A matrix or the `j` input to the B matrix
* `.block` is the block within the matrix, since some matrix multiplication instructions work on multiple separate blocks of $`NxMxK`$ matrix multiplications

The following is an example that requests the matrix entries which are contained in the 17th lane of register 1 of matrix A for the instruction V\_MFMA\_F32\_4X4X4F16 in the CDNA&trade; 2 architecture:
```
$ ./matrix_calculator.py --architecture cdna2 --instruction v_mfma_f32_4x4x4f16 --matrix-entry --register 1 --lane 17 --A-matrix
Architecture: CDNA2
Instruction: V_MFMA_F32_4X4X4F16
v1{17}.[15:0] = A[1][2].B4
v1{17}.[31:16] = A[1][3].B4
```

This output shows that this register contains two matrix entries, since the 32b register can hold two 16b FP16 inputs.
It prints both, and shows that the low 16b of this register holds the value at the 2nd column and 1st row, from the 4th block of 4x4 matrices in A.
The high 16b of this register holds the value at the 3rd column from the same row and block.

### Example of Querying the Source Matrix Entries for D Matrix Outputs
The `--matrix-entry` flag normally displays the matrix element associated with a single register and lane pair.
The `--output-calculation` option allows printing further information for the D\[\] output matrix.
When this option is set, the tool will also print out the A\[\], B\[\], and C\[\] input matrix entries that were used to calculate this output matrix entry.

This can be useful when debugging incorrect answers because it allows developers to quickly ascertain which input values were used to calculate the answer seen in the D\[\] matrix.

This option is only valid when the `--D-matrix` parameter is used.

The following is an example that requests the output matrix entry and input matrix entries used to calculate the value at coordinate the 33rd lane of register 2 for matrix D for the instruction V\_MFMA\_F32\_4X4X4F16 in the CDNA&trade; 2 architecture:
```
$ ./matrix_calculator.py --architecture cdna2 --instruction v_mfma_f32_4x4x4f16 --matrix-entry --register 2 --lane 33 --D-matrix --output-calculation
Architecture: CDNA2
Instruction: V_MFMA_F32_4X4X4F16
v2{33} = D[2][1].B8 = A[2][0].B8*B[0][1].B8 + A[2][1].B8*B[1][1].B8 + A[2][2].B8*B[2][1].B8 + A[2][3].B8*B[3][1].B8 + C[2][1].B8
```

This output shows the four multiplications and five additions that are used to calculate this output matrix entry, including what their `i`, `j`, and `k` coordinates are, as well as their block number.

Example of Printing the Registers and Lanes for an Entire Matrix
-------------------------------------------------------------------------------------------------------
The `--register-layout` flag allows users to query the register and lane locations for all entries in a target matrix.
The tool will translate each matrix coordinate into a register description, format the matrix as a table, and print it to the screen.

This flag requires the `--architecture` and `--instruction` flags to be set to pick the chip and instruction to query.
One of `--A-matrix`, `--B-matrix`, `--C-matrix`, `--D-matrix`, or `--compression` must be set to pick which matrix to query.

There are three optional parameters to this function which will change the format of the output: `--asciidoc`, `--csv`, and `--markdown`.
Setting `--asciidoc` causes the tool to print the output in an AsciiDoc format, rather than a visual tabular format.
Setting `--csv` causes the tool to print the output in a comma-separated-value format, and setting `--markdown` causes the tool to print the output in a Markdown format.
At most, one of these three output-modifying parameters can be used at a time.
In all cases, users may want to pipe the output into another tool such as `less -S`, because the outputs can become quite wide and difficult to read on normal screens or with line-wrap.

Finally, the optional `--transpose` parameter will swap output table's rows and column.
This will not cause any change to the matrix itself or the registers that hold the matrix entries; the `--transpose` parameter only changes the layout of the data output by this tool.

The following is an example that requests the layout of the matrix D for the V\_MFMA\_F64\_4X4X4F64 instruction in CDNA&trade; 2 architecture.
```
$ ./matrix_calculator.py --architecture cdna2 --instruction v_mfma_f64_4x4x4f64 --register-layout --D-matrix
Architecture: CDNA2
Instruction: V_MFMA_F64_4X4X4F64
Block 0
+-----------+------------+------------+------------+------------+
|   D[M][N] | 0          | 1          | 2          | 3          |
+===========+============+============+============+============+
|         0 | v[1:0]{0}  | v[1:0]{1}  | v[1:0]{2}  | v[1:0]{3}  |
+-----------+------------+------------+------------+------------+
|         1 | v[1:0]{16} | v[1:0]{17} | v[1:0]{18} | v[1:0]{19} |
+-----------+------------+------------+------------+------------+
|         2 | v[1:0]{32} | v[1:0]{33} | v[1:0]{34} | v[1:0]{35} |
+-----------+------------+------------+------------+------------+
|         3 | v[1:0]{48} | v[1:0]{49} | v[1:0]{50} | v[1:0]{51} |
+-----------+------------+------------+------------+------------+
Block 1
+-----------+------------+------------+------------+------------+
|   D[M][N] | 0          | 1          | 2          | 3          |
+===========+============+============+============+============+
|         0 | v[1:0]{4}  | v[1:0]{5}  | v[1:0]{6}  | v[1:0]{7}  |
+-----------+------------+------------+------------+------------+
|         1 | v[1:0]{20} | v[1:0]{21} | v[1:0]{22} | v[1:0]{23} |
+-----------+------------+------------+------------+------------+
|         2 | v[1:0]{36} | v[1:0]{37} | v[1:0]{38} | v[1:0]{39} |
+-----------+------------+------------+------------+------------+
|         3 | v[1:0]{52} | v[1:0]{53} | v[1:0]{54} | v[1:0]{55} |
+-----------+------------+------------+------------+------------+
Block 2
+-----------+------------+------------+------------+------------+
|   D[M][N] | 0          | 1          | 2          | 3          |
+===========+============+============+============+============+
|         0 | v[1:0]{8}  | v[1:0]{9}  | v[1:0]{10} | v[1:0]{11} |
+-----------+------------+------------+------------+------------+
|         1 | v[1:0]{24} | v[1:0]{25} | v[1:0]{26} | v[1:0]{27} |
+-----------+------------+------------+------------+------------+
|         2 | v[1:0]{40} | v[1:0]{41} | v[1:0]{42} | v[1:0]{43} |
+-----------+------------+------------+------------+------------+
|         3 | v[1:0]{56} | v[1:0]{57} | v[1:0]{58} | v[1:0]{59} |
+-----------+------------+------------+------------+------------+
Block 3
+-----------+------------+------------+------------+------------+
|   D[M][N] | 0          | 1          | 2          | 3          |
+===========+============+============+============+============+
|         0 | v[1:0]{12} | v[1:0]{13} | v[1:0]{14} | v[1:0]{15} |
+-----------+------------+------------+------------+------------+
|         1 | v[1:0]{28} | v[1:0]{29} | v[1:0]{30} | v[1:0]{31} |
+-----------+------------+------------+------------+------------+
|         2 | v[1:0]{44} | v[1:0]{45} | v[1:0]{46} | v[1:0]{47} |
+-----------+------------+------------+------------+------------+
|         3 | v[1:0]{60} | v[1:0]{61} | v[1:0]{62} | v[1:0]{63} |
+-----------+------------+------------+------------+------------+
```
This instruction has four output blocks, and they are printed one after another.
Each block is a 4x4 matrix; each entry is printed in the same `Vx{y}.z` format described in [the previous section](#example-of-querying-register-and-lane-for-a-single-matrix-element).

Example of Printing the Matrix Elements for all Registers and Lanes
-------------------------------------------------------------------------------------------------------
The `--matrix-layout` flag allows users to query the matrix entry locations for all the registers used by a matrix multiplication instruction for one of its input matrices.
The tool will translate the lanes in each matrix into the matrix entry at that coordinate, format this output as a table, and print it to the screen.

This flag requires the `--architecture` and `--instruction` flags to be set to pick the chip and instruction to query.
One of `--A-matrix`, `--B-matrix`, `--C-matrix`, `--D-matrix`, or `--compression` must be set to pick which matrix to query.

There are three optional parameters to this function which will change the format of the output: `--asciidoc`, `--csv`, and `--markdown`.
Setting `--asciidoc` causes the tool to print the output in an AsciiDoc format, rather than a visual tabular format.
Setting `--csv` causes the tool to print the output in a comma-separated-value format, and setting `--markdown` causes the tool to print the output in a Markdown format.
At most, one of these three output-modifying parameters can be used at a time.
In all cases, users may want to pipe the output into another tool such as `less -S`, because the outputs can become quite wide and difficult to read on normal screens or with line-wrap.

Finally, the optional `--transpose` parameter will swap output table's rows and column.
This will not cause any change to the matrix itself or the registers that hold the matrix entries; the `--transpose` parameter only changes the layout of the data output by this tool.

The following is an example that requests entries for all registers used by the V\_MFMA\_F64\_4X4X4F64 instruction in the CDNA&trade; 2 architecture.
```
./matrix_calculator.py --architecture cdna2 --instruction v_mfma_f64_4x4x4f64 --matrix-layout --D-matrix
Architecture: CDNA2
Instruction: V_MFMA_F64_4X4X4F64
+--------+------------+
|   lane | v[1:0]     |
+========+============+
|      0 | D[0][0].B0 |
+--------+------------+
|      1 | D[0][1].B0 |
+--------+------------+
|      2 | D[0][2].B0 |
+--------+------------+
|      3 | D[0][3].B0 |
+--------+------------+
|      4 | D[0][0].B1 |
+--------+------------+
|      5 | D[0][1].B1 |
+--------+------------+
|      6 | D[0][2].B1 |
+--------+------------+
|      7 | D[0][3].B1 |
+--------+------------+
|      8 | D[0][0].B2 |
+--------+------------+
|      9 | D[0][1].B2 |
+--------+------------+
|     10 | D[0][2].B2 |
+--------+------------+
|     11 | D[0][3].B2 |
+--------+------------+
|     12 | D[0][0].B3 |
+--------+------------+
|     13 | D[0][1].B3 |
+--------+------------+
|     14 | D[0][2].B3 |
+--------+------------+
|     15 | D[0][3].B3 |
+--------+------------+
|     16 | D[1][0].B0 |
+--------+------------+
|     17 | D[1][1].B0 |
+--------+------------+
|     18 | D[1][2].B0 |
+--------+------------+
|     19 | D[1][3].B0 |
+--------+------------+
|     20 | D[1][0].B1 |
+--------+------------+
|     21 | D[1][1].B1 |
+--------+------------+
|     22 | D[1][2].B1 |
+--------+------------+
|     23 | D[1][3].B1 |
+--------+------------+
|     24 | D[1][0].B2 |
+--------+------------+
|     25 | D[1][1].B2 |
+--------+------------+
|     26 | D[1][2].B2 |
+--------+------------+
|     27 | D[1][3].B2 |
+--------+------------+
|     28 | D[1][0].B3 |
+--------+------------+
|     29 | D[1][1].B3 |
+--------+------------+
|     30 | D[1][2].B3 |
+--------+------------+
|     31 | D[1][3].B3 |
+--------+------------+
|     32 | D[2][0].B0 |
+--------+------------+
|     33 | D[2][1].B0 |
+--------+------------+
|     34 | D[2][2].B0 |
+--------+------------+
|     35 | D[2][3].B0 |
+--------+------------+
|     36 | D[2][0].B1 |
+--------+------------+
|     37 | D[2][1].B1 |
+--------+------------+
|     38 | D[2][2].B1 |
+--------+------------+
|     39 | D[2][3].B1 |
+--------+------------+
|     40 | D[2][0].B2 |
+--------+------------+
|     41 | D[2][1].B2 |
+--------+------------+
|     42 | D[2][2].B2 |
+--------+------------+
|     43 | D[2][3].B2 |
+--------+------------+
|     44 | D[2][0].B3 |
+--------+------------+
|     45 | D[2][1].B3 |
+--------+------------+
|     46 | D[2][2].B3 |
+--------+------------+
|     47 | D[2][3].B3 |
+--------+------------+
|     48 | D[3][0].B0 |
+--------+------------+
|     49 | D[3][1].B0 |
+--------+------------+
|     50 | D[3][2].B0 |
+--------+------------+
|     51 | D[3][3].B0 |
+--------+------------+
|     52 | D[3][0].B1 |
+--------+------------+
|     53 | D[3][1].B1 |
+--------+------------+
|     54 | D[3][2].B1 |
+--------+------------+
|     55 | D[3][3].B1 |
+--------+------------+
|     56 | D[3][0].B2 |
+--------+------------+
|     57 | D[3][1].B2 |
+--------+------------+
|     58 | D[3][2].B2 |
+--------+------------+
|     59 | D[3][3].B2 |
+--------+------------+
|     60 | D[3][0].B3 |
+--------+------------+
|     61 | D[3][1].B3 |
+--------+------------+
|     62 | D[3][2].B3 |
+--------+------------+
|     63 | D[3][3].B3 |
+--------+------------+
```

This instruction only requires a single vector register-pair (because it has a 64b output), so it only has one column of 64 entries: one entry per lane.
Each entry is printed in the same `Matrix[row][col].block` format described in [the previous section](#example-of-querying-matrix-element-information-for-a-register-and-lane).

Examples of Setting Modifiers in AMD CDNA&trade; Architectures
-------------------------------------------------------------------------------------------------------
AMD Instinct&trade; accelerators that use CDNA architectures allow setting modifier fields that change how the matrix multiplication instructions access data out of registers.
Examples of using these fields, `CBSZ`, `ABID`, and `BLGP`, are shown below.
The CDNA 3 architecture also allows sparse matrix multiplication, which has a separate compression index matrix.
An example of querying information about this matrix is also shown below.

### Example of Using the CBSZ and ABID Modifiers to Change the A Matrix Layout
Some matrix multiplication instructions in AMD Instinct&trade; accelerators using the CDNA&trade; 1, CDNA 2, and CDNA 3 architectures have encoding modifiers that allow static changes of which input matrix values are fed into the Matrix Cores.
These can help the performance of matrix multiplication applications by allowing software to quickly perform multiple matrix multiplications without spending time rearranging data in the registers.

Two of these input modifiers (stored in bits 14:11 and 10:8 of the VOP3P-MAI instruction encoding, respectively) are Control Broadcast Size (CBSZ) and A-matrix Broadcast Identifier (ABID).
For instructions that support these modifiers, they allow a constrained set of broadcast operations between multiple blocks of values in the A\[\] matrix.

Instructions that support these modifiers have multiple (2, 4, 8, or 16) input blocks for the A\[\] matrix.
The CBSZ and ABID modifiers are paired to allow one input block to be broadcast to the Matrix Cores that would normally read a different block.

Setting CBSZ tells the instruction to broadcast one chosen block to the input of $`2^{CBSZ}`$ other blocks.
For example, an instruction that has 16 blocks of input data can set the CBSZ register to 0, 1, 2, 3, or 4.
The default value of `CBSZ=0` yields the traditional matrix multiplication instruction.
Setting the value to `CBSZ=4` for the 16-block instruction would cause the value from one chosen block to become the inputs for all 16 blocks' matrix multiplication operations.

The value in ABID picks which of these 16 blocks will be broadcast.
By way of example, setting `CBSZ=4` and `ABID=13` would cause the matrix multiplication instruction to use the values in block 13 of A\[\] for all 16 blocks' worth of matrix multiplications.

As another example, for a 16-block matrix, setting `CBSZ=1` would cause the matrix multiplication instruction to broadcast a single block to 2 blocks' worth of inputs.
Setting `CBSZ=1` and `ABID=1` would cause block 1 to be broadcast to the math units in place of both block 0 and block 1.
Block 3 would be used in place of blocks 2 and 3.
Block 5 would be used in place of blocks 4 and 5, etc.

As one final example, for a 16-block matrix, setting `CBSZ=2` would cause a single block to be broadcast as 4 other blocks' inputs.
Setting `CBSZ=2` and `ABID=1` would cause block 1 to be broadcast to the math units in place of blocks 0-3.
Block 5 would be used in place of blocks 4-7.
Block 9 would be used in place of blocks 8-11, and block 13 would be used in place of blocks 12-15.

CBSZ cannot be greater than $`log_2(blocks)`$, and ABID cannot be greater than $`2^{CBSZ}-1`$.
The CBSZ and ABID modifiers are only supported by a subset of instructions, which are indicated in the `--detail-instruction` listing.

The `--cbsz {#}` and `--abid {#}` flags for this tool allow users to observe the effects of these settings on the `--get-register`, `--matrix-entry`, `--register-layout`, and `--matrix-layout` outputs.

The following is an example of the register layout for the matrix A of the V\_MFMA\_F32\_16X16X2BF16 instruction in the CDNA&trade; 2 architecture without either of these modifiers set:
```
$ ./matrix_calculator.py --architecture cdna2 --instruction v_mfma_f32_16x16x2bf16 --register-layout --A-matrix
Architecture: CDNA2
Instruction: V_MFMA_F32_16X16X2BF16
Block 0
+-----------+---------------+----------------+
|   A[M][K] | 0             | 1              |
+===========+===============+================+
|         0 | v0{0}.[15:0]  | v0{0}.[31:16]  |
+-----------+---------------+----------------+
|         1 | v0{1}.[15:0]  | v0{1}.[31:16]  |
+-----------+---------------+----------------+
|         2 | v0{2}.[15:0]  | v0{2}.[31:16]  |
+-----------+---------------+----------------+
|         3 | v0{3}.[15:0]  | v0{3}.[31:16]  |
+-----------+---------------+----------------+
|         4 | v0{4}.[15:0]  | v0{4}.[31:16]  |
+-----------+---------------+----------------+
|         5 | v0{5}.[15:0]  | v0{5}.[31:16]  |
+-----------+---------------+----------------+
|         6 | v0{6}.[15:0]  | v0{6}.[31:16]  |
+-----------+---------------+----------------+
|         7 | v0{7}.[15:0]  | v0{7}.[31:16]  |
+-----------+---------------+----------------+
|         8 | v0{8}.[15:0]  | v0{8}.[31:16]  |
+-----------+---------------+----------------+
|         9 | v0{9}.[15:0]  | v0{9}.[31:16]  |
+-----------+---------------+----------------+
|        10 | v0{10}.[15:0] | v0{10}.[31:16] |
+-----------+---------------+----------------+
|        11 | v0{11}.[15:0] | v0{11}.[31:16] |
+-----------+---------------+----------------+
|        12 | v0{12}.[15:0] | v0{12}.[31:16] |
+-----------+---------------+----------------+
|        13 | v0{13}.[15:0] | v0{13}.[31:16] |
+-----------+---------------+----------------+
|        14 | v0{14}.[15:0] | v0{14}.[31:16] |
+-----------+---------------+----------------+
|        15 | v0{15}.[15:0] | v0{15}.[31:16] |
+-----------+---------------+----------------+
Block 1
+-----------+---------------+----------------+
|   A[M][K] | 0             | 1              |
+===========+===============+================+
|         0 | v0{16}.[15:0] | v0{16}.[31:16] |
+-----------+---------------+----------------+
|         1 | v0{17}.[15:0] | v0{17}.[31:16] |
+-----------+---------------+----------------+
|         2 | v0{18}.[15:0] | v0{18}.[31:16] |
+-----------+---------------+----------------+
|         3 | v0{19}.[15:0] | v0{19}.[31:16] |
+-----------+---------------+----------------+
|         4 | v0{20}.[15:0] | v0{20}.[31:16] |
+-----------+---------------+----------------+
|         5 | v0{21}.[15:0] | v0{21}.[31:16] |
+-----------+---------------+----------------+
|         6 | v0{22}.[15:0] | v0{22}.[31:16] |
+-----------+---------------+----------------+
|         7 | v0{23}.[15:0] | v0{23}.[31:16] |
+-----------+---------------+----------------+
|         8 | v0{24}.[15:0] | v0{24}.[31:16] |
+-----------+---------------+----------------+
|         9 | v0{25}.[15:0] | v0{25}.[31:16] |
+-----------+---------------+----------------+
|        10 | v0{26}.[15:0] | v0{26}.[31:16] |
+-----------+---------------+----------------+
|        11 | v0{27}.[15:0] | v0{27}.[31:16] |
+-----------+---------------+----------------+
|        12 | v0{28}.[15:0] | v0{28}.[31:16] |
+-----------+---------------+----------------+
|        13 | v0{29}.[15:0] | v0{29}.[31:16] |
+-----------+---------------+----------------+
|        14 | v0{30}.[15:0] | v0{30}.[31:16] |
+-----------+---------------+----------------+
|        15 | v0{31}.[15:0] | v0{31}.[31:16] |
+-----------+---------------+----------------+
Block 2
+-----------+---------------+----------------+
|   A[M][K] | 0             | 1              |
+===========+===============+================+
|         0 | v0{32}.[15:0] | v0{32}.[31:16] |
+-----------+---------------+----------------+
|         1 | v0{33}.[15:0] | v0{33}.[31:16] |
+-----------+---------------+----------------+
|         2 | v0{34}.[15:0] | v0{34}.[31:16] |
+-----------+---------------+----------------+
|         3 | v0{35}.[15:0] | v0{35}.[31:16] |
+-----------+---------------+----------------+
|         4 | v0{36}.[15:0] | v0{36}.[31:16] |
+-----------+---------------+----------------+
|         5 | v0{37}.[15:0] | v0{37}.[31:16] |
+-----------+---------------+----------------+
|         6 | v0{38}.[15:0] | v0{38}.[31:16] |
+-----------+---------------+----------------+
|         7 | v0{39}.[15:0] | v0{39}.[31:16] |
+-----------+---------------+----------------+
|         8 | v0{40}.[15:0] | v0{40}.[31:16] |
+-----------+---------------+----------------+
|         9 | v0{41}.[15:0] | v0{41}.[31:16] |
+-----------+---------------+----------------+
|        10 | v0{42}.[15:0] | v0{42}.[31:16] |
+-----------+---------------+----------------+
|        11 | v0{43}.[15:0] | v0{43}.[31:16] |
+-----------+---------------+----------------+
|        12 | v0{44}.[15:0] | v0{44}.[31:16] |
+-----------+---------------+----------------+
|        13 | v0{45}.[15:0] | v0{45}.[31:16] |
+-----------+---------------+----------------+
|        14 | v0{46}.[15:0] | v0{46}.[31:16] |
+-----------+---------------+----------------+
|        15 | v0{47}.[15:0] | v0{47}.[31:16] |
+-----------+---------------+----------------+
Block 3
+-----------+---------------+----------------+
|   A[M][K] | 0             | 1              |
+===========+===============+================+
|         0 | v0{48}.[15:0] | v0{48}.[31:16] |
+-----------+---------------+----------------+
|         1 | v0{49}.[15:0] | v0{49}.[31:16] |
+-----------+---------------+----------------+
|         2 | v0{50}.[15:0] | v0{50}.[31:16] |
+-----------+---------------+----------------+
|         3 | v0{51}.[15:0] | v0{51}.[31:16] |
+-----------+---------------+----------------+
|         4 | v0{52}.[15:0] | v0{52}.[31:16] |
+-----------+---------------+----------------+
|         5 | v0{53}.[15:0] | v0{53}.[31:16] |
+-----------+---------------+----------------+
|         6 | v0{54}.[15:0] | v0{54}.[31:16] |
+-----------+---------------+----------------+
|         7 | v0{55}.[15:0] | v0{55}.[31:16] |
+-----------+---------------+----------------+
|         8 | v0{56}.[15:0] | v0{56}.[31:16] |
+-----------+---------------+----------------+
|         9 | v0{57}.[15:0] | v0{57}.[31:16] |
+-----------+---------------+----------------+
|        10 | v0{58}.[15:0] | v0{58}.[31:16] |
+-----------+---------------+----------------+
|        11 | v0{59}.[15:0] | v0{59}.[31:16] |
+-----------+---------------+----------------+
|        12 | v0{60}.[15:0] | v0{60}.[31:16] |
+-----------+---------------+----------------+
|        13 | v0{61}.[15:0] | v0{61}.[31:16] |
+-----------+---------------+----------------+
|        14 | v0{62}.[15:0] | v0{62}.[31:16] |
+-----------+---------------+----------------+
|        15 | v0{63}.[15:0] | v0{63}.[31:16] |
+-----------+---------------+----------------+
```

This is the same output with the CBSZ modifier set to the value 2 (broadcast 1 block to 4 blocks), and with ABID set to 2 (broadcast block number 2)
This will change the values that will be multiplied by blocks 0, 1, and 3 of the B\[\] matrix, and what results are added to the C\[\] matrix values of blocks 0, 1, and 3.

```
./matrix_calculator.py --architecture cdna2 --instruction v_mfma_f32_16x16x2bf16 --register-layout --A-matrix --cbsz 2 --abid 2
Architecture: CDNA2
Instruction: V_MFMA_F32_16X16X2BF16
Blocks 0, 1, 2, 3
+-----------+---------------+----------------+
|   A[M][K] | 0             | 1              |
+===========+===============+================+
|         0 | v0{32}.[15:0] | v0{32}.[31:16] |
+-----------+---------------+----------------+
|         1 | v0{33}.[15:0] | v0{33}.[31:16] |
+-----------+---------------+----------------+
|         2 | v0{34}.[15:0] | v0{34}.[31:16] |
+-----------+---------------+----------------+
|         3 | v0{35}.[15:0] | v0{35}.[31:16] |
+-----------+---------------+----------------+
|         4 | v0{36}.[15:0] | v0{36}.[31:16] |
+-----------+---------------+----------------+
|         5 | v0{37}.[15:0] | v0{37}.[31:16] |
+-----------+---------------+----------------+
|         6 | v0{38}.[15:0] | v0{38}.[31:16] |
+-----------+---------------+----------------+
|         7 | v0{39}.[15:0] | v0{39}.[31:16] |
+-----------+---------------+----------------+
|         8 | v0{40}.[15:0] | v0{40}.[31:16] |
+-----------+---------------+----------------+
|         9 | v0{41}.[15:0] | v0{41}.[31:16] |
+-----------+---------------+----------------+
|        10 | v0{42}.[15:0] | v0{42}.[31:16] |
+-----------+---------------+----------------+
|        11 | v0{43}.[15:0] | v0{43}.[31:16] |
+-----------+---------------+----------------+
|        12 | v0{44}.[15:0] | v0{44}.[31:16] |
+-----------+---------------+----------------+
|        13 | v0{45}.[15:0] | v0{45}.[31:16] |
+-----------+---------------+----------------+
|        14 | v0{46}.[15:0] | v0{46}.[31:16] |
+-----------+---------------+----------------+
|        15 | v0{47}.[15:0] | v0{47}.[31:16] |
+-----------+---------------+----------------+
```

This output demonstrates that by setting this value, all 4 blocks' matrix multiplication use the same input values, contained in VGPR0 lanes 32-47.
The values contained in lanes 0-31 and 48-63 are ignored by the Matrix Cores for this instruction.

### Example of Using the BLGP Modifier to Change the B Matrix Layout
Some matrix multiplication instructions in AMD Instinct&trade; accelerators using the CDNA&trade; 1, CDNA 2, and CDNA 3 architectures have encoding modifiers that allow static changes of which input matrix values are fed into the Matrix Cores.
These can help the performance of matrix multiplication applications by allowing software to quickly perform multiple matrix multiplications without spending time rearranging data in the registers.

One of these input modifiers (stored in bits 63:61 of the VOP3P-MAI instruction encoding) is the B-matrix Lane Group Pattern, BLGP.
For instructions that support this modifier, this allows a constrained set of swizzling operations between lanes.
For example, the encoding `BLGP=1` causes the values contained in the B\[\] matrix registers from wavefront lanes 0-31 and broadcasts them to the matrix multiplier's inputs from lanes 32-63.
This temporarily replaces the values in the register's lanes 32-63 for that instruction.

This can be especially useful for matrix multiplication instructions that have multiple blocks because the blocks are generally spread across lanes.
By swizzling the lanes with BLGP, a developer is more quickly able to perform the outer product operation by moving the input blocks of B\[\] into a different position after their initial usage.

The BLGP modifier is only supported by a subset of instructions, which are indicated in the `--detail-instruction` listing.
For instructions that support BLGP, the following encodings are supported:

* `BLGP=0`: normal matrix layout for B\[\]
* `BLGP=1`: The B\[\] matrix data from lanes 0-31 is also broadcast into lanes 32-63
* `BLGP=2`: The B\[\] matrix data from lanes 32-63 is broadcast into lanes 0-31
* `BLGP=3`: The B\[\] matrix data from all lanes is rotated down by 16 (e.g., lane 0's data is put into lane 48, lane 16's data is put into lane 0)
* `BLGP=4`: The B\[\] matrix data from lanes 0-15 is broadcast into lanes 16-31, 32-47, and 48-63
* `BLGP=5`: The B\[\] matrix data from lanes 16-31 is broadcast into lanes 0-15, 32-47, and 48-63
* `BLGP=6`: The B\[\] matrix data from lanes 32-47 is broadcast into lanes 0-15, 16-31, and 48-63
* `BLGP=7`: The B\[\] matrix data from lanes 48-63 is broadcast into lanes 0-15, 16-31, and 32-47

The `--blgp {#}` flag for this tool allows users to observe the effects of these settings on the `--get-register`, `--matrix-entry`, `--register-layout`, and `--matrix-layout` outputs.

The following is an example of the register layout for the matrix B of the V\_MFMA\_F32\_16X16X2BF16 instruction in the CDNA&trade; 2 architecture without the BLGP modifier set:
```
$ ./matrix_calculator.py --architecture cdna2 --instruction v_mfma_f32_16x16x2bf16 --register-layout --B-matrix
Architecture: CDNA2
Instruction: V_MFMA_F32_16X16X2BF16
Block 0
+-----------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+----------------+----------------+----------------+----------------+----------------+----------------+
|   B[K][N] | 0             | 1             | 2             | 3             | 4             | 5             | 6             | 7             | 8             | 9             | 10             | 11             | 12             | 13             | 14             | 15             |
+===========+===============+===============+===============+===============+===============+===============+===============+===============+===============+===============+================+================+================+================+================+================+
|         0 | v0{0}.[15:0]  | v0{1}.[15:0]  | v0{2}.[15:0]  | v0{3}.[15:0]  | v0{4}.[15:0]  | v0{5}.[15:0]  | v0{6}.[15:0]  | v0{7}.[15:0]  | v0{8}.[15:0]  | v0{9}.[15:0]  | v0{10}.[15:0]  | v0{11}.[15:0]  | v0{12}.[15:0]  | v0{13}.[15:0]  | v0{14}.[15:0]  | v0{15}.[15:0]  |
+-----------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+----------------+----------------+----------------+----------------+----------------+----------------+
|         1 | v0{0}.[31:16] | v0{1}.[31:16] | v0{2}.[31:16] | v0{3}.[31:16] | v0{4}.[31:16] | v0{5}.[31:16] | v0{6}.[31:16] | v0{7}.[31:16] | v0{8}.[31:16] | v0{9}.[31:16] | v0{10}.[31:16] | v0{11}.[31:16] | v0{12}.[31:16] | v0{13}.[31:16] | v0{14}.[31:16] | v0{15}.[31:16] |
+-----------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+----------------+----------------+----------------+----------------+----------------+----------------+
Block 1
+-----------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
|   B[K][N] | 0              | 1              | 2              | 3              | 4              | 5              | 6              | 7              | 8              | 9              | 10             | 11             | 12             | 13             | 14             | 15             |
+===========+================+================+================+================+================+================+================+================+================+================+================+================+================+================+================+================+
|         0 | v0{16}.[15:0]  | v0{17}.[15:0]  | v0{18}.[15:0]  | v0{19}.[15:0]  | v0{20}.[15:0]  | v0{21}.[15:0]  | v0{22}.[15:0]  | v0{23}.[15:0]  | v0{24}.[15:0]  | v0{25}.[15:0]  | v0{26}.[15:0]  | v0{27}.[15:0]  | v0{28}.[15:0]  | v0{29}.[15:0]  | v0{30}.[15:0]  | v0{31}.[15:0]  |
+-----------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
|         1 | v0{16}.[31:16] | v0{17}.[31:16] | v0{18}.[31:16] | v0{19}.[31:16] | v0{20}.[31:16] | v0{21}.[31:16] | v0{22}.[31:16] | v0{23}.[31:16] | v0{24}.[31:16] | v0{25}.[31:16] | v0{26}.[31:16] | v0{27}.[31:16] | v0{28}.[31:16] | v0{29}.[31:16] | v0{30}.[31:16] | v0{31}.[31:16] |
+-----------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
Block 2
+-----------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
|   B[K][N] | 0              | 1              | 2              | 3              | 4              | 5              | 6              | 7              | 8              | 9              | 10             | 11             | 12             | 13             | 14             | 15             |
+===========+================+================+================+================+================+================+================+================+================+================+================+================+================+================+================+================+
|         0 | v0{32}.[15:0]  | v0{33}.[15:0]  | v0{34}.[15:0]  | v0{35}.[15:0]  | v0{36}.[15:0]  | v0{37}.[15:0]  | v0{38}.[15:0]  | v0{39}.[15:0]  | v0{40}.[15:0]  | v0{41}.[15:0]  | v0{42}.[15:0]  | v0{43}.[15:0]  | v0{44}.[15:0]  | v0{45}.[15:0]  | v0{46}.[15:0]  | v0{47}.[15:0]  |
+-----------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
|         1 | v0{32}.[31:16] | v0{33}.[31:16] | v0{34}.[31:16] | v0{35}.[31:16] | v0{36}.[31:16] | v0{37}.[31:16] | v0{38}.[31:16] | v0{39}.[31:16] | v0{40}.[31:16] | v0{41}.[31:16] | v0{42}.[31:16] | v0{43}.[31:16] | v0{44}.[31:16] | v0{45}.[31:16] | v0{46}.[31:16] | v0{47}.[31:16] |
+-----------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
Block 3
+-----------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
|   B[K][N] | 0              | 1              | 2              | 3              | 4              | 5              | 6              | 7              | 8              | 9              | 10             | 11             | 12             | 13             | 14             | 15             |
+===========+================+================+================+================+================+================+================+================+================+================+================+================+================+================+================+================+
|         0 | v0{48}.[15:0]  | v0{49}.[15:0]  | v0{50}.[15:0]  | v0{51}.[15:0]  | v0{52}.[15:0]  | v0{53}.[15:0]  | v0{54}.[15:0]  | v0{55}.[15:0]  | v0{56}.[15:0]  | v0{57}.[15:0]  | v0{58}.[15:0]  | v0{59}.[15:0]  | v0{60}.[15:0]  | v0{61}.[15:0]  | v0{62}.[15:0]  | v0{63}.[15:0]  |
+-----------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
|         1 | v0{48}.[31:16] | v0{49}.[31:16] | v0{50}.[31:16] | v0{51}.[31:16] | v0{52}.[31:16] | v0{53}.[31:16] | v0{54}.[31:16] | v0{55}.[31:16] | v0{56}.[31:16] | v0{57}.[31:16] | v0{58}.[31:16] | v0{59}.[31:16] | v0{60}.[31:16] | v0{61}.[31:16] | v0{62}.[31:16] | v0{63}.[31:16] |
+-----------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
```

This is the same output with the BLGP modifier set to the value 2, which causes blocks 0 and 1 to receive the same input as what was originally sent to only blocks 2 and 3.
This will change the values that will be multiplied by blocks 0 and 1 of the A\[\] matrix, and what results are added to the C\[\] matrix values of blocks 0 and 1.
```
$ ./matrix_calculator.py --architecture cdna2 --instruction v_mfma_f32_16x16x2bf16 --register-layout --B-matrix --blgp 2
Architecture: CDNA2
Instruction: V_MFMA_F32_16X16X2BF16
Block 0
+-----------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
|   B[K][N] | 0              | 1              | 2              | 3              | 4              | 5              | 6              | 7              | 8              | 9              | 10             | 11             | 12             | 13             | 14             | 15             |
+===========+================+================+================+================+================+================+================+================+================+================+================+================+================+================+================+================+
|         0 | v0{32}.[15:0]  | v0{33}.[15:0]  | v0{34}.[15:0]  | v0{35}.[15:0]  | v0{36}.[15:0]  | v0{37}.[15:0]  | v0{38}.[15:0]  | v0{39}.[15:0]  | v0{40}.[15:0]  | v0{41}.[15:0]  | v0{42}.[15:0]  | v0{43}.[15:0]  | v0{44}.[15:0]  | v0{45}.[15:0]  | v0{46}.[15:0]  | v0{47}.[15:0]  |
+-----------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
|         1 | v0{32}.[31:16] | v0{33}.[31:16] | v0{34}.[31:16] | v0{35}.[31:16] | v0{36}.[31:16] | v0{37}.[31:16] | v0{38}.[31:16] | v0{39}.[31:16] | v0{40}.[31:16] | v0{41}.[31:16] | v0{42}.[31:16] | v0{43}.[31:16] | v0{44}.[31:16] | v0{45}.[31:16] | v0{46}.[31:16] | v0{47}.[31:16] |
+-----------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
Block 1
+-----------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
|   B[K][N] | 0              | 1              | 2              | 3              | 4              | 5              | 6              | 7              | 8              | 9              | 10             | 11             | 12             | 13             | 14             | 15             |
+===========+================+================+================+================+================+================+================+================+================+================+================+================+================+================+================+================+
|         0 | v0{48}.[15:0]  | v0{49}.[15:0]  | v0{50}.[15:0]  | v0{51}.[15:0]  | v0{52}.[15:0]  | v0{53}.[15:0]  | v0{54}.[15:0]  | v0{55}.[15:0]  | v0{56}.[15:0]  | v0{57}.[15:0]  | v0{58}.[15:0]  | v0{59}.[15:0]  | v0{60}.[15:0]  | v0{61}.[15:0]  | v0{62}.[15:0]  | v0{63}.[15:0]  |
+-----------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
|         1 | v0{48}.[31:16] | v0{49}.[31:16] | v0{50}.[31:16] | v0{51}.[31:16] | v0{52}.[31:16] | v0{53}.[31:16] | v0{54}.[31:16] | v0{55}.[31:16] | v0{56}.[31:16] | v0{57}.[31:16] | v0{58}.[31:16] | v0{59}.[31:16] | v0{60}.[31:16] | v0{61}.[31:16] | v0{62}.[31:16] | v0{63}.[31:16] |
+-----------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
Block 2
+-----------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
|   B[K][N] | 0              | 1              | 2              | 3              | 4              | 5              | 6              | 7              | 8              | 9              | 10             | 11             | 12             | 13             | 14             | 15             |
+===========+================+================+================+================+================+================+================+================+================+================+================+================+================+================+================+================+
|         0 | v0{32}.[15:0]  | v0{33}.[15:0]  | v0{34}.[15:0]  | v0{35}.[15:0]  | v0{36}.[15:0]  | v0{37}.[15:0]  | v0{38}.[15:0]  | v0{39}.[15:0]  | v0{40}.[15:0]  | v0{41}.[15:0]  | v0{42}.[15:0]  | v0{43}.[15:0]  | v0{44}.[15:0]  | v0{45}.[15:0]  | v0{46}.[15:0]  | v0{47}.[15:0]  |
+-----------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
|         1 | v0{32}.[31:16] | v0{33}.[31:16] | v0{34}.[31:16] | v0{35}.[31:16] | v0{36}.[31:16] | v0{37}.[31:16] | v0{38}.[31:16] | v0{39}.[31:16] | v0{40}.[31:16] | v0{41}.[31:16] | v0{42}.[31:16] | v0{43}.[31:16] | v0{44}.[31:16] | v0{45}.[31:16] | v0{46}.[31:16] | v0{47}.[31:16] |
+-----------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
Block 3
+-----------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
|   B[K][N] | 0              | 1              | 2              | 3              | 4              | 5              | 6              | 7              | 8              | 9              | 10             | 11             | 12             | 13             | 14             | 15             |
+===========+================+================+================+================+================+================+================+================+================+================+================+================+================+================+================+================+
|         0 | v0{48}.[15:0]  | v0{49}.[15:0]  | v0{50}.[15:0]  | v0{51}.[15:0]  | v0{52}.[15:0]  | v0{53}.[15:0]  | v0{54}.[15:0]  | v0{55}.[15:0]  | v0{56}.[15:0]  | v0{57}.[15:0]  | v0{58}.[15:0]  | v0{59}.[15:0]  | v0{60}.[15:0]  | v0{61}.[15:0]  | v0{62}.[15:0]  | v0{63}.[15:0]  |
+-----------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
|         1 | v0{48}.[31:16] | v0{49}.[31:16] | v0{50}.[31:16] | v0{51}.[31:16] | v0{52}.[31:16] | v0{53}.[31:16] | v0{54}.[31:16] | v0{55}.[31:16] | v0{56}.[31:16] | v0{57}.[31:16] | v0{58}.[31:16] | v0{59}.[31:16] | v0{60}.[31:16] | v0{61}.[31:16] | v0{62}.[31:16] | v0{63}.[31:16] |
+-----------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
```

#### Example of Using the BLGP Modifier to Negate FP64 Input Matrices in the CDNA&trade; 3 Architecture
-------------------------------------------------------------------------------------------------------
The CDNA&trade; 2 architecture added support for matrix multiplication instructions that use double-precision floating point (FP64) numbers.
In the accelerators using the CDNA 2 architecture, the BLGP field is disallowed.
The AMD Instinct&trade; MI300 accelerators use the CDNA 3 architecture and enhance the 64-bit floating point matrix multiplication instructions to use the BLGP fields.
However, the BLGP modifier field does not affect these FP64 MFMA instructions [in the manner described above](#example-of-using-the-blgp-modifier-to-change-the-b-matrix-layout).

Instead, in CDNA 3 accelerators, the three bits of the BLGP modifier are used to indicate if the three input matrices (A, B, and/or C) should have their values negated before being sent to the Matrix Core.
Setting the lowest-order bit of BLGP will negate Src0, the values of the A matrix.
Setting the middle bit of BLGP will negate Src1, the values of the B matrix.
Setting the highest-order bit of BLGP will negate Src3, the values of the C matrix.
Any of the bits may be set at the same time.

The following is an example that requests the matrix layout of the matrix B for the V\_MFMA\_F64\_16X16X4\_F64 instruction in the CDNA 3 architecture, with the BLGP value of 6.
Because the value 6 has the middle bit set, the values of the B\[\] matrix are returned with a negative sign to indicate that their values will be negated by the instruction.
```
./matrix_calculator.py --architecture cdna3 --instruction v_mfma_f64_16x16x4_f64 --matrix-layout --B-matrix --blgp 6
Architecture: CDNA3
Instruction: V_MFMA_F64_16X16X4_F64
+--------+-----------+
|   lane | v[1:0]    |
+========+===========+
|      0 | -B[0][0]  |
+--------+-----------+
|      1 | -B[0][1]  |
+--------+-----------+
|      2 | -B[0][2]  |
+--------+-----------+
|      3 | -B[0][3]  |
+--------+-----------+
|      4 | -B[0][4]  |
+--------+-----------+
|      5 | -B[0][5]  |
+--------+-----------+
|      6 | -B[0][6]  |
+--------+-----------+
|      7 | -B[0][7]  |
+--------+-----------+
|      8 | -B[0][8]  |
+--------+-----------+
|      9 | -B[0][9]  |
+--------+-----------+
|     10 | -B[0][10] |
+--------+-----------+
|     11 | -B[0][11] |
+--------+-----------+
|     12 | -B[0][12] |
+--------+-----------+
|     13 | -B[0][13] |
+--------+-----------+
|     14 | -B[0][14] |
+--------+-----------+
|     15 | -B[0][15] |
+--------+-----------+
|     16 | -B[1][0]  |
+--------+-----------+
|     17 | -B[1][1]  |
+--------+-----------+
|     18 | -B[1][2]  |
+--------+-----------+
|     19 | -B[1][3]  |
+--------+-----------+
|     20 | -B[1][4]  |
+--------+-----------+
|     21 | -B[1][5]  |
+--------+-----------+
|     22 | -B[1][6]  |
+--------+-----------+
|     23 | -B[1][7]  |
+--------+-----------+
|     24 | -B[1][8]  |
+--------+-----------+
|     25 | -B[1][9]  |
+--------+-----------+
|     26 | -B[1][10] |
+--------+-----------+
|     27 | -B[1][11] |
+--------+-----------+
|     28 | -B[1][12] |
+--------+-----------+
|     29 | -B[1][13] |
+--------+-----------+
|     30 | -B[1][14] |
+--------+-----------+
|     31 | -B[1][15] |
+--------+-----------+
|     32 | -B[2][0]  |
+--------+-----------+
|     33 | -B[2][1]  |
+--------+-----------+
|     34 | -B[2][2]  |
+--------+-----------+
|     35 | -B[2][3]  |
+--------+-----------+
|     36 | -B[2][4]  |
+--------+-----------+
|     37 | -B[2][5]  |
+--------+-----------+
|     38 | -B[2][6]  |
+--------+-----------+
|     39 | -B[2][7]  |
+--------+-----------+
|     40 | -B[2][8]  |
+--------+-----------+
|     41 | -B[2][9]  |
+--------+-----------+
|     42 | -B[2][10] |
+--------+-----------+
|     43 | -B[2][11] |
+--------+-----------+
|     44 | -B[2][12] |
+--------+-----------+
|     45 | -B[2][13] |
+--------+-----------+
|     46 | -B[2][14] |
+--------+-----------+
|     47 | -B[2][15] |
+--------+-----------+
|     48 | -B[3][0]  |
+--------+-----------+
|     49 | -B[3][1]  |
+--------+-----------+
|     50 | -B[3][2]  |
+--------+-----------+
|     51 | -B[3][3]  |
+--------+-----------+
|     52 | -B[3][4]  |
+--------+-----------+
|     53 | -B[3][5]  |
+--------+-----------+
|     54 | -B[3][6]  |
+--------+-----------+
|     55 | -B[3][7]  |
+--------+-----------+
|     56 | -B[3][8]  |
+--------+-----------+
|     57 | -B[3][9]  |
+--------+-----------+
|     58 | -B[3][10] |
+--------+-----------+
|     59 | -B[3][11] |
+--------+-----------+
|     60 | -B[3][12] |
+--------+-----------+
|     61 | -B[3][13] |
+--------+-----------+
|     62 | -B[3][14] |
+--------+-----------+
|     63 | -B[3][15] |
+--------+-----------+
```

### Example of Printing Compression Index Information for a Sparse Matrix Instruction
The AMD Instinct&trade; MI300 accelerators use the CDNA&trade; 3 architecture and support instruction that accelerate matrix multiplication for A\[\] matrices compressed with 4:2 structural sparsity.
These sparse matrix fused multiply accumulate (SMFMAC) instructions perform `D += A*B`.
They do not allow a C input matrix that is not equal to the D output matrix; they instead accumulate into the destination matrix.
The Src2 input of these instructions is instead a VGPR that contains the compression information.

For every 4-contiguous-entry chunk of the A\[\] matrix, 2 of the input values are compressed to zero and not stored.
As such, the A\[\] matrix can be stored in half as much space and the matrix multiplication can be performed at twice the speed by skipping the 0-multiplications.
However, for every non-zero entry of A\[\] stored in registers, we need 2 bits to indicate which of the original 4 entries of A\[\] are stored in this register.
For example, if the first two entries in a register of A\[\] hold the values of `A[0][2]` and `A[0][3]`, because `A[0][0]` and `A[0][1]` are compressed out, the first two entries of the compression index would store `2` and `3`, respectively.

This compression index information is stored in the VGPR pointed to by the SMFMAC instruction's Src2 input register field.
This register must be stored in architected VGPRs (ArchVGPRs).
For sparse instructions, this tool can query the layout of this register using the same four options used for other matrix queries: `--get-register`, `--matrix-entry`, `--register-layout`, and `--matrix-layout`.

To access this compression index, the option `--compression` or `-k` should be passed instead of the options for matrices A-D.

The following is an example usage, where we are requesting the location within the compression indices for the 2nd row and 31st column for the V\_SMFMAC\_F32\_16X16X32\_F16 instruction in the CDNA 3 architecture:
```
$ ./matrix_calculator.py --architecture cdna3 --instruction v_smfmac_f32_16x16x32_f16 --get-register --I-coordinate 2 --K-coordinate 31 --compression
Architecture: CDNA3
Instruction: V_SMFMAC_F32_16X16X32_F16
K[2][31] = v0{50}.[7:4]
```

This indicates that the compression bits for this entry, if it exists in the matrix, should be put into the 4th and 5th bits of the Src2 VGPR's lane 50.
Or they should be placed into the 6th and 7th bits, if this is the second value contained in the post-compression block.

The `--cbsz` and `--abid` modifiers can also affect the output of the compression index.
Because the VGPR that holds the compression index is 32b long, sparse instructions with 16b inputs only need 1/4 of a register (8 bits per lane) to store the compression index.
Setting `CBSZ==0` allows setting ABID to 0, 1, 2, or 3. This allows storing different compression indices in any of the 8b segments.
Setting `CBSZ!=0` will cause the ABID field to be ignored, and the low 8 bits of each register will be used (as if `ABID==0`).

Similarly, sparse instructions with 8b inputs only need 1/2 of a register (16 bits per lane) to store the compression indices.
Setting `CBSZ==0` allows setting ABID to 0 or 1, which allows choosing the compression indices from the top or bottom half of the register.
Setting `CBSZ!=0` will cause the ABID field to be ignored, and the low 16 bits of each register will be used (as if `ABID==0`).

This use of CBSZ and ABID modifiers for SMFMAC instructions prevents their use for broadcasting A\[\] matrix values [in the manner described above](#example-of-using-the-cbsz-and-abid-modifiers-to-change-the-a-matrix-layout).

For example, the following is the same instruction as before, looking for the compression index of the 2nd row and 31st column of V\_SMFMAC\_F32\_16X16X32\_F16 in MI300, but setting CBSZ and ABID such that we are looking the upper-most bits of the compression indices' VGPR.

```
$ ./matrix_calculator.py --architecture cdna3 --instruction v_smfmac_f32_16x16x32_f16 --get-register --I-coordinate 2 --K-coordinate 31 --compression --cbsz 0 --abid 3
Architecture: CDNA3
Instruction: V_SMFMAC_F32_16X16X32_F16
K[2][31] = v0{50}.[31:28]
```

Examples of Setting Modifiers in the AMD RDNA&trade; 3 and RDNA 4 Architectures
-------------------------------------------------------------------------------------------------------
AMD Radeon&trade; GPUs that use the RDNA 3 and RDNA 4 architecture allow setting modifier fields that change how the matrix multiplication instructions access data out of registers.
Examples of using these fields, `OPSEL`, and `NEG`, are shown below.

### Example of Using the OPSEL Modifier to Change the C and D Matrix Storage in the AMD RDNA&trade; 3 Architecture
Some instructions in the AMD RDNA&trade; 3 architecture allow the elements in the C\[\] and D\[\] matrices to be 16-bit values.
The format of these matrices uses 16 bits out of each 32-bit register to store these values.
These instructions will read only half of each C\[\] input register and write to only half of each D\[\] output register.

The 3rd bit of the VOP3P field `OPSEL` allows a user to choose which half of these registers to use.
Setting `OPSEL[2]=0` will cause the WMMA instruction to read from the bottom half of each C\[\] input register (bits 15:0) and write to the bottom half of each D\[\] output register.
Setting `OPSEL[2]=1` will cause the WMMA instruction to read from the top half of each C\[\] input register (bits 31:16) and write to the top half of each D\[\] output register.

This tool allows setting of this field using the `--opsel` flag for architecture and instructions that support this modifier.
Legal values for this option are `--opsel 0`, which indicates `OPSEL[2]=0`, and `--opsel 4`, which indicates `OPSEL[2]=1`.

The following is an example of the register layout for the matrix D of the V\_WMMA\_F16\_16X16X16\_f16 instruction in the RDNA 3 architecture without the OPSEL modifier set:
```
$ ./matrix_calculator.py --architecture rdna3 --instruction v_wmma_f16_16x16x16_f16 --register-layout --D-matrix
Architecture: RDNA3
Instruction: V_WMMA_F16_16X16X16_F16
+-----------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+
|   D[M][N] | 0             | 1             | 2             | 3             | 4             | 5             | 6             | 7             | 8             | 9             | 10            | 11            | 12            | 13            | 14            | 15            |
+===========+===============+===============+===============+===============+===============+===============+===============+===============+===============+===============+===============+===============+===============+===============+===============+===============+
|         0 | v0{0}.[15:0]  | v0{1}.[15:0]  | v0{2}.[15:0]  | v0{3}.[15:0]  | v0{4}.[15:0]  | v0{5}.[15:0]  | v0{6}.[15:0]  | v0{7}.[15:0]  | v0{8}.[15:0]  | v0{9}.[15:0]  | v0{10}.[15:0] | v0{11}.[15:0] | v0{12}.[15:0] | v0{13}.[15:0] | v0{14}.[15:0] | v0{15}.[15:0] |
+-----------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+
|         1 | v0{16}.[15:0] | v0{17}.[15:0] | v0{18}.[15:0] | v0{19}.[15:0] | v0{20}.[15:0] | v0{21}.[15:0] | v0{22}.[15:0] | v0{23}.[15:0] | v0{24}.[15:0] | v0{25}.[15:0] | v0{26}.[15:0] | v0{27}.[15:0] | v0{28}.[15:0] | v0{29}.[15:0] | v0{30}.[15:0] | v0{31}.[15:0] |
+-----------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+
|         2 | v1{0}.[15:0]  | v1{1}.[15:0]  | v1{2}.[15:0]  | v1{3}.[15:0]  | v1{4}.[15:0]  | v1{5}.[15:0]  | v1{6}.[15:0]  | v1{7}.[15:0]  | v1{8}.[15:0]  | v1{9}.[15:0]  | v1{10}.[15:0] | v1{11}.[15:0] | v1{12}.[15:0] | v1{13}.[15:0] | v1{14}.[15:0] | v1{15}.[15:0] |
+-----------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+
|         3 | v1{16}.[15:0] | v1{17}.[15:0] | v1{18}.[15:0] | v1{19}.[15:0] | v1{20}.[15:0] | v1{21}.[15:0] | v1{22}.[15:0] | v1{23}.[15:0] | v1{24}.[15:0] | v1{25}.[15:0] | v1{26}.[15:0] | v1{27}.[15:0] | v1{28}.[15:0] | v1{29}.[15:0] | v1{30}.[15:0] | v1{31}.[15:0] |
+-----------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+
|         4 | v2{0}.[15:0]  | v2{1}.[15:0]  | v2{2}.[15:0]  | v2{3}.[15:0]  | v2{4}.[15:0]  | v2{5}.[15:0]  | v2{6}.[15:0]  | v2{7}.[15:0]  | v2{8}.[15:0]  | v2{9}.[15:0]  | v2{10}.[15:0] | v2{11}.[15:0] | v2{12}.[15:0] | v2{13}.[15:0] | v2{14}.[15:0] | v2{15}.[15:0] |
+-----------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+
|         5 | v2{16}.[15:0] | v2{17}.[15:0] | v2{18}.[15:0] | v2{19}.[15:0] | v2{20}.[15:0] | v2{21}.[15:0] | v2{22}.[15:0] | v2{23}.[15:0] | v2{24}.[15:0] | v2{25}.[15:0] | v2{26}.[15:0] | v2{27}.[15:0] | v2{28}.[15:0] | v2{29}.[15:0] | v2{30}.[15:0] | v2{31}.[15:0] |
+-----------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+
|         6 | v3{0}.[15:0]  | v3{1}.[15:0]  | v3{2}.[15:0]  | v3{3}.[15:0]  | v3{4}.[15:0]  | v3{5}.[15:0]  | v3{6}.[15:0]  | v3{7}.[15:0]  | v3{8}.[15:0]  | v3{9}.[15:0]  | v3{10}.[15:0] | v3{11}.[15:0] | v3{12}.[15:0] | v3{13}.[15:0] | v3{14}.[15:0] | v3{15}.[15:0] |
+-----------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+
|         7 | v3{16}.[15:0] | v3{17}.[15:0] | v3{18}.[15:0] | v3{19}.[15:0] | v3{20}.[15:0] | v3{21}.[15:0] | v3{22}.[15:0] | v3{23}.[15:0] | v3{24}.[15:0] | v3{25}.[15:0] | v3{26}.[15:0] | v3{27}.[15:0] | v3{28}.[15:0] | v3{29}.[15:0] | v3{30}.[15:0] | v3{31}.[15:0] |
+-----------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+
|         8 | v4{0}.[15:0]  | v4{1}.[15:0]  | v4{2}.[15:0]  | v4{3}.[15:0]  | v4{4}.[15:0]  | v4{5}.[15:0]  | v4{6}.[15:0]  | v4{7}.[15:0]  | v4{8}.[15:0]  | v4{9}.[15:0]  | v4{10}.[15:0] | v4{11}.[15:0] | v4{12}.[15:0] | v4{13}.[15:0] | v4{14}.[15:0] | v4{15}.[15:0] |
+-----------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+
|         9 | v4{16}.[15:0] | v4{17}.[15:0] | v4{18}.[15:0] | v4{19}.[15:0] | v4{20}.[15:0] | v4{21}.[15:0] | v4{22}.[15:0] | v4{23}.[15:0] | v4{24}.[15:0] | v4{25}.[15:0] | v4{26}.[15:0] | v4{27}.[15:0] | v4{28}.[15:0] | v4{29}.[15:0] | v4{30}.[15:0] | v4{31}.[15:0] |
+-----------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+
|        10 | v5{0}.[15:0]  | v5{1}.[15:0]  | v5{2}.[15:0]  | v5{3}.[15:0]  | v5{4}.[15:0]  | v5{5}.[15:0]  | v5{6}.[15:0]  | v5{7}.[15:0]  | v5{8}.[15:0]  | v5{9}.[15:0]  | v5{10}.[15:0] | v5{11}.[15:0] | v5{12}.[15:0] | v5{13}.[15:0] | v5{14}.[15:0] | v5{15}.[15:0] |
+-----------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+
|        11 | v5{16}.[15:0] | v5{17}.[15:0] | v5{18}.[15:0] | v5{19}.[15:0] | v5{20}.[15:0] | v5{21}.[15:0] | v5{22}.[15:0] | v5{23}.[15:0] | v5{24}.[15:0] | v5{25}.[15:0] | v5{26}.[15:0] | v5{27}.[15:0] | v5{28}.[15:0] | v5{29}.[15:0] | v5{30}.[15:0] | v5{31}.[15:0] |
+-----------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+
|        12 | v6{0}.[15:0]  | v6{1}.[15:0]  | v6{2}.[15:0]  | v6{3}.[15:0]  | v6{4}.[15:0]  | v6{5}.[15:0]  | v6{6}.[15:0]  | v6{7}.[15:0]  | v6{8}.[15:0]  | v6{9}.[15:0]  | v6{10}.[15:0] | v6{11}.[15:0] | v6{12}.[15:0] | v6{13}.[15:0] | v6{14}.[15:0] | v6{15}.[15:0] |
+-----------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+
|        13 | v6{16}.[15:0] | v6{17}.[15:0] | v6{18}.[15:0] | v6{19}.[15:0] | v6{20}.[15:0] | v6{21}.[15:0] | v6{22}.[15:0] | v6{23}.[15:0] | v6{24}.[15:0] | v6{25}.[15:0] | v6{26}.[15:0] | v6{27}.[15:0] | v6{28}.[15:0] | v6{29}.[15:0] | v6{30}.[15:0] | v6{31}.[15:0] |
+-----------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+
|        14 | v7{0}.[15:0]  | v7{1}.[15:0]  | v7{2}.[15:0]  | v7{3}.[15:0]  | v7{4}.[15:0]  | v7{5}.[15:0]  | v7{6}.[15:0]  | v7{7}.[15:0]  | v7{8}.[15:0]  | v7{9}.[15:0]  | v7{10}.[15:0] | v7{11}.[15:0] | v7{12}.[15:0] | v7{13}.[15:0] | v7{14}.[15:0] | v7{15}.[15:0] |
+-----------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+
|        15 | v7{16}.[15:0] | v7{17}.[15:0] | v7{18}.[15:0] | v7{19}.[15:0] | v7{20}.[15:0] | v7{21}.[15:0] | v7{22}.[15:0] | v7{23}.[15:0] | v7{24}.[15:0] | v7{25}.[15:0] | v7{26}.[15:0] | v7{27}.[15:0] | v7{28}.[15:0] | v7{29}.[15:0] | v7{30}.[15:0] | v7{31}.[15:0] |
+-----------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+---------------+
```

In the above case, because `OPSEL[2]=0`, only the bottom half of each register (bits 15:0) is used.

The following is an example of the register layout for the matrix D of the V\_WMMA\_F16\_16X16X16\_f16 instruction in the RDNA 3 architecture without the OPSEL modifier set:
```
$ ./matrix_calculator.py --architecture rdna3 --instruction v_wmma_f16_16x16x16_f16 --register-layout --D-matrix --opsel 4
Architecture: RDNA3
Instruction: V_WMMA_F16_16X16X16_F16
+-----------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
|   D[M][N] | 0              | 1              | 2              | 3              | 4              | 5              | 6              | 7              | 8              | 9              | 10             | 11             | 12             | 13             | 14             | 15             |
+===========+================+================+================+================+================+================+================+================+================+================+================+================+================+================+================+================+
|         0 | v0{0}.[31:16]  | v0{1}.[31:16]  | v0{2}.[31:16]  | v0{3}.[31:16]  | v0{4}.[31:16]  | v0{5}.[31:16]  | v0{6}.[31:16]  | v0{7}.[31:16]  | v0{8}.[31:16]  | v0{9}.[31:16]  | v0{10}.[31:16] | v0{11}.[31:16] | v0{12}.[31:16] | v0{13}.[31:16] | v0{14}.[31:16] | v0{15}.[31:16] |
+-----------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
|         1 | v0{16}.[31:16] | v0{17}.[31:16] | v0{18}.[31:16] | v0{19}.[31:16] | v0{20}.[31:16] | v0{21}.[31:16] | v0{22}.[31:16] | v0{23}.[31:16] | v0{24}.[31:16] | v0{25}.[31:16] | v0{26}.[31:16] | v0{27}.[31:16] | v0{28}.[31:16] | v0{29}.[31:16] | v0{30}.[31:16] | v0{31}.[31:16] |
+-----------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
|         2 | v1{0}.[31:16]  | v1{1}.[31:16]  | v1{2}.[31:16]  | v1{3}.[31:16]  | v1{4}.[31:16]  | v1{5}.[31:16]  | v1{6}.[31:16]  | v1{7}.[31:16]  | v1{8}.[31:16]  | v1{9}.[31:16]  | v1{10}.[31:16] | v1{11}.[31:16] | v1{12}.[31:16] | v1{13}.[31:16] | v1{14}.[31:16] | v1{15}.[31:16] |
+-----------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
|         3 | v1{16}.[31:16] | v1{17}.[31:16] | v1{18}.[31:16] | v1{19}.[31:16] | v1{20}.[31:16] | v1{21}.[31:16] | v1{22}.[31:16] | v1{23}.[31:16] | v1{24}.[31:16] | v1{25}.[31:16] | v1{26}.[31:16] | v1{27}.[31:16] | v1{28}.[31:16] | v1{29}.[31:16] | v1{30}.[31:16] | v1{31}.[31:16] |
+-----------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
|         4 | v2{0}.[31:16]  | v2{1}.[31:16]  | v2{2}.[31:16]  | v2{3}.[31:16]  | v2{4}.[31:16]  | v2{5}.[31:16]  | v2{6}.[31:16]  | v2{7}.[31:16]  | v2{8}.[31:16]  | v2{9}.[31:16]  | v2{10}.[31:16] | v2{11}.[31:16] | v2{12}.[31:16] | v2{13}.[31:16] | v2{14}.[31:16] | v2{15}.[31:16] |
+-----------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
|         5 | v2{16}.[31:16] | v2{17}.[31:16] | v2{18}.[31:16] | v2{19}.[31:16] | v2{20}.[31:16] | v2{21}.[31:16] | v2{22}.[31:16] | v2{23}.[31:16] | v2{24}.[31:16] | v2{25}.[31:16] | v2{26}.[31:16] | v2{27}.[31:16] | v2{28}.[31:16] | v2{29}.[31:16] | v2{30}.[31:16] | v2{31}.[31:16] |
+-----------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
|         6 | v3{0}.[31:16]  | v3{1}.[31:16]  | v3{2}.[31:16]  | v3{3}.[31:16]  | v3{4}.[31:16]  | v3{5}.[31:16]  | v3{6}.[31:16]  | v3{7}.[31:16]  | v3{8}.[31:16]  | v3{9}.[31:16]  | v3{10}.[31:16] | v3{11}.[31:16] | v3{12}.[31:16] | v3{13}.[31:16] | v3{14}.[31:16] | v3{15}.[31:16] |
+-----------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
|         7 | v3{16}.[31:16] | v3{17}.[31:16] | v3{18}.[31:16] | v3{19}.[31:16] | v3{20}.[31:16] | v3{21}.[31:16] | v3{22}.[31:16] | v3{23}.[31:16] | v3{24}.[31:16] | v3{25}.[31:16] | v3{26}.[31:16] | v3{27}.[31:16] | v3{28}.[31:16] | v3{29}.[31:16] | v3{30}.[31:16] | v3{31}.[31:16] |
+-----------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
|         8 | v4{0}.[31:16]  | v4{1}.[31:16]  | v4{2}.[31:16]  | v4{3}.[31:16]  | v4{4}.[31:16]  | v4{5}.[31:16]  | v4{6}.[31:16]  | v4{7}.[31:16]  | v4{8}.[31:16]  | v4{9}.[31:16]  | v4{10}.[31:16] | v4{11}.[31:16] | v4{12}.[31:16] | v4{13}.[31:16] | v4{14}.[31:16] | v4{15}.[31:16] |
+-----------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
|         9 | v4{16}.[31:16] | v4{17}.[31:16] | v4{18}.[31:16] | v4{19}.[31:16] | v4{20}.[31:16] | v4{21}.[31:16] | v4{22}.[31:16] | v4{23}.[31:16] | v4{24}.[31:16] | v4{25}.[31:16] | v4{26}.[31:16] | v4{27}.[31:16] | v4{28}.[31:16] | v4{29}.[31:16] | v4{30}.[31:16] | v4{31}.[31:16] |
+-----------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
|        10 | v5{0}.[31:16]  | v5{1}.[31:16]  | v5{2}.[31:16]  | v5{3}.[31:16]  | v5{4}.[31:16]  | v5{5}.[31:16]  | v5{6}.[31:16]  | v5{7}.[31:16]  | v5{8}.[31:16]  | v5{9}.[31:16]  | v5{10}.[31:16] | v5{11}.[31:16] | v5{12}.[31:16] | v5{13}.[31:16] | v5{14}.[31:16] | v5{15}.[31:16] |
+-----------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
|        11 | v5{16}.[31:16] | v5{17}.[31:16] | v5{18}.[31:16] | v5{19}.[31:16] | v5{20}.[31:16] | v5{21}.[31:16] | v5{22}.[31:16] | v5{23}.[31:16] | v5{24}.[31:16] | v5{25}.[31:16] | v5{26}.[31:16] | v5{27}.[31:16] | v5{28}.[31:16] | v5{29}.[31:16] | v5{30}.[31:16] | v5{31}.[31:16] |
+-----------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
|        12 | v6{0}.[31:16]  | v6{1}.[31:16]  | v6{2}.[31:16]  | v6{3}.[31:16]  | v6{4}.[31:16]  | v6{5}.[31:16]  | v6{6}.[31:16]  | v6{7}.[31:16]  | v6{8}.[31:16]  | v6{9}.[31:16]  | v6{10}.[31:16] | v6{11}.[31:16] | v6{12}.[31:16] | v6{13}.[31:16] | v6{14}.[31:16] | v6{15}.[31:16] |
+-----------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
|        13 | v6{16}.[31:16] | v6{17}.[31:16] | v6{18}.[31:16] | v6{19}.[31:16] | v6{20}.[31:16] | v6{21}.[31:16] | v6{22}.[31:16] | v6{23}.[31:16] | v6{24}.[31:16] | v6{25}.[31:16] | v6{26}.[31:16] | v6{27}.[31:16] | v6{28}.[31:16] | v6{29}.[31:16] | v6{30}.[31:16] | v6{31}.[31:16] |
+-----------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
|        14 | v7{0}.[31:16]  | v7{1}.[31:16]  | v7{2}.[31:16]  | v7{3}.[31:16]  | v7{4}.[31:16]  | v7{5}.[31:16]  | v7{6}.[31:16]  | v7{7}.[31:16]  | v7{8}.[31:16]  | v7{9}.[31:16]  | v7{10}.[31:16] | v7{11}.[31:16] | v7{12}.[31:16] | v7{13}.[31:16] | v7{14}.[31:16] | v7{15}.[31:16] |
+-----------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
|        15 | v7{16}.[31:16] | v7{17}.[31:16] | v7{18}.[31:16] | v7{19}.[31:16] | v7{20}.[31:16] | v7{21}.[31:16] | v7{22}.[31:16] | v7{23}.[31:16] | v7{24}.[31:16] | v7{25}.[31:16] | v7{26}.[31:16] | v7{27}.[31:16] | v7{28}.[31:16] | v7{29}.[31:16] | v7{30}.[31:16] | v7{31}.[31:16] |
+-----------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
```

In this case, because `OPSEL[2]=1`, only the top half of each register (bits 31:16) is used.

### Example of Using the OPSEL Modifier to Change the Compression Index Set in the AMD RDNA&trade; 4 Architecture
The AMD Radeon&trade; GPUs that use the RDNA&trade; 4 architecture support instruction that accelerate matrix multiplication for A\[\] matrices compressed with 4:2 structural sparsity.
These sparse wave matrix multiply accumulate (SWMMAC) instructions perform `D += A*B`.
They do not allow a C input matrix that is not equal to the D output matrix; they instead accumulate into the destination matrix.
The Src2 input of these instructions is instead a VGPR that contains the compression information.

For every 4-contiguous-entry chunk of the A\[\] matrix, 2 of the input values are compressed to zero and not stored.
As such, the A\[\] matrix can be stored in half as much space and the matrix multiplication can be performed at twice the speed by skipping the 0-multiplications.
However, for every non-zero entry of A\[\] stored in registers, we need 2 bits to indicate which of the original 4 entries of A\[\] are stored in this register.
For example, if the first two entries in a register of A\[\] hold the values of `A[0][2]` and `A[0][3]`, because `A[0][0]` and `A[0][1]` are compressed out, the first two entries of the compression index would store `2` and `3`, respectively.

This compression index information is stored in the VGPR pointed to by the SWMMAC instruction's Src2 input register field.
For sparse instructions, this tool can query the layout of this register using the same four options used for other matrix queries: `--get-register`, `--matrix-entry`, `--register-layout`, and `--matrix-layout`.

To access this compression index, the option `--compression` or `-k` should be passed instead of the options for matrices A-D.

The following is an example usage, where we are requesting the location within the compression indices for the 2nd row and 31st column for the V\_SWMMAC\_F32\_16X16X32\_F16 in the RDNA 4 architecture:
```
$ ./matrix_calculator.py --architecture rdna4 --instruction v_swmmac_f32_16x16x32_f16 --get-register --I-coordinate 2 --K-coordinate 31 --compression
Architecture: RDNA4
Instruction: V_SWMMAC_F32_16X16X32_F16
K[2][31] = v0{18}.[15:12]
```

This indicates that the compression bits for this entry should be put into the 13th and 12th bits of the Src2 VGPR's lane 18.
Or they should be placed into the 15th and 14th bits, if this is the second value contained in the post-compression block.

The `--opsel` modifier can also affect the output of the compression index.
Because the VGPR that holds the compression index is 32b long, some sparse matrix instructions do not need the entire all 32b of each 32- or 64-wide VGPR to hold the compression indices.
In this case, the remaining bits in each VGPR can be used to hold mutliple sets of compression indices.
Setting `OPSEL` allows picking between these sets.
The `OPSEL` modifier is only used for this purpose in the RDNA 4 architecture.

For example, the following is the same instruction as before, looking for the compression index of the 2nd row and 31st column of V\_SWMMAC\_F32\_16X16X32\_F16 in the RDNA4 architecture, but setting `OPSEL` modifier to 1 in order to access the 2nd set of compression indices within the register.

```
$ ./matrix_calculator.py --architecture rdna4 --instruction v_swmmac_f32_16x16x32_f16 --get-register --I-coordinate 2 --K-coordinate 31 --compression --opsel 1
Architecture: RDNA4
Instruction: V_SWMMAC_F32_16X16X32_F16
K[2][31] = v0{18}.[31:28]
```

The difference in this example is that the 1st compression index set for this coordinate was contained in the 18th lane's VGPR bits 15:12.
The 2nd compression index set for this coordinate is contained in the 18th lane's VGPR bits 31:28.

### Example of Using the NEG Modifier to Negate Input Matrices
The RDNA&trade; 3 and RDNA 4 architectures allow two separate three-bit modifiers, `NEG` and `NEG_HI`, which indicate if bits within the three input matrices (A, B, and/or C) should have their values negated before being sent to the AI Accelerator.
For 16-bit floating point WMMA and SWMMAC instructions, setting the lowest-order bit of these registers will negate Src0, the values of the A matrix.
`NEG[0]` will negate floating point values from bits 0-15 of Src0, while `NEG_HI[0]` will negate floating point values from bits 16-31 of Src0.
For 16-bit floating point WMMA and SWMMAC instructions, setting the middle bit of NEG will negate Src1, the values of the B matrix.
`NEG[1]` will negate floating point values from bits 0-15 of Src1, while `NEG_HI[1]` will negate floating point values from bits 16-31 of Src1.
For 16-bit floating point WMMA and SWMMAC instructions, setting the highest-order bit of NEG will negate Src3, the values of the C matrix, while setting the highest-order bit of `NEG_HI` will take the absolute value of the C matrix entry.
Any of the bits may be set at the same time; when `NEG[2]` and `NEG_HI[2]` are both set, the absolute value is calculated before the negation.

The `NEG` modifier is not supported by WMMA and SWMMAC instructions operating on 8-bit floating point values.

For integer WMMA and SWMMAC instructions, the two lower-order bits of `NEG` are used to control whether the values in the A\[\] and B\[\] matrices are treated as unsigned or signed integers.
`NEG[2]` and the entire `NEG_HI` modifier may not be used to negate the A\[] or B\[] matrices for integer WMMA instructions, and they should be set to 0.

This tool allows setting the `NEG` field using the `--neg` flag for architectures and instructions that support this modifier.
The  `--neg_hi` flag is used to set the `NEG_HI` field.
Because these are 3-bit fields, legal values for this option are integers between 0 and 7, inclusive.
See [the later section on instruction encodings](#details-of-the-vop3p-matrix-multiplication-instruction-encoding-in-the-amd-rdna-3-and-rdna-4-architectures) for how the `NEG` field and `NEG_HI` fields interact.
This tool does not let users manually change the `NEG_HI` field.

The following is an example that requests the matrix layout of the matrix B for the V\_WMMA\_F32\_16X16X16\_F16 instruction in the RDNA 3 architecture, with the `NEG` and `NEG_HI` value of 6.
Because the value 6 has the middle bit set, the values of the B\[\] matrix are returned with a negative sign to indicate that their values will be negated by the instruction.

```
$ ./matrix_calculator.py --architecture rdna3 --instruction v_wmma_f32_16x16x16_f16 --matrix-layout --B-matrix --neg 6 --neg_hi 6
Architecture: RDNA3
Instruction: V_WMMA_F32_16X16X16_F16
+--------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+
|   lane | v0.[15:0]   | v0.[31:16]   | v1.[15:0]   | v1.[31:16]   | v2.[15:0]   | v2.[31:16]   | v3.[15:0]   | v3.[31:16]   | v4.[15:0]   | v4.[31:16]   | v5.[15:0]   | v5.[31:16]   | v6.[15:0]   | v6.[31:16]   | v7.[15:0]   | v7.[31:16]   |
+========+=============+==============+=============+==============+=============+==============+=============+==============+=============+==============+=============+==============+=============+==============+=============+==============+
|      0 | -B[0][0]    | -B[1][0]     | -B[2][0]    | -B[3][0]     | -B[4][0]    | -B[5][0]     | -B[6][0]    | -B[7][0]     | -B[8][0]    | -B[9][0]     | -B[10][0]   | -B[11][0]    | -B[12][0]   | -B[13][0]    | -B[14][0]   | -B[15][0]    |
+--------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+
|      1 | -B[0][1]    | -B[1][1]     | -B[2][1]    | -B[3][1]     | -B[4][1]    | -B[5][1]     | -B[6][1]    | -B[7][1]     | -B[8][1]    | -B[9][1]     | -B[10][1]   | -B[11][1]    | -B[12][1]   | -B[13][1]    | -B[14][1]   | -B[15][1]    |
+--------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+
|      2 | -B[0][2]    | -B[1][2]     | -B[2][2]    | -B[3][2]     | -B[4][2]    | -B[5][2]     | -B[6][2]    | -B[7][2]     | -B[8][2]    | -B[9][2]     | -B[10][2]   | -B[11][2]    | -B[12][2]   | -B[13][2]    | -B[14][2]   | -B[15][2]    |
+--------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+
|      3 | -B[0][3]    | -B[1][3]     | -B[2][3]    | -B[3][3]     | -B[4][3]    | -B[5][3]     | -B[6][3]    | -B[7][3]     | -B[8][3]    | -B[9][3]     | -B[10][3]   | -B[11][3]    | -B[12][3]   | -B[13][3]    | -B[14][3]   | -B[15][3]    |
+--------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+
|      4 | -B[0][4]    | -B[1][4]     | -B[2][4]    | -B[3][4]     | -B[4][4]    | -B[5][4]     | -B[6][4]    | -B[7][4]     | -B[8][4]    | -B[9][4]     | -B[10][4]   | -B[11][4]    | -B[12][4]   | -B[13][4]    | -B[14][4]   | -B[15][4]    |
+--------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+
|      5 | -B[0][5]    | -B[1][5]     | -B[2][5]    | -B[3][5]     | -B[4][5]    | -B[5][5]     | -B[6][5]    | -B[7][5]     | -B[8][5]    | -B[9][5]     | -B[10][5]   | -B[11][5]    | -B[12][5]   | -B[13][5]    | -B[14][5]   | -B[15][5]    |
+--------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+
|      6 | -B[0][6]    | -B[1][6]     | -B[2][6]    | -B[3][6]     | -B[4][6]    | -B[5][6]     | -B[6][6]    | -B[7][6]     | -B[8][6]    | -B[9][6]     | -B[10][6]   | -B[11][6]    | -B[12][6]   | -B[13][6]    | -B[14][6]   | -B[15][6]    |
+--------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+
|      7 | -B[0][7]    | -B[1][7]     | -B[2][7]    | -B[3][7]     | -B[4][7]    | -B[5][7]     | -B[6][7]    | -B[7][7]     | -B[8][7]    | -B[9][7]     | -B[10][7]   | -B[11][7]    | -B[12][7]   | -B[13][7]    | -B[14][7]   | -B[15][7]    |
+--------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+
|      8 | -B[0][8]    | -B[1][8]     | -B[2][8]    | -B[3][8]     | -B[4][8]    | -B[5][8]     | -B[6][8]    | -B[7][8]     | -B[8][8]    | -B[9][8]     | -B[10][8]   | -B[11][8]    | -B[12][8]   | -B[13][8]    | -B[14][8]   | -B[15][8]    |
+--------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+
|      9 | -B[0][9]    | -B[1][9]     | -B[2][9]    | -B[3][9]     | -B[4][9]    | -B[5][9]     | -B[6][9]    | -B[7][9]     | -B[8][9]    | -B[9][9]     | -B[10][9]   | -B[11][9]    | -B[12][9]   | -B[13][9]    | -B[14][9]   | -B[15][9]    |
+--------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+
|     10 | -B[0][10]   | -B[1][10]    | -B[2][10]   | -B[3][10]    | -B[4][10]   | -B[5][10]    | -B[6][10]   | -B[7][10]    | -B[8][10]   | -B[9][10]    | -B[10][10]  | -B[11][10]   | -B[12][10]  | -B[13][10]   | -B[14][10]  | -B[15][10]   |
+--------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+
|     11 | -B[0][11]   | -B[1][11]    | -B[2][11]   | -B[3][11]    | -B[4][11]   | -B[5][11]    | -B[6][11]   | -B[7][11]    | -B[8][11]   | -B[9][11]    | -B[10][11]  | -B[11][11]   | -B[12][11]  | -B[13][11]   | -B[14][11]  | -B[15][11]   |
+--------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+
|     12 | -B[0][12]   | -B[1][12]    | -B[2][12]   | -B[3][12]    | -B[4][12]   | -B[5][12]    | -B[6][12]   | -B[7][12]    | -B[8][12]   | -B[9][12]    | -B[10][12]  | -B[11][12]   | -B[12][12]  | -B[13][12]   | -B[14][12]  | -B[15][12]   |
+--------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+
|     13 | -B[0][13]   | -B[1][13]    | -B[2][13]   | -B[3][13]    | -B[4][13]   | -B[5][13]    | -B[6][13]   | -B[7][13]    | -B[8][13]   | -B[9][13]    | -B[10][13]  | -B[11][13]   | -B[12][13]  | -B[13][13]   | -B[14][13]  | -B[15][13]   |
+--------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+
|     14 | -B[0][14]   | -B[1][14]    | -B[2][14]   | -B[3][14]    | -B[4][14]   | -B[5][14]    | -B[6][14]   | -B[7][14]    | -B[8][14]   | -B[9][14]    | -B[10][14]  | -B[11][14]   | -B[12][14]  | -B[13][14]   | -B[14][14]  | -B[15][14]   |
+--------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+
|     15 | -B[0][15]   | -B[1][15]    | -B[2][15]   | -B[3][15]    | -B[4][15]   | -B[5][15]    | -B[6][15]   | -B[7][15]    | -B[8][15]   | -B[9][15]    | -B[10][15]  | -B[11][15]   | -B[12][15]  | -B[13][15]   | -B[14][15]  | -B[15][15]   |
+--------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+
|     16 | -B[0][0]    | -B[1][0]     | -B[2][0]    | -B[3][0]     | -B[4][0]    | -B[5][0]     | -B[6][0]    | -B[7][0]     | -B[8][0]    | -B[9][0]     | -B[10][0]   | -B[11][0]    | -B[12][0]   | -B[13][0]    | -B[14][0]   | -B[15][0]    |
+--------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+
|     17 | -B[0][1]    | -B[1][1]     | -B[2][1]    | -B[3][1]     | -B[4][1]    | -B[5][1]     | -B[6][1]    | -B[7][1]     | -B[8][1]    | -B[9][1]     | -B[10][1]   | -B[11][1]    | -B[12][1]   | -B[13][1]    | -B[14][1]   | -B[15][1]    |
+--------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+
|     18 | -B[0][2]    | -B[1][2]     | -B[2][2]    | -B[3][2]     | -B[4][2]    | -B[5][2]     | -B[6][2]    | -B[7][2]     | -B[8][2]    | -B[9][2]     | -B[10][2]   | -B[11][2]    | -B[12][2]   | -B[13][2]    | -B[14][2]   | -B[15][2]    |
+--------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+
|     19 | -B[0][3]    | -B[1][3]     | -B[2][3]    | -B[3][3]     | -B[4][3]    | -B[5][3]     | -B[6][3]    | -B[7][3]     | -B[8][3]    | -B[9][3]     | -B[10][3]   | -B[11][3]    | -B[12][3]   | -B[13][3]    | -B[14][3]   | -B[15][3]    |
+--------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+
|     20 | -B[0][4]    | -B[1][4]     | -B[2][4]    | -B[3][4]     | -B[4][4]    | -B[5][4]     | -B[6][4]    | -B[7][4]     | -B[8][4]    | -B[9][4]     | -B[10][4]   | -B[11][4]    | -B[12][4]   | -B[13][4]    | -B[14][4]   | -B[15][4]    |
+--------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+
|     21 | -B[0][5]    | -B[1][5]     | -B[2][5]    | -B[3][5]     | -B[4][5]    | -B[5][5]     | -B[6][5]    | -B[7][5]     | -B[8][5]    | -B[9][5]     | -B[10][5]   | -B[11][5]    | -B[12][5]   | -B[13][5]    | -B[14][5]   | -B[15][5]    |
+--------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+
|     22 | -B[0][6]    | -B[1][6]     | -B[2][6]    | -B[3][6]     | -B[4][6]    | -B[5][6]     | -B[6][6]    | -B[7][6]     | -B[8][6]    | -B[9][6]     | -B[10][6]   | -B[11][6]    | -B[12][6]   | -B[13][6]    | -B[14][6]   | -B[15][6]    |
+--------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+
|     23 | -B[0][7]    | -B[1][7]     | -B[2][7]    | -B[3][7]     | -B[4][7]    | -B[5][7]     | -B[6][7]    | -B[7][7]     | -B[8][7]    | -B[9][7]     | -B[10][7]   | -B[11][7]    | -B[12][7]   | -B[13][7]    | -B[14][7]   | -B[15][7]    |
+--------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+
|     24 | -B[0][8]    | -B[1][8]     | -B[2][8]    | -B[3][8]     | -B[4][8]    | -B[5][8]     | -B[6][8]    | -B[7][8]     | -B[8][8]    | -B[9][8]     | -B[10][8]   | -B[11][8]    | -B[12][8]   | -B[13][8]    | -B[14][8]   | -B[15][8]    |
+--------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+
|     25 | -B[0][9]    | -B[1][9]     | -B[2][9]    | -B[3][9]     | -B[4][9]    | -B[5][9]     | -B[6][9]    | -B[7][9]     | -B[8][9]    | -B[9][9]     | -B[10][9]   | -B[11][9]    | -B[12][9]   | -B[13][9]    | -B[14][9]   | -B[15][9]    |
+--------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+
|     26 | -B[0][10]   | -B[1][10]    | -B[2][10]   | -B[3][10]    | -B[4][10]   | -B[5][10]    | -B[6][10]   | -B[7][10]    | -B[8][10]   | -B[9][10]    | -B[10][10]  | -B[11][10]   | -B[12][10]  | -B[13][10]   | -B[14][10]  | -B[15][10]   |
+--------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+
|     27 | -B[0][11]   | -B[1][11]    | -B[2][11]   | -B[3][11]    | -B[4][11]   | -B[5][11]    | -B[6][11]   | -B[7][11]    | -B[8][11]   | -B[9][11]    | -B[10][11]  | -B[11][11]   | -B[12][11]  | -B[13][11]   | -B[14][11]  | -B[15][11]   |
+--------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+
|     28 | -B[0][12]   | -B[1][12]    | -B[2][12]   | -B[3][12]    | -B[4][12]   | -B[5][12]    | -B[6][12]   | -B[7][12]    | -B[8][12]   | -B[9][12]    | -B[10][12]  | -B[11][12]   | -B[12][12]  | -B[13][12]   | -B[14][12]  | -B[15][12]   |
+--------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+
|     29 | -B[0][13]   | -B[1][13]    | -B[2][13]   | -B[3][13]    | -B[4][13]   | -B[5][13]    | -B[6][13]   | -B[7][13]    | -B[8][13]   | -B[9][13]    | -B[10][13]  | -B[11][13]   | -B[12][13]  | -B[13][13]   | -B[14][13]  | -B[15][13]   |
+--------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+
|     30 | -B[0][14]   | -B[1][14]    | -B[2][14]   | -B[3][14]    | -B[4][14]   | -B[5][14]    | -B[6][14]   | -B[7][14]    | -B[8][14]   | -B[9][14]    | -B[10][14]  | -B[11][14]   | -B[12][14]  | -B[13][14]   | -B[14][14]  | -B[15][14]   |
+--------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+
|     31 | -B[0][15]   | -B[1][15]    | -B[2][15]   | -B[3][15]    | -B[4][15]   | -B[5][15]    | -B[6][15]   | -B[7][15]    | -B[8][15]   | -B[9][15]    | -B[10][15]  | -B[11][15]   | -B[12][15]  | -B[13][15]   | -B[14][15]  | -B[15][15]   |
+--------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+-------------+--------------+
```

Details of AMD Matrix Multiplication Instruction Encodings
-------------------------------------------------------------------------------------------------------
This section details the encodings used by matrix multiplication instructions in AMD accelerators.

### Details of the VOP3P-MAI Matrix Multiplication Instruction Encoding in the AMD CDNA&trade; 1 - CDNA 3 Architectures
In the CDNA 1, 2, and 3 architectures, matrix multiplications instructions contain "MFMA", or matrix fused multiply add, in their mnemonics.
Instructions that execute matrix multiplication on 4:2 structured sparse matrices contain "SMFMAC", or sparse matrix fused multiply accumulate, in their mnemonics.
This section details the instruction encoding and the configuration fields for these instructions.

These operations are encoded in the VOP3P-MAI (**V**ector **Op** with **3** Inputs, Doing **P**acked Math - **M**atrix **A**rithmetic **I**nstructions) instruction encoding space.
VOP3P-MAI is a subset of the VOP3P (**V**ector **Op** with **3** Inputs, Doing **P**acked Math) encoding space.
The differentiator between these spaces is the upper-most bit in the 7-bit opcode field is always 0 in traditional VOP3P, while it is always 1 in VOP3P-MAI.
This is labelled as bit 54 in the chart below, and bit 22 in the AMD ISA guides.

The reason for this nomenclature difference is because the VOP3P-MAI is a 64b opcode, where bits \[63:32\] are held at bytes address X, and \[31:0\] are at bytes address X+4.
This 64b encoding space is how AMD's software tools, such as `llvm-objdump`, decode these 64b instructions.
Because the software tools will display these as a 64b number, we illustrate the encoding as such in this section.

If looking at these opcodes from the viewpoint of a little-endian architecture that primarily works on 4-byte values, one would swap bits 63:32 with bits 31:00.
This is how the AMD ISA guides present the numbers, because the hardware will decode the first 32b (here denoted as 63:32) before decoding the second 32b (here denoted as 31:0).
As such, in the ISA guides, these are denoted as 31:0 and 63:32, respectively.

The following illustrates the 64b VOP3P-MAI instruction format.
```
-------------------------------------------------------------------------------------------------
|63|62|61|60|59|58|57|56|55|54|53|52|51|50|49|48|47|46|45|44|43|42|41|40|39|38|37|36|35|34|33|32|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|                       |                       |                       |                       |
|                       |     |                 |AC|           |        |                       |
| 1  1  0  1  0  0  1  1| 1  1|  6-bit opcode   |C_|   ABID    |  CBSZ  |          Vdst         |
|                       |     |                 |CD|           |        |                       |
-------------------------------------------------------------------------------------------------
|31|30|29|28|27|26|25|24|23|22|21|20|19|18|17|16|15|14|13|12|11|10|09|08|07|06|05|04|03|02|01|00|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|                       |                       |                       |                       |
|        |     |                          |                          |                          |
|  BLGP  | ACC |           Vsrc2          |           Vsrc1          |           Vsrc0          |
|        |     |                          |                          |                          |
-------------------------------------------------------------------------------------------------
```

The following fields are illustrated:
* 6-bit opcode
    * Bits 53:48
    * This denotes the instruction, and it can be queried from the AMD Matrix Instruction Calculator by looking at the "VOP3P-MAI Opcode" field output by the `--detail-instruction` parameter.
* Vsrc0
    * Bits 8:0
    * Encoding for the A\[\] input matrix
    * This 9-bit field is encoded to denote sending the following into the Matrix Cores for the A\[\] matrix:
        * 0-127: Reserved
        * 128: Literal 0
        * 129-192: Signed integers between 1-64
        * 193-208: Signed integers between -1 and -16
        * 209-239: Reserved
        * 240: 0.5
        * 241: -0.5
        * 242: 1.0
        * 243: -1.0
        * 244: 2.0
        * 245: -2.0
        * 246: 4.0
        * 247: -4.0
        * 248: 1/(2*pi)
        * 249-255: Reserved
        * 256-511: Registers 0-255
    * Only 256 registers can be addressed by this field. The choice of ArchVGPR and AccVGPR is controlled by ACC\[0\], bit 27.
* Vsrc1
    * Bits 17:9
    * Encoding for the B\[\] input matrix
    * This is a 9-bit field that uses the same encoding described above for Vsrc0
    * Only 256 registers can be addressed by this field. The choice of ArchVGPR and AccVGPR is controlled by ACC\[1\], bit 28.
* Vsrc2
    * Bits 26:18
    * Encoding for the C\[\] input matrix
    * This is a 9-bit field that uses the same encoding described above for Vsrc0
    * Only 256 registers can be addressed by this field. The choice of ArchVGPR and AccVGPR is controlled by ACC\_CD, bit 47.
* ACC
    * Bits 28:27
    * Bit 27 configures Vsrc0 VGPR reads to target the ArchVGPR space (when set to 0) or the AccVGPR space (when set to 1)
    * Bit 28 configures Vsrc1 VGPR reads to target the ArchVGPR space (when set to 0) or the AccVGPR space (when set to 1)
* BLGP
    * Bits 31:29
    * Configures the B-matrix Lane Group Pattern parameter
        * Described in detail in the section: [Example of Using the BLGP Modifier to Change the B Matrix Layout](#example-of-using-the-blgp-modifier-to-change-the-b-matrix-layout)
    * For FP64 instructions in the CDNA 3 architecture, this field is used to negate input matrix values.
        * Described in detail in the section: [Example of Using the BLGP Modifier to Negate Input Matrices in the CDNA™ 3 Architecture](#example-of-using-the-blgp-modifier-to-negate-fp64-input-matrices-in-the-cdna-3-architecture)
* Vdst
    * Bits 39:32
    * Encoding for the D\[\] output matrix
    * Uses an 8-bit field that directly addresses the 256 possible target registers.
    * Only 256 registers can be addressed by this field. The choice of ArchVGPR and AccVGPR is controlled by ACC\_CD, bit 47.
* CBSZ
    * Bits 42:40
    * Configures the Control Broadcast Size parameter
    * Described in detail in the section: [Example of Using the CBSZ and ABID Modifiers to Change the A Matrix Layout](#example-of-using-the-cbsz-and-abid-modifiers-to-change-the-a-matrix-layout)
* ABID
    * Bits 46:43
    * Configures the A-matrix Broadcast Identifier parameter
    * Described in detail in the section: [Example of Using the CBSZ and ABID Modifiers to Change the A Matrix Layout](#example-of-using-the-cbsz-and-abid-modifiers-to-change-the-a-matrix-layout)
* ACC\_CD
    * Single bit that controls whether the C\[\] and D\[\] matrices use the ArchVGPR space (when set to 0) or the AccVGPR space (when set to 1)

### Details of the VOP3P Matrix Multiplication Instruction Encoding in the AMD RDNA&trade; 3 and RDNA 4 Architectures
In the RDNA 3 and RDNA 4 architectures, matrix multiplications instructions contain "WMMA", or wave matrix multiply accumulate, in their mnemonics.
Instructions that execute matrix multiplication on 4:2 structured sparse matrices contain "SWMMAC", or sparse wave matrix multiply accumulate, in their mnemonics.
This section details the instruction encoding and the configuration fields for these instructions.

These operations are encoded in the VOP3P (**V**ector **Op** with **3** Inputs, Doing **P**acked Math) instruction encoding space.
This field contains the 6-bit value `110011b` in bits 63:58, as labeled in the figure below.
This 6-bit value is listed as bits 31:26 in the AMD ISA guides.

The reason for this nomenclature difference is because the VOP3P is a 64b opcode, where bits \[63:32\] are held at bytes address X, and \[31:0\] are at bytes address X+4.
This 64b encoding space is how AMD's software tools, such as `llvm-objdump`, decode these 64b instructions.
Because the software tools will display these as a 64b number, we illustrate the encoding as such in this section.

If looking at these opcodes from the viewpoint of a little-endian architecture that primarily works on 4-byte values, one would swap bits 63:32 with bits 31:00.
This is how the AMD ISA guides present the numbers, because the hardware will decode the first 32b (here denoted as 63:32) before decoding the second 32b (here denoted as 31:0).
As such, in the ISA guides, these are denoted as 31:0 and 63:32, respectively.

The following illustrates the 64b VOP3P instruction format.
```
-------------------------------------------------------------------------------------------------
|63|62|61|60|59|58|57|56|55|54|53|52|51|50|49|48|47|46|45|44|43|42|41|40|39|38|37|36|35|34|33|32|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|                       |                       |                       |                       |
|                 |        |                    |CL|OP|        |        |                       |
| 1  1  0  0  1  1| 0  0  0|    7-bit opcode    |AM|SH|  OPSEL | NEG_HI |          Vdst         |
|                 |        |                    |P |I2|        |        |                       |
-------------------------------------------------------------------------------------------------
|31|30|29|28|27|26|25|24|23|22|21|20|19|18|17|16|15|14|13|12|11|10|09|08|07|06|05|04|03|02|01|00|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|                       |                       |                       |                       |
|        |OPSEL|                          |                          |                          |
|   NEG  |HI_01|           Vsrc2          |           Vsrc1          |           Vsrc0          |
|        |     |                          |                          |                          |
-------------------------------------------------------------------------------------------------
```

The following fields are illustrated:
* 7-bit opcode
    * Bits 54:48
    * This denotes the instruction, and it can be queried from the AMD Matrix Instruction Calculator by looking at the "VOP3P Opcode" field output by the `--detail-instruction` parameter.
* Vsrc0
    * Bits 8:0
    * Encoding for the A\[\] input matrix
    * This 9-bit field is encoded to denote sending the following into the Matrix Cores for the A\[\] matrix:
        * 0-255: Reserved
        * 256-511: Registers 0-255
* Vsrc1
    * Bits 17:9
    * Encoding for the B\[\] input matrix
    * This is a 9-bit field that uses the same encoding described above for Vsrc0
* Vsrc2
    * Bits 26:18
    * Encoding for the C\[\] input matrix
    * This is a 9-bit field that allows the use of inline constant values by using values that are reserved for the A\[\] and B\[\] encodings.
        * 0-127: Reserved
        * 128: Literal 0
        * 129-192: Signed integers between 1-64
        * 193-208: Signed integers between -1 and -16
        * 209-239: Reserved
        * 240: 0.5
        * 241: -0.5
        * 242: 1.0
        * 243: -1.0
        * 244: 2.0
        * 245: -2.0
        * 246: 4.0
        * 247: -4.0
        * 248: 1/(2*pi)
        * 249-255: Reserved
        * 256-511: Registers 0-255
* OPSEL\_HI
    * Bits 46, 28:27
        * Bit 46 is OPSEL_HI bit 2
        * Bit 28 is OPSEL_HI bit 1
        * Bit 27 is OPSEL_HI bit 0
    * Unused by WMMA and SWMMAC instructions in RDNA 3 and RDNA 4. Set to 0.
* NEG
    * Bits 31:29
    * Controls the values sent into the matrix multiplication units from the lower halves of each input register.
    * Bit 29:
        * For floating-point WMMA instructions, setting this to 1 will negate the values of the A\[\] matrix which are read from bits 0-15 of Vsrc0
            * This input modifier is not supported by 8-bit floating point operations in the RDNA 4 architecture
        * For integer WMMA instructions, this will indicate whether the integer type in A\[\] is signed or unsigned (0 = unsigned, 1 = signed)
    * Bit 30:
        * For floating-point WMMA instructions, setting this to 1 will negate the values of the B\[\] matrix which are read from bits 0-15 of Vsrc1
            * This input modifier is not supported by 8-bit floating point operations in the RDNA 4 architecture
        * For integer WMMA instructions, this will indicate whether the integer type in B\[\] is signed or unsigned (0 = unsigned, 1 = signed)
    * Bit 31:
        * For floating-point WMMA instructions, setting this to 1 will negate the values of the C\[\] matrix, which are read from Vsrc2
            * This input modifier is not supported by 8-bit floating point operations in the RDNA 4 architecture
        * For integer WMMA instructions, this bit must be zero
    * Described in detail in the section: [Example of Using the NEG Modifier to Negate Input Matrices](#example-of-using-the-neg-modifier-to-negate-input-matrices)
* Vdst
    * Bits 39:32
    * Encoding for the D\[\] output matrix
    * Uses an 8-bit field that directly addresses the 256 possible target registers.
* NEG\_HI
    * Bits 42:40
    * Bit 40:
        * For floating-point WMMA instructions, setting this to 1 will negate the values of the A\[\] matrix which are read from bits 16-31 of Vsrc0
            * This input modifier is not supported by 8-bit floating point operations in the RDNA 4 architecture
        * For integer WMMA instructions, this bit must be zero
    * Bit 41:
        * For floating-point WMMA instructions, setting this to 1 will negate the values of the B\[\] matrix which are read from bits 16-31 of Vsrc1
            * This input modifier is not supported by 8-bit floating point operations in the RDNA 4 architecture
        * For integer WMMA instructions, this bit must be zero
    * Bit 42:
        * For floating-point WMMA instructions, setting this to 1 will take the absolute values of the C\[\] matrix entries which are read from Vsrc2 (absolute value calculation occurs before the negation modifier)
            * This input modifier is not supported by 8-bit floating point operations in the RDNA 4 architecture
        * For integer WMMA instructions, this bit must be zero
* OPSEL
    * Bits 45:43
    * In the RDNA 3 architecture:
        * When working on 16-bit output instructions, `OPSEL[2]` (bit 45) controls whether to read from the low or high 16 bits are read from C\[\] (Vsrc2) and whether the output result is stored in the low or high 16 bits of each entry of D\[\] (Vdst)
        * Bits 44 and 43 are not used by WMMA instructions.
        * Bit 45 is not used by WMMA instructions with a 32-bit output.
        * Described in detail in the section: [Example of Using the OPSEL Modifier to Change the C and D Matrix Storage in the AMD RDNA™ 3 Architecture](#example-of-using-the-opsel-modifier-to-change-the-c-and-d-matrix-storage-in-the-amd-rdna-3-architecture)
    * In the RDNA 4 architecture:
        * In sparse matrix (SWMMAC) instructions, the OPSEL field is used to pick between the available sets of compression index values contained in the Vsrc2 register.
        * Described in detail in the section: [Example of Using the OPSEL Modifier to Change the Compression Index Set in the AMD RDNA™ 4 Architecture](#example-of-using-the-opsel-modifier-to-change-the-compression-index-set-in-the-amd-rdna-4-architecture)
* CLAMP
    * Single bit that controls whether integer output results saturate when they reach the maximum or minimum representable value for that integer type.
    * This field does not affect floating point WMMA operations.

Further Materials
-------------------------------------------------------------------------------------------------------
Further information about the MFMA and WMMA instructions, their data layout, and their capabilities can be found in AMD manuals, code repositories, and blog posts.
This section is meant to contains a list of as many of these documents as possible.

* CDNA&trade; Architecture Information
    * <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/instinct-mi100-cdna1-shader-instruction-set-architecture.pdf>
        * Public ISA guide for AMD Instinct&trade; MI100 Series Accelerators, which use the CDNA 1 ISA
    * <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/instinct-mi200-cdna2-instruction-set-architecture.pdf>
        * Public ISA guide for AMD Instinct MI200 Series Accelerators, which use the CDNA 2 ISA
    * <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf>
        * Public ISA guide for AMD Instinct MI300 Series Accelerators, which use the CDNA 3 ISA
* RDNA&trade; Architecture Information
    * <https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna3-shader-instruction-set-architecture-feb-2023_0.pdf>
        * Public ISA Guide for AMD Radeon&trade; GPUs using the RNDA3 ISA
    * <https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna4-instruction-set-architecture.pdf>
        * Public ISA Guide for AMD Radeon GPUs using the RDNA4 ISA
* AMD Blog Posts
    * <https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-README/>
        * AMD lab notes about AMD Matrix Cores, which covers the use of MFMA instructions in CDNA 1 and CDNA 2 accelerators
    * <https://gpuopen.com/learn/wmma_on_rdna3/>
        * "How to accelerate AI applications on RDNA 3 using WMMA", which covers the use of WMMA instructions on RDNA3 GPUs
* AMD Matrix Code Repositories
    * <https://github.com/amd/amd-lab-notes/tree/release/matrix-cores>
        * Code for examples that go along with the AMD lab notes blog post
        * This code demonstrates how to arrange data and execute various MFMA instructions
    * <https://github.com/ROCm/rocWMMA>
        * A C++ library that provides application-level access to MFMA and WMMA instructions without requiring users to write low-level assembly or compiler intrinsics
        * This library is the recommended path for developers to access these matrix acceleration instructions in the majority of cases

Trademark Attribution
-------------------------------------------------------------------------------------------------------
&copy; 2022-2025 Advanced Micro Devices, Inc. All rights reserved. AMD, the AMD Arrow logo, AMD CDNA, Instinct, Radeon, RDNA, and combinations thereof are trademarks of Advanced Micro Devices, Inc. in the United States and/or other jurisdictions. Other names are for informational purposes only and may be trademarks of their respective owners.
