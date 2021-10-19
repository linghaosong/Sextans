# Sextans

Sextans is an accelerator for general-purpose Sparse-Matrix Dense-Matrix Multiplication (SpMM). One exciting feature is that we only need to prototype Sextans once, and the hardware supports an arbitrary SpMM. Following are software and hardware dependencies.

+ Xilinx Vitis 2020.2
+ Alveo U280 HBM FPGA
+ Alveo U250 FPGA

To learn more about the techinqual details, please see [this link](https://arxiv.org/abs/2109.11081).


If you find this code useful, please cite:

    @misc{sextans2021,
    Author = {Linghao Song and Yuze Chi and Atefeh Sohrabizadeh and Young-kyu Choi and Jason Lau and Jason Cong},
    Title = {Sextans: A Streaming Accelerator for General-Purpose Sparse-Matrix Dense-Matrix Multiplication},
    Year = {2021},
    Eprint = {arXiv:2109.11081},
    }
