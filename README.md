[![DOI](https://zenodo.org/badge/408609900.svg)](https://zenodo.org/badge/latestdoi/408609900)
# Sextans

Sextans is an accelerator for general-purpose Sparse-Matrix Dense-Matrix Multiplication (SpMM). One exciting feature is that we only need to prototype Sextans once, and the hardware supports an arbitrary SpMM. Following are software and hardware dependencies.

+ TAPA + Autobridge, Gurobi
+ Xilinx Vitis 2021.2
+ Alveo U280 HBM FPGA

Dependencies: 
1. TAPA + Autobridge

+ Following [Install TAPA](https://tapa.readthedocs.io/en/release/installation.html) to install TAPA(Autobridge) and Gurobi.
+ Vitis 2021.2
+ Xilinx xilinx_u280_xdma_201920_3 shell and a Xilinx U280 FPGA card.

### Input matrix format & sample input
The host code takes martrix market format(https://math.nist.gov/MatrixMarket/formats.html). We test on sparse matrices from SuiteSparse(https://sparse.tamu.edu) collection. We have two example matices under the folder [matrices](https://github.com/linghaosong/Sextans/tree/tapa/matrices)
    
## To do software emulation:

    mkdir build
    cd build
    cmake ..
    make swsim
if you encounter cmake errors, just run the following under build

    g++ -o sextans -Wno-write-strings -Wunused-result -O2 ../src/sextans.cpp ../src/sextans-host.cpp -ltapa -lfrt -lglog -lgflags -lOpenCL 
To run an example matrix(software emulation), run

    ./sextans ../matrices/nasa4704/nasa4704.mtx 16
    
## To run HLS:

    cd build
    cp ../bitstream/run_tapa.sh ./
    sh run_tapa.sh

After HLS, a bitstream generator file *Sextans_generate_bitstream.sh* is generated under the build folder. Note that we leverages the DSE optition from Autobridge for better Place and Route. We use run-6 to generate hardware.

## To generate bitstream (hardware):

    sh Sextans_generate_bitstream.sh
    
## To run the accelerator on board:
We provide the generated bitstream. If you have a U280 FPGA ready, you don't need to generate the hwardware again, just run

    TAPAB=../bitstream/Sextans_xilinx_u280_xdma_201920_3.xclbin ./sextans ../matrices/nasa4704/nasa4704.mtx 16

To learn more about the techinqual details, please see [this link](https://dl.acm.org/doi/10.1145/3490422.3502357).


If you find this code useful, please cite:

    @inproceedings{song2022sextans,
        author = {Linghao Song and Yuze Chi and Atefeh Sohrabizadeh and Young-kyu Choi and Jason Lau and Jason Cong},
        title = {Sextans: A Streaming Accelerator for General-Purpose Sparse-Matrix Dense-Matrix Multiplication},
        booktitle={Proceedings of the 2022 ACM/SIGDA International Symposium on Field-Programmable Gate Arrays},
        pages={65--77},
        year = {2022}
    }
