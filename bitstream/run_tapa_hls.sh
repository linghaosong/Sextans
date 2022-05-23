tapac \
  --work-dir run \
  --top Sextans \
  --platform xilinx_u280_xdma_201920_3 \
  --clock-period 3.33 \
  -o Sextans.xo \
  --constraint Sextans_floorplan.tcl \
  --connectivity ../link_config.ini \
  --read-only-args edge_list_ptr \
  --read-only-args edge_list_ch* \
  --read-only-args mat_B_ch* \
  --read-only-args mat_C_ch_in* \
  --write-only-args mat_C_ch* \
  --enable-synth-util \
  --max-parallel-synth-jobs 16 \
  --enable-hbm-binding-adjustment \
  --run-floorplan-dse \
  ../src/sextans.cpp \
  2>&1 | tee tapa.log
