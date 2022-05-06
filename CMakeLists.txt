cmake_minimum_required(VERSION 3.14)

project(sextans)

set(PLATFORM
    xilinx_u280_xdma_201920_3
    CACHE STRING "Target FPGA platform")

set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -Wno-write-strings")

find_package(TAPA REQUIRED)
find_package(SDx REQUIRED)

add_executable(sextans)
target_sources(sextans PRIVATE src/sextans-host.cpp src/sextans.cpp)
target_link_libraries(sextans PRIVATE tapa::tapa)

add_tapa_target(
  hls
  --enable-synth-util
  INPUT src/sextans.cpp
  TOP Sextans
  CONNECTIVITY ${CMAKE_CURRENT_SOURCE_DIR}/link_config.ini
  CONSTRAINT ${CMAKE_CURRENT_BINARY_DIR}/constraint.tcl
  --enable-hbm-binding-adjustment
  --read-only-args edge_list_ptr
  --read-only-args edge_list_ch*
  --read-only-args mat_B_ch*
  --read-only-args mat_C_ch_in*
  --write-only-args mat_C_ch*
  --max-slr-width-limit 11000
  PLATFORM ${PLATFORM})

add_xocc_hw_link_targets(
  ${CMAKE_CURRENT_BINARY_DIR}
  --config=${CMAKE_CURRENT_SOURCE_DIR}/link_config.ini
  --vivado.prop run.impl_1.STEPS.PHYS_OPT_DESIGN.is_enabled=1
  --vivado.prop run.impl_1.STEPS.OPT_DESIGN.ARGS.DIRECTIVE=Explore
  --vivado.prop run.impl_1.STEPS.PLACE_DESIGN.ARGS.DIRECTIVE=EarlyBlockPlacement
  --vivado.prop run.impl_1.STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE=Explore
  --vivado.prop run.impl_1.STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE=Explore
  --vivado.prop run.impl_1.STEPS.OPT_DESIGN.TCL.PRE=${CMAKE_CURRENT_BINARY_DIR}/constraint.tcl
  INPUT hls
  HW_EMU_XCLBIN hw_emu_xclbin
  HW_XCLBIN hw_xclbin)

add_custom_target(
  swsim
  COMMAND $<TARGET_FILE:sextans> ../matrices/nasa4704/nasa4704.mtx 16
  DEPENDS sextans
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
add_custom_target(
  hwsim
  COMMAND TAPAB=$<TARGET_PROPERTY:${hw_emu_xclbin},FILE_NAME> $<TARGET_FILE:sextans>
          ../matrices/nasa4704/nasa4704.mtx 16
  DEPENDS sextans ${hw_emu_xclbin}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
add_custom_target(
  hw
  COMMAND sh ${CMAKE_CURRENT_BINARY_DIR}/Sextans.xilinx_u280_xdma_201920_3.hw_generate_bitstream.sh
  COMMAND TAPAB=$<TARGET_PROPERTY:${hw_xclbin},FILE_NAME> $<TARGET_FILE:sextans>
          ../matrices/nasa4704/nasa4704.mtx 16
  DEPENDS sextans hls
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
