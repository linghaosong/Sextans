.PHONY: help

help::
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make all TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<aarch32/aarch64/x86> SYSROOT=<sysroot_path>"
	$(ECHO) "      Command to generate the design for specified Target and Shell."
	$(ECHO) "      By default, HOST_ARCH=x86. HOST_ARCH and SYSROOT is required for SoC shells"
	$(ECHO) ""
	$(ECHO) "  make clean "
	$(ECHO) "      Command to remove the generated non-hardware files."
	$(ECHO) ""
	$(ECHO) "  make cleanall"
	$(ECHO) "      Command to remove all the generated files."
	$(ECHO) ""
	$(ECHO) "  make sd_card TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<aarch32/aarch64/x86> SYSROOT=<sysroot_path>"
	$(ECHO) "      Command to prepare sd_card files."
	$(ECHO) "      By default, HOST_ARCH=x86. HOST_ARCH and SYSROOT is required for SoC shells"
	$(ECHO) ""
	$(ECHO) "  make check TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<aarch32/aarch64/x86> SYSROOT=<sysroot_path>"
	$(ECHO) "      Command to run application in emulation."
	$(ECHO) "      By default, HOST_ARCH=x86. HOST_ARCH and SYSROOT is required for SoC shells"
	$(ECHO) ""
	$(ECHO) "  make build TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<aarch32/aarch64/x86> SYSROOT=<sysroot_path>"
	$(ECHO) "      Command to build xclbin application."
	$(ECHO) "      By default, HOST_ARCH=x86. HOST_ARCH and SYSROOT is required for SoC shells"
	$(ECHO) ""

# Points to top directory of Git repository
COMMON_REPO = .
PWD = $(shell readlink -f .)
ABS_COMMON_REPO = $(shell readlink -f $(COMMON_REPO))

TARGET := hw
HOST_ARCH := x86
SYSROOT :=
#DEVICE = xilinx_u50_gen3x16_xdma_201920_3
DEVICE = xilinx_u280_xdma_201920_3

include ./utils.mk

XSA := $(call device2xsa, $(DEVICE))
TEMP_DIR := ./_x.$(TARGET).$(XSA)
BUILD_DIR := ./build_dir.$(TARGET).$(XSA)

VPP := v++
SDCARD := sd_card

#Include Libraries
include $(ABS_COMMON_REPO)/common/includes/opencl/opencl.mk
include $(ABS_COMMON_REPO)/common/includes/xcl2/xcl2.mk
CXXFLAGS += $(xcl2_CXXFLAGS)
LDFLAGS += $(xcl2_LDFLAGS)
HOST_SRCS += $(xcl2_SRCS)
CXXFLAGS += $(opencl_CXXFLAGS) -Wall -O0 -g -std=c++11
LDFLAGS += $(opencl_LDFLAGS)

HOST_SRCS += src/host.cpp

# Host compiler global settings
CXXFLAGS += -fmessage-length=0
LDFLAGS += -lrt -lstdc++ -fopenmp

ifneq ($(HOST_ARCH), x86)
	LDFLAGS += --sysroot=$(SYSROOT)
endif

# Kernel compiler global settings
CLFLAGS += -t $(TARGET) --platform $(DEVICE) --save-temps
ifneq ($(TARGET), hw)
	CLFLAGS += -g
endif

#CLFLAGS += --sp sextans_1.m_axi_hbmcsrptr:HBM[0]
#CLFLAGS += --sp sextans_1.m_axi_hbmcsridx:HBM[0]
#CLFLAGS += --sp sextans_1.m_axi_hbmcsrval:HBM[0]

CLFLAGS += --sp sextans_1.m_axi_hbm0:HBM[0]

CLFLAGS += --sp sextans_1.m_axi_hbm1:HBM[1]
CLFLAGS += --sp sextans_1.m_axi_hbm2:HBM[2]
CLFLAGS += --sp sextans_1.m_axi_hbm3:HBM[3]
CLFLAGS += --sp sextans_1.m_axi_hbm4:HBM[4]

CLFLAGS += --sp sextans_1.m_axi_hbm5:HBM[5]
CLFLAGS += --sp sextans_1.m_axi_hbm6:HBM[6]
CLFLAGS += --sp sextans_1.m_axi_hbm7:HBM[7]
CLFLAGS += --sp sextans_1.m_axi_hbm8:HBM[8]

CLFLAGS += --sp sextans_1.m_axi_hbm9:HBM[9]
CLFLAGS += --sp sextans_1.m_axi_hbm10:HBM[10]
CLFLAGS += --sp sextans_1.m_axi_hbm11:HBM[11]
CLFLAGS += --sp sextans_1.m_axi_hbm12:HBM[12]

#CLFLAGS += --sp sextans_1.m_axi_hbm13:HBM[13]
#CLFLAGS += --sp sextans_1.m_axi_hbm14:HBM[14]
#CLFLAGS += --sp sextans_1.m_axi_hbm15:HBM[15]

CLFLAGS += --sp sextans_1.m_axi_hbm16:HBM[16]
CLFLAGS += --sp sextans_1.m_axi_hbm17:HBM[17]
CLFLAGS += --sp sextans_1.m_axi_hbm18:HBM[18]
CLFLAGS += --sp sextans_1.m_axi_hbm19:HBM[19]
CLFLAGS += --sp sextans_1.m_axi_hbm20:HBM[20]
CLFLAGS += --sp sextans_1.m_axi_hbm21:HBM[21]
CLFLAGS += --sp sextans_1.m_axi_hbm22:HBM[22]
CLFLAGS += --sp sextans_1.m_axi_hbm23:HBM[23]

CLFLAGS += --sp sextans_1.m_axi_hbm24:HBM[24]
CLFLAGS += --sp sextans_1.m_axi_hbm25:HBM[25]
CLFLAGS += --sp sextans_1.m_axi_hbm26:HBM[26]
CLFLAGS += --sp sextans_1.m_axi_hbm27:HBM[27]
CLFLAGS += --sp sextans_1.m_axi_hbm28:HBM[28]
CLFLAGS += --sp sextans_1.m_axi_hbm29:HBM[29]
CLFLAGS += --sp sextans_1.m_axi_hbm30:HBM[30]
CLFLAGS += --sp sextans_1.m_axi_hbm31:HBM[31]

CLFLAGS += --report_level 2

EXECUTABLE = host
CMD_ARGS = $(BUILD_DIR)/sextans.xclbin
EMCONFIG_DIR = $(TEMP_DIR)
EMU_DIR = $(SDCARD)/data/emulation

BINARY_CONTAINERS += $(BUILD_DIR)/sextans.xclbin
BINARY_CONTAINER_sextans_OBJS += $(TEMP_DIR)/sextans.xo

CP = cp -rf

.PHONY: all clean cleanall docs emconfig
all: check-devices $(EXECUTABLE) $(BINARY_CONTAINERS) emconfig sd_card

.PHONY: exe
exe: $(EXECUTABLE)

.PHONY: build
build: $(BINARY_CONTAINERS)

# Building kernel
$(TEMP_DIR)/sextans.xo: src/sextans.cpp
	mkdir -p $(TEMP_DIR)
	$(VPP) $(CLFLAGS) --temp_dir $(TEMP_DIR) -c -k sextans -I'$(<D)' -o'$@' '$<'
$(BUILD_DIR)/sextans.xclbin: $(BINARY_CONTAINER_sextans_OBJS)
	mkdir -p $(BUILD_DIR)
	$(VPP) $(CLFLAGS) --temp_dir $(BUILD_DIR) -l $(LDCLFLAGS) -o'$@' $(+)

# Building Host
$(EXECUTABLE): check-xrt $(HOST_SRCS) $(HOST_HDRS)
	$(CXX) $(CXXFLAGS) $(HOST_SRCS) $(HOST_HDRS) -o '$@' $(LDFLAGS)

emconfig:$(EMCONFIG_DIR)/emconfig.json
$(EMCONFIG_DIR)/emconfig.json:
	emconfigutil --platform $(DEVICE) --od $(EMCONFIG_DIR)

check: all
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
ifeq ($(HOST_ARCH), x86)
	$(CP) $(EMCONFIG_DIR)/emconfig.json .
	XCL_EMULATION_MODE=$(TARGET) ./$(EXECUTABLE) $(BUILD_DIR)/sextans.xclbin /home/slh/suitesparse/rgg010/rgg010.mtx 16
else
	mkdir -p $(EMU_DIR)
	$(CP) $(XILINX_VITIS)/data/emulation/unified $(EMU_DIR)
	mkfatimg $(SDCARD) $(SDCARD).img 500000
	launch_emulator -no-reboot -runtime ocl -t $(TARGET) -sd-card-image $(SDCARD).img -device-family $(DEV_FAM)
endif
else
ifeq ($(HOST_ARCH), x86)
	./$(EXECUTABLE) $(BUILD_DIR)/sextans.xclbin
endif
endif
ifeq ($(HOST_ARCH), x86)
	perf_analyze profile -i profile_summary.csv -f html
endif

sd_card: $(EXECUTABLE) $(BINARY_CONTAINERS) emconfig
ifneq ($(HOST_ARCH), x86)
	mkdir -p $(SDCARD)/$(BUILD_DIR)
	$(CP) $(B_NAME)/sw/$(XSA)/boot/generic.readme $(B_NAME)/sw/$(XSA)/xrt/image/* xrt.ini $(EXECUTABLE) $(SDCARD)
	$(CP) $(BUILD_DIR)/*.xclbin $(SDCARD)/$(BUILD_DIR)/
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
	$(ECHO) 'cd /mnt/' >> $(SDCARD)/init.sh
	$(ECHO) 'export XILINX_VITIS=$$PWD' >> $(SDCARD)/init.sh
	$(ECHO) 'export XCL_EMULATION_MODE=$(TARGET)' >> $(SDCARD)/init.sh
	$(ECHO) './$(EXECUTABLE) $(CMD_ARGS)' >> $(SDCARD)/init.sh
	$(ECHO) 'reboot' >> $(SDCARD)/init.sh
else
	[ -f $(SDCARD)/BOOT.BIN ] && echo "INFO: BOOT.BIN already exists" || $(CP) $(BUILD_DIR)/sd_card/BOOT.BIN $(SDCARD)/
	$(ECHO) './$(EXECUTABLE) $(CMD_ARGS)' >> $(SDCARD)/init.sh
endif
endif

# Cleaning stuff
clean:
	-$(RMDIR) $(EXECUTABLE) $(XCLBIN)/{*sw_emu*,*hw_emu*}
	-$(RMDIR) profile_* TempConfig system_estimate.xtxt *.rpt *.csv
	-$(RMDIR) src/*.ll *v++* .Xil emconfig.json dltmp* xmltmp* *.log *.jou *.wcfg *.wdb

cleanall: clean
	-$(RMDIR) build_dir* sd_card*
	-$(RMDIR) _x.* *xclbin.run_summary qemu-memory-_* emulation/ _vimage/ pl* start_simulation.sh *.xclbin
