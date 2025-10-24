# F37X Platform Support

This directory provides the Vivado HLS 2019.1 entry point for building the
Sextans SpMM accelerator as an IP core targeting the **F37X** development
board.  The accelerator keeps the computation micro-architecture identical to
the original Alveo design: matrix tiles are streamed from HBM into on-chip
BRAM/URAM buffers and processed by eight processing elements.

## Key files

- `src/sextans.cpp` – F37X specific top level that maps kernel memories to HBM
  bundles and exposes an AXI4-Lite control interface.  The core computational
  logic is shared through `common/includes/sextans_kernel.hpp`.
- `hls_sextans_f37x.tcl` – Helper script for Vivado HLS 2019.1 that
  synthesises the kernel and exports an IP catalog component.

## Building with Vivado HLS 2019.1

1. Launch the tool from this directory:

   ```bash
   vivado_hls -f hls_sextans_f37x.tcl -tclargs <part_name> [clock_period_ns]
   ```

   - `part_name` should match the FPGA device used on the F37X board.
     A default of `xcvu37p-fsvh2892-2L-e` is used when no argument is
     provided.
   - `clock_period_ns` optionally overrides the default 3.3ns constraint.

2. After synthesis, the generated IP core is placed under the `ip/` directory.

## Customisation options

The F37X wrapper exposes several compile-time knobs:

- Override `F37X_AXI_BUNDLE(n)` to match the HBM AXI port naming used in your
  platform project.
- Override `F37X_CONTROL_BUNDLE` to change the AXI4-Lite control bundle name.
- The shared kernel honours the following optional macros before inclusion:
  `SEXTANS_WINDOW_SIZE`, `SEXTANS_DEP_DIST_LOAD_STORE`,
  `SEXTANS_B_PARTITION_FACTOR` and `SEXTANS_URAM_DEPTH`.

These options make it straightforward to retarget the design without modifying
the kernel implementation.

