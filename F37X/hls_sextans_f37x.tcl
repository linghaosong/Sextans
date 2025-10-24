set default_part "xcvu37p-fsvh2892-2L-e"
set default_period 3.3

if { $argc > 0 } {
    set part_name [lindex $argv 0]
} else {
    set part_name $default_part
}

if { $argc > 1 } {
    set clock_period [lindex $argv 1]
} else {
    set clock_period $default_period
}

open_project -reset sextans_f37x
set_top sextans
add_files src/sextans.cpp
open_solution -reset solution1
set_part $part_name
create_clock -period $clock_period -name default
csynth_design
export_design -format ip_catalog -output ./ip
exit

