#include "../../common/includes/sextans_kernel.hpp"

#ifndef HLS
extern "C" {
#endif

void sextans(
        const ap_uint<32> *edge_list_ptr,

        const ap_uint<512> *edge_list_ch0,
        const ap_uint<512> *edge_list_ch1,
        const ap_uint<512> *edge_list_ch2,
        const ap_uint<512> *edge_list_ch3,
        const ap_uint<512> *edge_list_ch4,
        const ap_uint<512> *edge_list_ch5,
        const ap_uint<512> *edge_list_ch6,
        const ap_uint<512> *edge_list_ch7,

        const ap_uint<512>  *mat_B_ch0,
        const ap_uint<512>  *mat_B_ch1,
        const ap_uint<512>  *mat_B_ch2,
        const ap_uint<512>  *mat_B_ch3,

        ap_uint<512>  *mat_C_ch0_in,
        ap_uint<512>  *mat_C_ch1_in,
        ap_uint<512>  *mat_C_ch2_in,
        ap_uint<512>  *mat_C_ch3_in,
        ap_uint<512>  *mat_C_ch4_in,
        ap_uint<512>  *mat_C_ch5_in,
        ap_uint<512>  *mat_C_ch6_in,
        ap_uint<512>  *mat_C_ch7_in,

        ap_uint<512>  *mat_C_ch0,
        ap_uint<512>  *mat_C_ch1,
        ap_uint<512>  *mat_C_ch2,
        ap_uint<512>  *mat_C_ch3,
        ap_uint<512>  *mat_C_ch4,
        ap_uint<512>  *mat_C_ch5,
        ap_uint<512>  *mat_C_ch6,
        ap_uint<512>  *mat_C_ch7,

        const int NUM_ITE,
        const int NUM_A_LEN,
        const int M,
        const int K,
        const int P_N,
        const unsigned int alpha_u,
        const unsigned int beta_u
) {
#pragma HLS INTERFACE m_axi port = edge_list_ptr offset = slave bundle = hbm0

#pragma HLS INTERFACE m_axi port = edge_list_ch0 offset = slave bundle = hbm1
#pragma HLS INTERFACE m_axi port = edge_list_ch1 offset = slave bundle = hbm2
#pragma HLS INTERFACE m_axi port = edge_list_ch2 offset = slave bundle = hbm3
#pragma HLS INTERFACE m_axi port = edge_list_ch3 offset = slave bundle = hbm4
#pragma HLS INTERFACE m_axi port = edge_list_ch4 offset = slave bundle = hbm5
#pragma HLS INTERFACE m_axi port = edge_list_ch5 offset = slave bundle = hbm6
#pragma HLS INTERFACE m_axi port = edge_list_ch6 offset = slave bundle = hbm7
#pragma HLS INTERFACE m_axi port = edge_list_ch7 offset = slave bundle = hbm8

#pragma HLS INTERFACE m_axi port = mat_B_ch0 offset = slave bundle = hbm9
#pragma HLS INTERFACE m_axi port = mat_B_ch1 offset = slave bundle = hbm10
#pragma HLS INTERFACE m_axi port = mat_B_ch2 offset = slave bundle = hbm11
#pragma HLS INTERFACE m_axi port = mat_B_ch3 offset = slave bundle = hbm12

#pragma HLS INTERFACE m_axi port = mat_C_ch0 offset = slave bundle = hbm16
#pragma HLS INTERFACE m_axi port = mat_C_ch1 offset = slave bundle = hbm17
#pragma HLS INTERFACE m_axi port = mat_C_ch2 offset = slave bundle = hbm18
#pragma HLS INTERFACE m_axi port = mat_C_ch3 offset = slave bundle = hbm19
#pragma HLS INTERFACE m_axi port = mat_C_ch4 offset = slave bundle = hbm20
#pragma HLS INTERFACE m_axi port = mat_C_ch5 offset = slave bundle = hbm21
#pragma HLS INTERFACE m_axi port = mat_C_ch6 offset = slave bundle = hbm22
#pragma HLS INTERFACE m_axi port = mat_C_ch7 offset = slave bundle = hbm23

#pragma HLS INTERFACE m_axi port = mat_C_ch0_in offset = slave bundle = hbm24
#pragma HLS INTERFACE m_axi port = mat_C_ch1_in offset = slave bundle = hbm25
#pragma HLS INTERFACE m_axi port = mat_C_ch2_in offset = slave bundle = hbm26
#pragma HLS INTERFACE m_axi port = mat_C_ch3_in offset = slave bundle = hbm27
#pragma HLS INTERFACE m_axi port = mat_C_ch4_in offset = slave bundle = hbm28
#pragma HLS INTERFACE m_axi port = mat_C_ch5_in offset = slave bundle = hbm29
#pragma HLS INTERFACE m_axi port = mat_C_ch6_in offset = slave bundle = hbm30
#pragma HLS INTERFACE m_axi port = mat_C_ch7_in offset = slave bundle = hbm31

    sextans_kernel(
            edge_list_ptr,
            edge_list_ch0,
            edge_list_ch1,
            edge_list_ch2,
            edge_list_ch3,
            edge_list_ch4,
            edge_list_ch5,
            edge_list_ch6,
            edge_list_ch7,
            mat_B_ch0,
            mat_B_ch1,
            mat_B_ch2,
            mat_B_ch3,
            mat_C_ch0_in,
            mat_C_ch1_in,
            mat_C_ch2_in,
            mat_C_ch3_in,
            mat_C_ch4_in,
            mat_C_ch5_in,
            mat_C_ch6_in,
            mat_C_ch7_in,
            mat_C_ch0,
            mat_C_ch1,
            mat_C_ch2,
            mat_C_ch3,
            mat_C_ch4,
            mat_C_ch5,
            mat_C_ch6,
            mat_C_ch7,
            NUM_ITE,
            NUM_A_LEN,
            M,
            K,
            P_N,
            alpha_u,
            beta_u);
}

#ifndef HLS
}
#endif

