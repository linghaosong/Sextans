#ifndef F37X_AXI_BUNDLE
#define F37X_AXI_BUNDLE(n) gmem##n
#endif

#ifndef F37X_CONTROL_BUNDLE
#define F37X_CONTROL_BUNDLE control
#endif

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
#pragma HLS INTERFACE m_axi port = edge_list_ptr offset = slave bundle = F37X_AXI_BUNDLE(0)

#pragma HLS INTERFACE m_axi port = edge_list_ch0 offset = slave bundle = F37X_AXI_BUNDLE(1)
#pragma HLS INTERFACE m_axi port = edge_list_ch1 offset = slave bundle = F37X_AXI_BUNDLE(2)
#pragma HLS INTERFACE m_axi port = edge_list_ch2 offset = slave bundle = F37X_AXI_BUNDLE(3)
#pragma HLS INTERFACE m_axi port = edge_list_ch3 offset = slave bundle = F37X_AXI_BUNDLE(4)
#pragma HLS INTERFACE m_axi port = edge_list_ch4 offset = slave bundle = F37X_AXI_BUNDLE(5)
#pragma HLS INTERFACE m_axi port = edge_list_ch5 offset = slave bundle = F37X_AXI_BUNDLE(6)
#pragma HLS INTERFACE m_axi port = edge_list_ch6 offset = slave bundle = F37X_AXI_BUNDLE(7)
#pragma HLS INTERFACE m_axi port = edge_list_ch7 offset = slave bundle = F37X_AXI_BUNDLE(8)

#pragma HLS INTERFACE m_axi port = mat_B_ch0 offset = slave bundle = F37X_AXI_BUNDLE(9)
#pragma HLS INTERFACE m_axi port = mat_B_ch1 offset = slave bundle = F37X_AXI_BUNDLE(10)
#pragma HLS INTERFACE m_axi port = mat_B_ch2 offset = slave bundle = F37X_AXI_BUNDLE(11)
#pragma HLS INTERFACE m_axi port = mat_B_ch3 offset = slave bundle = F37X_AXI_BUNDLE(12)

#pragma HLS INTERFACE m_axi port = mat_C_ch0 offset = slave bundle = F37X_AXI_BUNDLE(13)
#pragma HLS INTERFACE m_axi port = mat_C_ch1 offset = slave bundle = F37X_AXI_BUNDLE(14)
#pragma HLS INTERFACE m_axi port = mat_C_ch2 offset = slave bundle = F37X_AXI_BUNDLE(15)
#pragma HLS INTERFACE m_axi port = mat_C_ch3 offset = slave bundle = F37X_AXI_BUNDLE(16)
#pragma HLS INTERFACE m_axi port = mat_C_ch4 offset = slave bundle = F37X_AXI_BUNDLE(17)
#pragma HLS INTERFACE m_axi port = mat_C_ch5 offset = slave bundle = F37X_AXI_BUNDLE(18)
#pragma HLS INTERFACE m_axi port = mat_C_ch6 offset = slave bundle = F37X_AXI_BUNDLE(19)
#pragma HLS INTERFACE m_axi port = mat_C_ch7 offset = slave bundle = F37X_AXI_BUNDLE(20)

#pragma HLS INTERFACE m_axi port = mat_C_ch0_in offset = slave bundle = F37X_AXI_BUNDLE(21)
#pragma HLS INTERFACE m_axi port = mat_C_ch1_in offset = slave bundle = F37X_AXI_BUNDLE(22)
#pragma HLS INTERFACE m_axi port = mat_C_ch2_in offset = slave bundle = F37X_AXI_BUNDLE(23)
#pragma HLS INTERFACE m_axi port = mat_C_ch3_in offset = slave bundle = F37X_AXI_BUNDLE(24)
#pragma HLS INTERFACE m_axi port = mat_C_ch4_in offset = slave bundle = F37X_AXI_BUNDLE(25)
#pragma HLS INTERFACE m_axi port = mat_C_ch5_in offset = slave bundle = F37X_AXI_BUNDLE(26)
#pragma HLS INTERFACE m_axi port = mat_C_ch6_in offset = slave bundle = F37X_AXI_BUNDLE(27)
#pragma HLS INTERFACE m_axi port = mat_C_ch7_in offset = slave bundle = F37X_AXI_BUNDLE(28)

#pragma HLS INTERFACE s_axilite port = edge_list_ptr bundle = F37X_CONTROL_BUNDLE
#pragma HLS INTERFACE s_axilite port = edge_list_ch0 bundle = F37X_CONTROL_BUNDLE
#pragma HLS INTERFACE s_axilite port = edge_list_ch1 bundle = F37X_CONTROL_BUNDLE
#pragma HLS INTERFACE s_axilite port = edge_list_ch2 bundle = F37X_CONTROL_BUNDLE
#pragma HLS INTERFACE s_axilite port = edge_list_ch3 bundle = F37X_CONTROL_BUNDLE
#pragma HLS INTERFACE s_axilite port = edge_list_ch4 bundle = F37X_CONTROL_BUNDLE
#pragma HLS INTERFACE s_axilite port = edge_list_ch5 bundle = F37X_CONTROL_BUNDLE
#pragma HLS INTERFACE s_axilite port = edge_list_ch6 bundle = F37X_CONTROL_BUNDLE
#pragma HLS INTERFACE s_axilite port = edge_list_ch7 bundle = F37X_CONTROL_BUNDLE

#pragma HLS INTERFACE s_axilite port = mat_B_ch0 bundle = F37X_CONTROL_BUNDLE
#pragma HLS INTERFACE s_axilite port = mat_B_ch1 bundle = F37X_CONTROL_BUNDLE
#pragma HLS INTERFACE s_axilite port = mat_B_ch2 bundle = F37X_CONTROL_BUNDLE
#pragma HLS INTERFACE s_axilite port = mat_B_ch3 bundle = F37X_CONTROL_BUNDLE

#pragma HLS INTERFACE s_axilite port = mat_C_ch0_in bundle = F37X_CONTROL_BUNDLE
#pragma HLS INTERFACE s_axilite port = mat_C_ch1_in bundle = F37X_CONTROL_BUNDLE
#pragma HLS INTERFACE s_axilite port = mat_C_ch2_in bundle = F37X_CONTROL_BUNDLE
#pragma HLS INTERFACE s_axilite port = mat_C_ch3_in bundle = F37X_CONTROL_BUNDLE
#pragma HLS INTERFACE s_axilite port = mat_C_ch4_in bundle = F37X_CONTROL_BUNDLE
#pragma HLS INTERFACE s_axilite port = mat_C_ch5_in bundle = F37X_CONTROL_BUNDLE
#pragma HLS INTERFACE s_axilite port = mat_C_ch6_in bundle = F37X_CONTROL_BUNDLE
#pragma HLS INTERFACE s_axilite port = mat_C_ch7_in bundle = F37X_CONTROL_BUNDLE

#pragma HLS INTERFACE s_axilite port = mat_C_ch0 bundle = F37X_CONTROL_BUNDLE
#pragma HLS INTERFACE s_axilite port = mat_C_ch1 bundle = F37X_CONTROL_BUNDLE
#pragma HLS INTERFACE s_axilite port = mat_C_ch2 bundle = F37X_CONTROL_BUNDLE
#pragma HLS INTERFACE s_axilite port = mat_C_ch3 bundle = F37X_CONTROL_BUNDLE
#pragma HLS INTERFACE s_axilite port = mat_C_ch4 bundle = F37X_CONTROL_BUNDLE
#pragma HLS INTERFACE s_axilite port = mat_C_ch5 bundle = F37X_CONTROL_BUNDLE
#pragma HLS INTERFACE s_axilite port = mat_C_ch6 bundle = F37X_CONTROL_BUNDLE
#pragma HLS INTERFACE s_axilite port = mat_C_ch7 bundle = F37X_CONTROL_BUNDLE

#pragma HLS INTERFACE s_axilite port = NUM_ITE bundle = F37X_CONTROL_BUNDLE
#pragma HLS INTERFACE s_axilite port = NUM_A_LEN bundle = F37X_CONTROL_BUNDLE
#pragma HLS INTERFACE s_axilite port = M bundle = F37X_CONTROL_BUNDLE
#pragma HLS INTERFACE s_axilite port = K bundle = F37X_CONTROL_BUNDLE
#pragma HLS INTERFACE s_axilite port = P_N bundle = F37X_CONTROL_BUNDLE
#pragma HLS INTERFACE s_axilite port = alpha_u bundle = F37X_CONTROL_BUNDLE
#pragma HLS INTERFACE s_axilite port = beta_u bundle = F37X_CONTROL_BUNDLE
#pragma HLS INTERFACE s_axilite port = return bundle = F37X_CONTROL_BUNDLE

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

