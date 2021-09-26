#ifndef SEXTANS_H
#define SEXTANS_H

#include <ap_int.h>
#include <tapa.h>

constexpr int NUM_CH_SPARSE = 8;

constexpr int NUM_CH_B = 4;

constexpr int NUM_CH_C = 8;

void sextans(tapa::mmap<const ap_uint<32>> edge_list_ptr,
             tapa::mmaps<const ap_uint<512>, NUM_CH_SPARSE> edge_list_ch,
             tapa::mmaps<const ap_uint<512>, NUM_CH_B> mat_B_ch,
             tapa::mmaps<ap_uint<512>, NUM_CH_C> mat_C_ch_in,
             tapa::mmaps<ap_uint<512>, NUM_CH_C> mat_C_ch, const int NUM_ITE,
             const int NUM_A_LEN, const int M, const int K, const int P_N,
             const unsigned int alpha_u, const unsigned int beta_u);

#endif  // SEXTANS_H
