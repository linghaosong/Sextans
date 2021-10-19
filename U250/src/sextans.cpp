#include <ap_int.h>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <hls_stream.h>

//#define DEBUG_PRINT_KERNEL

#ifdef DEBUG_PRINT_KERNEL
#include <iostream>
using std::cout;
using std::endl;
#endif

const int WINDOW_SIZE = 4096;
const int DEP_DIST_LOAD_STORE = 10;
const int B_PARTITION_FACTOR = 4;
const int URAM_DEPTH = 24576;
const int T_FACTOR_C = 768;

template<class T>
T HLS_REG(T in) {
#pragma HLS pipeline II=1
#pragma HLS inline off
#pragma HLS LATENCY min=1 max=1
	return in;
}

float uint32_to_float(ap_uint<32> u) {
#pragma HLS inline
//#pragma HLS pipeline II=1
	float * tmpPointer_v = (float*) & u;
	return (*tmpPointer_v);
}

ap_uint<32> float_to_uint32(float u) {
#pragma HLS inline
//#pragma HLS pipeline II=1
	ap_uint<32> * tmpPointer_v = (ap_uint<32>*) & u;
	return (*tmpPointer_v);
}

void read_edge_list_ptr(
	const ap_uint<32> num_ite_in,
	const ap_uint<32> M_in,
	const ap_uint<32> P_N_in, // bit 31 - 16: repeat time, bit 15 - 0: N
	const ap_uint<32> K_in,
	const ap_uint<32> alpha_u_in,
	const ap_uint<32> beta_u_in,
	const ap_uint<32> *edge_list_ptr,
	hls::stream<ap_uint<32> > & fifo_edge_list_ptr
	) {
#pragma HLS inline off
	const ap_uint<32> num_ite = num_ite_in;
	fifo_edge_list_ptr.write(num_ite);

	const ap_uint<32> M = M_in;
	fifo_edge_list_ptr.write(M);

	const ap_uint<32> N = P_N_in & ((ap_uint<32>) 0x0000FFFF);
	fifo_edge_list_ptr.write(P_N_in);

	const ap_uint<32> K = K_in;
	fifo_edge_list_ptr.write(K);

	fifo_edge_list_ptr.write(alpha_u_in);
	fifo_edge_list_ptr.write(beta_u_in);

	const ap_uint<16> N16 = P_N_in(31, 16);
	const ap_uint<16> rp_time = (N16 == 0)? ((ap_uint<16>) 1) : N16;

	l_rp: for(ap_uint<16> rp = 0; rp < rp_time; rp++) {
#pragma HLS loop_flatten off
#pragma HLS loop_tripcount min=1 max=16
		l_N: for(ap_uint<32> nn = 0; nn < (N + 7)/8; nn++) {
#pragma HLS loop_flatten off
#pragma HLS loop_tripcount min=1 max=16
			rd_ptr: for(ap_uint<32> i = 0; i < num_ite + 1; i++) {
#pragma HLS loop_tripcount min=1 max=800
#pragma HLS pipeline II=1
				ap_uint<32> tmp = edge_list_ptr[i];
				fifo_edge_list_ptr.write(tmp);
			}
		}
	}
}

template <int ch>
void read_A(
	const ap_uint<512> *A,
	hls::stream<ap_uint<512> > & fifo_A,
	const ap_uint<32> A_len,
	const ap_uint<32> P_N
	) {
#pragma HLS inline off
	const ap_uint<16> N16 = P_N(31, 16);
	const ap_uint<16> rp_time = (N16 == 0)? ((ap_uint<16>) 1) : N16;
	const ap_uint<32> N = P_N & ((ap_uint<32>) 0x0000FFFF);

	l_rp: for(ap_uint<16> rp = 0; rp < rp_time; rp++) {
#pragma HLS loop_flatten off
#pragma HLS loop_tripcount min=1 max=16
		l_N: for(ap_uint<32> nn = 0; nn < (N + 7)/8; nn++) {
#pragma HLS loop_flatten off
#pragma HLS loop_tripcount min=1 max=16
			rd_A: for(ap_uint<32> i = 0; i < A_len; i++) {
#pragma HLS loop_tripcount min=1 max=10000
#pragma HLS pipeline II=1
				ap_uint<512> tmp_A = A[i];
				fifo_A.write(tmp_A);
			}
		}
	}
}

template <int chb>
void read_B(
	const ap_uint<512>* B,
	hls::stream<ap_uint<512> > & fifo_B,
	const ap_uint<32> K,
	const ap_uint<32> P_N
	) {
#pragma HLS inline off
	const ap_uint<16> N16 = P_N(31, 16);
	const ap_uint<16> rp_time = (N16 == 0)? ((ap_uint<16>) 1) : N16;
	const ap_uint<32> N = P_N & ((ap_uint<32>) 0x0000FFFF);	
	const ap_uint<32> num_ite_B = ((K + 7) / 8) * ((N + 7) / 8);

	l_rp: for(ap_uint<16> rp = 0; rp < rp_time; rp++) {
#pragma HLS loop_flatten off
#pragma HLS loop_tripcount min=1 max=16
		rd_B: for(ap_uint<32> i = 0; i < num_ite_B; i++) {
#pragma HLS loop_tripcount min=1 max=500000
#pragma HLS pipeline II=1
			ap_uint<512> tmp_B = B[i];
			fifo_B.write(tmp_B);
		}
	}
}

template <int ch>
void PU2core(
	ap_uint<18> & addr_c,
	float & a_val_f,
	float & b_val_d0_f,
	float & b_val_d1_f,

	ap_uint<64> * local_C_pe0_d0_d1
	) {
#pragma HLS inline
	ap_uint<64> c_val_d0_d1_u64 = local_C_pe0_d0_d1[addr_c];

	ap_uint<32> c_val_d0_u = c_val_d0_d1_u64(31,  0);
	ap_uint<32> c_val_d1_u = c_val_d0_d1_u64(63, 32);

	float c_val_d0_f = uint32_to_float(c_val_d0_u);
	float c_val_d1_f = uint32_to_float(c_val_d1_u);

	c_val_d0_f += HLS_REG(a_val_f) * b_val_d0_f;
	c_val_d1_f += HLS_REG(a_val_f) * b_val_d1_f;

	c_val_d0_u = float_to_uint32(c_val_d0_f);
	c_val_d1_u = float_to_uint32(c_val_d1_f);

	c_val_d0_d1_u64(31,  0) = c_val_d0_u;
	c_val_d0_d1_u64(63, 32) = c_val_d1_u;

	local_C_pe0_d0_d1[addr_c] = c_val_d0_d1_u64;
}

template <int ch>
void PEcore(
	ap_uint<14> & addr_b,
	ap_uint<18> & addr_c,
	ap_uint<32> & a_val_u,

	ap_uint<64> * local_C_pe0_d0_d1,
	ap_uint<64> * local_C_pe0_d2_d3,
	ap_uint<64> * local_C_pe0_d4_d5,
	ap_uint<64> * local_C_pe0_d6_d7,

	ap_uint<32> * local_B_pe0_pe1_d0,
	ap_uint<32> * local_B_pe0_pe1_d1,
	ap_uint<32> * local_B_pe0_pe1_d2,
	ap_uint<32> * local_B_pe0_pe1_d3,
	ap_uint<32> * local_B_pe0_pe1_d4,
	ap_uint<32> * local_B_pe0_pe1_d5,
	ap_uint<32> * local_B_pe0_pe1_d6,
	ap_uint<32> * local_B_pe0_pe1_d7
	) {
#pragma HLS inline
	if (addr_c != ((ap_uint<18>) 0x3FFFF)) {
		float a_val_f = uint32_to_float(a_val_u);

		ap_uint<32> b_val_d0_u = local_B_pe0_pe1_d0[addr_b];
		ap_uint<32> b_val_d1_u = local_B_pe0_pe1_d1[addr_b];
		ap_uint<32> b_val_d2_u = local_B_pe0_pe1_d2[addr_b];
		ap_uint<32> b_val_d3_u = local_B_pe0_pe1_d3[addr_b];
		ap_uint<32> b_val_d4_u = local_B_pe0_pe1_d4[addr_b];
		ap_uint<32> b_val_d5_u = local_B_pe0_pe1_d5[addr_b];
		ap_uint<32> b_val_d6_u = local_B_pe0_pe1_d6[addr_b];
		ap_uint<32> b_val_d7_u = local_B_pe0_pe1_d7[addr_b];

		float b_val_d0_f = uint32_to_float(b_val_d0_u);
		float b_val_d1_f = uint32_to_float(b_val_d1_u);
		float b_val_d2_f = uint32_to_float(b_val_d2_u);
		float b_val_d3_f = uint32_to_float(b_val_d3_u);
		float b_val_d4_f = uint32_to_float(b_val_d4_u);
		float b_val_d5_f = uint32_to_float(b_val_d5_u);
		float b_val_d6_f = uint32_to_float(b_val_d6_u);
		float b_val_d7_f = uint32_to_float(b_val_d7_u);

		PU2core<0>(
			addr_c,
			a_val_f,
			b_val_d0_f,
			b_val_d1_f,
			local_C_pe0_d0_d1
		);

		PU2core<1>(
			addr_c,
			a_val_f,
			b_val_d2_f,
			b_val_d3_f,
			local_C_pe0_d2_d3
		);

		PU2core<2>(
			addr_c,
			a_val_f,
			b_val_d4_f,
			b_val_d5_f,
			local_C_pe0_d4_d5
		);

		PU2core<3>(
			addr_c,
			a_val_f,
			b_val_d6_f,
			b_val_d7_f,
			local_C_pe0_d6_d7
		);
	}
}

void peg16mult(
	ap_uint<512> opa512,
	ap_uint<32> alpha_u,
	ap_uint<512> & mult512
	) {
#pragma HLS inline
//#pragma HLS pipeline II=1
		float alpha_f = uint32_to_float(alpha_u);
		ap_uint<512> c_out;

		float op_a[16];
#pragma HLS array_partition variable=op_a complete
		float op_result[16];
#pragma HLS array_partition variable=op_result complete

		for(ap_uint<5> p = 0; p < 16; ++p) {
			op_a[p]      = uint32_to_float(opa512(31 + p * 32, p * 32));
			op_result[p] = HLS_REG(alpha_f) * op_a[p];
			c_out(31 + p * 32, p * 32) = float_to_uint32(op_result[p]);
		}
		mult512 = HLS_REG(c_out);
}

template <int ch>
void PEG(
	hls::stream<ap_uint<32> > & fifo_inst,
	hls::stream<ap_uint<512> > & fifo_A,
	hls::stream<ap_uint<512> > & fifo_B_x0, // [32bits * 8 (256)] * 2
	hls::stream<ap_uint<512> > & fifo_B_x1, // [32bits * 8 (256)] * 2
	hls::stream<ap_uint<512> > & fifo_B_x2, // [32bits * 8 (256)] * 2
	hls::stream<ap_uint<512> > & fifo_B_x3, // [32bits * 8 (256)] * 2
	hls::stream<ap_uint<32> > & fifo_inst_out, // to next PE
	hls::stream<ap_uint<512> > & fifo_B_out_x0, // output to next PE [32bits * 8 (256)] * 2
	hls::stream<ap_uint<512> > & fifo_B_out_x1, // output to next PE [32bits * 8 (256)] * 2
	hls::stream<ap_uint<512> > & fifo_B_out_x2, // output to next PE [32bits * 8 (256)] * 2
	hls::stream<ap_uint<512> > & fifo_B_out_x3, // output to next PE [32bits * 8 (256)] * 2
	hls::stream<ap_uint<512> > & fifo_C_out0 // [256(32bits * 8 d)] * 2
	) {
#pragma HLS inline off
	ap_uint<32> NUM_ITE;
	ap_uint<32> M;
	ap_uint<32> P_N;
	ap_uint<32> K;
	ap_uint<32> alpha_u;
	ap_uint<32> beta_u;

	ap_uint<32> parameter;
	w_ITE: for (ap_uint<3> i = 0; i < 6; ) {
#pragma HLS loop_tripcount min=1 max=10
#pragma HLS pipeline II=1
		bool parameter_ready = fifo_inst.read_nb(parameter);
		if (parameter_ready) {
			ap_uint<32> parameter_dealy = HLS_REG(parameter);
			switch (i) {
				case 0: NUM_ITE = parameter_dealy; break;
				case 1: M = parameter_dealy; break;
				case 2: P_N = parameter_dealy; break;
				case 3: K = parameter_dealy; break;
				case 4: alpha_u = parameter_dealy; break;
				case 5: beta_u = parameter_dealy; break;
			}
			++i;
			fifo_inst_out.write(parameter_dealy);
		}
	}

	const ap_uint<16> N16 = P_N(31, 16);
	const ap_uint<16> rp_time = (N16 == 0)? ((ap_uint<16>) 1) : N16;
	const ap_uint<32> N = P_N & ((ap_uint<32>) 0x0000FFFF);

	ap_uint<512> MN512;
	MN512(31,  0) = M;
	MN512(63, 32) = P_N;
	MN512(95, 64) = beta_u;
	fifo_C_out0.write(MN512);

	//define local C buffer and pragma to URAM
	ap_uint<64> local_C_pe0_d0_d1[URAM_DEPTH];
	ap_uint<64> local_C_pe0_d2_d3[URAM_DEPTH];
	ap_uint<64> local_C_pe0_d4_d5[URAM_DEPTH];
	ap_uint<64> local_C_pe0_d6_d7[URAM_DEPTH];

#pragma HLS bind_storage variable=local_C_pe0_d0_d1 type=RAM_2P impl=URAM
#pragma HLS bind_storage variable=local_C_pe0_d2_d3 type=RAM_2P impl=URAM
#pragma HLS bind_storage variable=local_C_pe0_d4_d5 type=RAM_2P impl=URAM
#pragma HLS bind_storage variable=local_C_pe0_d6_d7 type=RAM_2P impl=URAM

	ap_uint<64> local_C_pe1_d0_d1[URAM_DEPTH];
	ap_uint<64> local_C_pe1_d2_d3[URAM_DEPTH];
	ap_uint<64> local_C_pe1_d4_d5[URAM_DEPTH];
	ap_uint<64> local_C_pe1_d6_d7[URAM_DEPTH];

#pragma HLS bind_storage variable=local_C_pe1_d0_d1 type=RAM_2P impl=URAM
#pragma HLS bind_storage variable=local_C_pe1_d2_d3 type=RAM_2P impl=URAM
#pragma HLS bind_storage variable=local_C_pe1_d4_d5 type=RAM_2P impl=URAM
#pragma HLS bind_storage variable=local_C_pe1_d6_d7 type=RAM_2P impl=URAM

	ap_uint<64> local_C_pe2_d0_d1[URAM_DEPTH];
	ap_uint<64> local_C_pe2_d2_d3[URAM_DEPTH];
	ap_uint<64> local_C_pe2_d4_d5[URAM_DEPTH];
	ap_uint<64> local_C_pe2_d6_d7[URAM_DEPTH];

#pragma HLS bind_storage variable=local_C_pe2_d0_d1 type=RAM_2P impl=URAM
#pragma HLS bind_storage variable=local_C_pe2_d2_d3 type=RAM_2P impl=URAM
#pragma HLS bind_storage variable=local_C_pe2_d4_d5 type=RAM_2P impl=URAM
#pragma HLS bind_storage variable=local_C_pe2_d6_d7 type=RAM_2P impl=URAM

	ap_uint<64> local_C_pe3_d0_d1[URAM_DEPTH];
	ap_uint<64> local_C_pe3_d2_d3[URAM_DEPTH];
	ap_uint<64> local_C_pe3_d4_d5[URAM_DEPTH];
	ap_uint<64> local_C_pe3_d6_d7[URAM_DEPTH];

#pragma HLS bind_storage variable=local_C_pe3_d0_d1 type=RAM_2P impl=URAM
#pragma HLS bind_storage variable=local_C_pe3_d2_d3 type=RAM_2P impl=URAM
#pragma HLS bind_storage variable=local_C_pe3_d4_d5 type=RAM_2P impl=URAM
#pragma HLS bind_storage variable=local_C_pe3_d6_d7 type=RAM_2P impl=URAM

	ap_uint<64> local_C_pe4_d0_d1[URAM_DEPTH];
	ap_uint<64> local_C_pe4_d2_d3[URAM_DEPTH];
	ap_uint<64> local_C_pe4_d4_d5[URAM_DEPTH];
	ap_uint<64> local_C_pe4_d6_d7[URAM_DEPTH];

#pragma HLS bind_storage variable=local_C_pe4_d0_d1 type=RAM_2P impl=URAM
#pragma HLS bind_storage variable=local_C_pe4_d2_d3 type=RAM_2P impl=URAM
#pragma HLS bind_storage variable=local_C_pe4_d4_d5 type=RAM_2P impl=URAM
#pragma HLS bind_storage variable=local_C_pe4_d6_d7 type=RAM_2P impl=URAM

	ap_uint<64> local_C_pe5_d0_d1[URAM_DEPTH];
	ap_uint<64> local_C_pe5_d2_d3[URAM_DEPTH];
	ap_uint<64> local_C_pe5_d4_d5[URAM_DEPTH];
	ap_uint<64> local_C_pe5_d6_d7[URAM_DEPTH];

#pragma HLS bind_storage variable=local_C_pe5_d0_d1 type=RAM_2P impl=URAM
#pragma HLS bind_storage variable=local_C_pe5_d2_d3 type=RAM_2P impl=URAM
#pragma HLS bind_storage variable=local_C_pe5_d4_d5 type=RAM_2P impl=URAM
#pragma HLS bind_storage variable=local_C_pe5_d6_d7 type=RAM_2P impl=URAM

	ap_uint<64> local_C_pe6_d0_d1[URAM_DEPTH];
	ap_uint<64> local_C_pe6_d2_d3[URAM_DEPTH];
	ap_uint<64> local_C_pe6_d4_d5[URAM_DEPTH];
	ap_uint<64> local_C_pe6_d6_d7[URAM_DEPTH];

#pragma HLS bind_storage variable=local_C_pe6_d0_d1 type=RAM_2P impl=URAM
#pragma HLS bind_storage variable=local_C_pe6_d2_d3 type=RAM_2P impl=URAM
#pragma HLS bind_storage variable=local_C_pe6_d4_d5 type=RAM_2P impl=URAM
#pragma HLS bind_storage variable=local_C_pe6_d6_d7 type=RAM_2P impl=URAM

	ap_uint<64> local_C_pe7_d0_d1[URAM_DEPTH];
	ap_uint<64> local_C_pe7_d2_d3[URAM_DEPTH];
	ap_uint<64> local_C_pe7_d4_d5[URAM_DEPTH];
	ap_uint<64> local_C_pe7_d6_d7[URAM_DEPTH];

#pragma HLS bind_storage variable=local_C_pe7_d0_d1 type=RAM_2P impl=URAM
#pragma HLS bind_storage variable=local_C_pe7_d2_d3 type=RAM_2P impl=URAM
#pragma HLS bind_storage variable=local_C_pe7_d4_d5 type=RAM_2P impl=URAM
#pragma HLS bind_storage variable=local_C_pe7_d6_d7 type=RAM_2P impl=URAM

	l_rp: for(ap_uint<16> rp = 0; rp < rp_time; rp++) {
#pragma HLS loop_flatten off
#pragma HLS loop_tripcount min=1 max=16
		l_N: for(ap_uint<32> nn = 0; nn < (N + 7)/8; nn++) {
#pragma HLS loop_flatten off
#pragma HLS loop_tripcount min=1 max=16

			//init local C
			init_C: for (ap_uint<32> i = 0; i < ((M + 31) / 32); ++i) {
#pragma HLS loop_tripcount min=1 max=800
#pragma HLS pipeline II=1
				local_C_pe0_d0_d1[i] = 0;
				local_C_pe0_d2_d3[i] = 0;
				local_C_pe0_d4_d5[i] = 0;
				local_C_pe0_d6_d7[i] = 0;
				local_C_pe1_d0_d1[i] = 0;
				local_C_pe1_d2_d3[i] = 0;
				local_C_pe1_d4_d5[i] = 0;
				local_C_pe1_d6_d7[i] = 0;
				local_C_pe2_d0_d1[i] = 0;
				local_C_pe2_d2_d3[i] = 0;
				local_C_pe2_d4_d5[i] = 0;
				local_C_pe2_d6_d7[i] = 0;
				local_C_pe3_d0_d1[i] = 0;
				local_C_pe3_d2_d3[i] = 0;
				local_C_pe3_d4_d5[i] = 0;
				local_C_pe3_d6_d7[i] = 0;
				local_C_pe4_d0_d1[i] = 0;
				local_C_pe4_d2_d3[i] = 0;
				local_C_pe4_d4_d5[i] = 0;
				local_C_pe4_d6_d7[i] = 0;
				local_C_pe5_d0_d1[i] = 0;
				local_C_pe5_d2_d3[i] = 0;
				local_C_pe5_d4_d5[i] = 0;
				local_C_pe5_d6_d7[i] = 0;
				local_C_pe6_d0_d1[i] = 0;
				local_C_pe6_d2_d3[i] = 0;
				local_C_pe6_d4_d5[i] = 0;
				local_C_pe6_d6_d7[i] = 0;
				local_C_pe7_d0_d1[i] = 0;
				local_C_pe7_d2_d3[i] = 0;
				local_C_pe7_d4_d5[i] = 0;
				local_C_pe7_d6_d7[i] = 0;
			}
			//define local B buffer and pragma local B buffer if partition factor > 1

			ap_uint<32> local_B_pe0_pe1_d0[WINDOW_SIZE];
			ap_uint<32> local_B_pe0_pe1_d1[WINDOW_SIZE];
			ap_uint<32> local_B_pe0_pe1_d2[WINDOW_SIZE];
			ap_uint<32> local_B_pe0_pe1_d3[WINDOW_SIZE];
			ap_uint<32> local_B_pe0_pe1_d4[WINDOW_SIZE];
			ap_uint<32> local_B_pe0_pe1_d5[WINDOW_SIZE];
			ap_uint<32> local_B_pe0_pe1_d6[WINDOW_SIZE];
			ap_uint<32> local_B_pe0_pe1_d7[WINDOW_SIZE];

#pragma HLS bind_storage variable=local_B_pe0_pe1_d0 latency=3
#pragma HLS bind_storage variable=local_B_pe0_pe1_d1 latency=3
#pragma HLS bind_storage variable=local_B_pe0_pe1_d2 latency=3
#pragma HLS bind_storage variable=local_B_pe0_pe1_d3 latency=3
#pragma HLS bind_storage variable=local_B_pe0_pe1_d4 latency=3
#pragma HLS bind_storage variable=local_B_pe0_pe1_d5 latency=3
#pragma HLS bind_storage variable=local_B_pe0_pe1_d6 latency=3
#pragma HLS bind_storage variable=local_B_pe0_pe1_d7 latency=3

#pragma HLS array_partition variable=local_B_pe0_pe1_d0 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe0_pe1_d1 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe0_pe1_d2 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe0_pe1_d3 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe0_pe1_d4 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe0_pe1_d5 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe0_pe1_d6 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe0_pe1_d7 cyclic factor=B_PARTITION_FACTOR

			ap_uint<32> local_B_pe2_pe3_d0[WINDOW_SIZE];
			ap_uint<32> local_B_pe2_pe3_d1[WINDOW_SIZE];
			ap_uint<32> local_B_pe2_pe3_d2[WINDOW_SIZE];
			ap_uint<32> local_B_pe2_pe3_d3[WINDOW_SIZE];
			ap_uint<32> local_B_pe2_pe3_d4[WINDOW_SIZE];
			ap_uint<32> local_B_pe2_pe3_d5[WINDOW_SIZE];
			ap_uint<32> local_B_pe2_pe3_d6[WINDOW_SIZE];
			ap_uint<32> local_B_pe2_pe3_d7[WINDOW_SIZE];

#pragma HLS bind_storage variable=local_B_pe2_pe3_d0 latency=3
#pragma HLS bind_storage variable=local_B_pe2_pe3_d1 latency=3
#pragma HLS bind_storage variable=local_B_pe2_pe3_d2 latency=3
#pragma HLS bind_storage variable=local_B_pe2_pe3_d3 latency=3
#pragma HLS bind_storage variable=local_B_pe2_pe3_d4 latency=3
#pragma HLS bind_storage variable=local_B_pe2_pe3_d5 latency=3
#pragma HLS bind_storage variable=local_B_pe2_pe3_d6 latency=3
#pragma HLS bind_storage variable=local_B_pe2_pe3_d7 latency=3

#pragma HLS array_partition variable=local_B_pe2_pe3_d0 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe2_pe3_d1 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe2_pe3_d2 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe2_pe3_d3 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe2_pe3_d4 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe2_pe3_d5 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe2_pe3_d6 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe2_pe3_d7 cyclic factor=B_PARTITION_FACTOR

			ap_uint<32> local_B_pe4_pe5_d0[WINDOW_SIZE];
			ap_uint<32> local_B_pe4_pe5_d1[WINDOW_SIZE];
			ap_uint<32> local_B_pe4_pe5_d2[WINDOW_SIZE];
			ap_uint<32> local_B_pe4_pe5_d3[WINDOW_SIZE];
			ap_uint<32> local_B_pe4_pe5_d4[WINDOW_SIZE];
			ap_uint<32> local_B_pe4_pe5_d5[WINDOW_SIZE];
			ap_uint<32> local_B_pe4_pe5_d6[WINDOW_SIZE];
			ap_uint<32> local_B_pe4_pe5_d7[WINDOW_SIZE];

#pragma HLS bind_storage variable=local_B_pe4_pe5_d0 latency=3
#pragma HLS bind_storage variable=local_B_pe4_pe5_d1 latency=3
#pragma HLS bind_storage variable=local_B_pe4_pe5_d2 latency=3
#pragma HLS bind_storage variable=local_B_pe4_pe5_d3 latency=3
#pragma HLS bind_storage variable=local_B_pe4_pe5_d4 latency=3
#pragma HLS bind_storage variable=local_B_pe4_pe5_d5 latency=3
#pragma HLS bind_storage variable=local_B_pe4_pe5_d6 latency=3
#pragma HLS bind_storage variable=local_B_pe4_pe5_d7 latency=3

#pragma HLS array_partition variable=local_B_pe4_pe5_d0 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe4_pe5_d1 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe4_pe5_d2 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe4_pe5_d3 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe4_pe5_d4 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe4_pe5_d5 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe4_pe5_d6 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe4_pe5_d7 cyclic factor=B_PARTITION_FACTOR

			ap_uint<32> local_B_pe6_pe7_d0[WINDOW_SIZE];
			ap_uint<32> local_B_pe6_pe7_d1[WINDOW_SIZE];
			ap_uint<32> local_B_pe6_pe7_d2[WINDOW_SIZE];
			ap_uint<32> local_B_pe6_pe7_d3[WINDOW_SIZE];
			ap_uint<32> local_B_pe6_pe7_d4[WINDOW_SIZE];
			ap_uint<32> local_B_pe6_pe7_d5[WINDOW_SIZE];
			ap_uint<32> local_B_pe6_pe7_d6[WINDOW_SIZE];
			ap_uint<32> local_B_pe6_pe7_d7[WINDOW_SIZE];

#pragma HLS bind_storage variable=local_B_pe6_pe7_d0 latency=3
#pragma HLS bind_storage variable=local_B_pe6_pe7_d1 latency=3
#pragma HLS bind_storage variable=local_B_pe6_pe7_d2 latency=3
#pragma HLS bind_storage variable=local_B_pe6_pe7_d3 latency=3
#pragma HLS bind_storage variable=local_B_pe6_pe7_d4 latency=3
#pragma HLS bind_storage variable=local_B_pe6_pe7_d5 latency=3
#pragma HLS bind_storage variable=local_B_pe6_pe7_d6 latency=3
#pragma HLS bind_storage variable=local_B_pe6_pe7_d7 latency=3

#pragma HLS array_partition variable=local_B_pe6_pe7_d0 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe6_pe7_d1 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe6_pe7_d2 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe6_pe7_d3 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe6_pe7_d4 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe6_pe7_d5 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe6_pe7_d6 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe6_pe7_d7 cyclic factor=B_PARTITION_FACTOR

			ap_uint<32> start_32_in;
			bool start_32_in_ready = false;
			w1: while(!start_32_in_ready) {
#pragma HLS loop_tripcount min=1 max=10
#pragma HLS pipeline II=1
				start_32_in_ready = fifo_inst.read_nb(start_32_in);
			}
			ap_uint<32> start_32 = HLS_REG(start_32_in);

			fifo_inst_out.write(start_32);

			main: for (ap_uint<32> i = 0; i < NUM_ITE; ++i) {
#pragma HLS loop_tripcount min=1 max=49

				// fill onchip B
				ap_uint<512> b_512_x0;
				ap_uint<512> b_512_x1;
				ap_uint<512> b_512_x2;
				ap_uint<512> b_512_x3;

				bool b_512_x0_ready = false;
				bool b_512_x1_ready = false;
				bool b_512_x2_ready = false;
				bool b_512_x3_ready = false;

				read_B: for (ap_uint<14> j = 0; (j < WINDOW_SIZE/8) && (j < (K + 7) / 8 - i*WINDOW_SIZE/8); ) {
#pragma HLS loop_tripcount min=1 max=512
#pragma HLS pipeline II=1
					if (!b_512_x0_ready) {
						b_512_x0_ready = fifo_B_x0.read_nb(b_512_x0);
					}
					if (!b_512_x1_ready) {
						b_512_x1_ready = fifo_B_x1.read_nb(b_512_x1);
					}
					if (!b_512_x2_ready) {
						b_512_x2_ready = fifo_B_x2.read_nb(b_512_x2);
					}
					if (!b_512_x3_ready) {
						b_512_x3_ready = fifo_B_x3.read_nb(b_512_x3);
					}

					bool b_2048_ready = b_512_x0_ready && b_512_x1_ready && b_512_x2_ready && b_512_x3_ready;

					if (b_2048_ready) {
						ap_uint<512> b_512_x0_delay = HLS_REG(HLS_REG(b_512_x0));
						ap_uint<512> b_512_x1_delay = HLS_REG(HLS_REG(b_512_x1));
						ap_uint<512> b_512_x2_delay = HLS_REG(HLS_REG(b_512_x2));
						ap_uint<512> b_512_x3_delay = HLS_REG(HLS_REG(b_512_x3));

						fifo_B_out_x0.write(b_512_x0_delay);
						fifo_B_out_x1.write(b_512_x1_delay);
						fifo_B_out_x2.write(b_512_x2_delay);
						fifo_B_out_x3.write(b_512_x3_delay);

						ap_uint<256> b_d_seg[8]; // d 0 - 7, 256bits
#pragma HLS array_partition variable=b_d_seg complete

						b_d_seg[0] = b_512_x0_delay(255,   0);
						b_d_seg[1] = b_512_x0_delay(511, 256);
						b_d_seg[2] = b_512_x1_delay(255,   0);
						b_d_seg[3] = b_512_x1_delay(511, 256);
						b_d_seg[4] = b_512_x2_delay(255,   0);
						b_d_seg[5] = b_512_x2_delay(511, 256);
						b_d_seg[6] = b_512_x3_delay(255,   0);
						b_d_seg[7] = b_512_x3_delay(511, 256);

						read_B_p: for (ap_uint<4> k = 0; k < 8; ++k) {
							ap_uint<32> b_pe_d0 = (b_d_seg[k])( 31,   0);
							ap_uint<32> b_pe_d1 = (b_d_seg[k])( 63,  32);
							ap_uint<32> b_pe_d2 = (b_d_seg[k])( 95,  64);
							ap_uint<32> b_pe_d3 = (b_d_seg[k])(127,  96);
							ap_uint<32> b_pe_d4 = (b_d_seg[k])(159, 128);
							ap_uint<32> b_pe_d5 = (b_d_seg[k])(191, 160);
							ap_uint<32> b_pe_d6 = (b_d_seg[k])(223, 192);
							ap_uint<32> b_pe_d7 = (b_d_seg[k])(255, 224);

							local_B_pe0_pe1_d0[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d0;
							local_B_pe0_pe1_d1[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d1;
							local_B_pe0_pe1_d2[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d2;
							local_B_pe0_pe1_d3[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d3;
							local_B_pe0_pe1_d4[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d4;
							local_B_pe0_pe1_d5[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d5;
							local_B_pe0_pe1_d6[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d6;
							local_B_pe0_pe1_d7[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d7;

							local_B_pe2_pe3_d0[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d0;
							local_B_pe2_pe3_d1[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d1;
							local_B_pe2_pe3_d2[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d2;
							local_B_pe2_pe3_d3[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d3;
							local_B_pe2_pe3_d4[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d4;
							local_B_pe2_pe3_d5[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d5;
							local_B_pe2_pe3_d6[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d6;
							local_B_pe2_pe3_d7[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d7;

							local_B_pe4_pe5_d0[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d0;
							local_B_pe4_pe5_d1[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d1;
							local_B_pe4_pe5_d2[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d2;
							local_B_pe4_pe5_d3[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d3;
							local_B_pe4_pe5_d4[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d4;
							local_B_pe4_pe5_d5[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d5;
							local_B_pe4_pe5_d6[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d6;
							local_B_pe4_pe5_d7[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d7;

							local_B_pe6_pe7_d0[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d0;
							local_B_pe6_pe7_d1[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d1;
							local_B_pe6_pe7_d2[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d2;
							local_B_pe6_pe7_d3[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d3;
							local_B_pe6_pe7_d4[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d4;
							local_B_pe6_pe7_d5[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d5;
							local_B_pe6_pe7_d6[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d6;
							local_B_pe6_pe7_d7[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d7;
						}
						b_512_x0_ready = false;
						b_512_x1_ready = false;
						b_512_x2_ready = false;
						b_512_x3_ready = false;
						++j;
					}
				}

				// computation
				ap_uint<32> end_32_in;
				bool end_32_ready = false;
				w2: while(!end_32_ready) {
#pragma HLS loop_tripcount min=1 max=10
#pragma HLS pipeline II=1
					end_32_ready = fifo_inst.read_nb(end_32_in);
				}
				ap_uint<32> end_32 = HLS_REG(end_32_in);

				fifo_inst_out.write(end_32);

				computation: for (ap_uint<32> j = start_32; j < end_32; ) {
#pragma HLS loop_tripcount min=1 max=200
#pragma HLS pipeline II=1

#pragma HLS dependence true variable=local_C_pe0_d0_d1 distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=local_C_pe0_d2_d3 distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=local_C_pe0_d4_d5 distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=local_C_pe0_d6_d7 distance=DEP_DIST_LOAD_STORE

#pragma HLS dependence true variable=local_C_pe1_d0_d1 distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=local_C_pe1_d2_d3 distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=local_C_pe1_d4_d5 distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=local_C_pe1_d6_d7 distance=DEP_DIST_LOAD_STORE

#pragma HLS dependence true variable=local_C_pe2_d0_d1 distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=local_C_pe2_d2_d3 distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=local_C_pe2_d4_d5 distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=local_C_pe2_d6_d7 distance=DEP_DIST_LOAD_STORE

#pragma HLS dependence true variable=local_C_pe3_d0_d1 distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=local_C_pe3_d2_d3 distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=local_C_pe3_d4_d5 distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=local_C_pe3_d6_d7 distance=DEP_DIST_LOAD_STORE

#pragma HLS dependence true variable=local_C_pe4_d0_d1 distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=local_C_pe4_d2_d3 distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=local_C_pe4_d4_d5 distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=local_C_pe4_d6_d7 distance=DEP_DIST_LOAD_STORE

#pragma HLS dependence true variable=local_C_pe5_d0_d1 distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=local_C_pe5_d2_d3 distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=local_C_pe5_d4_d5 distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=local_C_pe5_d6_d7 distance=DEP_DIST_LOAD_STORE

#pragma HLS dependence true variable=local_C_pe6_d0_d1 distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=local_C_pe6_d2_d3 distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=local_C_pe6_d4_d5 distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=local_C_pe6_d6_d7 distance=DEP_DIST_LOAD_STORE

#pragma HLS dependence true variable=local_C_pe7_d0_d1 distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=local_C_pe7_d2_d3 distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=local_C_pe7_d4_d5 distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=local_C_pe7_d6_d7 distance=DEP_DIST_LOAD_STORE

					ap_uint<512> a_pes;
					bool a_pes_ready = fifo_A.read_nb(a_pes);

					if (a_pes_ready) {
						ap_uint<512> a_pes_delay = HLS_REG(a_pes);

						ap_uint<64> a[8];
#pragma HLS array_partition variable=a complete

						ap_uint<14> a_col[8];
#pragma HLS array_partition variable=a_col complete

						ap_uint<18> a_row[8];
#pragma HLS array_partition variable=a_row complete

						ap_uint<32> a_val[8];
#pragma HLS array_partition variable=a_val complete

						for (ap_uint<4> p = 0; p < 8; ++p) {
							a[p] = a_pes_delay(63 + p * 64, p * 64);
						}

						for (ap_uint<4> p = 0; p < 8; ++p) {
							a_col[p] = (a[p])(63, 50);
							a_row[p] = (a[p])(49, 32);
							a_val[p] = (a[p])(31,  0);
						}

						// PE process
						PEcore<0>(
							a_col[0],
							a_row[0],
							a_val[0],
							local_C_pe0_d0_d1,
							local_C_pe0_d2_d3,
							local_C_pe0_d4_d5,
							local_C_pe0_d6_d7,
							local_B_pe0_pe1_d0,
							local_B_pe0_pe1_d1,
							local_B_pe0_pe1_d2,
							local_B_pe0_pe1_d3,
							local_B_pe0_pe1_d4,
							local_B_pe0_pe1_d5,
							local_B_pe0_pe1_d6,
							local_B_pe0_pe1_d7
							);

						PEcore<1>(
							a_col[1],
							a_row[1],
							a_val[1],
							local_C_pe1_d0_d1,
							local_C_pe1_d2_d3,
							local_C_pe1_d4_d5,
							local_C_pe1_d6_d7,
							local_B_pe0_pe1_d0,
							local_B_pe0_pe1_d1,
							local_B_pe0_pe1_d2,
							local_B_pe0_pe1_d3,
							local_B_pe0_pe1_d4,
							local_B_pe0_pe1_d5,
							local_B_pe0_pe1_d6,
							local_B_pe0_pe1_d7
							);

						PEcore<2>(
							a_col[2],
							a_row[2],
							a_val[2],
							local_C_pe2_d0_d1,
							local_C_pe2_d2_d3,
							local_C_pe2_d4_d5,
							local_C_pe2_d6_d7,
							local_B_pe2_pe3_d0,
							local_B_pe2_pe3_d1,
							local_B_pe2_pe3_d2,
							local_B_pe2_pe3_d3,
							local_B_pe2_pe3_d4,
							local_B_pe2_pe3_d5,
							local_B_pe2_pe3_d6,
							local_B_pe2_pe3_d7
							);

						PEcore<3>(
							a_col[3],
							a_row[3],
							a_val[3],
							local_C_pe3_d0_d1,
							local_C_pe3_d2_d3,
							local_C_pe3_d4_d5,
							local_C_pe3_d6_d7,
							local_B_pe2_pe3_d0,
							local_B_pe2_pe3_d1,
							local_B_pe2_pe3_d2,
							local_B_pe2_pe3_d3,
							local_B_pe2_pe3_d4,
							local_B_pe2_pe3_d5,
							local_B_pe2_pe3_d6,
							local_B_pe2_pe3_d7
							);

						PEcore<4>(
							a_col[4],
							a_row[4],
							a_val[4],
							local_C_pe4_d0_d1,
							local_C_pe4_d2_d3,
							local_C_pe4_d4_d5,
							local_C_pe4_d6_d7,
							local_B_pe4_pe5_d0,
							local_B_pe4_pe5_d1,
							local_B_pe4_pe5_d2,
							local_B_pe4_pe5_d3,
							local_B_pe4_pe5_d4,
							local_B_pe4_pe5_d5,
							local_B_pe4_pe5_d6,
							local_B_pe4_pe5_d7
							);

						PEcore<5>(
							a_col[5],
							a_row[5],
							a_val[5],
							local_C_pe5_d0_d1,
							local_C_pe5_d2_d3,
							local_C_pe5_d4_d5,
							local_C_pe5_d6_d7,
							local_B_pe4_pe5_d0,
							local_B_pe4_pe5_d1,
							local_B_pe4_pe5_d2,
							local_B_pe4_pe5_d3,
							local_B_pe4_pe5_d4,
							local_B_pe4_pe5_d5,
							local_B_pe4_pe5_d6,
							local_B_pe4_pe5_d7
							);

						PEcore<6>(
							a_col[6],
							a_row[6],
							a_val[6],
							local_C_pe6_d0_d1,
							local_C_pe6_d2_d3,
							local_C_pe6_d4_d5,
							local_C_pe6_d6_d7,
							local_B_pe6_pe7_d0,
							local_B_pe6_pe7_d1,
							local_B_pe6_pe7_d2,
							local_B_pe6_pe7_d3,
							local_B_pe6_pe7_d4,
							local_B_pe6_pe7_d5,
							local_B_pe6_pe7_d6,
							local_B_pe6_pe7_d7
							);

						PEcore<7>(
							a_col[7],
							a_row[7],
							a_val[7],
							local_C_pe7_d0_d1,
							local_C_pe7_d2_d3,
							local_C_pe7_d4_d5,
							local_C_pe7_d6_d7,
							local_B_pe6_pe7_d0,
							local_B_pe6_pe7_d1,
							local_B_pe6_pe7_d2,
							local_B_pe6_pe7_d3,
							local_B_pe6_pe7_d4,
							local_B_pe6_pe7_d5,
							local_B_pe6_pe7_d6,
							local_B_pe6_pe7_d7
							);

						++j;
					}
				}
				start_32 = end_32;
			}

			ap_uint<512> out_u0;

			//write C fifo
			write_C_outer: for (ap_uint<32> i = 0; i < (M + 7)/8; ++i) {
#pragma HLS loop_tripcount min=1 max=1800
#pragma HLS pipeline II=1
				ap_uint<64> u_64_pe_d[2][4];
#pragma HLS array_partition variable=u_64_pe_d complete

				switch (i % 4) {
					//0,  1,  8,  9, 16, 17, 24, 25
					//2,  3, 10, 11, 18, 19, 26, 27
					//4,  5, 12, 13, 20, 21, 28, 29
					//6,  7, 14, 15, 22, 23, 30, 31
					case 0:
						u_64_pe_d[0][0] = local_C_pe0_d0_d1[i/4];
						u_64_pe_d[0][1] = local_C_pe0_d2_d3[i/4];
						u_64_pe_d[0][2] = local_C_pe0_d4_d5[i/4];
						u_64_pe_d[0][3] = local_C_pe0_d6_d7[i/4];

						u_64_pe_d[1][0] = local_C_pe1_d0_d1[i/4];
						u_64_pe_d[1][1] = local_C_pe1_d2_d3[i/4];
						u_64_pe_d[1][2] = local_C_pe1_d4_d5[i/4];
						u_64_pe_d[1][3] = local_C_pe1_d6_d7[i/4];

						break;
					case 1:
						u_64_pe_d[0][0] = local_C_pe2_d0_d1[i/4];
						u_64_pe_d[0][1] = local_C_pe2_d2_d3[i/4];
						u_64_pe_d[0][2] = local_C_pe2_d4_d5[i/4];
						u_64_pe_d[0][3] = local_C_pe2_d6_d7[i/4];

						u_64_pe_d[1][0] = local_C_pe3_d0_d1[i/4];
						u_64_pe_d[1][1] = local_C_pe3_d2_d3[i/4];
						u_64_pe_d[1][2] = local_C_pe3_d4_d5[i/4];
						u_64_pe_d[1][3] = local_C_pe3_d6_d7[i/4];

						break;
					case 2:
						u_64_pe_d[0][0] = local_C_pe4_d0_d1[i/4];
						u_64_pe_d[0][1] = local_C_pe4_d2_d3[i/4];
						u_64_pe_d[0][2] = local_C_pe4_d4_d5[i/4];
						u_64_pe_d[0][3] = local_C_pe4_d6_d7[i/4];

						u_64_pe_d[1][0] = local_C_pe5_d0_d1[i/4];
						u_64_pe_d[1][1] = local_C_pe5_d2_d3[i/4];
						u_64_pe_d[1][2] = local_C_pe5_d4_d5[i/4];
						u_64_pe_d[1][3] = local_C_pe5_d6_d7[i/4];

						break;
					case 3:
						u_64_pe_d[0][0] = local_C_pe6_d0_d1[i/4];
						u_64_pe_d[0][1] = local_C_pe6_d2_d3[i/4];
						u_64_pe_d[0][2] = local_C_pe6_d4_d5[i/4];
						u_64_pe_d[0][3] = local_C_pe6_d6_d7[i/4];

						u_64_pe_d[1][0] = local_C_pe7_d0_d1[i/4];
						u_64_pe_d[1][1] = local_C_pe7_d2_d3[i/4];
						u_64_pe_d[1][2] = local_C_pe7_d4_d5[i/4];
						u_64_pe_d[1][3] = local_C_pe7_d6_d7[i/4];

						break;
				}

				for (ap_uint<2> pe = 0; pe < 2; ++pe) {
					for (ap_uint<3> d = 0; d < 4; ++d) {
						out_u0(63 + pe * 256 + d * 64, pe * 256 + d * 64) = u_64_pe_d[pe][d];
					}
				}

				ap_uint<512> out_u0_mult;
				peg16mult(out_u0, alpha_u, out_u0_mult);
				fifo_C_out0.write(HLS_REG(out_u0_mult));
			}
		}
	}
}

template <int ch>
void PEG_last(
	hls::stream<ap_uint<32> > & fifo_inst,
	hls::stream<ap_uint<512> > & fifo_A,
	hls::stream<ap_uint<512> > & fifo_B_x0, // [32bits * 8 (256)] * 2
	hls::stream<ap_uint<512> > & fifo_B_x1, // [32bits * 8 (256)] * 2
	hls::stream<ap_uint<512> > & fifo_B_x2, // [32bits * 8 (256)] * 2
	hls::stream<ap_uint<512> > & fifo_B_x3, // [32bits * 8 (256)] * 2
	hls::stream<ap_uint<512> > & fifo_C_out0 // [256(32bits * 8 d)] * 2
	) {
#pragma HLS inline off
	ap_uint<32> NUM_ITE;
	ap_uint<32> M;
	ap_uint<32> P_N;
	ap_uint<32> K;
	ap_uint<32> alpha_u;
	ap_uint<32> beta_u;

	ap_uint<32> parameter;
	w_ITE: for (ap_uint<3> i = 0; i < 6; ) {
#pragma HLS loop_tripcount min=1 max=10
#pragma HLS pipeline II=1
		bool parameter_ready = fifo_inst.read_nb(parameter);
		if (parameter_ready) {
			ap_uint<32> parameter_dealy = HLS_REG(parameter);
			switch (i) {
				case 0: NUM_ITE = parameter_dealy; break;
				case 1: M = parameter_dealy; break;
				case 2: P_N = parameter_dealy; break;
				case 3: K = parameter_dealy; break;
				case 4: alpha_u = parameter_dealy; break;
				case 5: beta_u = parameter_dealy; break;
			}
			++i;
		}
	}

	const ap_uint<16> N16 = P_N(31, 16);
	const ap_uint<16> rp_time = (N16 == 0)? ((ap_uint<16>) 1) : N16;
	const ap_uint<32> N = P_N & ((ap_uint<32>) 0x0000FFFF);

	ap_uint<512> MN512;
	MN512(31,  0) = M;
	MN512(63, 32) = P_N;
	MN512(95, 64) = beta_u;
	fifo_C_out0.write(MN512);

	//define local C buffer and pragma to URAM
	ap_uint<64> local_C_pe0_d0_d1[URAM_DEPTH];
	ap_uint<64> local_C_pe0_d2_d3[URAM_DEPTH];
	ap_uint<64> local_C_pe0_d4_d5[URAM_DEPTH];
	ap_uint<64> local_C_pe0_d6_d7[URAM_DEPTH];

#pragma HLS bind_storage variable=local_C_pe0_d0_d1 type=RAM_2P impl=URAM
#pragma HLS bind_storage variable=local_C_pe0_d2_d3 type=RAM_2P impl=URAM
#pragma HLS bind_storage variable=local_C_pe0_d4_d5 type=RAM_2P impl=URAM
#pragma HLS bind_storage variable=local_C_pe0_d6_d7 type=RAM_2P impl=URAM

	ap_uint<64> local_C_pe1_d0_d1[URAM_DEPTH];
	ap_uint<64> local_C_pe1_d2_d3[URAM_DEPTH];
	ap_uint<64> local_C_pe1_d4_d5[URAM_DEPTH];
	ap_uint<64> local_C_pe1_d6_d7[URAM_DEPTH];

#pragma HLS bind_storage variable=local_C_pe1_d0_d1 type=RAM_2P impl=URAM
#pragma HLS bind_storage variable=local_C_pe1_d2_d3 type=RAM_2P impl=URAM
#pragma HLS bind_storage variable=local_C_pe1_d4_d5 type=RAM_2P impl=URAM
#pragma HLS bind_storage variable=local_C_pe1_d6_d7 type=RAM_2P impl=URAM

	ap_uint<64> local_C_pe2_d0_d1[URAM_DEPTH];
	ap_uint<64> local_C_pe2_d2_d3[URAM_DEPTH];
	ap_uint<64> local_C_pe2_d4_d5[URAM_DEPTH];
	ap_uint<64> local_C_pe2_d6_d7[URAM_DEPTH];

#pragma HLS bind_storage variable=local_C_pe2_d0_d1 type=RAM_2P impl=URAM
#pragma HLS bind_storage variable=local_C_pe2_d2_d3 type=RAM_2P impl=URAM
#pragma HLS bind_storage variable=local_C_pe2_d4_d5 type=RAM_2P impl=URAM
#pragma HLS bind_storage variable=local_C_pe2_d6_d7 type=RAM_2P impl=URAM

	ap_uint<64> local_C_pe3_d0_d1[URAM_DEPTH];
	ap_uint<64> local_C_pe3_d2_d3[URAM_DEPTH];
	ap_uint<64> local_C_pe3_d4_d5[URAM_DEPTH];
	ap_uint<64> local_C_pe3_d6_d7[URAM_DEPTH];

#pragma HLS bind_storage variable=local_C_pe3_d0_d1 type=RAM_2P impl=URAM
#pragma HLS bind_storage variable=local_C_pe3_d2_d3 type=RAM_2P impl=URAM
#pragma HLS bind_storage variable=local_C_pe3_d4_d5 type=RAM_2P impl=URAM
#pragma HLS bind_storage variable=local_C_pe3_d6_d7 type=RAM_2P impl=URAM

	ap_uint<64> local_C_pe4_d0_d1[URAM_DEPTH];
	ap_uint<64> local_C_pe4_d2_d3[URAM_DEPTH];
	ap_uint<64> local_C_pe4_d4_d5[URAM_DEPTH];
	ap_uint<64> local_C_pe4_d6_d7[URAM_DEPTH];

#pragma HLS bind_storage variable=local_C_pe4_d0_d1 type=RAM_2P impl=URAM
#pragma HLS bind_storage variable=local_C_pe4_d2_d3 type=RAM_2P impl=URAM
#pragma HLS bind_storage variable=local_C_pe4_d4_d5 type=RAM_2P impl=URAM
#pragma HLS bind_storage variable=local_C_pe4_d6_d7 type=RAM_2P impl=URAM

	ap_uint<64> local_C_pe5_d0_d1[URAM_DEPTH];
	ap_uint<64> local_C_pe5_d2_d3[URAM_DEPTH];
	ap_uint<64> local_C_pe5_d4_d5[URAM_DEPTH];
	ap_uint<64> local_C_pe5_d6_d7[URAM_DEPTH];

#pragma HLS bind_storage variable=local_C_pe5_d0_d1 type=RAM_2P impl=URAM
#pragma HLS bind_storage variable=local_C_pe5_d2_d3 type=RAM_2P impl=URAM
#pragma HLS bind_storage variable=local_C_pe5_d4_d5 type=RAM_2P impl=URAM
#pragma HLS bind_storage variable=local_C_pe5_d6_d7 type=RAM_2P impl=URAM

	ap_uint<64> local_C_pe6_d0_d1[URAM_DEPTH];
	ap_uint<64> local_C_pe6_d2_d3[URAM_DEPTH];
	ap_uint<64> local_C_pe6_d4_d5[URAM_DEPTH];
	ap_uint<64> local_C_pe6_d6_d7[URAM_DEPTH];

#pragma HLS bind_storage variable=local_C_pe6_d0_d1 type=RAM_2P impl=URAM
#pragma HLS bind_storage variable=local_C_pe6_d2_d3 type=RAM_2P impl=URAM
#pragma HLS bind_storage variable=local_C_pe6_d4_d5 type=RAM_2P impl=URAM
#pragma HLS bind_storage variable=local_C_pe6_d6_d7 type=RAM_2P impl=URAM

	ap_uint<64> local_C_pe7_d0_d1[URAM_DEPTH];
	ap_uint<64> local_C_pe7_d2_d3[URAM_DEPTH];
	ap_uint<64> local_C_pe7_d4_d5[URAM_DEPTH];
	ap_uint<64> local_C_pe7_d6_d7[URAM_DEPTH];

#pragma HLS bind_storage variable=local_C_pe7_d0_d1 type=RAM_2P impl=URAM
#pragma HLS bind_storage variable=local_C_pe7_d2_d3 type=RAM_2P impl=URAM
#pragma HLS bind_storage variable=local_C_pe7_d4_d5 type=RAM_2P impl=URAM
#pragma HLS bind_storage variable=local_C_pe7_d6_d7 type=RAM_2P impl=URAM

	l_rp: for(ap_uint<16> rp = 0; rp < rp_time; rp++) {
#pragma HLS loop_flatten off
#pragma HLS loop_tripcount min=1 max=16
		l_N: for(ap_uint<32> nn = 0; nn < (N + 7)/8; nn++) {
#pragma HLS loop_flatten off
#pragma HLS loop_tripcount min=1 max=16

			//init local C
			init_C: for (ap_uint<32> i = 0; i < ((M + 31) / 32); ++i) {
#pragma HLS loop_tripcount min=1 max=800
#pragma HLS pipeline II=1
				local_C_pe0_d0_d1[i] = 0;
				local_C_pe0_d2_d3[i] = 0;
				local_C_pe0_d4_d5[i] = 0;
				local_C_pe0_d6_d7[i] = 0;
				local_C_pe1_d0_d1[i] = 0;
				local_C_pe1_d2_d3[i] = 0;
				local_C_pe1_d4_d5[i] = 0;
				local_C_pe1_d6_d7[i] = 0;
				local_C_pe2_d0_d1[i] = 0;
				local_C_pe2_d2_d3[i] = 0;
				local_C_pe2_d4_d5[i] = 0;
				local_C_pe2_d6_d7[i] = 0;
				local_C_pe3_d0_d1[i] = 0;
				local_C_pe3_d2_d3[i] = 0;
				local_C_pe3_d4_d5[i] = 0;
				local_C_pe3_d6_d7[i] = 0;
				local_C_pe4_d0_d1[i] = 0;
				local_C_pe4_d2_d3[i] = 0;
				local_C_pe4_d4_d5[i] = 0;
				local_C_pe4_d6_d7[i] = 0;
				local_C_pe5_d0_d1[i] = 0;
				local_C_pe5_d2_d3[i] = 0;
				local_C_pe5_d4_d5[i] = 0;
				local_C_pe5_d6_d7[i] = 0;
				local_C_pe6_d0_d1[i] = 0;
				local_C_pe6_d2_d3[i] = 0;
				local_C_pe6_d4_d5[i] = 0;
				local_C_pe6_d6_d7[i] = 0;
				local_C_pe7_d0_d1[i] = 0;
				local_C_pe7_d2_d3[i] = 0;
				local_C_pe7_d4_d5[i] = 0;
				local_C_pe7_d6_d7[i] = 0;
			}
			//define local B buffer and pragma local B buffer if partition factor > 1

			ap_uint<32> local_B_pe0_pe1_d0[WINDOW_SIZE];
			ap_uint<32> local_B_pe0_pe1_d1[WINDOW_SIZE];
			ap_uint<32> local_B_pe0_pe1_d2[WINDOW_SIZE];
			ap_uint<32> local_B_pe0_pe1_d3[WINDOW_SIZE];
			ap_uint<32> local_B_pe0_pe1_d4[WINDOW_SIZE];
			ap_uint<32> local_B_pe0_pe1_d5[WINDOW_SIZE];
			ap_uint<32> local_B_pe0_pe1_d6[WINDOW_SIZE];
			ap_uint<32> local_B_pe0_pe1_d7[WINDOW_SIZE];

#pragma HLS bind_storage variable=local_B_pe0_pe1_d0 latency=3
#pragma HLS bind_storage variable=local_B_pe0_pe1_d1 latency=3
#pragma HLS bind_storage variable=local_B_pe0_pe1_d2 latency=3
#pragma HLS bind_storage variable=local_B_pe0_pe1_d3 latency=3
#pragma HLS bind_storage variable=local_B_pe0_pe1_d4 latency=3
#pragma HLS bind_storage variable=local_B_pe0_pe1_d5 latency=3
#pragma HLS bind_storage variable=local_B_pe0_pe1_d6 latency=3
#pragma HLS bind_storage variable=local_B_pe0_pe1_d7 latency=3

#pragma HLS array_partition variable=local_B_pe0_pe1_d0 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe0_pe1_d1 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe0_pe1_d2 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe0_pe1_d3 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe0_pe1_d4 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe0_pe1_d5 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe0_pe1_d6 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe0_pe1_d7 cyclic factor=B_PARTITION_FACTOR

			ap_uint<32> local_B_pe2_pe3_d0[WINDOW_SIZE];
			ap_uint<32> local_B_pe2_pe3_d1[WINDOW_SIZE];
			ap_uint<32> local_B_pe2_pe3_d2[WINDOW_SIZE];
			ap_uint<32> local_B_pe2_pe3_d3[WINDOW_SIZE];
			ap_uint<32> local_B_pe2_pe3_d4[WINDOW_SIZE];
			ap_uint<32> local_B_pe2_pe3_d5[WINDOW_SIZE];
			ap_uint<32> local_B_pe2_pe3_d6[WINDOW_SIZE];
			ap_uint<32> local_B_pe2_pe3_d7[WINDOW_SIZE];

#pragma HLS bind_storage variable=local_B_pe2_pe3_d0 latency=3
#pragma HLS bind_storage variable=local_B_pe2_pe3_d1 latency=3
#pragma HLS bind_storage variable=local_B_pe2_pe3_d2 latency=3
#pragma HLS bind_storage variable=local_B_pe2_pe3_d3 latency=3
#pragma HLS bind_storage variable=local_B_pe2_pe3_d4 latency=3
#pragma HLS bind_storage variable=local_B_pe2_pe3_d5 latency=3
#pragma HLS bind_storage variable=local_B_pe2_pe3_d6 latency=3
#pragma HLS bind_storage variable=local_B_pe2_pe3_d7 latency=3

#pragma HLS array_partition variable=local_B_pe2_pe3_d0 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe2_pe3_d1 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe2_pe3_d2 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe2_pe3_d3 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe2_pe3_d4 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe2_pe3_d5 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe2_pe3_d6 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe2_pe3_d7 cyclic factor=B_PARTITION_FACTOR

			ap_uint<32> local_B_pe4_pe5_d0[WINDOW_SIZE];
			ap_uint<32> local_B_pe4_pe5_d1[WINDOW_SIZE];
			ap_uint<32> local_B_pe4_pe5_d2[WINDOW_SIZE];
			ap_uint<32> local_B_pe4_pe5_d3[WINDOW_SIZE];
			ap_uint<32> local_B_pe4_pe5_d4[WINDOW_SIZE];
			ap_uint<32> local_B_pe4_pe5_d5[WINDOW_SIZE];
			ap_uint<32> local_B_pe4_pe5_d6[WINDOW_SIZE];
			ap_uint<32> local_B_pe4_pe5_d7[WINDOW_SIZE];

#pragma HLS bind_storage variable=local_B_pe4_pe5_d0 latency=3
#pragma HLS bind_storage variable=local_B_pe4_pe5_d1 latency=3
#pragma HLS bind_storage variable=local_B_pe4_pe5_d2 latency=3
#pragma HLS bind_storage variable=local_B_pe4_pe5_d3 latency=3
#pragma HLS bind_storage variable=local_B_pe4_pe5_d4 latency=3
#pragma HLS bind_storage variable=local_B_pe4_pe5_d5 latency=3
#pragma HLS bind_storage variable=local_B_pe4_pe5_d6 latency=3
#pragma HLS bind_storage variable=local_B_pe4_pe5_d7 latency=3

#pragma HLS array_partition variable=local_B_pe4_pe5_d0 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe4_pe5_d1 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe4_pe5_d2 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe4_pe5_d3 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe4_pe5_d4 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe4_pe5_d5 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe4_pe5_d6 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe4_pe5_d7 cyclic factor=B_PARTITION_FACTOR

			ap_uint<32> local_B_pe6_pe7_d0[WINDOW_SIZE];
			ap_uint<32> local_B_pe6_pe7_d1[WINDOW_SIZE];
			ap_uint<32> local_B_pe6_pe7_d2[WINDOW_SIZE];
			ap_uint<32> local_B_pe6_pe7_d3[WINDOW_SIZE];
			ap_uint<32> local_B_pe6_pe7_d4[WINDOW_SIZE];
			ap_uint<32> local_B_pe6_pe7_d5[WINDOW_SIZE];
			ap_uint<32> local_B_pe6_pe7_d6[WINDOW_SIZE];
			ap_uint<32> local_B_pe6_pe7_d7[WINDOW_SIZE];

#pragma HLS bind_storage variable=local_B_pe6_pe7_d0 latency=3
#pragma HLS bind_storage variable=local_B_pe6_pe7_d1 latency=3
#pragma HLS bind_storage variable=local_B_pe6_pe7_d2 latency=3
#pragma HLS bind_storage variable=local_B_pe6_pe7_d3 latency=3
#pragma HLS bind_storage variable=local_B_pe6_pe7_d4 latency=3
#pragma HLS bind_storage variable=local_B_pe6_pe7_d5 latency=3
#pragma HLS bind_storage variable=local_B_pe6_pe7_d6 latency=3
#pragma HLS bind_storage variable=local_B_pe6_pe7_d7 latency=3

#pragma HLS array_partition variable=local_B_pe6_pe7_d0 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe6_pe7_d1 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe6_pe7_d2 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe6_pe7_d3 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe6_pe7_d4 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe6_pe7_d5 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe6_pe7_d6 cyclic factor=B_PARTITION_FACTOR
#pragma HLS array_partition variable=local_B_pe6_pe7_d7 cyclic factor=B_PARTITION_FACTOR

			ap_uint<32> start_32_in;
			bool start_32_in_ready = false;
			w1: while(!start_32_in_ready) {
#pragma HLS loop_tripcount min=1 max=10
#pragma HLS pipeline II=1
				start_32_in_ready = fifo_inst.read_nb(start_32_in);
			}
			ap_uint<32> start_32 = HLS_REG(start_32_in);

			main: for (ap_uint<32> i = 0; i < NUM_ITE; ++i) {
#pragma HLS loop_tripcount min=1 max=49

				// fill onchip B
				ap_uint<512> b_512_x0;
				ap_uint<512> b_512_x1;
				ap_uint<512> b_512_x2;
				ap_uint<512> b_512_x3;

				bool b_512_x0_ready = false;
				bool b_512_x1_ready = false;
				bool b_512_x2_ready = false;
				bool b_512_x3_ready = false;

				read_B: for (ap_uint<14> j = 0; (j < WINDOW_SIZE/8) && (j < (K + 7) / 8 - i*WINDOW_SIZE/8); ) {
#pragma HLS loop_tripcount min=1 max=512
#pragma HLS pipeline II=1
					if (!b_512_x0_ready) {
						b_512_x0_ready = fifo_B_x0.read_nb(b_512_x0);
					}
					if (!b_512_x1_ready) {
						b_512_x1_ready = fifo_B_x1.read_nb(b_512_x1);
					}
					if (!b_512_x2_ready) {
						b_512_x2_ready = fifo_B_x2.read_nb(b_512_x2);
					}
					if (!b_512_x3_ready) {
						b_512_x3_ready = fifo_B_x3.read_nb(b_512_x3);
					}

					bool b_2048_ready = b_512_x0_ready && b_512_x1_ready && b_512_x2_ready && b_512_x3_ready;

					if (b_2048_ready) {
						ap_uint<512> b_512_x0_delay = HLS_REG(HLS_REG(b_512_x0));
						ap_uint<512> b_512_x1_delay = HLS_REG(HLS_REG(b_512_x1));
						ap_uint<512> b_512_x2_delay = HLS_REG(HLS_REG(b_512_x2));
						ap_uint<512> b_512_x3_delay = HLS_REG(HLS_REG(b_512_x3));

						ap_uint<256> b_d_seg[8]; // d 0 - 7, 256bits
#pragma HLS array_partition variable=b_d_seg complete

						b_d_seg[0] = b_512_x0_delay(255,   0);
						b_d_seg[1] = b_512_x0_delay(511, 256);
						b_d_seg[2] = b_512_x1_delay(255,   0);
						b_d_seg[3] = b_512_x1_delay(511, 256);
						b_d_seg[4] = b_512_x2_delay(255,   0);
						b_d_seg[5] = b_512_x2_delay(511, 256);
						b_d_seg[6] = b_512_x3_delay(255,   0);
						b_d_seg[7] = b_512_x3_delay(511, 256);

						read_B_p: for (ap_uint<4> k = 0; k < 8; ++k) {
							ap_uint<32> b_pe_d0 = (b_d_seg[k])( 31,   0);
							ap_uint<32> b_pe_d1 = (b_d_seg[k])( 63,  32);
							ap_uint<32> b_pe_d2 = (b_d_seg[k])( 95,  64);
							ap_uint<32> b_pe_d3 = (b_d_seg[k])(127,  96);
							ap_uint<32> b_pe_d4 = (b_d_seg[k])(159, 128);
							ap_uint<32> b_pe_d5 = (b_d_seg[k])(191, 160);
							ap_uint<32> b_pe_d6 = (b_d_seg[k])(223, 192);
							ap_uint<32> b_pe_d7 = (b_d_seg[k])(255, 224);

							local_B_pe0_pe1_d0[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d0;
							local_B_pe0_pe1_d1[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d1;
							local_B_pe0_pe1_d2[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d2;
							local_B_pe0_pe1_d3[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d3;
							local_B_pe0_pe1_d4[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d4;
							local_B_pe0_pe1_d5[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d5;
							local_B_pe0_pe1_d6[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d6;
							local_B_pe0_pe1_d7[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d7;

							local_B_pe2_pe3_d0[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d0;
							local_B_pe2_pe3_d1[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d1;
							local_B_pe2_pe3_d2[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d2;
							local_B_pe2_pe3_d3[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d3;
							local_B_pe2_pe3_d4[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d4;
							local_B_pe2_pe3_d5[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d5;
							local_B_pe2_pe3_d6[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d6;
							local_B_pe2_pe3_d7[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d7;

							local_B_pe4_pe5_d0[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d0;
							local_B_pe4_pe5_d1[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d1;
							local_B_pe4_pe5_d2[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d2;
							local_B_pe4_pe5_d3[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d3;
							local_B_pe4_pe5_d4[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d4;
							local_B_pe4_pe5_d5[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d5;
							local_B_pe4_pe5_d6[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d6;
							local_B_pe4_pe5_d7[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d7;

							local_B_pe6_pe7_d0[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d0;
							local_B_pe6_pe7_d1[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d1;
							local_B_pe6_pe7_d2[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d2;
							local_B_pe6_pe7_d3[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d3;
							local_B_pe6_pe7_d4[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d4;
							local_B_pe6_pe7_d5[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d5;
							local_B_pe6_pe7_d6[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d6;
							local_B_pe6_pe7_d7[HLS_REG(HLS_REG(j)) * 8 + k] = b_pe_d7;
						}
						b_512_x0_ready = false;
						b_512_x1_ready = false;
						b_512_x2_ready = false;
						b_512_x3_ready = false;
						++j;
					}
				}

				// computation
				ap_uint<32> end_32_in;
				bool end_32_ready = false;
				w2: while(!end_32_ready) {
#pragma HLS loop_tripcount min=1 max=10
#pragma HLS pipeline II=1
					end_32_ready = fifo_inst.read_nb(end_32_in);
				}
				ap_uint<32> end_32 = HLS_REG(end_32_in);


				computation: for (ap_uint<32> j = start_32; j < end_32; ) {
#pragma HLS loop_tripcount min=1 max=200
#pragma HLS pipeline II=1

#pragma HLS dependence true variable=local_C_pe0_d0_d1 distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=local_C_pe0_d2_d3 distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=local_C_pe0_d4_d5 distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=local_C_pe0_d6_d7 distance=DEP_DIST_LOAD_STORE

#pragma HLS dependence true variable=local_C_pe1_d0_d1 distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=local_C_pe1_d2_d3 distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=local_C_pe1_d4_d5 distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=local_C_pe1_d6_d7 distance=DEP_DIST_LOAD_STORE

#pragma HLS dependence true variable=local_C_pe2_d0_d1 distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=local_C_pe2_d2_d3 distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=local_C_pe2_d4_d5 distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=local_C_pe2_d6_d7 distance=DEP_DIST_LOAD_STORE

#pragma HLS dependence true variable=local_C_pe3_d0_d1 distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=local_C_pe3_d2_d3 distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=local_C_pe3_d4_d5 distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=local_C_pe3_d6_d7 distance=DEP_DIST_LOAD_STORE

#pragma HLS dependence true variable=local_C_pe4_d0_d1 distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=local_C_pe4_d2_d3 distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=local_C_pe4_d4_d5 distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=local_C_pe4_d6_d7 distance=DEP_DIST_LOAD_STORE

#pragma HLS dependence true variable=local_C_pe5_d0_d1 distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=local_C_pe5_d2_d3 distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=local_C_pe5_d4_d5 distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=local_C_pe5_d6_d7 distance=DEP_DIST_LOAD_STORE

#pragma HLS dependence true variable=local_C_pe6_d0_d1 distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=local_C_pe6_d2_d3 distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=local_C_pe6_d4_d5 distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=local_C_pe6_d6_d7 distance=DEP_DIST_LOAD_STORE

#pragma HLS dependence true variable=local_C_pe7_d0_d1 distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=local_C_pe7_d2_d3 distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=local_C_pe7_d4_d5 distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=local_C_pe7_d6_d7 distance=DEP_DIST_LOAD_STORE

					ap_uint<512> a_pes;
					bool a_pes_ready = fifo_A.read_nb(a_pes);

					if (a_pes_ready) {
						ap_uint<512> a_pes_delay = HLS_REG(a_pes);

						ap_uint<64> a[8];
#pragma HLS array_partition variable=a complete

						ap_uint<14> a_col[8];
#pragma HLS array_partition variable=a_col complete

						ap_uint<18> a_row[8];
#pragma HLS array_partition variable=a_row complete

						ap_uint<32> a_val[8];
#pragma HLS array_partition variable=a_val complete

						for (ap_uint<4> p = 0; p < 8; ++p) {
							a[p] = a_pes_delay(63 + p * 64, p * 64);
						}

						for (ap_uint<4> p = 0; p < 8; ++p) {
							a_col[p] = (a[p])(63, 50);
							a_row[p] = (a[p])(49, 32);
							a_val[p] = (a[p])(31,  0);
						}

						// PE process
						PEcore<0>(
							a_col[0],
							a_row[0],
							a_val[0],
							local_C_pe0_d0_d1,
							local_C_pe0_d2_d3,
							local_C_pe0_d4_d5,
							local_C_pe0_d6_d7,
							local_B_pe0_pe1_d0,
							local_B_pe0_pe1_d1,
							local_B_pe0_pe1_d2,
							local_B_pe0_pe1_d3,
							local_B_pe0_pe1_d4,
							local_B_pe0_pe1_d5,
							local_B_pe0_pe1_d6,
							local_B_pe0_pe1_d7
							);

						PEcore<1>(
							a_col[1],
							a_row[1],
							a_val[1],
							local_C_pe1_d0_d1,
							local_C_pe1_d2_d3,
							local_C_pe1_d4_d5,
							local_C_pe1_d6_d7,
							local_B_pe0_pe1_d0,
							local_B_pe0_pe1_d1,
							local_B_pe0_pe1_d2,
							local_B_pe0_pe1_d3,
							local_B_pe0_pe1_d4,
							local_B_pe0_pe1_d5,
							local_B_pe0_pe1_d6,
							local_B_pe0_pe1_d7
							);

						PEcore<2>(
							a_col[2],
							a_row[2],
							a_val[2],
							local_C_pe2_d0_d1,
							local_C_pe2_d2_d3,
							local_C_pe2_d4_d5,
							local_C_pe2_d6_d7,
							local_B_pe2_pe3_d0,
							local_B_pe2_pe3_d1,
							local_B_pe2_pe3_d2,
							local_B_pe2_pe3_d3,
							local_B_pe2_pe3_d4,
							local_B_pe2_pe3_d5,
							local_B_pe2_pe3_d6,
							local_B_pe2_pe3_d7
							);

						PEcore<3>(
							a_col[3],
							a_row[3],
							a_val[3],
							local_C_pe3_d0_d1,
							local_C_pe3_d2_d3,
							local_C_pe3_d4_d5,
							local_C_pe3_d6_d7,
							local_B_pe2_pe3_d0,
							local_B_pe2_pe3_d1,
							local_B_pe2_pe3_d2,
							local_B_pe2_pe3_d3,
							local_B_pe2_pe3_d4,
							local_B_pe2_pe3_d5,
							local_B_pe2_pe3_d6,
							local_B_pe2_pe3_d7
							);

						PEcore<4>(
							a_col[4],
							a_row[4],
							a_val[4],
							local_C_pe4_d0_d1,
							local_C_pe4_d2_d3,
							local_C_pe4_d4_d5,
							local_C_pe4_d6_d7,
							local_B_pe4_pe5_d0,
							local_B_pe4_pe5_d1,
							local_B_pe4_pe5_d2,
							local_B_pe4_pe5_d3,
							local_B_pe4_pe5_d4,
							local_B_pe4_pe5_d5,
							local_B_pe4_pe5_d6,
							local_B_pe4_pe5_d7
							);

						PEcore<5>(
							a_col[5],
							a_row[5],
							a_val[5],
							local_C_pe5_d0_d1,
							local_C_pe5_d2_d3,
							local_C_pe5_d4_d5,
							local_C_pe5_d6_d7,
							local_B_pe4_pe5_d0,
							local_B_pe4_pe5_d1,
							local_B_pe4_pe5_d2,
							local_B_pe4_pe5_d3,
							local_B_pe4_pe5_d4,
							local_B_pe4_pe5_d5,
							local_B_pe4_pe5_d6,
							local_B_pe4_pe5_d7
							);

						PEcore<6>(
							a_col[6],
							a_row[6],
							a_val[6],
							local_C_pe6_d0_d1,
							local_C_pe6_d2_d3,
							local_C_pe6_d4_d5,
							local_C_pe6_d6_d7,
							local_B_pe6_pe7_d0,
							local_B_pe6_pe7_d1,
							local_B_pe6_pe7_d2,
							local_B_pe6_pe7_d3,
							local_B_pe6_pe7_d4,
							local_B_pe6_pe7_d5,
							local_B_pe6_pe7_d6,
							local_B_pe6_pe7_d7
							);

						PEcore<7>(
							a_col[7],
							a_row[7],
							a_val[7],
							local_C_pe7_d0_d1,
							local_C_pe7_d2_d3,
							local_C_pe7_d4_d5,
							local_C_pe7_d6_d7,
							local_B_pe6_pe7_d0,
							local_B_pe6_pe7_d1,
							local_B_pe6_pe7_d2,
							local_B_pe6_pe7_d3,
							local_B_pe6_pe7_d4,
							local_B_pe6_pe7_d5,
							local_B_pe6_pe7_d6,
							local_B_pe6_pe7_d7
							);

						++j;
					}
				}
				start_32 = end_32;
			}

			ap_uint<512> out_u0;

			//write C fifo
			write_C_outer: for (ap_uint<32> i = 0; i < (M + 7)/8; ++i) {
#pragma HLS loop_tripcount min=1 max=1800
#pragma HLS pipeline II=1
				ap_uint<64> u_64_pe_d[2][4];
#pragma HLS array_partition variable=u_64_pe_d complete

				switch (i % 4) {
					//0,  1,  8,  9, 16, 17, 24, 25
					//2,  3, 10, 11, 18, 19, 26, 27
					//4,  5, 12, 13, 20, 21, 28, 29
					//6,  7, 14, 15, 22, 23, 30, 31
					case 0:
						u_64_pe_d[0][0] = local_C_pe0_d0_d1[i/4];
						u_64_pe_d[0][1] = local_C_pe0_d2_d3[i/4];
						u_64_pe_d[0][2] = local_C_pe0_d4_d5[i/4];
						u_64_pe_d[0][3] = local_C_pe0_d6_d7[i/4];

						u_64_pe_d[1][0] = local_C_pe1_d0_d1[i/4];
						u_64_pe_d[1][1] = local_C_pe1_d2_d3[i/4];
						u_64_pe_d[1][2] = local_C_pe1_d4_d5[i/4];
						u_64_pe_d[1][3] = local_C_pe1_d6_d7[i/4];

						break;
					case 1:
						u_64_pe_d[0][0] = local_C_pe2_d0_d1[i/4];
						u_64_pe_d[0][1] = local_C_pe2_d2_d3[i/4];
						u_64_pe_d[0][2] = local_C_pe2_d4_d5[i/4];
						u_64_pe_d[0][3] = local_C_pe2_d6_d7[i/4];

						u_64_pe_d[1][0] = local_C_pe3_d0_d1[i/4];
						u_64_pe_d[1][1] = local_C_pe3_d2_d3[i/4];
						u_64_pe_d[1][2] = local_C_pe3_d4_d5[i/4];
						u_64_pe_d[1][3] = local_C_pe3_d6_d7[i/4];

						break;
					case 2:
						u_64_pe_d[0][0] = local_C_pe4_d0_d1[i/4];
						u_64_pe_d[0][1] = local_C_pe4_d2_d3[i/4];
						u_64_pe_d[0][2] = local_C_pe4_d4_d5[i/4];
						u_64_pe_d[0][3] = local_C_pe4_d6_d7[i/4];

						u_64_pe_d[1][0] = local_C_pe5_d0_d1[i/4];
						u_64_pe_d[1][1] = local_C_pe5_d2_d3[i/4];
						u_64_pe_d[1][2] = local_C_pe5_d4_d5[i/4];
						u_64_pe_d[1][3] = local_C_pe5_d6_d7[i/4];

						break;
					case 3:
						u_64_pe_d[0][0] = local_C_pe6_d0_d1[i/4];
						u_64_pe_d[0][1] = local_C_pe6_d2_d3[i/4];
						u_64_pe_d[0][2] = local_C_pe6_d4_d5[i/4];
						u_64_pe_d[0][3] = local_C_pe6_d6_d7[i/4];

						u_64_pe_d[1][0] = local_C_pe7_d0_d1[i/4];
						u_64_pe_d[1][1] = local_C_pe7_d2_d3[i/4];
						u_64_pe_d[1][2] = local_C_pe7_d4_d5[i/4];
						u_64_pe_d[1][3] = local_C_pe7_d6_d7[i/4];

						break;
				}

				for (ap_uint<2> pe = 0; pe < 2; ++pe) {
					for (ap_uint<3> d = 0; d < 4; ++d) {
						out_u0(63 + pe * 256 + d * 64, pe * 256 + d * 64) = u_64_pe_d[pe][d];
					}
				}

				ap_uint<512> out_u0_mult;
				peg16mult(out_u0, alpha_u, out_u0_mult);
				fifo_C_out0.write(HLS_REG(out_u0_mult));
			}
		}
	}
}

template <int ch>
void C_IO(
	hls::stream<ap_uint<512> > & fifo_C_in,
	hls::stream<ap_uint<512> > & fifo_C_out,
	ap_uint<512>* C_mem
	) {
#pragma HLS inline off
	ap_uint<512> M_u512;
	bool M_ready = false;
	w_M: while(!M_ready) {
#pragma HLS loop_tripcount min=1 max=10
#pragma HLS pipeline II=1
		M_ready = fifo_C_in.read_nb(M_u512);
	}
	ap_uint<32> M = M_u512(31, 0);
	ap_uint<32> P_N = M_u512(63, 32);
	ap_uint<32> beta_u = M_u512(95, 64);
	float beta_f  = uint32_to_float(beta_u);

	const ap_uint<16> N16 = P_N(31, 16);
	const ap_uint<16> rp_time = (N16 == 0)? ((ap_uint<16>) 1) : N16;
	const ap_uint<32> N = P_N & ((ap_uint<32>) 0x0000FFFF);
	const ap_uint<32> num_ite_C = ((M + 7)/8) * ((N+7)/8);

	const ap_uint<32> c_write_base = (((((M + 8 - 1) >> 3) << 4)  * (N >> 3) + 1023) >> 10) << 6;

	l_rp: for(ap_uint<16> rp = 0; rp < rp_time; rp++) {
#pragma HLS loop_flatten off
#pragma HLS loop_tripcount min=1 max=16
		tiled_C: for(ap_uint<32> i = 0; i < num_ite_C; i=i+T_FACTOR_C) {
#pragma HLS loop_tripcount min=1 max=500
			ap_uint<32> tmp = i + T_FACTOR_C;
			const ap_uint<32> num_tile = (tmp < num_ite_C)? (ap_uint<32>)T_FACTOR_C : (ap_uint<32>)(num_ite_C - i);

			rd_C: for(ap_uint<32> j = 0; j < num_tile; j++) {
#pragma HLS loop_tripcount min=1 max=T_FACTOR_C
#pragma HLS pipeline II=1
				ap_uint<512> tmp_c = C_mem[i+j];
				fifo_C_out.write(tmp_c);
			}

			wr_C: for(ap_uint<32> j = 0; j < num_tile; j++) {
#pragma HLS loop_tripcount min=1 max=T_FACTOR_C
#pragma HLS pipeline II=1
				ap_uint<512> tmp_c = fifo_C_in.read();
				C_mem[c_write_base + i + j] = tmp_c;
			}

		}
	}
}

template <int ch>
void comp_C(
	hls::stream<ap_uint<512> > & fifo_C_read_in,
	hls::stream<ap_uint<512> > & fifo_C_pe_in,
	hls::stream<ap_uint<512> > & fifo_C_out
	) {
#pragma HLS inline off
	bool M_ready = false;
	ap_uint<512> M512;
	w_Mxx: while(!M_ready) {
#pragma HLS loop_tripcount min=1 max=10
#pragma HLS pipeline II=1
		M_ready = fifo_C_pe_in.read_nb(M512);
	}
	fifo_C_out.write(M512);
	ap_uint<32> M = M512(31, 0);
	ap_uint<32> P_N = M512(63, 32);
	ap_uint<32> beta_u = M512(95, 64);

	float beta_f  = uint32_to_float(beta_u);

	ap_uint<512> c_read;
	bool c_read_ready = false;
	ap_uint<512> c_pe;
	bool c_pe_ready = false;

	const ap_uint<16> N16 = P_N(31, 16);
	const ap_uint<16> rp_time = (N16 == 0)? ((ap_uint<16>) 1) : N16;
	const ap_uint<32> N = P_N & ((ap_uint<32>) 0x0000FFFF);
	const ap_uint<32> num_ite_C = ((M + 7) / 8) * ((N+7)/8);

	l_rp: for(ap_uint<16> rp = 0; rp < rp_time; rp++) {
#pragma HLS loop_flatten off
#pragma HLS loop_tripcount min=1 max=16

		cc: for (ap_uint<32> i = 0; i < num_ite_C; ) {
#pragma HLS loop_tripcount min=1 max=5000
#pragma HLS pipeline II=1
			if (!c_read_ready) {
				c_read_ready = fifo_C_read_in.read_nb(c_read);
			}
			if (!c_pe_ready) {
				c_pe_ready = fifo_C_pe_in.read_nb(c_pe);
			}
			if (c_read_ready && c_pe_ready) {
				ap_uint<512> c_pe_delay = HLS_REG(c_pe);
				ap_uint<512> c_read_delay = HLS_REG(c_read);

				ap_uint<512> c_out;
				for(ap_uint<5> p = 0; p < 16; ++p) {
					float op_ab = uint32_to_float(c_pe_delay(31 + p * 32, p * 32));
					float op_c  = uint32_to_float(c_read_delay(31 + p * 32, p * 32));
					float op_result = op_ab + HLS_REG(beta_f) * op_c;
					c_out(31 + p * 32, p * 32) = float_to_uint32(op_result);
				}

				ap_uint<512> c_out_reg = HLS_REG(c_out);
				fifo_C_out.write(c_out_reg);
				c_read_ready = false;
				c_pe_ready = false;
				++i;
			}
		}
	}
}


#ifndef HLS
extern "C" {
#endif

void sextans(
	const ap_uint<32> *edge_list_ptr,

	const ap_uint<512> *edge_list_ch0,
	const ap_uint<512> *edge_list_ch1,
	const ap_uint<512> *edge_list_ch2,
	const ap_uint<512> *edge_list_ch3,

	const ap_uint<512>  *mat_B_ch0,
	const ap_uint<512>  *mat_B_ch1,
	const ap_uint<512>  *mat_B_ch2,
	const ap_uint<512>  *mat_B_ch3,

	ap_uint<512>  *mat_C_ch0,
	ap_uint<512>  *mat_C_ch1,
	ap_uint<512>  *mat_C_ch2,
	ap_uint<512>  *mat_C_ch3,

	const int NUM_ITE,
	const int NUM_A_LEN,
	const int M,
	const int K,
	const int P_N,
	const unsigned int alpha_u,
	const unsigned int beta_u
) {
#pragma HLS INTERFACE m_axi port = edge_list_ptr offset = slave bundle = gmemptr

#pragma HLS INTERFACE m_axi port = edge_list_ch0 offset = slave bundle = gmemA0
#pragma HLS INTERFACE m_axi port = edge_list_ch1 offset = slave bundle = gmemA1
#pragma HLS INTERFACE m_axi port = edge_list_ch2 offset = slave bundle = gmemA2
#pragma HLS INTERFACE m_axi port = edge_list_ch3 offset = slave bundle = gmemA3

#pragma HLS INTERFACE m_axi port = mat_B_ch0 offset = slave bundle = gmemB0
#pragma HLS INTERFACE m_axi port = mat_B_ch1 offset = slave bundle = gmemB1
#pragma HLS INTERFACE m_axi port = mat_B_ch2 offset = slave bundle = gmemB2
#pragma HLS INTERFACE m_axi port = mat_B_ch3 offset = slave bundle = gmemB3

#pragma HLS INTERFACE m_axi port = mat_C_ch0 offset = slave bundle = gmemC0
#pragma HLS INTERFACE m_axi port = mat_C_ch1 offset = slave bundle = gmemC1
#pragma HLS INTERFACE m_axi port = mat_C_ch2 offset = slave bundle = gmemC2
#pragma HLS INTERFACE m_axi port = mat_C_ch3 offset = slave bundle = gmemC3

	hls::stream<ap_uint<32> > fifo_edge_list_ptr_pe0("fifo_edge_list_ptr_pe0");
	hls::stream<ap_uint<32> > fifo_edge_list_ptr_pe1("fifo_edge_list_ptr_pe1");
	hls::stream<ap_uint<32> > fifo_edge_list_ptr_pe2("fifo_edge_list_ptr_pe2");
	hls::stream<ap_uint<32> > fifo_edge_list_ptr_pe3("fifo_edge_list_ptr_pe3");

#pragma HLS STREAM variable=fifo_edge_list_ptr_pe0 depth=8
#pragma HLS STREAM variable=fifo_edge_list_ptr_pe1 depth=8
#pragma HLS STREAM variable=fifo_edge_list_ptr_pe2 depth=8
#pragma HLS STREAM variable=fifo_edge_list_ptr_pe3 depth=8

#pragma HLS bind_storage variable=fifo_edge_list_ptr_pe0 type=FIFO impl=SRL
#pragma HLS bind_storage variable=fifo_edge_list_ptr_pe1 type=FIFO impl=SRL
#pragma HLS bind_storage variable=fifo_edge_list_ptr_pe2 type=FIFO impl=SRL
#pragma HLS bind_storage variable=fifo_edge_list_ptr_pe3 type=FIFO impl=SRL

	hls::stream<ap_uint<512> > fifo_A_pe0("fifo_A_pe0");
	hls::stream<ap_uint<512> > fifo_A_pe1("fifo_A_pe1");
	hls::stream<ap_uint<512> > fifo_A_pe2("fifo_A_pe2");
	hls::stream<ap_uint<512> > fifo_A_pe3("fifo_A_pe3");

#pragma HLS STREAM variable=fifo_A_pe0 depth=8
#pragma HLS STREAM variable=fifo_A_pe1 depth=8
#pragma HLS STREAM variable=fifo_A_pe2 depth=8
#pragma HLS STREAM variable=fifo_A_pe3 depth=8

#pragma HLS bind_storage variable=fifo_A_pe0 type=FIFO impl=SRL
#pragma HLS bind_storage variable=fifo_A_pe1 type=FIFO impl=SRL
#pragma HLS bind_storage variable=fifo_A_pe2 type=FIFO impl=SRL
#pragma HLS bind_storage variable=fifo_A_pe3 type=FIFO impl=SRL

	hls::stream<ap_uint<512> > fifo_B_pe0_x0("fifo_B_pe0_x0");
	hls::stream<ap_uint<512> > fifo_B_pe0_x1("fifo_B_pe0_x1");
	hls::stream<ap_uint<512> > fifo_B_pe0_x2("fifo_B_pe0_x2");
	hls::stream<ap_uint<512> > fifo_B_pe0_x3("fifo_B_pe0_x3");
	hls::stream<ap_uint<512> > fifo_B_pe1_x0("fifo_B_pe1_x0");
	hls::stream<ap_uint<512> > fifo_B_pe1_x1("fifo_B_pe1_x1");
	hls::stream<ap_uint<512> > fifo_B_pe1_x2("fifo_B_pe1_x2");
	hls::stream<ap_uint<512> > fifo_B_pe1_x3("fifo_B_pe1_x3");
	hls::stream<ap_uint<512> > fifo_B_pe2_x0("fifo_B_pe2_x0");
	hls::stream<ap_uint<512> > fifo_B_pe2_x1("fifo_B_pe2_x1");
	hls::stream<ap_uint<512> > fifo_B_pe2_x2("fifo_B_pe2_x2");
	hls::stream<ap_uint<512> > fifo_B_pe2_x3("fifo_B_pe2_x3");
	hls::stream<ap_uint<512> > fifo_B_pe3_x0("fifo_B_pe3_x0");
	hls::stream<ap_uint<512> > fifo_B_pe3_x1("fifo_B_pe3_x1");
	hls::stream<ap_uint<512> > fifo_B_pe3_x2("fifo_B_pe3_x2");
	hls::stream<ap_uint<512> > fifo_B_pe3_x3("fifo_B_pe3_x3");

#pragma HLS STREAM variable=fifo_B_pe0_x0 depth=8
#pragma HLS STREAM variable=fifo_B_pe0_x1 depth=8
#pragma HLS STREAM variable=fifo_B_pe0_x2 depth=8
#pragma HLS STREAM variable=fifo_B_pe0_x3 depth=8
#pragma HLS STREAM variable=fifo_B_pe1_x0 depth=8
#pragma HLS STREAM variable=fifo_B_pe1_x1 depth=8
#pragma HLS STREAM variable=fifo_B_pe1_x2 depth=8
#pragma HLS STREAM variable=fifo_B_pe1_x3 depth=8
#pragma HLS STREAM variable=fifo_B_pe2_x0 depth=8
#pragma HLS STREAM variable=fifo_B_pe2_x1 depth=8
#pragma HLS STREAM variable=fifo_B_pe2_x2 depth=8
#pragma HLS STREAM variable=fifo_B_pe2_x3 depth=8
#pragma HLS STREAM variable=fifo_B_pe3_x0 depth=8
#pragma HLS STREAM variable=fifo_B_pe3_x1 depth=8
#pragma HLS STREAM variable=fifo_B_pe3_x2 depth=8
#pragma HLS STREAM variable=fifo_B_pe3_x3 depth=8

#pragma HLS bind_storage variable=fifo_B_pe0_x0 type=FIFO impl=SRL
#pragma HLS bind_storage variable=fifo_B_pe0_x1 type=FIFO impl=SRL
#pragma HLS bind_storage variable=fifo_B_pe0_x2 type=FIFO impl=SRL
#pragma HLS bind_storage variable=fifo_B_pe0_x3 type=FIFO impl=SRL
#pragma HLS bind_storage variable=fifo_B_pe1_x0 type=FIFO impl=SRL
#pragma HLS bind_storage variable=fifo_B_pe1_x1 type=FIFO impl=SRL
#pragma HLS bind_storage variable=fifo_B_pe1_x2 type=FIFO impl=SRL
#pragma HLS bind_storage variable=fifo_B_pe1_x3 type=FIFO impl=SRL
#pragma HLS bind_storage variable=fifo_B_pe2_x0 type=FIFO impl=SRL
#pragma HLS bind_storage variable=fifo_B_pe2_x1 type=FIFO impl=SRL
#pragma HLS bind_storage variable=fifo_B_pe2_x2 type=FIFO impl=SRL
#pragma HLS bind_storage variable=fifo_B_pe2_x3 type=FIFO impl=SRL
#pragma HLS bind_storage variable=fifo_B_pe3_x0 type=FIFO impl=SRL
#pragma HLS bind_storage variable=fifo_B_pe3_x1 type=FIFO impl=SRL
#pragma HLS bind_storage variable=fifo_B_pe3_x2 type=FIFO impl=SRL
#pragma HLS bind_storage variable=fifo_B_pe3_x3 type=FIFO impl=SRL

	hls::stream<ap_uint<512> > fifo_C_pe0("fifo_C_pe0");
	hls::stream<ap_uint<512> > fifo_C_pe1("fifo_C_pe1");
	hls::stream<ap_uint<512> > fifo_C_pe2("fifo_C_pe2");
	hls::stream<ap_uint<512> > fifo_C_pe3("fifo_C_pe3");

#pragma HLS STREAM variable=fifo_C_pe0 depth=8
#pragma HLS STREAM variable=fifo_C_pe1 depth=8
#pragma HLS STREAM variable=fifo_C_pe2 depth=8
#pragma HLS STREAM variable=fifo_C_pe3 depth=8

#pragma HLS bind_storage variable=fifo_C_pe0 type=FIFO impl=SRL
#pragma HLS bind_storage variable=fifo_C_pe1 type=FIFO impl=SRL
#pragma HLS bind_storage variable=fifo_C_pe2 type=FIFO impl=SRL
#pragma HLS bind_storage variable=fifo_C_pe3 type=FIFO impl=SRL

	hls::stream<ap_uint<512> > fifo_C_ch0("fifo_C_ch0");
	hls::stream<ap_uint<512> > fifo_C_ch1("fifo_C_ch1");
	hls::stream<ap_uint<512> > fifo_C_ch2("fifo_C_ch2");
	hls::stream<ap_uint<512> > fifo_C_ch3("fifo_C_ch3");

#pragma HLS STREAM variable=fifo_C_ch0 depth=32
#pragma HLS STREAM variable=fifo_C_ch1 depth=32
#pragma HLS STREAM variable=fifo_C_ch2 depth=32
#pragma HLS STREAM variable=fifo_C_ch3 depth=32

#pragma HLS bind_storage variable=fifo_C_ch0 type=FIFO impl=SRL
#pragma HLS bind_storage variable=fifo_C_ch1 type=FIFO impl=SRL
#pragma HLS bind_storage variable=fifo_C_ch2 type=FIFO impl=SRL
#pragma HLS bind_storage variable=fifo_C_ch3 type=FIFO impl=SRL

	hls::stream<ap_uint<512> > fifo_C_read_in0("fifo_C_ch_in0");
	hls::stream<ap_uint<512> > fifo_C_read_in1("fifo_C_ch_in1");
	hls::stream<ap_uint<512> > fifo_C_read_in2("fifo_C_ch_in2");
	hls::stream<ap_uint<512> > fifo_C_read_in3("fifo_C_ch_in3");

#pragma HLS STREAM variable=fifo_C_read_in0 depth=800
#pragma HLS STREAM variable=fifo_C_read_in1 depth=800
#pragma HLS STREAM variable=fifo_C_read_in2 depth=800
#pragma HLS STREAM variable=fifo_C_read_in3 depth=800

#pragma HLS bind_storage variable=fifo_C_read_in0 type=FIFO impl=BRAM
#pragma HLS bind_storage variable=fifo_C_read_in1 type=FIFO impl=BRAM
#pragma HLS bind_storage variable=fifo_C_read_in2 type=FIFO impl=BRAM
#pragma HLS bind_storage variable=fifo_C_read_in3 type=FIFO impl=BRAM

#pragma HLS dataflow

	ap_uint<32> A_LEN0 = HLS_REG(NUM_A_LEN);
	ap_uint<32> A_LEN1 = HLS_REG(A_LEN0);
	ap_uint<32> A_LEN2 = HLS_REG(A_LEN1);
	ap_uint<32> A_LEN3 = HLS_REG(A_LEN2);

	ap_uint<32> N_rdA0 = HLS_REG(P_N);
	ap_uint<32> N_rdA1 = HLS_REG(N_rdA0);
	ap_uint<32> N_rdA2 = HLS_REG(N_rdA1);
	ap_uint<32> N_rdA3 = HLS_REG(N_rdA2);

	ap_uint<32> N_rdB0 = HLS_REG(P_N);
	ap_uint<32> N_rdB1 = HLS_REG(N_rdB0);
	ap_uint<32> N_rdB2 = HLS_REG(N_rdB1);
	ap_uint<32> N_rdB3 = HLS_REG(N_rdB2);

	ap_uint<32> K0 = HLS_REG(K);
	ap_uint<32> K1 = HLS_REG(K0);
	ap_uint<32> K2 = HLS_REG(K1);
	ap_uint<32> K3 = HLS_REG(K2);

	read_edge_list_ptr(
		NUM_ITE,
		M,
		P_N,
		K,
		alpha_u,
		beta_u,
		edge_list_ptr,
		fifo_edge_list_ptr_pe0
		);

	read_A<0>(
		edge_list_ch0,
		fifo_A_pe0,
		A_LEN0,
		N_rdA0
		);

	read_A<1>(
		edge_list_ch1,
		fifo_A_pe1,
		A_LEN1,
		N_rdA1
		);

	read_A<2>(
		edge_list_ch2,
		fifo_A_pe2,
		A_LEN2,
		N_rdA2
		);

	read_A<3>(
		edge_list_ch3,
		fifo_A_pe3,
		A_LEN3,
		N_rdA3
		);

	read_B<0>(
		mat_B_ch0,
		fifo_B_pe0_x0,
		K0,
		N_rdB0
		);

	read_B<1>(
		mat_B_ch1,
		fifo_B_pe0_x1,
		K1,
		N_rdB1
		);

	read_B<2>(
		mat_B_ch2,
		fifo_B_pe0_x2,
		K2,
		N_rdB2
		);

	read_B<3>(
		mat_B_ch3,
		fifo_B_pe0_x3,
		K3,
		N_rdB3
		);

	PEG<0>(
		fifo_edge_list_ptr_pe0,
		fifo_A_pe0,
		fifo_B_pe0_x0,
		fifo_B_pe0_x1,
		fifo_B_pe0_x2,
		fifo_B_pe0_x3,
		fifo_edge_list_ptr_pe1,
		fifo_B_pe1_x0,
		fifo_B_pe1_x1,
		fifo_B_pe1_x2,
		fifo_B_pe1_x3,
		fifo_C_pe0
		);

	PEG<1>(
		fifo_edge_list_ptr_pe1,
		fifo_A_pe1,
		fifo_B_pe1_x0,
		fifo_B_pe1_x1,
		fifo_B_pe1_x2,
		fifo_B_pe1_x3,
		fifo_edge_list_ptr_pe2,
		fifo_B_pe2_x0,
		fifo_B_pe2_x1,
		fifo_B_pe2_x2,
		fifo_B_pe2_x3,
		fifo_C_pe1
		);

	PEG<2>(
		fifo_edge_list_ptr_pe2,
		fifo_A_pe2,
		fifo_B_pe2_x0,
		fifo_B_pe2_x1,
		fifo_B_pe2_x2,
		fifo_B_pe2_x3,
		fifo_edge_list_ptr_pe3,
		fifo_B_pe3_x0,
		fifo_B_pe3_x1,
		fifo_B_pe3_x2,
		fifo_B_pe3_x3,
		fifo_C_pe2
		);

	PEG_last<3>(
		fifo_edge_list_ptr_pe3,
		fifo_A_pe3,
		fifo_B_pe3_x0,
		fifo_B_pe3_x1,
		fifo_B_pe3_x2,
		fifo_B_pe3_x3,
		fifo_C_pe3
		);

	comp_C<0>(
		fifo_C_read_in0,
		fifo_C_pe0,
		fifo_C_ch0
		);

	comp_C<1>(
		fifo_C_read_in1,
		fifo_C_pe1,
		fifo_C_ch1
		);

	comp_C<2>(
		fifo_C_read_in2,
		fifo_C_pe2,
		fifo_C_ch2
		);

	comp_C<3>(
		fifo_C_read_in3,
		fifo_C_pe3,
		fifo_C_ch3
		);

	C_IO<0>(
		fifo_C_ch0,
		fifo_C_read_in0,
		mat_C_ch0
		);

	C_IO<1>(
		fifo_C_ch1,
		fifo_C_read_in1,
		mat_C_ch1
		);

	C_IO<2>(
		fifo_C_ch2,
		fifo_C_read_in2,
		mat_C_ch2
		);

	C_IO<3>(
		fifo_C_ch3,
		fifo_C_read_in3,
		mat_C_ch3
		);
}

#ifndef HLS
} // end of extern C
#endif

