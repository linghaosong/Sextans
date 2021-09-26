#include <ap_int.h>
#include <cstdio>
#include <cstring>
#include <cassert>

#include <tapa.h>

const int WINDOW_SIZE = 4096;
const int DEP_DIST_LOAD_STORE = 10;
const int B_PARTITION_FACTOR = 4;
const int URAM_DEPTH = 12288;

template<class T>
T HLS_REG(T in) {
#pragma HLS pipeline II=1
#pragma HLS inline off
#pragma HLS LATENCY min=1 max=1
	return in;
}

float uint32_to_float(ap_uint<32> u) {
#pragma HLS inline
#pragma HLS pipeline II=1
	float * tmpPointer_v = (float*) & u;
	return (*tmpPointer_v);
}

ap_uint<32> float_to_uint32(float u) {
#pragma HLS inline
#pragma HLS pipeline II=1
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
	tapa::ostream<ap_uint<32> > & fifo_edge_list_ptr
	) {
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
	tapa::ostream<ap_uint<512> > & fifo_A,
	const ap_uint<32> A_len,
	const ap_uint<32> P_N
	) {
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
	tapa::ostream<ap_uint<512> > & fifo_B,
	const ap_uint<32> K,
	const ap_uint<32> P_N
	) {
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
	tapa::istream<ap_uint<32> > & fifo_inst,
	tapa::istream<ap_uint<512> > & fifo_A,
	tapa::istream<ap_uint<512> > & fifo_B_x0, // [256(16)] * 2, 2: dim d
	tapa::istream<ap_uint<512> > & fifo_B_x1, // [256(16)] * 2, 2: dim d
	tapa::istream<ap_uint<512> > & fifo_B_x2, // [256(16)] * 2, 2: dim d
	tapa::istream<ap_uint<512> > & fifo_B_x3, // [256(16)] * 2, 2: dim d
	tapa::ostream<ap_uint<32> > & fifo_inst_out, // to next PE
	tapa::ostream<ap_uint<512> > & fifo_B_out_x0, // output to next PE [256(16)] * 2, 2: dim d
	tapa::ostream<ap_uint<512> > & fifo_B_out_x1, // output to next PE [256(16)] * 2, 2: dim d
	tapa::ostream<ap_uint<512> > & fifo_B_out_x2, // output to next PE [256(16)] * 2, 2: dim d
	tapa::ostream<ap_uint<512> > & fifo_B_out_x3, // output to next PE [256(16)] * 2, 2: dim d
	tapa::ostream<ap_uint<512> > & fifo_C_out0 // [64(32bits * 2.0)] * 8 dims
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
		bool parameter_ready = fifo_inst.try_read(parameter);
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
			init_C: for (ap_uint<32> i = 0; i < ((M + 63) / 64); ++i) {
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

			ap_uint<32> start_32;
			bool start_32_ready = false;
			w1: while(!start_32_ready) {
#pragma HLS loop_tripcount min=1 max=10
#pragma HLS pipeline II=1
				start_32_ready = fifo_inst.try_read(start_32);
			};

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
#pragma HLS pipeline II = 1
					if (!b_512_x0_ready) {
						b_512_x0_ready = fifo_B_x0.try_read(b_512_x0);
					}
					if (!b_512_x1_ready) {
						b_512_x1_ready = fifo_B_x1.try_read(b_512_x1);
					}
					if (!b_512_x2_ready) {
						b_512_x2_ready = fifo_B_x2.try_read(b_512_x2);
					}
					if (!b_512_x3_ready) {
						b_512_x3_ready = fifo_B_x3.try_read(b_512_x3);
					}

					bool b_2048_ready = b_512_x0_ready && b_512_x1_ready && b_512_x2_ready && b_512_x3_ready;

					if (b_2048_ready) {
						ap_uint<512> b_512_x0_delay = HLS_REG(b_512_x0);
						ap_uint<512> b_512_x1_delay = HLS_REG(b_512_x1);
						ap_uint<512> b_512_x2_delay = HLS_REG(b_512_x2);
						ap_uint<512> b_512_x3_delay = HLS_REG(b_512_x3);

						fifo_B_out_x0.write(b_512_x0_delay);
						fifo_B_out_x1.write(b_512_x1_delay);
						fifo_B_out_x2.write(b_512_x2_delay);
						fifo_B_out_x3.write(b_512_x3_delay);

						read_B_p: for (ap_uint<4> k = 0; k < 8; ++k) {
							ap_uint<32> b_pe_d0 = b_512_x0_delay(31 + k * 32 +   0,  k * 32 +   0);
							ap_uint<32> b_pe_d1 = b_512_x0_delay(31 + k * 32 + 256,  k * 32 + 256);
							ap_uint<32> b_pe_d2 = b_512_x1_delay(31 + k * 32 +   0,  k * 32 +   0);
							ap_uint<32> b_pe_d3 = b_512_x1_delay(31 + k * 32 + 256,  k * 32 + 256);
							ap_uint<32> b_pe_d4 = b_512_x2_delay(31 + k * 32 +   0,  k * 32 +   0);
							ap_uint<32> b_pe_d5 = b_512_x2_delay(31 + k * 32 + 256,  k * 32 + 256);
							ap_uint<32> b_pe_d6 = b_512_x3_delay(31 + k * 32 +   0,  k * 32 +   0);
							ap_uint<32> b_pe_d7 = b_512_x3_delay(31 + k * 32 + 256,  k * 32 + 256);

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
				ap_uint<32> end_32;
				bool end_32_ready = false;
				w2: while(!end_32_ready) {
#pragma HLS loop_tripcount min=1 max=10
#pragma HLS pipeline II=1
					end_32_ready = fifo_inst.try_read(end_32);
				};

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
					bool a_pes_ready = fifo_A.try_read(a_pes);

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
			write_C_outer: for (ap_uint<32> i = 0; i < (M + 15)/16; ++i) {
#pragma HLS loop_tripcount min=1 max=1800
#pragma HLS pipeline II=1
				ap_uint<64> u_64_pe_d[2][4];
#pragma HLS array_partition variable=u_64_pe_d complete

				ap_uint<32> u_32_d_pe[8][2];
#pragma HLS array_partition variable=u_32_d_pe complete

				switch (i % 4) {
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

				for (ap_uint<4> pe = 0; pe < 2; ++pe) {
					for (ap_uint<4> d = 0; d < 4; ++d) {
						u_32_d_pe[2 * d    ][pe] = (u_64_pe_d[pe][d])(31,  0);
						u_32_d_pe[2 * d + 1][pe] = (u_64_pe_d[pe][d])(63, 32);
					}
				}

				for (ap_uint<4> d = 0; d < 8; ++d) {
					for (ap_uint<4> pe = 0; pe < 2; ++pe) {
						out_u0(31 + pe * 32 + d * 64, pe * 32 + d * 64) = u_32_d_pe[d + 0][pe];
					}
				}

				ap_uint<512> out_u0_mult;
				peg16mult(out_u0, alpha_u, out_u0_mult);
				fifo_C_out0.write(out_u0_mult);
			}
		}
	}
}

template <int ch>
void PEG_last(
	tapa::istream<ap_uint<32> > & fifo_inst,
	tapa::istream<ap_uint<512> > & fifo_A,
	tapa::istream<ap_uint<512> > & fifo_B_x0, // [256(16)] * 2, 2: dim d
	tapa::istream<ap_uint<512> > & fifo_B_x1, // [256(16)] * 2, 2: dim d
	tapa::istream<ap_uint<512> > & fifo_B_x2, // [256(16)] * 2, 2: dim d
	tapa::istream<ap_uint<512> > & fifo_B_x3, // [256(16)] * 2, 2: dim d
	tapa::ostream<ap_uint<512> > & fifo_C_out0 // [64(32bits * 2.0)] * 8 dims
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
		bool parameter_ready = fifo_inst.try_read(parameter);
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
			init_C: for (ap_uint<32> i = 0; i < ((M + 63) / 64); ++i) {
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

			ap_uint<32> start_32;
			bool start_32_ready = false;
			w1: while(!start_32_ready) {
#pragma HLS loop_tripcount min=1 max=10
#pragma HLS pipeline II=1
				start_32_ready = fifo_inst.try_read(start_32);
			};

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
						b_512_x0_ready = fifo_B_x0.try_read(b_512_x0);
					}
					if (!b_512_x1_ready) {
						b_512_x1_ready = fifo_B_x1.try_read(b_512_x1);
					}
					if (!b_512_x2_ready) {
						b_512_x2_ready = fifo_B_x2.try_read(b_512_x2);
					}
					if (!b_512_x3_ready) {
						b_512_x3_ready = fifo_B_x3.try_read(b_512_x3);
					}

					bool b_2048_ready = b_512_x0_ready && b_512_x1_ready && b_512_x2_ready && b_512_x3_ready;

					if (b_2048_ready) {
						ap_uint<512> b_512_x0_delay = HLS_REG(b_512_x0);
						ap_uint<512> b_512_x1_delay = HLS_REG(b_512_x1);
						ap_uint<512> b_512_x2_delay = HLS_REG(b_512_x2);
						ap_uint<512> b_512_x3_delay = HLS_REG(b_512_x3);


						read_B_p: for (ap_uint<4> k = 0; k < 8; ++k) {
							ap_uint<32> b_pe_d0 = b_512_x0_delay(31 + k * 32 +   0,  k * 32 +   0);
							ap_uint<32> b_pe_d1 = b_512_x0_delay(31 + k * 32 + 256,  k * 32 + 256);
							ap_uint<32> b_pe_d2 = b_512_x1_delay(31 + k * 32 +   0,  k * 32 +   0);
							ap_uint<32> b_pe_d3 = b_512_x1_delay(31 + k * 32 + 256,  k * 32 + 256);
							ap_uint<32> b_pe_d4 = b_512_x2_delay(31 + k * 32 +   0,  k * 32 +   0);
							ap_uint<32> b_pe_d5 = b_512_x2_delay(31 + k * 32 + 256,  k * 32 + 256);
							ap_uint<32> b_pe_d6 = b_512_x3_delay(31 + k * 32 +   0,  k * 32 +   0);
							ap_uint<32> b_pe_d7 = b_512_x3_delay(31 + k * 32 + 256,  k * 32 + 256);

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
				ap_uint<32> end_32;
				bool end_32_ready = false;
				w2: while(!end_32_ready) {
#pragma HLS loop_tripcount min=1 max=10
#pragma HLS pipeline II=1
					end_32_ready = fifo_inst.try_read(end_32);
				};

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
					bool a_pes_ready = fifo_A.try_read(a_pes);

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
			write_C_outer: for (ap_uint<32> i = 0; i < (M + 15)/16; ++i) {
#pragma HLS loop_tripcount min=1 max=1800
#pragma HLS pipeline II=1
				ap_uint<64> u_64_pe_d[2][4];
#pragma HLS array_partition variable=u_64_pe_d complete

				ap_uint<32> u_32_d_pe[8][2];
#pragma HLS array_partition variable=u_32_d_pe complete

				switch (i % 4) {
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

				for (ap_uint<4> pe = 0; pe < 2; ++pe) {
					for (ap_uint<4> d = 0; d < 4; ++d) {
						u_32_d_pe[2 * d    ][pe] = (u_64_pe_d[pe][d])(31,  0);
						u_32_d_pe[2 * d + 1][pe] = (u_64_pe_d[pe][d])(63, 32);
					}
				}

				for (ap_uint<4> d = 0; d < 8; ++d) {
					for (ap_uint<4> pe = 0; pe < 2; ++pe) {
						out_u0(31 + pe * 32 + d * 64, pe * 32 + d * 64) = u_32_d_pe[d + 0][pe];
					}
				}

				ap_uint<512> out_u0_mult;
				peg16mult(out_u0, alpha_u, out_u0_mult);
				fifo_C_out0.write(out_u0_mult);
			}
		}
	}
}

template <int id>
void C_collect(
    tapa::istream<ap_uint<512> > & fifo_C_in0,
    tapa::istream<ap_uint<512> > & fifo_C_in1,
    tapa::istream<ap_uint<512> > & fifo_C_in2,
    tapa::istream<ap_uint<512> > & fifo_C_in3,
    tapa::istream<ap_uint<512> > & fifo_C_in4,
    tapa::istream<ap_uint<512> > & fifo_C_in5,
    tapa::istream<ap_uint<512> > & fifo_C_in6,
    tapa::istream<ap_uint<512> > & fifo_C_in7,

    tapa::ostream<ap_uint<512> > & fifo_C_out0,
    tapa::ostream<ap_uint<512> > & fifo_C_out1,
    tapa::ostream<ap_uint<512> > & fifo_C_out2,
    tapa::ostream<ap_uint<512> > & fifo_C_out3,
    tapa::ostream<ap_uint<512> > & fifo_C_out4,
    tapa::ostream<ap_uint<512> > & fifo_C_out5,
    tapa::ostream<ap_uint<512> > & fifo_C_out6,
    tapa::ostream<ap_uint<512> > & fifo_C_out7
    ) {
    ap_uint<512> tmp_c0;
    ap_uint<512> tmp_c1;
    ap_uint<512> tmp_c2;
    ap_uint<512> tmp_c3;
    ap_uint<512> tmp_c4;
    ap_uint<512> tmp_c5;
    ap_uint<512> tmp_c6;
    ap_uint<512> tmp_c7;

    bool c0_ready = false;
    bool c1_ready = false;
    bool c2_ready = false;
    bool c3_ready = false;
    bool c4_ready = false;
    bool c5_ready = false;
    bool c6_ready = false;
    bool c7_ready = false;

    ap_uint<512> MN512;
    bool M512_ready = false;
    w_Mxx: while(!M512_ready) {
#pragma HLS loop_tripcount min=1 max=10
#pragma HLS pipeline II=1
        M512_ready = fifo_C_in7.try_read(MN512);
    };
    ap_uint<32> M = MN512(31, 0);
    ap_uint<32> P_N = MN512(63, 32);

    ap_uint<512> out_c0;
    ap_uint<512> out_c1;
    ap_uint<512> out_c2;
    ap_uint<512> out_c3;
    ap_uint<512> out_c4;
    ap_uint<512> out_c5;
    ap_uint<512> out_c6;
    ap_uint<512> out_c7;

    fifo_C_out0.write(HLS_REG(MN512));
    fifo_C_out1.write(HLS_REG(MN512));
    fifo_C_out2.write(HLS_REG(MN512));
    fifo_C_out3.write(HLS_REG(MN512));
    fifo_C_out4.write(HLS_REG(MN512));
    fifo_C_out5.write(HLS_REG(MN512));
    fifo_C_out6.write(HLS_REG(MN512));
    fifo_C_out7.write(HLS_REG(MN512));

    const ap_uint<16> N16 = P_N(31, 16);
    const ap_uint<16> rp_time = (N16 == 0)? ((ap_uint<16>) 1) : N16;
    const ap_uint<32> N = P_N & ((ap_uint<32>) 0x0000FFFF);
    const ap_uint<32> num_ite = ((M + 15) / 16) * ((N+7)/8);

    l_rp: for(ap_uint<16> rp = 0; rp < rp_time; rp++) {
#pragma HLS loop_flatten off
#pragma HLS loop_tripcount min=1 max=16
        cvt_b: for(ap_uint<32> i = 0; i < num_ite; ) {
#pragma HLS loop_tripcount min=1 max=800
#pragma HLS pipeline II=1
            if (!c0_ready) {
                c0_ready = fifo_C_in0.try_read(tmp_c0);
            }
            if (!c1_ready) {
                c1_ready = fifo_C_in1.try_read(tmp_c1);
            }
            if (!c2_ready) {
                c2_ready = fifo_C_in2.try_read(tmp_c2);
            }
            if (!c3_ready) {
                c3_ready = fifo_C_in3.try_read(tmp_c3);
            }
            if (!c4_ready) {
                c4_ready = fifo_C_in4.try_read(tmp_c4);
            }
            if (!c5_ready) {
                c5_ready = fifo_C_in5.try_read(tmp_c5);
            }
            if (!c6_ready) {
                c6_ready = fifo_C_in6.try_read(tmp_c6);
            }
            if (!c7_ready) {
                c7_ready = fifo_C_in7.try_read(tmp_c7);
            }

            bool all_c_ready = c0_ready && c1_ready && c2_ready && c3_ready && c4_ready && c5_ready && c6_ready && c7_ready;

            if (all_c_ready) {
                ap_uint<512> tmp_c0_delay = HLS_REG(tmp_c0);
                ap_uint<512> tmp_c1_delay = HLS_REG(tmp_c1);
                ap_uint<512> tmp_c2_delay = HLS_REG(tmp_c2);
                ap_uint<512> tmp_c3_delay = HLS_REG(tmp_c3);
                ap_uint<512> tmp_c4_delay = HLS_REG(tmp_c4);
                ap_uint<512> tmp_c5_delay = HLS_REG(tmp_c5);
                ap_uint<512> tmp_c6_delay = HLS_REG(tmp_c6);
                ap_uint<512> tmp_c7_delay = HLS_REG(tmp_c7);

                out_c0( 63,   0) = tmp_c0_delay(  63,    0);
                out_c0(127,  64) = tmp_c1_delay(  63,    0);
                out_c0(191, 128) = tmp_c2_delay(  63,    0);
                out_c0(255, 192) = tmp_c3_delay(  63,    0);
                out_c0(319, 256) = tmp_c4_delay(  63,    0);
                out_c0(383, 320) = tmp_c5_delay(  63,    0);
                out_c0(447, 384) = tmp_c6_delay(  63,    0);
                out_c0(511, 448) = tmp_c7_delay(  63,    0);

                out_c1( 63,   0) = tmp_c0_delay( 127,   64);
                out_c1(127,  64) = tmp_c1_delay( 127,   64);
                out_c1(191, 128) = tmp_c2_delay( 127,   64);
                out_c1(255, 192) = tmp_c3_delay( 127,   64);
                out_c1(319, 256) = tmp_c4_delay( 127,   64);
                out_c1(383, 320) = tmp_c5_delay( 127,   64);
                out_c1(447, 384) = tmp_c6_delay( 127,   64);
                out_c1(511, 448) = tmp_c7_delay( 127,   64);

                out_c2( 63,   0) = tmp_c0_delay( 191,  128);
                out_c2(127,  64) = tmp_c1_delay( 191,  128);
                out_c2(191, 128) = tmp_c2_delay( 191,  128);
                out_c2(255, 192) = tmp_c3_delay( 191,  128);
                out_c2(319, 256) = tmp_c4_delay( 191,  128);
                out_c2(383, 320) = tmp_c5_delay( 191,  128);
                out_c2(447, 384) = tmp_c6_delay( 191,  128);
                out_c2(511, 448) = tmp_c7_delay( 191,  128);

                out_c3( 63,   0) = tmp_c0_delay( 255,  192);
                out_c3(127,  64) = tmp_c1_delay( 255,  192);
                out_c3(191, 128) = tmp_c2_delay( 255,  192);
                out_c3(255, 192) = tmp_c3_delay( 255,  192);
                out_c3(319, 256) = tmp_c4_delay( 255,  192);
                out_c3(383, 320) = tmp_c5_delay( 255,  192);
                out_c3(447, 384) = tmp_c6_delay( 255,  192);
                out_c3(511, 448) = tmp_c7_delay( 255,  192);

                out_c4( 63,   0) = tmp_c0_delay( 319,  256);
                out_c4(127,  64) = tmp_c1_delay( 319,  256);
                out_c4(191, 128) = tmp_c2_delay( 319,  256);
                out_c4(255, 192) = tmp_c3_delay( 319,  256);
                out_c4(319, 256) = tmp_c4_delay( 319,  256);
                out_c4(383, 320) = tmp_c5_delay( 319,  256);
                out_c4(447, 384) = tmp_c6_delay( 319,  256);
                out_c4(511, 448) = tmp_c7_delay( 319,  256);

                out_c5( 63,   0) = tmp_c0_delay( 383,  320);
                out_c5(127,  64) = tmp_c1_delay( 383,  320);
                out_c5(191, 128) = tmp_c2_delay( 383,  320);
                out_c5(255, 192) = tmp_c3_delay( 383,  320);
                out_c5(319, 256) = tmp_c4_delay( 383,  320);
                out_c5(383, 320) = tmp_c5_delay( 383,  320);
                out_c5(447, 384) = tmp_c6_delay( 383,  320);
                out_c5(511, 448) = tmp_c7_delay( 383,  320);

                out_c6( 63,   0) = tmp_c0_delay( 447,  384);
                out_c6(127,  64) = tmp_c1_delay( 447,  384);
                out_c6(191, 128) = tmp_c2_delay( 447,  384);
                out_c6(255, 192) = tmp_c3_delay( 447,  384);
                out_c6(319, 256) = tmp_c4_delay( 447,  384);
                out_c6(383, 320) = tmp_c5_delay( 447,  384);
                out_c6(447, 384) = tmp_c6_delay( 447,  384);
                out_c6(511, 448) = tmp_c7_delay( 447,  384);

                out_c7( 63,   0) = tmp_c0_delay( 511,  448);
                out_c7(127,  64) = tmp_c1_delay( 511,  448);
                out_c7(191, 128) = tmp_c2_delay( 511,  448);
                out_c7(255, 192) = tmp_c3_delay( 511,  448);
                out_c7(319, 256) = tmp_c4_delay( 511,  448);
                out_c7(383, 320) = tmp_c5_delay( 511,  448);
                out_c7(447, 384) = tmp_c6_delay( 511,  448);
                out_c7(511, 448) = tmp_c7_delay( 511,  448);

                fifo_C_out0.write(out_c0);
                fifo_C_out1.write(out_c1);
                fifo_C_out2.write(out_c2);
                fifo_C_out3.write(out_c3);
                fifo_C_out4.write(out_c4);
                fifo_C_out5.write(out_c5);
                fifo_C_out6.write(out_c6);
                fifo_C_out7.write(out_c7);

                c0_ready = false;
                c1_ready = false;
                c2_ready = false;
                c3_ready = false;
                c4_ready = false;
                c5_ready = false;
                c6_ready = false;
                c7_ready = false;
                ++i;
            }
        }
    }
}


template <int ch>
void write_C(
	tapa::istream<ap_uint<512> > & fifo_C,
	ap_uint<512>* C_out
	) {
	ap_uint<512> M_u512;
	bool M_ready = false;
	w_M: while(!M_ready) {
#pragma HLS loop_tripcount min=1 max=10
#pragma HLS pipeline II=1
		M_ready = fifo_C.try_read(M_u512);
	};
	ap_uint<32> M = M_u512(31, 0);
	ap_uint<32> P_N = M_u512(63, 32);

	const ap_uint<16> N16 = P_N(31, 16);
	const ap_uint<16> rp_time = (N16 == 0)? ((ap_uint<16>) 1) : N16;
	const ap_uint<32> N = P_N & ((ap_uint<32>) 0x0000FFFF);
	const ap_uint<32> num_ite_C = ((M + 15)/16) * ((N+7)/8);

	l_rp: for(ap_uint<16> rp = 0; rp < rp_time; rp++) {
#pragma HLS loop_flatten off
#pragma HLS loop_tripcount min=1 max=16
		wr_C: for(ap_uint<32> i = 0; i < num_ite_C; i++) {
#pragma HLS loop_tripcount min=1 max=500000
#pragma HLS pipeline II=1
			ap_uint<512> tmp_c = fifo_C.read();
			C_out[i] = tmp_c;
		}
	}
}

template <int ch>
void read_C(
	ap_uint<512>* C,
	tapa::ostream<ap_uint<512> > & fifo_C,
	const ap_uint<32> M,
	const ap_uint<32> P_N
	) {
	const ap_uint<16> N16 = P_N(31, 16);
	const ap_uint<16> rp_time = (N16 == 0)? ((ap_uint<16>) 1) : N16;
	const ap_uint<32> N = P_N & ((ap_uint<32>) 0x0000FFFF);
	const ap_uint<32> num_ite_C = ((M + 15) / 16) * ((N+7)/8);

	l_rp: for(ap_uint<16> rp = 0; rp < rp_time; rp++) {
#pragma HLS loop_flatten off
#pragma HLS loop_tripcount min=1 max=16
		rd_C: for(ap_uint<32> i = 0; i < num_ite_C; i++) {
#pragma HLS loop_tripcount min=1 max=500000
#pragma HLS pipeline II=1
			ap_uint<512> tmp_c = C[i];
			fifo_C.write(tmp_c);
		}
	}
}

template <int ch>
void comp_C(
	tapa::istream<ap_uint<512> > & fifo_C_read_in,
	tapa::istream<ap_uint<512> > & fifo_C_pe_in,
	tapa::ostream<ap_uint<512> > & fifo_C_out
	) {
	bool M_ready = false;
	ap_uint<512> M512;
	w_Mxx: while(!M_ready) {
#pragma HLS loop_tripcount min=1 max=10
#pragma HLS pipeline II=1
		M_ready = fifo_C_pe_in.try_read(M512);
	};
	fifo_C_out.write(M512);
	ap_uint<32> M = M512(31, 0);
	ap_uint<32> P_N = M512(63, 32);
	ap_uint<32> beta_u = M512(95, 64);

	float beta_f  = uint32_to_float(beta_u);

	ap_uint<512> c_out;
	ap_uint<512> c_read;
	ap_uint<512> c_pe;

	bool c_read_ready = false;
	bool c_pe_ready = false;

	const ap_uint<16> N16 = P_N(31, 16);
	const ap_uint<16> rp_time = (N16 == 0)? ((ap_uint<16>) 1) : N16;
	const ap_uint<32> N = P_N & ((ap_uint<32>) 0x0000FFFF);
	const ap_uint<32> num_ite_C = ((M + 15) / 16) * ((N+7)/8);

	l_rp: for(ap_uint<16> rp = 0; rp < rp_time; rp++) {
#pragma HLS loop_flatten off
#pragma HLS loop_tripcount min=1 max=16

		cc: for (ap_uint<32> i = 0; i < num_ite_C; ) {
#pragma HLS loop_tripcount min=1 max=5000
#pragma HLS pipeline II=1
			if (!c_read_ready) {
				c_read_ready = fifo_C_read_in.try_read(c_read);
			}
			if (!c_pe_ready) {
				c_pe_ready = fifo_C_pe_in.try_read(c_pe);
			}

			if (c_read_ready && c_pe_ready) {
				ap_uint<512> c_pe_delay = HLS_REG(c_pe);
				ap_uint<512> c_read_delay = HLS_REG(c_read);

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

	tapa::stream<ap_uint<32>, 8> fifo_edge_list_ptr_pe0("fifo_edge_list_ptr_pe0");
	tapa::stream<ap_uint<32>, 8> fifo_edge_list_ptr_pe1("fifo_edge_list_ptr_pe1");
	tapa::stream<ap_uint<32>, 8> fifo_edge_list_ptr_pe2("fifo_edge_list_ptr_pe2");
	tapa::stream<ap_uint<32>, 8> fifo_edge_list_ptr_pe3("fifo_edge_list_ptr_pe3");
	tapa::stream<ap_uint<32>, 8> fifo_edge_list_ptr_pe4("fifo_edge_list_ptr_pe4");
	tapa::stream<ap_uint<32>, 8> fifo_edge_list_ptr_pe5("fifo_edge_list_ptr_pe5");
	tapa::stream<ap_uint<32>, 8> fifo_edge_list_ptr_pe6("fifo_edge_list_ptr_pe6");
	tapa::stream<ap_uint<32>, 8> fifo_edge_list_ptr_pe7("fifo_edge_list_ptr_pe7");

	tapa::stream<ap_uint<512>, 8> fifo_A_pe0("fifo_A_pe0");
	tapa::stream<ap_uint<512>, 8> fifo_A_pe1("fifo_A_pe1");
	tapa::stream<ap_uint<512>, 8> fifo_A_pe2("fifo_A_pe2");
	tapa::stream<ap_uint<512>, 8> fifo_A_pe3("fifo_A_pe3");
	tapa::stream<ap_uint<512>, 8> fifo_A_pe4("fifo_A_pe4");
	tapa::stream<ap_uint<512>, 8> fifo_A_pe5("fifo_A_pe5");
	tapa::stream<ap_uint<512>, 8> fifo_A_pe6("fifo_A_pe6");
	tapa::stream<ap_uint<512>, 8> fifo_A_pe7("fifo_A_pe7");

	tapa::stream<ap_uint<512>, 8> fifo_B_pe0_x0("fifo_B_pe0_x0");
	tapa::stream<ap_uint<512>, 8> fifo_B_pe0_x1("fifo_B_pe0_x1");
	tapa::stream<ap_uint<512>, 8> fifo_B_pe0_x2("fifo_B_pe0_x2");
	tapa::stream<ap_uint<512>, 8> fifo_B_pe0_x3("fifo_B_pe0_x3");
	tapa::stream<ap_uint<512>, 8> fifo_B_pe1_x0("fifo_B_pe1_x0");
	tapa::stream<ap_uint<512>, 8> fifo_B_pe1_x1("fifo_B_pe1_x1");
	tapa::stream<ap_uint<512>, 8> fifo_B_pe1_x2("fifo_B_pe1_x2");
	tapa::stream<ap_uint<512>, 8> fifo_B_pe1_x3("fifo_B_pe1_x3");
	tapa::stream<ap_uint<512>, 8> fifo_B_pe2_x0("fifo_B_pe2_x0");
	tapa::stream<ap_uint<512>, 8> fifo_B_pe2_x1("fifo_B_pe2_x1");
	tapa::stream<ap_uint<512>, 8> fifo_B_pe2_x2("fifo_B_pe2_x2");
	tapa::stream<ap_uint<512>, 8> fifo_B_pe2_x3("fifo_B_pe2_x3");
	tapa::stream<ap_uint<512>, 8> fifo_B_pe3_x0("fifo_B_pe3_x0");
	tapa::stream<ap_uint<512>, 8> fifo_B_pe3_x1("fifo_B_pe3_x1");
	tapa::stream<ap_uint<512>, 8> fifo_B_pe3_x2("fifo_B_pe3_x2");
	tapa::stream<ap_uint<512>, 8> fifo_B_pe3_x3("fifo_B_pe3_x3");
	tapa::stream<ap_uint<512>, 8> fifo_B_pe4_x0("fifo_B_pe4_x0");
	tapa::stream<ap_uint<512>, 8> fifo_B_pe4_x1("fifo_B_pe4_x1");
	tapa::stream<ap_uint<512>, 8> fifo_B_pe4_x2("fifo_B_pe4_x2");
	tapa::stream<ap_uint<512>, 8> fifo_B_pe4_x3("fifo_B_pe4_x3");
	tapa::stream<ap_uint<512>, 8> fifo_B_pe5_x0("fifo_B_pe5_x0");
	tapa::stream<ap_uint<512>, 8> fifo_B_pe5_x1("fifo_B_pe5_x1");
	tapa::stream<ap_uint<512>, 8> fifo_B_pe5_x2("fifo_B_pe5_x2");
	tapa::stream<ap_uint<512>, 8> fifo_B_pe5_x3("fifo_B_pe5_x3");
	tapa::stream<ap_uint<512>, 8> fifo_B_pe6_x0("fifo_B_pe6_x0");
	tapa::stream<ap_uint<512>, 8> fifo_B_pe6_x1("fifo_B_pe6_x1");
	tapa::stream<ap_uint<512>, 8> fifo_B_pe6_x2("fifo_B_pe6_x2");
	tapa::stream<ap_uint<512>, 8> fifo_B_pe6_x3("fifo_B_pe6_x3");
	tapa::stream<ap_uint<512>, 8> fifo_B_pe7_x0("fifo_B_pe7_x0");
	tapa::stream<ap_uint<512>, 8> fifo_B_pe7_x1("fifo_B_pe7_x1");
	tapa::stream<ap_uint<512>, 8> fifo_B_pe7_x2("fifo_B_pe7_x2");
	tapa::stream<ap_uint<512>, 8> fifo_B_pe7_x3("fifo_B_pe7_x3");

	tapa::stream<ap_uint<512>, 8> fifo_C_pe0("fifo_C_pe0");
	tapa::stream<ap_uint<512>, 8> fifo_C_pe1("fifo_C_pe1");
	tapa::stream<ap_uint<512>, 8> fifo_C_pe2("fifo_C_pe2");
	tapa::stream<ap_uint<512>, 8> fifo_C_pe3("fifo_C_pe3");
	tapa::stream<ap_uint<512>, 8> fifo_C_pe4("fifo_C_pe4");
	tapa::stream<ap_uint<512>, 8> fifo_C_pe5("fifo_C_pe5");
	tapa::stream<ap_uint<512>, 8> fifo_C_pe6("fifo_C_pe6");
	tapa::stream<ap_uint<512>, 8> fifo_C_pe7("fifo_C_pe7");

	tapa::stream<ap_uint<512>, 8> fifo_C_read_in0("fifo_C_read_in0");
	tapa::stream<ap_uint<512>, 8> fifo_C_read_in1("fifo_C_read_in1");
	tapa::stream<ap_uint<512>, 8> fifo_C_read_in2("fifo_C_read_in2");
	tapa::stream<ap_uint<512>, 8> fifo_C_read_in3("fifo_C_read_in3");
	tapa::stream<ap_uint<512>, 8> fifo_C_read_in4("fifo_C_read_in4");
	tapa::stream<ap_uint<512>, 8> fifo_C_read_in5("fifo_C_read_in5");
	tapa::stream<ap_uint<512>, 8> fifo_C_read_in6("fifo_C_read_in6");
	tapa::stream<ap_uint<512>, 8> fifo_C_read_in7("fifo_C_read_in7");

	tapa::stream<ap_uint<512>, 8> fifo_C_ch0_result("fifo_C_ch0_result");
	tapa::stream<ap_uint<512>, 8> fifo_C_ch1_result("fifo_C_ch1_result");
	tapa::stream<ap_uint<512>, 8> fifo_C_ch2_result("fifo_C_ch2_result");
	tapa::stream<ap_uint<512>, 8> fifo_C_ch3_result("fifo_C_ch3_result");
	tapa::stream<ap_uint<512>, 8> fifo_C_ch4_result("fifo_C_ch4_result");
	tapa::stream<ap_uint<512>, 8> fifo_C_ch5_result("fifo_C_ch5_result");
	tapa::stream<ap_uint<512>, 8> fifo_C_ch6_result("fifo_C_ch6_result");
	tapa::stream<ap_uint<512>, 8> fifo_C_ch7_result("fifo_C_ch7_result");

	tapa::stream<ap_uint<512>, 8> fifo_C_ch0("fifo_C_ch0");
	tapa::stream<ap_uint<512>, 8> fifo_C_ch1("fifo_C_ch1");
	tapa::stream<ap_uint<512>, 8> fifo_C_ch2("fifo_C_ch2");
	tapa::stream<ap_uint<512>, 8> fifo_C_ch3("fifo_C_ch3");
	tapa::stream<ap_uint<512>, 8> fifo_C_ch4("fifo_C_ch4");
	tapa::stream<ap_uint<512>, 8> fifo_C_ch5("fifo_C_ch5");
	tapa::stream<ap_uint<512>, 8> fifo_C_ch6("fifo_C_ch6");
	tapa::stream<ap_uint<512>, 8> fifo_C_ch7("fifo_C_ch7");

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
		NUM_A_LEN,
		P_N
		);

	read_A<1>(
		edge_list_ch1,
		fifo_A_pe1,
		NUM_A_LEN,
		P_N
		);

	read_A<2>(
		edge_list_ch2,
		fifo_A_pe2,
		NUM_A_LEN,
		P_N
		);

	read_A<3>(
		edge_list_ch3,
		fifo_A_pe3,
		NUM_A_LEN,
		P_N
		);

	read_A<4>(
		edge_list_ch4,
		fifo_A_pe4,
		NUM_A_LEN,
		P_N
		);

	read_A<5>(
		edge_list_ch5,
		fifo_A_pe5,
		NUM_A_LEN,
		P_N
		);

	read_A<6>(
		edge_list_ch6,
		fifo_A_pe6,
		NUM_A_LEN,
		P_N
		);

	read_A<7>(
		edge_list_ch7,
		fifo_A_pe7,
		NUM_A_LEN,
		P_N
		);

	read_B<0>(
		mat_B_ch0,
		fifo_B_pe0_x0,
		K,
		P_N
		);

	read_B<1>(
		mat_B_ch1,
		fifo_B_pe0_x1,
		K,
		P_N
		);

	read_B<2>(
		mat_B_ch2,
		fifo_B_pe0_x2,
		K,
		P_N
		);

	read_B<3>(
		mat_B_ch3,
		fifo_B_pe0_x3,
		K,
		P_N
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

	PEG<3>(
		fifo_edge_list_ptr_pe3,
		fifo_A_pe3,
		fifo_B_pe3_x0,
		fifo_B_pe3_x1,
		fifo_B_pe3_x2,
		fifo_B_pe3_x3,
		fifo_edge_list_ptr_pe4,
		fifo_B_pe4_x0,
		fifo_B_pe4_x1,
		fifo_B_pe4_x2,
		fifo_B_pe4_x3,
		fifo_C_pe3
		);

	PEG<4>(
		fifo_edge_list_ptr_pe4,
		fifo_A_pe4,
		fifo_B_pe4_x0,
		fifo_B_pe4_x1,
		fifo_B_pe4_x2,
		fifo_B_pe4_x3,
		fifo_edge_list_ptr_pe5,
		fifo_B_pe5_x0,
		fifo_B_pe5_x1,
		fifo_B_pe5_x2,
		fifo_B_pe5_x3,
		fifo_C_pe4
		);

	PEG<5>(
		fifo_edge_list_ptr_pe5,
		fifo_A_pe5,
		fifo_B_pe5_x0,
		fifo_B_pe5_x1,
		fifo_B_pe5_x2,
		fifo_B_pe5_x3,
		fifo_edge_list_ptr_pe6,
		fifo_B_pe6_x0,
		fifo_B_pe6_x1,
		fifo_B_pe6_x2,
		fifo_B_pe6_x3,
		fifo_C_pe5
		);

	PEG<6>(
		fifo_edge_list_ptr_pe6,
		fifo_A_pe6,
		fifo_B_pe6_x0,
		fifo_B_pe6_x1,
		fifo_B_pe6_x2,
		fifo_B_pe6_x3,
		fifo_edge_list_ptr_pe7,
		fifo_B_pe7_x0,
		fifo_B_pe7_x1,
		fifo_B_pe7_x2,
		fifo_B_pe7_x3,
		fifo_C_pe6
		);

	PEG_last<7>(
		fifo_edge_list_ptr_pe7,
		fifo_A_pe7,
		fifo_B_pe7_x0,
		fifo_B_pe7_x1,
		fifo_B_pe7_x2,
		fifo_B_pe7_x3,
		fifo_C_pe7
		);

    C_collect<0>(
		fifo_C_pe0,
		fifo_C_pe1,
		fifo_C_pe2,
		fifo_C_pe3,
		fifo_C_pe4,
		fifo_C_pe5,
		fifo_C_pe6,
		fifo_C_pe7,

		fifo_C_ch0_result,
		fifo_C_ch1_result,
		fifo_C_ch2_result,
		fifo_C_ch3_result,
		fifo_C_ch4_result,
		fifo_C_ch5_result,
		fifo_C_ch6_result,
		fifo_C_ch7_result
		);

	read_C<0>(
		mat_C_ch0_in,
		fifo_C_read_in0,
		M,
		P_N
		);

	read_C<1>(
		mat_C_ch1_in,
		fifo_C_read_in1,
		M,
		P_N
		);

	read_C<2>(
		mat_C_ch2_in,
		fifo_C_read_in2,
		M,
		P_N
		);

	read_C<3>(
		mat_C_ch3_in,
		fifo_C_read_in3,
		M,
		P_N
		);

	read_C<4>(
		mat_C_ch4_in,
		fifo_C_read_in4,
		M,
		P_N
		);

	read_C<5>(
		mat_C_ch5_in,
		fifo_C_read_in5,
		M,
		P_N
		);

	read_C<6>(
		mat_C_ch6_in,
		fifo_C_read_in6,
		M,
		P_N
		);

	read_C<7>(
		mat_C_ch7_in,
		fifo_C_read_in7,
		M,
		P_N
		);

	comp_C<0>(
		fifo_C_read_in0,
		fifo_C_ch0_result,
		fifo_C_ch0
		);

	comp_C<1>(
		fifo_C_read_in1,
		fifo_C_ch1_result,
		fifo_C_ch1
		);

	comp_C<2>(
		fifo_C_read_in2,
		fifo_C_ch2_result,
		fifo_C_ch2
		);

	comp_C<3>(
		fifo_C_read_in3,
		fifo_C_ch3_result,
		fifo_C_ch3
		);

	comp_C<4>(
		fifo_C_read_in4,
		fifo_C_ch4_result,
		fifo_C_ch4
		);

	comp_C<5>(
		fifo_C_read_in5,
		fifo_C_ch5_result,
		fifo_C_ch5
		);

	comp_C<6>(
		fifo_C_read_in6,
		fifo_C_ch6_result,
		fifo_C_ch6
		);

	comp_C<7>(
		fifo_C_read_in7,
		fifo_C_ch7_result,
		fifo_C_ch7
		);

	write_C<0>(
		fifo_C_ch0,
		mat_C_ch0
		);

	write_C<1>(
		fifo_C_ch1,
		mat_C_ch1
		);

	write_C<2>(
		fifo_C_ch2,
		mat_C_ch2
		);

	write_C<3>(
		fifo_C_ch3,
		mat_C_ch3
		);

	write_C<4>(
		fifo_C_ch4,
		mat_C_ch4
		);

	write_C<5>(
		fifo_C_ch5,
		mat_C_ch5
		);

	write_C<6>(
		fifo_C_ch6,
		mat_C_ch6
		);

	write_C<7>(
		fifo_C_ch7,
		mat_C_ch7
		);
}
