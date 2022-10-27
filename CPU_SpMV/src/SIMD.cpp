#include "../include/macro.h"
using namespace std;
// TODO: alter the "SUM"
void power_iteration_SpMV_AVX(int methodIdx) {
	int iter = ITERATIONS;
	float totaltime = 0.0;
	__m256 vR, vR_past, vY, vTemp, v1Q;
	v1Q = _mm256_set1_ps(1-Q);
	vTemp = _mm256_setzero_ps();
	while (iter--)
	{
		float* R_past = new float[N_ROW];
		for (int i = 0; i < N_ROW; i++) {
			if (i == 0) {
				R[i] = 1.0;
			}
			else {
				R[i] = 0.0;
			}
			Y[i] = 0.0;
		}
		float error;
		
		begin_count;
		do {
			error = 0.0;
			for (int i = 0; i < N_ROW; i = i+8) {
                vR_past = _mm256_loadu_ps(R_past+i);
                vR = _mm256_loadu_ps(R+i);
                vY = _mm256_loadu_ps(Y+i);
                vR_past = vR;
                _mm256_storeu_ps(R_past+i, vR_past);
                vTemp = _mm256_mul_ps(v1Q, vR_past);
                vR = _mm256_add_ps(vTemp, vY);
                _mm256_storeu_ps(R+i, vR);
				vTemp = _mm256_sub_ps(vR, vR_past);
				error += abs(vTemp[0]) + abs(vTemp[1]) + abs(vTemp[2]) + abs(vTemp[3]) + abs(vTemp[4]) + abs(vTemp[5]) + abs(vTemp[6]) + abs(vTemp[7]);
			}
		} while (error >= EPSILON);
		end_count;
		totaltime += total;
	}
	output(methodName_AVX[methodIdx], totaltime/ITERATIONS);
}

float AVX_SpMV_unrolled() {
	float ret = 0.0;
	__m256 vec1, vec2, vec3, vec_acc, vec4, vec5, vec6, vec_acc2;
	float acc;
	for (int i = 0; i < N_ROW; i++) {
		int r_begin = Ap[i];
		int r_end = Ap[i + 1];
		acc = 0;
		int q = r_begin;
		vec_acc = _mm256_setzero_ps();
		vec_acc2 = _mm256_setzero_ps();
		float temp[16];
		memset(temp, 0, sizeof(temp));
		for (; q < r_end - 15; q += 16) {
			vec1 = _mm256_loadu_ps(Av + q);
			vec2 = _mm256_set_ps(R[Aj[q]], R[Aj[q + 1]], R[Aj[q + 2]], R[Aj[q + 3]], R[Aj[q+4]], R[Aj[q + 5]], R[Aj[q + 6]], R[Aj[q + 7]]);
			vec3 = _mm256_mul_ps(vec1, vec2);
			vec_acc = _mm256_add_ps(vec3, vec_acc);
			_mm256_storeu_ps(temp, vec_acc);
			vec4 = _mm256_loadu_ps(Av + q + 8);
			vec5 = _mm256_set_ps(R[Aj[q+8]], R[Aj[q + 9]], R[Aj[q + 10]], R[Aj[q + 11]], R[Aj[q + 12]], R[Aj[q + 13]], R[Aj[q + 14]], R[Aj[q + 15]]);
			vec6 = _mm256_mul_ps(vec4, vec5);
			vec_acc2 = _mm256_add_ps(vec6, vec_acc2);
			_mm256_storeu_ps(temp + 8, vec_acc2);
		}
		for (int i = 0; i < 16; i++) {
			acc += temp[i];
		}
		for (; q < r_end; q++) {
			acc += Av[q] * R[Aj[q]];
		}
		/*acc += Av[c] * R[Aj[c]];*/
		Y[i] = Q * acc;
		ret += Y[i];
	}
	return ret;
}

float AVX_SpMV() {
	float ret = 0.0;
	__m256 vec1, vec2, vec3, vec_acc;
	float acc;
	for (int i = 0; i < N_ROW; i++) {
		int r_begin = Ap[i];
		int r_end = Ap[i + 1];
		acc = 0;
		int q = r_begin;
		vec_acc = _mm256_setzero_ps();
		float temp[8];
		memset(temp, 0, sizeof(temp));
		for (; q < r_end - 7; q += 8) {
			vec1 = _mm256_loadu_ps(Av + q);
			vec2 = _mm256_set_ps(R[Aj[q]], R[Aj[q + 1]], R[Aj[q + 2]], R[Aj[q + 3]], R[Aj[q + 4]], R[Aj[q + 5]], R[Aj[q + 6]], R[Aj[q + 7]]);
			vec3 = _mm256_mul_ps(vec1, vec2);
			vec_acc = _mm256_add_ps(vec3, vec_acc);
			_mm256_storeu_ps(temp, vec_acc);
		}
		acc += temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
		for (; q < r_end; q++) {
			acc += Av[q] * R[Aj[q]];
		}
		/*acc += Av[c] * R[Aj[c]];*/
		Y[i] = Q * acc;
		ret += Y[i];
	}
	return ret;
}

float AVX_SpMV_aligned() {
	float ret = 0.0;
	__m256 vec1, vec2, vec3, vec_acc;
	float acc;
	for (int i = 0; i < N_ROW; i++) {
		int r_begin = Ap[i];
		int r_end = Ap[i + 1];
		acc = 0;
		int q = r_begin;
		vec_acc = _mm256_setzero_ps();
		float *temp = (float*) aligned_alloc(8 * sizeof(float),sizeof(float));
		memset(temp, 0, 8*sizeof(float));
		if(q % 8){
            for(; q % 8; q++){
                acc += Av[q] * R[Aj[q]];
            }
		}
		for (; q < r_end - 7; q += 8) {
			vec1 = _mm256_load_ps(Av + q);
			vec2 = _mm256_set_ps(R[Aj[q]], R[Aj[q + 1]], R[Aj[q + 2]], R[Aj[q + 3]], R[Aj[q + 4]], R[Aj[q + 5]], R[Aj[q + 6]], R[Aj[q + 7]]);
			vec3 = _mm256_mul_ps(vec1, vec2);
			vec_acc = _mm256_add_ps(vec3, vec_acc);
			_mm256_store_ps(temp, vec_acc);
		}
		acc += temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
		//_aligned_free(temp);
		for (; q < r_end; q++) {
			acc += Av[q] * R[Aj[q]];
		}
		/*acc += Av[c] * R[Aj[c]];*/
		Y[i] = Q * acc;
		ret += Y[i];
	}
	return ret;
}
