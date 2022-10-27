#include "../include/macro.h"

using namespace std;

// Output Formats
float (*SpMVs[])() = { naive_SpMV, AVX_SpMV, AVX_SpMV_aligned, AVX_SpMV_unrolled };
const char* methodName[] = { "naive SpMv aligned", "AVX", "AVX aligned", "AVX unrolled",  };
const char* methodName_AVX[] = { "naive SpMv aligned AVX", "AVX++", "AVX++ aligned ", "AVX++ unrolled",  };
long long head, tail, freq;


// naive version of Sequential Sparse Matrix - Vector Multiplication (SpMV) (aligned)
float naive_SpMV() {
	float ret = 0.0;
	for (int i = 0; i < N_ROW; i++) {
		int r_begin = Ap[i];
		int r_end = Ap[i + 1];
		float acc = 0;
		for (int c = r_begin; c < r_end; c++) {
			float a = Av[c];
			float b = R[Aj[c]];
			acc += a * b;
		}
		Y[i] = Q * acc;
		ret += Y[i];
	}
	return ret;
}

void power_iteration_SpMV(int methodIdx) {
	int iter = ITERATIONS;
	float totaltime = 0.0;
	while (iter--)
	{
		float* R_past = new float[N_ROW];
		for (int i = 0; i < N_ROW; i++) {
			R[i] = 1/N_ROW;
		}
		memcpy(Y, R, sizeof(float)*N_ROW);
		Y[0] = 0.0;
		float error;
		begin_count;
		do {
			float d = SpMVs[methodIdx]();
			error = 0.0;
			for (int i = 0; i < N_ROW; i++) {
				R_past[i] = R[i];
				R[i] = Y[i] + (1 - d) /N_ROW;
				error += abs(R[i] - R_past[i]);
			}
		} while (error >= EPSILON);
		end_count;
		totaltime += total;
	}
	memcpy(stdR, R, sizeof(float)*N_ROW);
	output(methodName[methodIdx], totaltime/ITERATIONS);
}

void debug_output(int flag) {
	switch (flag) {
	case 0x0000: {
		float error = 0.0;
		for (int i = 0; i < N_ROW; i++)
			error += abs(R[i] - stdR[i]);
		cout << "Error: " << error << endl;
		break;
	}

	default:
		break;
	}
}


