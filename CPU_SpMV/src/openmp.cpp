#include "../include/macro.h"


void naive_SpMV_omp() {
	# pragma omp parallel for num_threads(THREAD)
	for (int i = 0; i < N_ROW; i++) {
		int r_begin = Ap[i];
		int r_end = Ap[i + 1];
		float acc = 0;
		for (int c = r_begin; c < r_end; c++) {
			acc += Av[c] * R[Aj[c]];
		}
		Y[i] = Q * acc;
	}
}

// PIS - Power Iteration with SpMV
void simple_openmp_PIS(int methodIdx) {
	int iter = ITERATIONS;
	float totaltime = 0.0;
	while (iter--)
	{
		R_past = new float[N_ROW];
		for (int i = 0; i < N_ROW; i++) {
			if (i == 0) {
				R[i] = 1.0;
			}
			else {
				R[i] = 0.0;
			}
			Y[i] = 0.0;
		}
		
		begin_count;
		do {
			naive_SpMV_omp();
			error = 0.0;
			for (int i = 0; i < N_ROW; i++) {
				R_past[i] = R[i];
				R[i] = Y[i] + (1 - Q) * R_past[i];
				error += abs(R[i] - R_past[i]);
			}
		} while (error >= EPSILON);
		end_count;
		totaltime += total;
	}
	output("OpenMp", totaltime/ITERATIONS);
}

void SpMV_omp_opt() {
	# pragma omp parallel for num_threads(THREAD)
	for (int i = 0; i < N_ROW; i++) {
		int r_begin = Ap[i];
		int r_end = Ap[i + 1];
		float acc = 0;
		for (int c = r_begin; c < r_end; c++) {
			acc += Av[c] * R[Aj[c]];
		}
		Y[i] = Q * acc;
		
		R_past[i] = R[i];
		R_copy[i] = Y[i] + (1 - Q) * R_past[i];
		
		error_list[i] = abs(R_copy[i] - R_past[i]);
	}
}

// PIS - Power Iteration with SpMV
void openmp_opt_PIS(int methodIdx) {
	int iter = ITERATIONS;
	float totaltime = 0.0;
	while (iter--)
	{
		R_past = new float[N_ROW];
		for (int i = 0; i < N_ROW; i++) {
			if (i == 0) {
				R[i] = 1.0;
			}
			else {
				R[i] = 0.0;
			}
			Y[i] = 0.0;
			error_list[i] = 0.0;
		}
		
		begin_count;
		do {
			error = 0.0;
			SpMV_omp_opt();
			memcpy(R, R_copy, sizeof(float) * N_ROW);
			for(int i = 0; i < N_ROW; i += 4){
				float sum = error_list[i] + error_list[i+1] + error_list[i+2] + error_list[i+3];
				error += sum;
			}
		} while (error >= EPSILON);
		end_count;
		totaltime += total;
	}
	output("OpenMp Optimized", totaltime/ITERATIONS);
}

void SpMV_omp_opt_dynamic() {
	# pragma omp parallel for schedule(dynamic, 32) num_threads(THREAD)
	for (int i = 0; i < N_ROW; i++) {
		int r_begin = Ap[i];
		int r_end = Ap[i + 1];
		float acc = 0;
		for (int c = r_begin; c < r_end; c++) {
			acc += Av[c] * R[Aj[c]];
		}
		Y[i] = Q * acc;
		
		R_past[i] = R[i];
		R_copy[i] = Y[i] + (1 - Q) * R_past[i];
		error_list[i] = abs(R_copy[i] - R_past[i]);
	}
}

// PIS - Power Iteration with SpMV
void openmp_opt_dynamic_PIS(int methodIdx) {
	int iter = ITERATIONS;
	float totaltime = 0.0;
	while (iter--)
	{
		R_past = new float[N_ROW];
		for (int i = 0; i < N_ROW; i++) {
			if (i == 0) {
				R[i] = 1.0;
			}
			else {
				R[i] = 0.0;
			}
			Y[i] = 0.0;
			error_list[i] = 0.0;
		}
		
		begin_count;
		do {
			error = 0.0;
			SpMV_omp_opt_dynamic();
			memcpy(R, R_copy, sizeof(float) * N_ROW);
			for(int i = 0; i < N_ROW; i += 4){
				float sum = error_list[i] + error_list[i+1] + error_list[i+2] + error_list[i+3];
				error += sum;
			}
		} while (error >= EPSILON);
		end_count;
		totaltime += total;
	}
	output("OpenMp Opt Dynamic", totaltime/ITERATIONS);
}

void SpMV_omp_opt_guided() {
	# pragma omp parallel for schedule(guided, 32) num_threads(THREAD)
	for (int i = 0; i < N_ROW; i++) {
		int r_begin = Ap[i];
		int r_end = Ap[i + 1];
		float acc = 0;
		for (int c = r_begin; c < r_end; c++) {
			acc += Av[c] * R[Aj[c]];
		}
		Y[i] = Q * acc;
		
		R_past[i] = R[i];
		R_copy[i] = Y[i] + (1 - Q) * R_past[i];
		error_list[i] = abs(R_copy[i] - R_past[i]);
	}
}

// PIS - Power Iteration with SpMV
void openmp_opt_guided_PIS(int methodIdx) {
	int iter = ITERATIONS;
	float totaltime = 0.0;
	while (iter--)
	{
		R_past = new float[N_ROW];
		for (int i = 0; i < N_ROW; i++) {
			if (i == 0) {
				R[i] = 1.0;
			}
			else {
				R[i] = 0.0;
			}
			Y[i] = 0.0;
			error_list[i] = 0.0;
		}
		
		begin_count;
		do {
			error = 0.0;
			SpMV_omp_opt_guided();
			memcpy(R, R_copy, sizeof(float) * N_ROW);
			for(int i = 0; i < N_ROW; i += 4){
				float sum = error_list[i] + error_list[i+1] + error_list[i+2] + error_list[i+3];
				error += sum;
			}
		} while (error >= EPSILON);
		end_count;
		totaltime += total;
	}
	output("OpenMp Opt guided", totaltime/ITERATIONS);
}

void SpMV_SIMD_opt_dynamic() {
	# pragma omp parallel for schedule(dynamic, 32) num_threads(THREAD)
	for (int i = 0; i < N_ROW; i++) {
		int r_begin = Ap[i];
		int r_end = Ap[i + 1];
		float acc = 0;
		# pragma omp simd
		for (int c = r_begin; c < r_end; c++) {
			acc += Av[c] * R[Aj[c]];
		}
		Y[i] = Q * acc;
		R_past[i] = R[i];
		R_copy[i] = Y[i] + (1 - Q) * R_past[i];
		error_list[i] = abs(R_copy[i] - R_past[i]);
	}
}

void openmp_SIMD_dynamic_PIS(int methodIdx) {
	int iter = ITERATIONS;
	float totaltime = 0.0;
	while (iter--)
	{
		R_past = new float[N_ROW];
		for (int i = 0; i < N_ROW; i++) {
			if (i == 0) {
				R[i] = 1.0;
			}
			else {
				R[i] = 0.0;
			}
			Y[i] = 0.0;
			error_list[i] = 0.0;
		}
		
		begin_count;
		do {
			error = 0.0;
			SpMV_SIMD_opt_dynamic();
			memcpy(R, R_copy, sizeof(float) * N_ROW);
			# pragma omp simd
			for(int i = 0; i < N_ROW; i += 4){
				float sum = error_list[i] + error_list[i+1] + error_list[i+2] + error_list[i+3];
				error += sum;
			}
		} while (error >= EPSILON);
		end_count;
		totaltime += total;
	}
	output("OpenMp SIMD Dynamic", totaltime/ITERATIONS);
}