#ifndef __MARCO_H_
#define __MACRO_H_

#include <cmath>
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <immintrin.h>
#include <sys/time.h>
#include <memory.h>
#include <string>
#include <assert.h>
#include <omp.h>
#include "threadpool.h"

/* MACROS */
extern int N_ROW;
constexpr auto EPSILON = 0.001f;
constexpr auto DEBUG = 0;
constexpr auto Q = 0.85f;
constexpr auto ITERATIONS = 10;
constexpr auto THREAD = 8;
constexpr auto LOG = 0;

#define begin_count gettimeofday(&t_begin, NULL)
#define end_count gettimeofday(&t_end, NULL)
#define total (t_end.tv_sec*1000000 + t_end.tv_usec - t_begin.tv_sec*1000000 - t_begin.tv_usec)
#define output(method, t) (std::cout<<std::left<<std::setw(18)<<method<<" time: "<<std::setw(8)<<t/1000.0<<"ms"<<"    ")
extern struct timeval t_begin, t_end;

/* GLOBAL VARIABLES*/
// initial R vector
extern float *R;

// initial Y
extern float* Y;

// initial R standard output
extern float *stdR;

// the number of nonzero elements in transfer matrix
extern int nonzero;

// CSR: Av - values, Aj - column indices, Ap - offsets, uAv - unaligned values
extern int *Ap;
extern float* Av;
extern int* Aj;

// TimeCount
extern long long head, tail, freq;

extern float (*SpMVs[])();
extern const char* methodName[];
extern const char* methodName_AVX[];
/* PageRank Functions with SpMV */
extern float naive_SpMV();
extern void power_iteration_SpMV(int);
extern void debug_output(int);

extern float error;
extern float* R_past;
extern float* R_copy;
extern float* error_list;
/* TODO: PageRank Functions with Monte Carlo Method */

/* Parallel Optimizations */
// Read the Datasets
struct dataItem {
	uint32_t row;
	uint32_t col;
    float val;
};
extern void readData();


// SIMD Optimization
extern float AVX_SpMV();
extern float AVX_SpMV_unrolled();
extern float AVX_SpMV_aligned();
extern void power_iteration_SpMV_AVX(int);

// Pthread Optimization
extern void task_SpMV_AVX(void*);
extern void pool_SpMV_AVX(void);
extern void task_SpMV(void*);
extern void pool_SpMV(void);
extern void task_SpMV_unrolled(void*);
extern void pool_SpMV_unrolled(void);
extern void task_SpMV_improved(void*);
extern void pool_SpMV_improved(void);
extern void task_SpMV_cond(void*);
extern void pool_SpMV_cond(void);

// OpenMP Optimization
extern void simple_openmp_PIS(int);
extern void naive_SpMV_omp();
extern void openmp_opt_PIS(int);
extern void SpMV_omp_opt();
extern void openmp_opt_dynamic_PIS(int);
extern void SpMV_omp_opt_dynamic();
extern void openmp_opt_guided_PIS(int);
extern void SpMV_omp_opt_guided();
extern void openmp_SIMD_dynamic_PIS(int);
extern void SpMV_SIMD_opt_dynamic();

// MPI Optimization
extern void mpi_Master_Slave_SpMV();
extern double mpi_Master_Slave_PIS();
extern void mpi_Master_Slave();
extern float mpi_Collective_SpMV(int, int);
extern void mpi_Collective_PIS();
extern void mpi_RMA();
extern void mpi_RMA_omp();
#endif
