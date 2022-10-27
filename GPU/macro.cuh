#pragma once
#ifdef __INTELLISENSE__
#include "intellisense_cuda_intrinsics.h"
#endif
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <immintrin.h>
#include <Windows.h>
#include <memory.h>
#include <string>
#include <fstream>
#include <sstream>
#include <assert.h>


#define timer(x) QueryPerformanceCounter((LARGE_INTEGER*)&(x))
#define total(op,ed) ((ed-op)*1000000 /freq)
#define output(method, t) std::cout<<std::left<<method<<t/1000.f<<"ms."<<std::endl
#define prefetch
extern long long head, tail, freq;

// random walk iterations
constexpr auto ITERATIONS = 50;
constexpr auto ITER = 30;
constexpr auto Q = 0.85f;
constexpr auto TOLERANCE = 0.0001f;
constexpr auto EPOCH = 1;

// Read the Datasets
extern void readData();
extern void init();

// CUDA relevant features
constexpr auto WARP_SIZE = 32;
constexpr auto THREADS_PER_BLOCK = 512;

extern void monte_carlo_cu(
	const int pages,		// number of total pages
	const int nonzeros,		// number of total edges
	float* r,				// pagerank value of each page
	const int* row,			// row index of each edge
	const int* col,			// column index of each edge
	const int* rowptr,		// row pointer of each row (CSR)
	const int* rowcount,	// number of edges in each row 
	const float* value);	// value of each edge

void power_iter_cu(
	const int pages,		// number of total pages
	const int nonzeros,		// number of total edges
	float* r,				// pagerank value of each page
	const int* row,			// row index of each edge
	const int* col,			// column index of each edge
	const int* rowptr,		// row pointer of each row (CSR)
	const int* rowcount,	// number of edges in each row 
	const float* value);	// value of each edge	