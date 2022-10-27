#pragma once

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
extern long long head, tail, freq;

// random walk iterations
constexpr auto ITERATIONS = 50;
constexpr auto Q = 0.85f;
constexpr auto TOLERANCE = 0.0001f;
// Read the Datasets

extern void readData();

extern void init();

extern void monte_carlo(
	const int pages,		// number of total pages
	const int nonzeros,		// number of total edges
	float* r,				// pagerank value of each page
	const int* row,			// row index of each edge
	const int* col,			// column index of each edge
	const int* rowptr,		// row pointer of each row (CSR)
	const int* rowcount,	// number of edges in each row 
	const float* value);	// value of each edge	

extern void power_iter(
	const int pages,		// number of total pages
	const int nonzeros,		// number of total edges
	float* r,				// pagerank value of each page
	const int* row,			// row index of each edge
	const int* col,			// column index of each edge
	const int* rowptr,		// row pointer of each row (CSR)
	const int* rowcount,	// number of edges in each row 
	const float* value);	// value of each edge	
