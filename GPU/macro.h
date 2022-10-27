#pragma once

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
extern long long head, tail, freq;

// random walk iterations
constexpr auto ITERATIONS = 50;
constexpr auto Q = 0.85f;
constexpr auto TOLERANCE = 0.0001f;

// Read the Datasets
extern void readData();
extern void init();

// CUDA relevant features
constexpr auto WARP_SIZE = 32;
//extern float atomicAdd(float* address, float val) {};