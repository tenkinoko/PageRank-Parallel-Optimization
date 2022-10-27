#include "macro.cuh"
using namespace std;

__global__
void init(const int pages, float* r, float* r_, float* y) {
	const int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < pages) {
		r[tid] = 1.f / pages;
		y[tid] = 0.f;
		r_[tid] = 0.f;
	}
}

__global__
void SpMV(
	const int pages,		// number of total pages
	float* r,				// pagerank value of each page
	float* y,				// temp value before r calculated
	float* r_,				// r value of last iteration
	const int* col,			// column index of each edge
	const int* rowptr,		// row pointer of each row (CSR)
	const float* value)		// value of each edge	
{
	const int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < pages) {
		int r_begin = rowptr[tid];
		int r_end = rowptr[tid + 1];
		float acc = 0.f;
		for (int c = r_begin; c < r_end; c++) {
			float a = value[c];
			float b = r[col[c]];
			acc += a * b;
		}
		y[tid] = Q * acc;
		atomicAdd(&y[pages], y[tid]);
	}
}
__global__ 
void renew(const int pages, float* y) {
	const int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid == 0)
		y[pages] = 0.f;
}

__global__
void param_update(const int pages, float *r, float *r_, float *y)
{
	const int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < pages) {
		r_[tid] = r[tid];
		r[tid] = y[tid] + (1.f - y[pages]) / pages;
	}
}

void power_iter_cu(
	const int pages,		// number of total pages
	const int nonzeros,		// number of total edges
	float* r,				// pagerank value of each page
	const int* row,			// row index of each edge
	const int* col,			// column index of each edge
	const int* rowptr,		// row pointer of each row (CSR)
	const int* rowcount,	// number of edges in each row 
	const float* value)		// value of each edge		
{
	int deviceId;
	cudaGetDevice(&deviceId);
	const int blocks = pages / THREADS_PER_BLOCK + 1;
	float* _r, *_r_, *_y, *_Av;
	int* _Ap, * _Aj;

	cudaMalloc(&_r, sizeof(float) * pages);
	cudaMalloc(&_r_, sizeof(float) * pages);
	cudaMalloc(&_y, sizeof(float) * (1 + pages));
	
#ifdef prefetch
	cudaMemPrefetchAsync(rowptr, sizeof(int) * (pages + 1), deviceId);
	cudaMemPrefetchAsync(col, sizeof(int) * (nonzeros), deviceId);
	cudaMemPrefetchAsync(value, sizeof(float) * (nonzeros), deviceId);
	init << < blocks, THREADS_PER_BLOCK >> > (pages, _r, _r_, _y);
	for (int i = 0; i < ITER; i++) {
		renew << < blocks, THREADS_PER_BLOCK >> > (pages, _y);
		SpMV << < blocks, THREADS_PER_BLOCK >> > (pages, _r, _y, _r_, col, rowptr, value);
		param_update << < blocks, THREADS_PER_BLOCK >> > (pages, _r, _r_, _y);
	};
	cudaMemPrefetchAsync(r, sizeof(float) * pages, cudaCpuDeviceId);
#else
	cudaMalloc(&_Ap, sizeof(int) * (pages + 1));
	cudaMalloc(&_Aj, sizeof(int) * (nonzeros));
	cudaMalloc(&_Av, sizeof(float) * (nonzeros));
	cudaMemcpy(_Ap, rowptr, sizeof(int) * (pages + 1), cudaMemcpyHostToDevice);
	cudaMemcpy(_Aj, col, sizeof(int) * (nonzeros), cudaMemcpyHostToDevice);
	cudaMemcpy(_Av, value, sizeof(float) * (nonzeros), cudaMemcpyHostToDevice);

	init << < blocks, THREADS_PER_BLOCK >> > (pages, _r, _r_, _y);
	for (int i = 0; i < ITER; i++) {
		renew << < blocks, THREADS_PER_BLOCK >> > (pages, _y);
		SpMV << < blocks, THREADS_PER_BLOCK >> > (pages, _r, _y, _r_, _Aj, _Ap, _Av);
		param_update << < blocks, THREADS_PER_BLOCK >> > (pages, _r, _r_, _y);
	};
	cudaMemcpy(r, _r, sizeof(float) * pages, cudaMemcpyDeviceToHost);
	cudaFree(_Ap);
	cudaFree(_Aj);
	cudaFree(_Av);
#endif
	
	cudaDeviceSynchronize();
	cudaFree(_r);
	cudaFree(_r_);
	cudaFree(_y);
	
}