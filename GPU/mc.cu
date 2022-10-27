#include "macro.cuh"

using namespace std;
__global__
void mc_init(const int pages, float* r, curandStateMRG32k3a* state, unsigned long seed) {
	const int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < pages)
		r[tid] = 0;

	if (threadIdx.x < WARP_SIZE) {
		int rid = threadIdx.x + blockIdx.x * WARP_SIZE;
		curand_init(seed, rid, 0, &state[rid]);
	}
}

__global__
void rand_walk(
	const int pages,			// number of total pages
	const int nonzeros,			// number of total edges
	float* r,					// pagerank value of each page
	const int* col,				// column index of each edge
	const int* rowptr,			// row pointer of each row (CSR)
	curandStateMRG32k3a* state)	// state of random number generator
{
	const int tid = threadIdx.x + blockIdx.x * blockDim.x;
	const int rid = (threadIdx.x % WARP_SIZE) + blockIdx.x * WARP_SIZE;

	if (tid < pages) {
		int cur = tid;
		for (int i = 0; i < ITERATIONS; i++) {
			int count = rowptr[cur + 1] - rowptr[cur];
			if (curand_uniform(&state[rid]) < Q) {
				cur = count == 0 ? cur : col[rowptr[cur] + (int)(curand_uniform(&state[rid]) * count)];
			}
			else
				cur = tid;
			atomicAdd(&r[cur], 1);
		}
	}
}

__global__
void norm(const int pages, float* r, float *sum) {
	const int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < pages) {
		r[tid] /= ITERATIONS;
	}
}

void monte_carlo_cu(
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
	float* _r;
	int* _Ap, * _Aj;
	curandStateMRG32k3a* state;

	float* sum = new float(0);
	const int blocks = pages / THREADS_PER_BLOCK + 1;
	cudaMalloc(&state, sizeof(curandStateMRG32k3a) * blocks * WARP_SIZE);
	cudaMalloc(&_r, sizeof(float) * pages);
	
#ifdef prefetch
	cudaMemPrefetchAsync(rowptr, sizeof(int) * (pages + 1), deviceId);
	cudaMemPrefetchAsync(col, sizeof(int) * (nonzeros), deviceId);
	mc_init << < blocks, THREADS_PER_BLOCK >> > (pages, _r, state, time(NULL));
	rand_walk << < blocks, THREADS_PER_BLOCK >> > (pages, nonzeros, _r, col, rowptr, state);
	norm << < blocks, THREADS_PER_BLOCK >> > (pages, _r, sum);
	cudaMemPrefetchAsync(r, sizeof(float) * pages, cudaCpuDeviceId);
#else
	cudaMalloc(&_Ap, sizeof(int) * (pages + 1));
	cudaMalloc(&_Aj, sizeof(int) * (nonzeros));
	cudaMemcpy(_Ap, rowptr, sizeof(int) * (pages + 1), cudaMemcpyHostToDevice);
	cudaMemcpy(_Aj, col, sizeof(int) * (nonzeros), cudaMemcpyHostToDevice);

	mc_init <<< blocks, THREADS_PER_BLOCK >>> (pages, _r, state, time(NULL));
	rand_walk <<< blocks, THREADS_PER_BLOCK >>> (pages, nonzeros, _r, _Aj, _Ap, state);
	norm <<< blocks, THREADS_PER_BLOCK >>> (pages, _r, sum);
	cudaMemcpy(r, _r, sizeof(float) * pages, cudaMemcpyDeviceToHost);
	cudaFree(_Ap);
	cudaFree(_Aj);
#endif
	cudaDeviceSynchronize();
	
	cudaFree(state);
	cudaFree(_r);
	
}