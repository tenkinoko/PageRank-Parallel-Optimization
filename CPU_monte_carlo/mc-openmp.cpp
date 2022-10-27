#include "macro.h"
#include <random>
using namespace std;

void monte_carlo(
	const int pages,		// number of total pages
	const int nonzeros,		// number of total edges
	float* r,				// pagerank value of each page
	const int* row,			// row index of each edge
	const int* col,			// column index of each edge
	const int* rowptr,		// row pointer of each row (CSR)
	const int* rowcount,	// number of edges in each row 
	const float* value)		// value of each edge		
{
	int threads = omp_get_max_threads();
	
	float *r_temp = new float[pages * threads];
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	timer(head);
	#pragma omp parallel shared(threads)
	{
		// High-quality randomization
		std::random_device rand_dev;
		std::mt19937 generator(rand_dev());
		std::uniform_real_distribution<float> distribution(0.f, 1.f);
		for (int i = 0; i < pages; i++) {
			int cur = i;
			for (int j = 0; j < ITERATIONS; j++) {
				if (distribution(generator) < Q)
					cur = rowcount[cur] == 0 ? cur : col[rowptr[cur] + (int)(distribution(generator) * rowcount[cur])];
				else
					cur = i;
				r_temp[cur] += 1;
			}
		}
	}
	
	float sum = 0.f;
	#pragma omp parallel for
	for (int i = 0; i < pages; i++) {
		for(int j = 0; j < threads; j++){
			r[i] += r_temp[i * threads +j];
		}
		r[i] /= ITERATIONS;
		sum += r[i];
	}
	#pragma omp parallel for
	for (int i = 0; i < pages; i++) {
		r[i] /= sum;
	}
	timer(tail);
	output("Monte Carlo:\t\t", total(head, tail));
}