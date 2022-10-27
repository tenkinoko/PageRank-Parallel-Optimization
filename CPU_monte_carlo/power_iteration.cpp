#include "macro.h"
using namespace std;
void power_iter(
	const int pages,		// number of total pages
	const int nonzeros,		// number of total edges
	float* r,				// pagerank value of each page
	const int* row,			// row index of each edge
	const int* col,			// column index of each edge
	const int* rowptr,		// row pointer of each row (CSR)
	const int* rowcount,	// number of edges in each row 
	const float* value)		// value of each edge		
{
	// Initialization
	float* r_ = new float[pages];
	float* y = new float[pages];
	memset(r_, 0, sizeof(float) * pages);
	for (int i = 0; i < pages; i++) {
		r[i] = 1.f / pages;
		y[i] = 0.f;
	}
	float error;
	float d;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	timer(head);
	do {
		d = 0.f;
		error = 0.f;
		for (int i = 0; i < pages; i++) {
			int r_begin = rowptr[i];
			int r_end = rowptr[i + 1];
			float acc = 0.f;
			for (int c = r_begin; c < r_end; c++) {
				float a = value[c];
				float b = r[col[c]];
				acc += a * b;
			}
			y[i] = Q * acc;
			d += y[i];
		}
		for (int i = 0; i < pages; i++) {
			r_[i] = r[i];
			r[i] = y[i] + (1.f - d) / pages;
			error += abs(r[i] - r_[i]);
		}
	} while (error >= TOLERANCE);
	timer(tail);
	output("Power Iteration:\t", total(head, tail));
}