#include<iostream>
#include<fstream>
#include "../include/macro.h"
using namespace std;
dataItem* WebGraph;
int M, N, Nonzeros;
int N_ROW;
// CSR: Av - values, Aj - column indices, Ap - offsets
int *Ap;
float* Av;
int* Aj;
int nonzero;
// initial R vector (aligned)
float *R;
// initial Y vector
float *Y;
// initial R standard output
float *stdR;
float *error_list;
float* R_past;
float* R_copy;
struct timeval t_begin, t_end;

void readData() {
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	// Open the file:
	ifstream fin("dataset/wikipedia-20051105.mtx");
	// Ignore headers and comments:
	while (fin.peek() == '%') fin.ignore(2048, '\n');

	// Read defining parameters:
	fin >> M >> N >> Nonzeros;
	nonzero = Nonzeros;
	N_ROW = M;
	WebGraph = new dataItem [Nonzeros];
	int count = 1;
	int begin = 0;
	float newval = 0.0;
	if(LOG)
		cout<<"Start Reading Dataset..."<<endl;
	begin_count;
	for (int l=0; l<Nonzeros; l++)
	{
		fin >> WebGraph[l].row >> WebGraph[l].col;
		if(l>0 && WebGraph[l].col != WebGraph[l-1].col){
			newval = 1.0/count;
			for(int i=begin; i<=l-1; i++)
				WebGraph[i].val = newval;
			count = 1;
			begin = l;
		}
		else{
			count++;
			if(l == Nonzeros -1){
				newval = 1.0/count;
				for(int i=begin; i<Nonzeros; i++)
					WebGraph[i].val = newval;
			}
		}
	}
	end_count;
	if(LOG)
		cout<<"Read Dataset Success! Time: "<<total/1000000.0<<"s"<<endl;
	int *num_nonzeros = new int[N_ROW];
	memset(num_nonzeros, 0, N_ROW*sizeof(int));
	for(int i=0; i<Nonzeros; i++){
		num_nonzeros[WebGraph[i].row]++;
	}
	int maxcount = 0;
	int l0=0, l1=0, l2=0;
	for(int i=0; i<N_ROW; i++){
		if(num_nonzeros[i] > maxcount)
			maxcount = num_nonzeros[i];
		if(num_nonzeros[i] > 96)
			l2++;
		else if(num_nonzeros[i] > 6)
			l1++;
		else
			l0++;
	}
	if(LOG)
		cout<<"The row with the most nonzeros includes "<<maxcount<<" elements"<<endl;
	if(LOG)
		cout<<"l0: "<<l0<<" l1: "<<l1<<" l2: "<<l2<<endl;
	int *csr_row = new int [M+1];
	memset(csr_row, 0, (M+1)*sizeof(int));
	for(int i=0; i<Nonzeros; i++)
		csr_row[WebGraph[i].row + 1]++;
	for(int i=0; i<M; i++)
		csr_row[i+1] += csr_row[i];
	if(LOG)
		cout<<"CSR Format Generation Success!"<<endl;
	fin.close();
	int counts = (Nonzeros/8)*8;
	Ap = new int[M+1];
	Av = (float*) aligned_alloc(8 * sizeof(float),counts * sizeof(float));
	Aj = new int[Nonzeros];
	for(int i=0; i<Nonzeros; i++){
		Aj[i] = WebGraph[i].col;
		Av[i] = WebGraph[i].val;
	}
	memcpy(Ap, csr_row, (M+1)*sizeof(int));
	if(LOG)
		cout<<"CSR Format Initialization Success!"<<endl;
	R = (float*) aligned_alloc(8 * sizeof(float),N_ROW * sizeof(float));
	Y = (float*) aligned_alloc(8 * sizeof(float),N_ROW * sizeof(float));
	stdR = (float*) aligned_alloc(8 * sizeof(float),N_ROW * sizeof(float));
	error_list = (float*) aligned_alloc(8 * sizeof(float),N_ROW * sizeof(float));
	R_past = (float*) aligned_alloc(8 * sizeof(float),N_ROW * sizeof(float));
	R_copy = (float*) aligned_alloc(8 * sizeof(float),N_ROW * sizeof(float));
	cout<<"Process "<<rank<<" initialization complete!"<<endl;
}
