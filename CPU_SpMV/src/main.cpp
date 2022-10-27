#include <iostream>
#include <pthread.h>
#include <unistd.h>
#include <mpi.h>
#include "../include/macro.h"

int main()
{
	MPI::Init();
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	readData();
	if(rank == 0){
	power_iteration_SpMV(0);
	debug_output(0x0000);
	// simple_openmp_PIS(0);
	// openmp_opt_PIS(0);
	// openmp_opt_dynamic_PIS(0);
	}
	mpi_Master_Slave();
	if(rank == 0)
		debug_output(0x0000);
	mpi_Collective_PIS();
	if(rank == 0)
		debug_output(0x0000);
	mpi_RMA();
	if(rank == 0)
		debug_output(0x0000);
	mpi_RMA_omp();
	if(rank == 0)
		debug_output(0x0000);
	MPI::Finalize();
    return 0;
}
