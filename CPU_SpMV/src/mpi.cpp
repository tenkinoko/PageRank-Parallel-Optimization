#include "../include/macro.h"
using namespace std;
#define EPOCH_ACCOMPLISHED_TAG (N_ROW+1)
#define UPDATE_R_TAG (N_ROW+2)
#define UPDATE_R_FINISHED_TAG (N_ROW+3)
#define ALL_END_TAG (N_ROW+4)
#define DEFAULT_TASKSIZE 16384
int taskSize = DEFAULT_TASKSIZE;

// Master-Slave
void mpi_Master_Slave_SpMV() {
    // Initialization
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Status status;
    int rowBegin = 0;
    float *tempR = (float*) aligned_alloc(8 * sizeof(float),N_ROW * sizeof(float));
    for (int i = 0; i < N_ROW; i++) {
            R[i] = 1.0/N_ROW;
            Y[i] = 0.0;
    }
    // Start taking tasks
    while(1){
        MPI_Recv(&rowBegin, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        if(status.MPI_TAG == ALL_END_TAG)
        {
            return;
        }
        // if one epoch is finished, wait for the master to renew R
        else if(status.MPI_TAG == EPOCH_ACCOMPLISHED_TAG){
            MPI_Recv(tempR, N_ROW, MPI_FLOAT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            if(status.MPI_TAG == UPDATE_R_TAG){
                memcpy(R, tempR, sizeof(float)*N_ROW);
                MPI_Send(&rowBegin, 1, MPI_INT, 0, UPDATE_R_FINISHED_TAG, MPI_COMM_WORLD);
            }
        }
        else{
            int taskSizeReal = (rowBegin + taskSize < N_ROW)? taskSize : (N_ROW - rowBegin);
            // float sum = 0.0;
            for (int i = rowBegin; i < rowBegin + taskSizeReal; i++) {
                int r_begin = Ap[i];
                int r_end = Ap[i + 1];
                float acc = 0;
                for (int c = r_begin; c < r_end; c++) {
                    acc += Av[c] * R[Aj[c]];
                }
                Y[i] = Q * acc;
                // sum += Y[i];
            }
            // std::cout<<rowBegin<<" "<<sum<<std::endl;
            MPI_Send(&Y[rowBegin], taskSizeReal, MPI_FLOAT, 0, rowBegin, MPI_COMM_WORLD);
        }
    }
}

// PIS - Power Iteration with SpMV
double mpi_Master_Slave_PIS() {
    double totaltime = 0.0;
    double begin = 0.0;
    double end = 0.0;
    int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
    R_past = new float[N_ROW];
    for (int i = 0; i < N_ROW; i++) {
        R[i] = 1.0/N_ROW;
        Y[i] = 0.0;
    }
    /* Global Variables */
    int finished = 1;
    int renewed = 1;
    int rowBegin = 0;
    MPI_Status status;
    float error = 0.0;
    float sumR = 0.0;
    float sumY = 0.0;
    float *tempY = (float*) aligned_alloc(8 * sizeof(float),taskSize * sizeof(float));
    begin = MPI_Wtime();
    do{
        sumY = sumR = error = 0.0;
        rowBegin = 0;
        finished = renewed = 1;
        for(int i=1; i < size; i++){
            MPI_Send(&rowBegin, 1, MPI_INT, i, rowBegin, MPI_COMM_WORLD);
            //std::cout<<"main process sent: "<<rowBegin<<std::endl;
            rowBegin += taskSize;
        }
        while(finished < size){
            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            int taskSizeReal = 0;
            MPI_Get_count(&status, MPI_FLOAT, &taskSizeReal);
            //cout<<taskSizeReal<<endl;
            MPI_Recv(tempY, taskSize, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            memcpy(Y+status.MPI_TAG, tempY, sizeof(float)*taskSizeReal);
            // if one epoch has not accomplished
            if(rowBegin  < N_ROW){
                MPI_Send(&rowBegin, 1, MPI_INT, status.MPI_SOURCE, rowBegin, MPI_COMM_WORLD);
                //std::cout<<"main process sent: "<<rowBegin<<std::endl;
                rowBegin += taskSizeReal;
            }
            // one epoch accomplished
            else{
                MPI_Send(&rowBegin, 1, MPI_INT, status.MPI_SOURCE, EPOCH_ACCOMPLISHED_TAG, MPI_COMM_WORLD);
                finished++;
            }
        }
        for(int i=0; i<N_ROW; i++){
            sumY += Y[i];
            sumR += R[i];
        }
        //cout<<"sumY: "<<sumY<<" sumR: "<<sumR;
        for (int i = 0; i < N_ROW; i++) {
            R_past[i] = R[i];
            R[i] = Y[i] + (1 - sumY)/N_ROW;
            error += abs(R[i] - R_past[i]);
        }
        //cout<<" error: "<<error<<endl;
        for(int i=1; i<size;i++){
            MPI_Send(R, N_ROW, MPI_FLOAT, i, UPDATE_R_TAG, MPI_COMM_WORLD);
        }
        while(renewed <size){
            MPI_Recv(&rowBegin, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            if(status.MPI_TAG == UPDATE_R_FINISHED_TAG)
                renewed++;
        }  
    }while(error >= EPSILON);
    end = MPI_Wtime();
    totaltime = (end-begin);
    
    int _ = 0;
    for(int i=1; i<size; i++){
        MPI_Send(&_, 1, MPI_INT, i, ALL_END_TAG, MPI_COMM_WORLD);
    }
    return totaltime;
}

void mpi_Master_Slave(){
    int iter = ITERATIONS;
    int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
    double totaltime = 0.0;
    while(iter--){
        if(rank != 0)
		    mpi_Master_Slave_SpMV();
        else
            totaltime += mpi_Master_Slave_PIS();
    }
    if(rank == 0)
        output("MPI", totaltime*1000000/ITERATIONS);
}

void mpi_Collective_PIS(){
    int iter = ITERATIONS;
    int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
    double totaltime = 0.0;
    double begin = 0.0;
    double end = 0.0;
    while (iter--)
    {
        R_past = new float[N_ROW];
        for (int i = 0; i < N_ROW; i++) {
            R[i] = 1/N_ROW;
            Y[i] = 0.0;
        }

        /* Global Variables */
        float error = 0.0;
        float sumY = 0.0;
        int rowsPerProc = (N_ROW - N_ROW % size) / size;
        float *tempY = (float*) aligned_alloc(8 * sizeof(float),rowsPerProc * sizeof(float));
        int rowsLeft = N_ROW % size;
        int *rowsBegin = new int[size];
        int rowBegin = 0;
        rowsBegin[0] = 0;
        for(int i=1; i<size; i++){
            rowsBegin[i] = rowsBegin[i-1] + rowsPerProc; 
        }
        begin = MPI_Wtime();
        do{
            error = 0.0;
            MPI_Scatter(rowsBegin, 1, MPI_INT, &rowBegin, 1, MPI_INT, 0, MPI_COMM_WORLD);
            sumY = mpi_Collective_SpMV(rowBegin, rowsPerProc);
            //cout<<"rank: "<<rank<<" sumY: "<<sumY<<endl;
            memcpy(tempY, Y+rowBegin, rowsPerProc*sizeof(float));
            MPI_Gather(tempY, rowsPerProc, MPI_FLOAT, Y, rowsPerProc, MPI_FLOAT, 0, MPI_COMM_WORLD);
            float YGlobal = 0.0;
            MPI_Reduce(&sumY, &YGlobal, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
            if(rank==0){
                float YLeft = 0.0;
                YLeft = mpi_Collective_SpMV(rowsPerProc*size, rowsLeft);
                YGlobal += YLeft;
                for(int i=0 ;i<N_ROW; i++){
                    R_past[i] = R[i];
		            R[i] = Y[i] + (1 - YGlobal) / N_ROW;
		            error += abs(R[i] - R_past[i]);
                }
            }
            //std::cout<<"rank : "<<rank<<" iter: "<<iter<<" error: "<< error<<std::endl;
            MPI_Bcast(R, N_ROW, MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&error, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }while(error >= EPSILON);
        end = MPI_Wtime();
        totaltime += (end-begin);
    }
    if(rank == 0)
    {
        output("MPI Collective", totaltime*1000000/ITERATIONS);
    }
}

float mpi_Collective_SpMV(int begin, int num){
    float sum = 0.0;
    for (int i = begin; i < begin+num; i++) {
		int r_begin = Ap[i];
		int r_end = Ap[i + 1];
		float acc = 0;
		for (int c = r_begin; c < r_end; c++) {
			acc += Av[c] * R[Aj[c]];
		}
		Y[i] = Q * acc;
        sum += Y[i];
	}
    return sum;
}

void mpi_RMA(){
    int iter = ITERATIONS;
    int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
    double totaltime = 0.0;
    while(iter--){
        // flag[0]: rowBegin flag[1]: process finish number flag[2]: update r
        int *flag;
        MPI_Win flagwin;
        MPI_Win_allocate(3*sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &flag, &flagwin);
        MPI_Win Ywin;
        MPI_Win_create(Y, N_ROW*sizeof(float), sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD,  &Ywin);
        MPI_Win Rwin;
        float *commR;
        MPI_Win_allocate((N_ROW+1)*sizeof(float), sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &commR, &Rwin);
        float *errorStore;
        MPI_Win ewin;
        MPI_Win_allocate(sizeof(float), sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &errorStore, &ewin);
        if(rank == 0)
            flag[0] = flag[1] = flag[2] = 0;
        //cout<<"Windows Initialization Complete!"<<endl;
        if(rank != 0)
		{
            int rowBeginLocal = 0;
            int putRes = 0;
            int epochFinish = 0;
            int startNextEpoch = 0;
            float errorLocal = 1.0;
            for (int i = 0; i < N_ROW; i++) {
                    R[i] = 1.0/N_ROW;
                    Y[i] = 0.0;
            }
            // Start taking tasks
            while(errorLocal >= EPSILON){
                startNextEpoch = 0;
                epochFinish = 0;
                MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, flagwin);
                MPI_Get(&rowBeginLocal, 1, MPI_INT, 0, 0, 1, MPI_INT, flagwin);
                //cout<<"rank: "<<rank<<" row: "<<rowBeginLocal<<endl;
                if(rowBeginLocal >= N_ROW)
                {
                    epochFinish = 1;
                    int add1 = 1;
                    MPI_Get(&add1, 1, MPI_INT, 0, 1, 1, MPI_INT, flagwin);
                    add1 += 1;
                    //cout<<"flag1: "<<add1<<endl;
                    MPI_Put(&add1, 1, MPI_INT, 0, 1, 1, MPI_INT, flagwin);
                }
                else
                {
                    putRes = rowBeginLocal + taskSize;
                    MPI_Put(&putRes, 1, MPI_INT, 0, 0, 1, MPI_INT, flagwin);
                }
                MPI_Win_unlock(0, flagwin);
                //cout<<"rank: "<<rank<<" finish epoch"<<endl;
                // if one epoch is finished, wait for the master to renew R
                if(epochFinish == 1){
                    MPI_Status status;
                    MPI_Recv(&startNextEpoch, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                    MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, Rwin);
                    MPI_Get(&R[0], N_ROW, MPI_FLOAT, 0, 0, N_ROW, MPI_FLOAT, Rwin);
                    MPI_Win_flush(0, Rwin);
                    MPI_Win_unlock(0, Rwin);
                    MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, ewin);
                    MPI_Get(&errorLocal, 1, MPI_FLOAT, 0, 0, 1, MPI_FLOAT, ewin);
                    MPI_Win_flush(0, ewin);
                    MPI_Win_unlock(0, ewin);
                }
                else{
                    int taskSizeReal = (rowBeginLocal + taskSize < N_ROW)? taskSize : (N_ROW - rowBeginLocal);
                    // float sum = 0.0;
                    for (int i = rowBeginLocal; i < rowBeginLocal + taskSizeReal; i++) {
                        int r_begin = Ap[i];
                        int r_end = Ap[i + 1];
                        float acc = 0;
                        for (int c = r_begin; c < r_end; c++) {
                            acc += Av[c] * R[Aj[c]];
                        }
                        Y[i] = Q * acc;
                        // sum += Y[i];
                    }
                    //cout<<"rank: "<<rank<<" rowput:"<<rowBeginLocal<<endl;
                    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, Ywin);
                    MPI_Put(&Y[rowBeginLocal], taskSizeReal, MPI_FLOAT, 0, rowBeginLocal, taskSizeReal, MPI_FLOAT, Ywin);
                    // MPI_Win_unlock(0, Ywin);
                    MPI_Win_flush(0, Ywin);
                    MPI_Win_unlock(0, Ywin);
                    //cout<<"rank: "<<rank<<" compute finish"<<endl;
                }
            }
        }
        else
        {
            double begin = 0.0;
            double end = 0.0;
            float sumY = 0.0; 
            float error = 0.0;
            R_past = new float[N_ROW];
            for (int i = 0; i < N_ROW; i++) {
                R[i] = 1.0/N_ROW;
                Y[i] = 0.0;
            }
            /* Global Variables */

            begin = MPI_Wtime();
            do{
                sumY = 0.0;
                error = 0.0;
                while(1){
                    if(flag[1] == size - 1)
                        break;
                }
                for(int i=0; i<N_ROW; i++)
                    sumY += Y[i];
                for (int i = 0; i < N_ROW; i++) {
                    R_past[i] = R[i];
                    R[i] = Y[i] + (1 - sumY)/N_ROW;
                    error += abs(R[i] - R_past[i]);
                }
                memcpy(commR, R, sizeof(float)*N_ROW);
                errorStore[0] = error;
                //cout<<"sumY: "<<sumY<<" sumR: "<<sumR<<" error: "<<error<<endl;
                int sendflag = -1;
                for(int i = 1; i<size; i++){
                    MPI_Send(&sendflag, 1, MPI_INT, i, EPOCH_ACCOMPLISHED_TAG, MPI_COMM_WORLD);
                }
                flag[0] = 0;
                flag[1] = 0;
            }while(error >= EPSILON);
            end = MPI_Wtime();
            totaltime += (end-begin);
        }
        MPI_Win_free(&Rwin);
        MPI_Win_free(&Ywin);
        MPI_Win_free(&flagwin);
        MPI_Win_free(&ewin);
    }
    
    if(rank == 0)
        output("MPI RMA", totaltime*1000000/ITERATIONS);
}

void mpi_RMA_omp(){
    int iter = ITERATIONS;
    int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
    double totaltime = 0.0;
    while(iter--){
        // flag[0]: rowBegin flag[1]: process finish number flag[2]: update r
        int *flag;
        MPI_Win flagwin;
        MPI_Win_allocate(3*sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &flag, &flagwin);
        MPI_Win Ywin;
        MPI_Win_create(Y, N_ROW*sizeof(float), sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD,  &Ywin);
        MPI_Win Rwin;
        float *commR;
        MPI_Win_allocate((N_ROW+1)*sizeof(float), sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &commR, &Rwin);
        float *errorStore;
        MPI_Win ewin;
        MPI_Win_allocate(sizeof(float), sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &errorStore, &ewin);
        if(rank == 0)
            flag[0] = flag[1] = flag[2] = 0;
        //cout<<"Windows Initialization Complete!"<<endl;
        if(rank != 0)
		{
            int rowBeginLocal = 0;
            int putRes = 0;
            int epochFinish = 0;
            int startNextEpoch = 0;
            float errorLocal = 1.0;
            for (int i = 0; i < N_ROW; i++) {
                    R[i] = 1.0/N_ROW;
                    Y[i] = 0.0;
            }
            // Start taking tasks
            while(errorLocal >= EPSILON){
                startNextEpoch = 0;
                epochFinish = 0;
                MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, flagwin);
                MPI_Get(&rowBeginLocal, 1, MPI_INT, 0, 0, 1, MPI_INT, flagwin);
                //cout<<"rank: "<<rank<<" row: "<<rowBeginLocal<<endl;
                if(rowBeginLocal >= N_ROW)
                {
                    epochFinish = 1;
                    int add1 = 1;
                    MPI_Get(&add1, 1, MPI_INT, 0, 1, 1, MPI_INT, flagwin);
                    add1 += 1;
                    //cout<<"flag1: "<<add1<<endl;
                    MPI_Put(&add1, 1, MPI_INT, 0, 1, 1, MPI_INT, flagwin);
                }
                else
                {
                    putRes = rowBeginLocal + taskSize;
                    MPI_Put(&putRes, 1, MPI_INT, 0, 0, 1, MPI_INT, flagwin);
                }
                MPI_Win_unlock(0, flagwin);
                //cout<<"rank: "<<rank<<" finish epoch"<<endl;
                // if one epoch is finished, wait for the master to renew R
                if(epochFinish == 1){
                    MPI_Status status;
                    MPI_Recv(&startNextEpoch, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                    MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, Rwin);
                    MPI_Get(&R[0], N_ROW, MPI_FLOAT, 0, 0, N_ROW, MPI_FLOAT, Rwin);
                    MPI_Win_flush(0, Rwin);
                    MPI_Win_unlock(0, Rwin);
                    MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, ewin);
                    MPI_Get(&errorLocal, 1, MPI_FLOAT, 0, 0, 1, MPI_FLOAT, ewin);
                    MPI_Win_flush(0, ewin);
                    MPI_Win_unlock(0, ewin);
                }
                else{
                    int taskSizeReal = (rowBeginLocal + taskSize < N_ROW)? taskSize : (N_ROW - rowBeginLocal);
                    // float sum = 0.0;
                    # pragma omp parallel for schedule(dynamic, 32) num_threads(THREAD)
                    for (int i = rowBeginLocal; i < rowBeginLocal + taskSizeReal; i++) {
                        int r_begin = Ap[i];
                        int r_end = Ap[i + 1];
                        float acc = 0;
                        for (int c = r_begin; c < r_end; c++) {
                            acc += Av[c] * R[Aj[c]];
                        }
                        Y[i] = Q * acc;
                        // sum += Y[i];
                    }
                    //cout<<"rank: "<<rank<<" rowput:"<<rowBeginLocal<<endl;
                    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, Ywin);
                    MPI_Put(&Y[rowBeginLocal], taskSizeReal, MPI_FLOAT, 0, rowBeginLocal, taskSizeReal, MPI_FLOAT, Ywin);
                    // MPI_Win_unlock(0, Ywin);
                    MPI_Win_flush(0, Ywin);
                    MPI_Win_unlock(0, Ywin);
                    //cout<<"rank: "<<rank<<" compute finish"<<endl;
                }
            }
        }
        else
        {
            double begin = 0.0;
            double end = 0.0;
            float sumY = 0.0;
            float error = 0.0;
            R_past = new float[N_ROW];
            for (int i = 0; i < N_ROW; i++) {
                R[i] = 1.0/N_ROW;
                Y[i] = 0.0;
            }
            /* Global Variables */

            begin = MPI_Wtime();
            do{
                sumY = 0.0;
                error = 0.0;
                while(1){
                    if(flag[1] == size - 1)
                        break;
                }
                //# pragma omp parallel for schedule(dynamic, 32) num_threads(THREAD)
                for(int i=0; i<N_ROW; i++)
                    sumY += Y[i];
                //# pragma omp parallel for schedule(dynamic, 32) num_threads(THREAD)
                for (int i = 0; i < N_ROW; i++) {
                    R_past[i] = R[i];
                    R[i] = Y[i] + (1 - sumY)/N_ROW;
                    error += abs(R[i] - R_past[i]);
                }
                memcpy(commR, R, sizeof(float)*N_ROW);
                errorStore[0] = error;
                //cout<<"sumY: "<<sumY<<" sumR: "<<sumR<<" error: "<<error<<endl;
                int sendflag = -1;
                for(int i = 1; i<size; i++){
                    MPI_Send(&sendflag, 1, MPI_INT, i, EPOCH_ACCOMPLISHED_TAG, MPI_COMM_WORLD);
                }
                flag[0] = 0;
                flag[1] = 0;
            }while(error >= EPSILON);
            end = MPI_Wtime();
            totaltime += (end-begin);
        }
        MPI_Win_free(&Rwin);
        MPI_Win_free(&Ywin);
        MPI_Win_free(&flagwin);
        MPI_Win_free(&ewin);
    }
    
    if(rank == 0)
        output("MPI RMA omp", totaltime*1000000/ITERATIONS);
}