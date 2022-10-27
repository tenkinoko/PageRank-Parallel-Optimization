#include "../include/macro.h"
#include <semaphore.h>
using namespace std;
int LENGTH = N_ROW;
int QUEUE = N_ROW  / THREAD;
float error = 0.0;

// TODO: may need to alter for the D (sum)
// trivial version - use mutex
pthread_mutex_t lock;
int tasks = 0;
bool roundFinish = false;
bool roundBegin = false;
int threadReady = 0;
int threadFinish = 0;

pthread_cond_t threshold;

void task_SpMV_AVX(void *arg) {
    int start = *(int*) arg;
    int endRow = start + LENGTH;
    int i = start;
    __m256 vec1, vec2, vec3, vec_acc;
    while(1){
        pthread_mutex_lock(&lock);
        threadReady ++;
        pthread_mutex_unlock(&lock);
        while(1){
            if(roundBegin)
                break;
        }
        while(i < endRow) {
            int r_begin = Ap[i];
            int r_end = Ap[i + 1];
            float acc = 0.0;
            int q = r_begin;
            vec_acc = _mm256_setzero_ps();
            float temp[8];
            memset(temp, 0, sizeof(temp));
            for (; q < r_end - 7; q += 8) {
                vec1 = _mm256_loadu_ps(Av + q);
                vec2 = _mm256_set_ps(R[Aj[q]], R[Aj[q + 1]], R[Aj[q + 2]], R[Aj[q + 3]], R[Aj[q + 4]], R[Aj[q + 5]], R[Aj[q + 6]], R[Aj[q + 7]]);
                vec3 = _mm256_mul_ps(vec1, vec2);
                vec_acc = _mm256_add_ps(vec3, vec_acc);
                _mm256_storeu_ps(temp, vec_acc);
            }
            acc += temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
            for (; q < r_end; q++) {
                acc += Av[q] * R[Aj[q]];
            }
            /*acc += Av[c] * R[Aj[c]];*/
            Y[i] = Q * acc;
            i++;
        }
        pthread_mutex_lock(&lock);
        threadFinish++;
        pthread_mutex_unlock(&lock);
        i = start;
        while(1){
            if(roundFinish)
                break;
        }
    if(error<EPSILON)
        return;
    }
}

void pool_SpMV_AVX(){
    int iter = ITERATIONS;
    float totaltime =0.0;
    while(iter--){
        for (int i = 0; i < N_ROW; i++) {
            if (i == 0) {
                R[i] = 1.0;
            }
            else {
                R[i] = 0.0;
            }
            Y[i] = 0.0;
        }
        threadpool_t* pool;
        pthread_mutex_init(&lock, NULL);
        assert((pool = threadpool_create(THREAD, QUEUE, 0)) != NULL);
        //cout<<"Pool Started with "<<THREAD<<" threads, queue size of "<<QUEUE<<endl;
        for(int i = 0; i<N_ROW; i+=LENGTH){
            int *start = new int();
            *start = i;
            if(threadpool_add(pool, &task_SpMV_AVX, (void*)start, 0)==0){
                pthread_mutex_lock(&lock);
                tasks++;
                pthread_mutex_unlock(&lock);
            }
            R_past[i] = R[i];
        }
        //assert(tasks == N_ROW);
        
        begin_count;
        do{
            while(1){
                if(threadReady == THREAD)
                    break;
            }
            pthread_mutex_lock(&lock);
            threadReady = 0;
            error = 0.0;
            roundBegin = true;
            roundFinish = false;
            pthread_mutex_unlock(&lock);
            while(1){
                if(threadFinish == THREAD)
                    break;
            }
            for(int i=0; i<N_ROW; i++){
                R_past[i] = R[i];
                R[i] = Y[i] + (1 - Q) * R_past[i];
                error += abs(R[i] - R_past[i]);
            }
            pthread_mutex_lock(&lock);
            threadFinish = 0;
            roundFinish = true;
            roundBegin = false;
            pthread_mutex_unlock(&lock);
            //cout<<error<<endl;
        }while(error>=EPSILON);
        end_count;
        assert(threadpool_destroy(pool, 0) == 0);
        totaltime += total;
    }
    output("pthread_AVX", totaltime/ITERATIONS);
}

void task_SpMV(void *arg) {
    int start = *(int*) arg;
    int endRow = start + LENGTH;
    int i = start;
    while(1){
        pthread_mutex_lock(&lock);
        threadReady ++;
        pthread_mutex_unlock(&lock);
        while(1){
            if(roundBegin)
                break;
        }
        while(i < endRow) {
            int r_begin = Ap[i];
            int r_end = Ap[i + 1];
            float acc = 0;
            int c = r_begin;
            for (; c < r_end; c++) {
                acc += Av[c] * R[Aj[c]];
            }
            Y[i] = Q * acc;
            i++;
        }
        pthread_mutex_lock(&lock);
        threadFinish++;
        pthread_mutex_unlock(&lock);
        i = start;
        while(1){
            if(roundFinish)
                break;
        }
    if(error<EPSILON)
        return;
    }
}


void pool_SpMV(){
    int iter = ITERATIONS;
    float totaltime =0.0;
    while(iter--){
        for (int i = 0; i < N_ROW; i++) {
            if (i == 0) {
                R[i] = 1.0;
            }
            else {
                R[i] = 0.0;
            }
            Y[i] = 0.0;
        }
        threadpool_t* pool;
        pthread_mutex_init(&lock, NULL);
        assert((pool = threadpool_create(THREAD, QUEUE, 0)) != NULL);
        //cout<<"Pool Started with "<<THREAD<<" threads, queue size of "<<QUEUE<<endl;
        for(int i = 0; i<N_ROW; i+=LENGTH){
            int *start = new int();
            *start = i;
            if(threadpool_add(pool, &task_SpMV, (void*)start, 0)==0){
                pthread_mutex_lock(&lock);
                tasks++;
                pthread_mutex_unlock(&lock);
            }
            R_past[i] = R[i];
        }
        //assert(tasks == N_ROW);
        
        begin_count;
        do{
            while(1){
                if(threadReady == THREAD)
                    break;
            }
            pthread_mutex_lock(&lock);
            threadReady = 0;
            error = 0.0;
            roundBegin = true;
            roundFinish = false;
            pthread_mutex_unlock(&lock);
            while(1){
                if(threadFinish == THREAD)
                    break;
            }
            for(int i=0; i<N_ROW; i++){
                R_past[i] = R[i];
                R[i] = Y[i] + (1 - Q) * R_past[i];
                error += abs(R[i] - R_past[i]);
            }
            pthread_mutex_lock(&lock);
            threadFinish = 0;
            roundFinish = true;
            roundBegin = false;
            pthread_mutex_unlock(&lock);
            //cout<<error<<endl;
        }while(error>=EPSILON);
        end_count;
        assert(threadpool_destroy(pool, 0) == 0);
        totaltime += total;
    }
    output("pthread", totaltime/ITERATIONS);
}

void task_SpMV_unrolled(void *arg) {
    int start = *(int*) arg;
    int endRow = start + LENGTH;
    int i = start;
    while(1){
        pthread_mutex_lock(&lock);
        threadReady ++;
        //cout<<threadReady<<endl;
        pthread_mutex_unlock(&lock);

        while(1){
            if(roundBegin)
                break;
        }
        while(i < endRow) {
            int r_begin = Ap[i];
            int r_end = Ap[i + 1];
            float acc = 0;
            int c = r_begin;
            for (; c < r_end-3; c = c+4) {
                acc += Av[c] * R[Aj[c]];
                acc += Av[c+1] * R[Aj[c+1]];
                acc += Av[c+2] * R[Aj[c+2]];
                acc += Av[c+3] * R[Aj[c+3]];
            }
            for(; c<r_end; c++)
                acc += Av[c] * R[Aj[c]];
            Y[i] = Q * acc;
            i++;
        }
        pthread_mutex_lock(&lock);
        threadFinish++;
        pthread_mutex_unlock(&lock);
        i = start;
        while(1){
            if(roundFinish)
                break;
        }
    if(error<EPSILON)
        return;
    }

}

void pool_SpMV_unrolled(){
    int iter = ITERATIONS;
    float totaltime =0.0;
    while(iter--){
        for (int i = 0; i < N_ROW; i++) {
            if (i == 0) {
                R[i] = 1.0;
            }
            else {
                R[i] = 0.0;
            }
            Y[i] = 0.0;
        }
        threadpool_t* pool;
        pthread_mutex_init(&lock, NULL);
        assert((pool = threadpool_create(THREAD, QUEUE, 0)) != NULL);
        //cout<<"Pool Started with "<<THREAD<<" threads, queue size of "<<QUEUE<<endl;
        for(int i = 0; i<N_ROW; i+=LENGTH){
            int *start = new int();
            *start = i;
            if(threadpool_add(pool, &task_SpMV_unrolled, (void*)start, 0)==0){
                pthread_mutex_lock(&lock);
                tasks++;
                pthread_mutex_unlock(&lock);
            }
            R_past[i] = R[i];
        }
        //assert(tasks == N_ROW);
        
        begin_count;
        do{
            while(1){
                if(threadReady == THREAD)
                    break;
            }
            pthread_mutex_lock(&lock);
            threadReady = 0;
            error = 0.0;
            roundBegin = true;
            roundFinish = false;
            pthread_mutex_unlock(&lock);
            while(1){
                if(threadFinish == THREAD)
                    break;
            }
            for(int i=0; i<N_ROW; i++){
                R_past[i] = R[i];
                R[i] = Y[i] + (1 - Q) * R_past[i];
                error += abs(R[i] - R_past[i]);
            }
            pthread_mutex_lock(&lock);
            threadFinish = 0;
            roundFinish = true;
            roundBegin = false;
            pthread_mutex_unlock(&lock);
            //cout<<error<<endl;
        }while(error>=EPSILON);
        end_count;
        assert(threadpool_destroy(pool, 0) == 0);
        totaltime += total;
    }
    output("pthread_unrolled", totaltime/ITERATIONS);
}

void task_SpMV_improved(void *arg) {
    int start = *(int*) arg;
    int endRow = start + LENGTH;
    int i = start;
    while(1){
        pthread_mutex_lock(&lock);
        threadReady ++;
        //cout<<threadReady<<endl;
        pthread_mutex_unlock(&lock);

        while(1){
            if(roundBegin)
                break;
        }
        while(i < endRow) {
            int r_begin = Ap[i];
            int r_end = Ap[i + 1];
            float acc = 0;
            for (int c = r_begin; c < r_end; c++) {
                acc += Av[c] * R[Aj[c]];
            }
            Y[i] = Q * acc;
            R_past[i] = R[i];
            R_copy[i] = Y[i] + (1 - Q) * R_past[i];
            i++;
        }
        pthread_mutex_lock(&lock);
        threadFinish++;
        pthread_mutex_unlock(&lock);
        i = start;
        while(1){
            if(roundFinish)
                break;
        }
    if(error<EPSILON)
        return;
    }

}

void pool_SpMV_improved(){
    int iter = ITERATIONS;
    float totaltime = 0.0;
    while(iter--){
        for (int i = 0; i < N_ROW; i++) {
            if (i == 0) {
                R[i] = 1.0;
            }
            else {
                R[i] = 0.0;
            }
            Y[i] = 0.0;
        }
        threadpool_t* pool;
        pthread_mutex_init(&lock, NULL);
        assert((pool = threadpool_create(THREAD, QUEUE, 0)) != NULL);
        //cout<<"Pool Started with "<<THREAD<<" threads, queue size of "<<QUEUE<<endl;
        for(int i = 0; i<N_ROW; i+=LENGTH){
            int *start = new int();
            *start = i;
            if(threadpool_add(pool, &task_SpMV_improved, (void*)start, 0)==0){
                pthread_mutex_lock(&lock);
                tasks++;
                pthread_mutex_unlock(&lock);
            }
            R_past[i] = R[i];
        }
        //assert(tasks == N_ROW);
        
        begin_count;
        do{
            while(1){
                if(threadReady == THREAD)
                    break;
            }
            pthread_mutex_lock(&lock);
            threadReady = 0;
            error = 0.0;
            roundBegin = true;
            roundFinish = false;
            pthread_mutex_unlock(&lock);
            while(1){
                if(threadFinish == THREAD)
                    break;
            }
            memcpy(R, R_copy, sizeof(float) * N_ROW);
            for(int i=0;i<N_ROW; i++){
                error += abs(R[i] - R_past[i]);
            }
            pthread_mutex_lock(&lock);
            threadFinish = 0;
            roundFinish = true;
            roundBegin = false;
            pthread_mutex_unlock(&lock);
        }while(error>=EPSILON);
        end_count;
        assert(threadpool_destroy(pool, 0) == 0);
        totaltime += total;
    }
    output("pthread_improved", totaltime / ITERATIONS);
}

void task_SpMV_cond(void *arg) {
    int start = *(int*) arg;
    int endRow = start + LENGTH;
    int i = start;
    while(1){
        pthread_mutex_lock(&lock);
        threadReady ++;
        if(threadReady == THREAD)
            pthread_cond_signal(&threshold);
        pthread_mutex_unlock(&lock);
        while(1){
            if(roundBegin)
                break;
        }
        while(i < endRow) {
            int r_begin = Ap[i];
            int r_end = Ap[i + 1];
            float acc = 0;
            int c = r_begin;
            for (; c < r_end; c++) {
                acc += Av[c] * R[Aj[c]];
            }
            Y[i] = Q * acc;
            i++;
        }
        pthread_mutex_lock(&lock);
        threadFinish++;
        if(threadFinish == THREAD)
            pthread_cond_signal(&threshold);
        pthread_mutex_unlock(&lock);
        i = start;
        while(1){
            if(roundFinish)
                break;
        }
    if(error<EPSILON)
        return;
    }
}


void pool_SpMV_cond(){
    int iter = ITERATIONS;
    float totaltime =0.0;
    while(iter--){
        for (int i = 0; i < N_ROW; i++) {
            if (i == 0) {
                R[i] = 1.0;
            }
            else {
                R[i] = 0.0;
            }
            Y[i] = 0.0;
        }
        threadpool_t* pool;
        pthread_mutex_init(&lock, NULL);
        pthread_cond_init(&threshold, NULL);
        assert((pool = threadpool_create(THREAD, QUEUE, 0)) != NULL);
        for(int i = 0; i<N_ROW; i+=LENGTH){
            int *start = new int();
            *start = i;
            if(threadpool_add(pool, &task_SpMV_cond, (void*)start, 0)==0){
                pthread_mutex_lock(&lock);
                tasks++;
                pthread_mutex_unlock(&lock);
            }
            R_past[i] = R[i];
        }
        
        begin_count;
        do{
            pthread_mutex_lock(&lock);
            while(threadReady<THREAD){
                pthread_cond_wait(&threshold, &lock);
            }
            threadReady = 0;
            error = 0.0;
            roundBegin = true;
            roundFinish = false;
            pthread_mutex_unlock(&lock);
            while(threadFinish<THREAD){
                pthread_cond_wait(&threshold, &lock);
            }
            pthread_mutex_unlock(&lock);
            for(int i=0; i<N_ROW; i++){
                R_past[i] = R[i];
                R[i] = Y[i] + (1 - Q) * R_past[i];
                error += abs(R[i] - R_past[i]);
            }
            pthread_mutex_lock(&lock);
            threadFinish = 0;
            roundFinish = true;
            roundBegin = false;
            pthread_mutex_unlock(&lock);
            //cout<<error<<endl;
        }while(error>=EPSILON);
        end_count;
        assert(threadpool_destroy(pool, 0) == 0);
        totaltime += total;
    }
    output("pthread_cond", totaltime/ITERATIONS);
}
