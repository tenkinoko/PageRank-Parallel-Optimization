#include <iostream>
#include "../include/threadpool.h"
enum threadpool_shutdown_t {
    immediate_shutdown = 1,
    graceful_shutdown = 2
};

struct threadpool_task_t {
    void (*function) (void*);
    void* argument;
};

struct threadpool_t {
    pthread_mutex_t lock;
    pthread_cond_t notify;
    pthread_t* threads;
    threadpool_task_t* queue;
    int thread_count;
    int queue_size;
    int head;
    int tail;
    int count;
    int shutdown;
    int started;
};

static void* threadpool_thread(void* threadpool);

int threadpool_free(threadpool_t* pool);

threadpool_t* threadpool_create(int thread_count, int queue_size, int flags) {
    threadpool_t* pool;
    int i;
    (void)flags;
    bool free_pool_flag = false;
    if (thread_count <= 0 || thread_count > MAX_THREADS || queue_size > MAX_QUEUE) {
        free_pool_flag = true;
    }
    try {
        pool = new threadpool_t();
    }
    catch (...) {
        std::cout << "Bad alloc" << std::endl;
        return nullptr;
    }

    pool->thread_count = 0;
    pool->queue_size = queue_size;
    pool->head = pool->tail = pool->count = 0;
    pool->shutdown = pool->started = 0;

    pool->threads = new pthread_t[thread_count];
    pool->queue = new threadpool_task_t[queue_size];

    if ((pthread_mutex_init(&(pool->lock), NULL) != 0) ||
        (pthread_cond_init(&(pool->notify), NULL) != 0) ||
        (pool->threads == NULL) ||
        (pool->queue == NULL)) {
        free_pool_flag = true;
    }

    for (i = 0; i < thread_count; i++) {
            int ret = pthread_create(&(pool->threads[i]), NULL, threadpool_thread, (void*)pool) != 0;
        if (ret != 0) {
            threadpool_destroy(pool, 0);
            return NULL;
        }
        pool->thread_count++;
        pool->started++;
    }
    if (free_pool_flag) {
        threadpool_free(pool);
        return nullptr;
    }
    return pool;
}

int threadpool_add(threadpool_t* pool, void (*function)(void*),
    void* argument, int flags) {
    int err = 0;
    int next;
    (void)flags;

    if (pool == NULL || function == NULL) {
        return threadpool_invalid;
    }

    if (pthread_mutex_lock(&(pool->lock)) != 0) {
        return threadpool_lock_failure;
    }

    next = (pool->tail + 1) % pool->queue_size;

    do {
        /* Are we full ? */
        if (pool->count == pool->queue_size) {
            err = threadpool_queue_full;
            break;
        }

        /* Are we shutting down ? */
        if (pool->shutdown) {
            err = threadpool_shutdown;
            break;
        }

        /* Add task to queue */
        pool->queue[pool->tail].function = function;
        pool->queue[pool->tail].argument = argument;
        pool->tail = next;
        pool->count += 1;

        /* pthread_cond_broadcast */
        if (pthread_cond_signal(&(pool->notify)) != 0) {
            err = threadpool_lock_failure;
            break;
        }
    } while (0);

    if (pthread_mutex_unlock(&pool->lock) != 0) {
        err = threadpool_lock_failure;
    }

    return err;
}

int threadpool_destroy(threadpool_t* pool, int flags)
{
    int i, err = 0;

    if (pool == NULL) {
        return threadpool_invalid;
    }

    if (pthread_mutex_lock(&(pool->lock)) != 0) {
        return threadpool_lock_failure;
    }

    do {
        /* Already shutting down */
        if (pool->shutdown) {
            err = threadpool_shutdown;
            break;
        }

        pool->shutdown = (flags & threadpool_graceful) ?
            graceful_shutdown : immediate_shutdown;

        /* Wake up all worker threads */
        if ((pthread_cond_broadcast(&(pool->notify)) != 0) ||
            (pthread_mutex_unlock(&(pool->lock)) != 0)) {
            err = threadpool_lock_failure;
            break;
        }

        /* Join all worker thread */
        for (i = 0; i < pool->thread_count; i++) {
            if (pthread_join(pool->threads[i], NULL) != 0) {
                err = threadpool_thread_failure;
            }
        }
    } while (0);

    /* Only if everything went well do we deallocate the pool */
    if (!err) {
        threadpool_free(pool);
    }
    return err;
}

int threadpool_free(threadpool_t* pool)
{
    if (pool == NULL || pool->started > 0) {
        return -1;
    }

    /* Did we manage to allocate ? */
    if (pool->threads) {
        delete[] pool->threads;
        delete[] pool->queue;

        /* Because we allocate pool->threads after initializing the
           mutex and condition variable, we're sure they're
           initialized. Let's lock the mutex just in case. */
        pthread_mutex_lock(&(pool->lock));
        pthread_mutex_destroy(&(pool->lock));
        pthread_cond_destroy(&(pool->notify));
    }
    delete pool;
    return 0;
}

static void* threadpool_thread(void* threadpool)
{
    threadpool_t* pool = (threadpool_t*)threadpool;
    threadpool_task_t task;

    for (;;) {
        /* Lock must be taken to wait on conditional variable */
        pthread_mutex_lock(&(pool->lock));

        /* Wait on condition variable, check for spurious wakeups.
           When returning from pthread_cond_wait(), we own the lock. */
        while ((pool->count == 0) && (!pool->shutdown)) {
            pthread_cond_wait(&(pool->notify), &(pool->lock));
        }

        if ((pool->shutdown == immediate_shutdown) ||
            ((pool->shutdown == graceful_shutdown) &&
                (pool->count == 0))) {
            break;
        }

        /* Grab our task */
        task.function = pool->queue[pool->head].function;
        task.argument = pool->queue[pool->head].argument;
        pool->head = (pool->head + 1) % pool->queue_size;
        pool->count -= 1;

        /* Unlock */
        pthread_mutex_unlock(&(pool->lock));

        /* Get to work */
        (*(task.function))(task.argument);
    }

    pool->started--;

    pthread_mutex_unlock(&(pool->lock));
    pthread_exit(NULL);
    return(NULL);
}
