<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [deprecated/hw2/hw2/lock.cu](https://github.com/aanickode/cis6010/blob/main/deprecated/hw2/hw2/lock.cu)
- [deprecated/hw2/hw2/lock.cuh](https://github.com/aanickode/cis6010/blob/main/deprecated/hw2/hw2/lock.cuh)
- [deprecated/hw2/hw2/utils.cuh](https://github.com/aanickode/cis6010/blob/main/deprecated/hw2/hw2/utils.cuh)
- [deprecated/hw2/hw2/utils.cu](https://github.com/aanickode/cis6010/blob/main/deprecated/hw2/hw2/utils.cu)
- [deprecated/hw2/hw2/main.cu](https://github.com/aanickode/cis6010/blob/main/deprecated/hw2/hw2/main.cu)

</details>

# Lock-based Synchronization

## Introduction

Lock-based synchronization is a mechanism used in parallel computing to ensure thread-safe access to shared resources, preventing race conditions and maintaining data integrity. This wiki page focuses on the implementation of lock-based synchronization within the context of the provided project, which appears to be a CUDA-based application for parallel computing on GPUs.

The lock-based synchronization implementation in this project is designed to work with both CPU threads and GPU threads (CUDA threads). It provides a unified interface for acquiring and releasing locks, allowing for synchronization across CPU and GPU code.

Sources: [lock.cuh:1-10](), [lock.cu:1-10]()

## Lock Implementation

### Lock Data Structure

The lock implementation is based on a `Lock` struct, which encapsulates the necessary data and functionality for lock management. The `Lock` struct contains the following members:

```cpp
struct Lock {
    uint32_t *lock_addr;
    uint32_t *lock_array;
    uint32_t num_locks;
};
```

- `lock_addr`: A pointer to the shared memory location where the lock value is stored.
- `lock_array`: A pointer to an array of lock values, used for managing multiple locks.
- `num_locks`: The number of locks in the `lock_array`.

Sources: [lock.cuh:13-18]()

### Lock Initialization

The `initLock` function is used to initialize a `Lock` struct with a single lock value stored in shared memory. It takes a pointer to the shared memory location as an argument and returns a `Lock` struct with the `lock_addr` set to the provided memory location, and `lock_array` and `num_locks` set to `nullptr` and `0`, respectively.

```cpp
__device__ Lock initLock(uint32_t *lock_addr) {
    Lock lock = {lock_addr, nullptr, 0};
    return lock;
}
```

Sources: [lock.cuh:20-25]()

### Lock Acquisition

The `acquireLock` function is used to acquire a lock. It takes a `Lock` struct as an argument and performs an atomic operation to attempt to acquire the lock. If the lock is already acquired by another thread, the function spins (busy-waits) until the lock is released.

```cpp
__device__ void acquireLock(Lock lock) {
    while (atomicExch(lock.lock_addr, 1) != 0)
        ;
}
```

Sources: [lock.cuh:27-31]()

### Lock Release

The `releaseLock` function is used to release a previously acquired lock. It takes a `Lock` struct as an argument and performs an atomic operation to reset the lock value to zero, allowing other threads to acquire the lock.

```cpp
__device__ void releaseLock(Lock lock) {
    atomicExch(lock.lock_addr, 0);
}
```

Sources: [lock.cuh:33-36]()

## Multi-Lock Management

The lock implementation also provides functionality for managing multiple locks through the `initLocks` and `acquireMultipleLocks` functions.

### Multi-Lock Initialization

The `initLocks` function is used to initialize a `Lock` struct with multiple locks stored in an array. It takes a pointer to the shared memory array and the number of locks as arguments, and returns a `Lock` struct with the `lock_array` and `num_locks` members set accordingly.

```cpp
__device__ Lock initLocks(uint32_t *lock_array, uint32_t num_locks) {
    Lock lock = {nullptr, lock_array, num_locks};
    return lock;
}
```

Sources: [lock.cuh:38-43]()

### Multi-Lock Acquisition

The `acquireMultipleLocks` function is used to acquire multiple locks simultaneously. It takes a `Lock` struct as an argument and attempts to acquire all locks in the `lock_array` using a spin-lock approach. If any lock is already acquired by another thread, the function releases all previously acquired locks and retries until all locks are successfully acquired.

```cpp
__device__ void acquireMultipleLocks(Lock lock) {
    bool locked = false;
    while (!locked) {
        locked = true;
        for (uint32_t i = 0; i < lock.num_locks; i++) {
            if (atomicExch(&lock.lock_array[i], 1) != 0) {
                for (uint32_t j = 0; j < i; j++) {
                    atomicExch(&lock.lock_array[j], 0);
                }
                locked = false;
                break;
            }
        }
    }
}
```

Sources: [lock.cuh:45-58]()

## Usage Examples

The lock-based synchronization implementation is used in various parts of the project, particularly in the `main.cu` file, where it is employed to ensure thread-safe access to shared resources during parallel computations.

### Example 1: Single Lock Usage

In this example, a single lock is used to protect a shared counter variable during a parallel reduction operation.

```cpp
__global__ void reduce(uint32_t *g_data, uint32_t *g_odata, uint32_t n, uint32_t *lock_addr) {
    extern __shared__ uint32_t sdata[];
    uint32_t tid = threadIdx.x;
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_data[i] : 0;
    __syncthreads();

    // Perform reduction in shared memory
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result to global memory
    if (tid == 0) {
        Lock lock = initLock(lock_addr);
        acquireLock(lock);
        g_odata[blockIdx.x] = sdata[0];
        releaseLock(lock);
    }
}
```

In this example, each thread block performs a parallel reduction on a portion of the input data. The final result from each block is written to global memory, protected by a lock to ensure thread-safe access.

Sources: [main.cu:48-72]()

### Example 2: Multiple Lock Usage

In this example, multiple locks are used to protect shared resources during a parallel matrix multiplication operation.

```cpp
__global__ void matrixMul(uint32_t *a, uint32_t *b, uint32_t *c, uint32_t n, uint32_t *locks) {
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t sum = 0;
    for (uint32_t i = 0; i < n; i++) {
        sum += a[row * n + i] * b[i * n + col];
    }

    Lock lock = initLocks(locks, n);
    acquireMultipleLocks(lock);
    c[row * n + col] = sum;
    releaseMultipleLocks(lock);
}
```

In this example, each thread computes a single element of the output matrix by performing a dot product between a row of the first input matrix and a column of the second input matrix. Multiple locks are used to protect the shared output matrix during the write operation, ensuring that each element is updated atomically.

Sources: [main.cu:74-89]()

## Mermaid Diagrams

### Lock Acquisition Sequence Diagram

```mermaid
sequenceDiagram
    participant Thread
    participant Lock
    participant SharedMemory

    Thread->>Lock: acquireLock(lock)
    Loop Spin-lock
        Lock->>SharedMemory: atomicExch(lock_addr, 1)
        SharedMemory-->>Lock: Previous lock value
        Note right of Lock: If previous value is 0, lock is acquired
        Alt Previous value is not 0
            Note right of Lock: Lock is held by another thread, spin again
        Else Lock is acquired
            Break
        End
    End

    Note over Thread,Lock: Lock is acquired

    Thread->>Lock: Critical section code

    Thread->>Lock: releaseLock(lock)
    Lock->>SharedMemory: atomicExch(lock_addr, 0)
    SharedMemory-->>Lock: Previous lock value

    Note over Thread,Lock: Lock is released
```

This sequence diagram illustrates the process of acquiring and releasing a lock using the `acquireLock` and `releaseLock` functions, respectively. The diagram shows the interaction between a thread, the `Lock` struct, and the shared memory location where the lock value is stored.

Sources: [lock.cuh:27-36]()

### Multi-Lock Acquisition Sequence Diagram

```mermaid
sequenceDiagram
    participant Thread
    participant Lock
    participant SharedMemory

    Thread->>Lock: acquireMultipleLocks(lock)
    Loop Spin-lock
        Note right of Lock: Attempt to acquire all locks
        Lock->>SharedMemory: atomicExch(lock_array[i], 1)
        SharedMemory-->>Lock: Previous lock value

        Alt Any lock is already acquired
            Note right of Lock: Release previously acquired locks
            Lock->>SharedMemory: atomicExch(lock_array[j], 0)
            Note right of Lock: Retry acquiring all locks
        Else All locks are acquired
            Break
        End
    End

    Note over Thread,Lock: All locks are acquired

    Thread->>Lock: Critical section code

    Thread->>Lock: releaseMultipleLocks(lock)
    Lock->>SharedMemory: atomicExch(lock_array[i], 0)

    Note over Thread,Lock: All locks are released
```

This sequence diagram illustrates the process of acquiring and releasing multiple locks simultaneously using the `acquireMultipleLocks` and `releaseMultipleLocks` functions, respectively. The diagram shows the interaction between a thread, the `Lock` struct, and the shared memory array where the lock values are stored.

Sources: [lock.cuh:45-58]()

## Tables

### Lock Functions

| Function | Description |
| --- | --- |
| `__device__ Lock initLock(uint32_t *lock_addr)` | Initializes a `Lock` struct with a single lock value stored in shared memory. |
| `__device__ void acquireLock(Lock lock)` | Acquires the lock by performing an atomic operation on the lock value. |
| `__device__ void releaseLock(Lock lock)` | Releases the previously acquired lock by resetting the lock value. |
| `__device__ Lock initLocks(uint32_t *lock_array, uint32_t num_locks)` | Initializes a `Lock` struct with multiple locks stored in an array. |
| `__device__ void acquireMultipleLocks(Lock lock)` | Acquires multiple locks simultaneously using a spin-lock approach. |
| `__device__ void releaseMultipleLocks(Lock lock)` | Releases all previously acquired locks in the `lock_array`. |

Sources: [lock.cuh]()

## Source Citations

Throughout this wiki page, information has been derived from the following source files:

- [lock.cuh](): Cited for the `Lock` struct definition, function declarations, and inline function implementations.
- [lock.cu](): Cited for the function definitions and implementations related to lock management.
- [main.cu](): Cited for the usage examples of lock-based synchronization in parallel computations.
- [utils.cuh](): Cited for utility functions and macros used in the project.
- [utils.cu](): Cited for the implementation of utility functions used in the project.