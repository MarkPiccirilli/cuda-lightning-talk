#include <omp.h>
#include <cstdlib>
#include <iostream>
#include <math.h>

#ifndef ARRAYSIZE
#define ARRAYSIZE 1000000
#endif

//#ifndef THREADARRAY
//#define THREADARRAY (int[]){1, 2, 4, 6, 8, 12, 16}
//#endif

#ifndef BLOCKSIZE
#define BLOCKSIZE 256
#endif

#ifndef NUMBLOCKS
#define NUMBLOCKS 0
#endif

using std::cout;
using std::cerr;

__global__
void cudaMultiply(int *array1, int *array2, int *array3, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < size; i += stride)
        array3[i] = array1[i] * array2[i];
}

void printArray(int *array, int arraySize) {
    for(int i = 0; i < arraySize; i++) {
        cout << array[i] << " ";
    }
    cout << '\n';
}

int main(int argc, char **argv) {

    #ifndef _OPENMP
    cerr << "No OpenMP support";
    return 1;
    #endif

    int arraySize = ARRAYSIZE;
    int threadArray[] = {1, 2, 4, 6, 8, 12, 16};
    int threadArraySize = sizeof(threadArray)/sizeof(threadArray[0]);

    int* hostArray1 = new int[arraySize];
    int* hostArray2 = new int[arraySize];
    int* hostArray3 = new int[arraySize];

    for(int i = 0; i < arraySize; i++) {
        hostArray1[i] = rand() % 100;
        hostArray2[i] = rand() % 100;
    }

    for(int i = 0; i < threadArraySize; i++) {
        omp_set_num_threads(threadArray[i]);
        double timeStart = omp_get_wtime();

        #pragma omp parallel for
        for(int j = 0; j < arraySize; j++) {
            hostArray3[j] = hostArray1[j] * hostArray2[j];
        }

        double timeComplete = omp_get_wtime();
        double totalTime = timeComplete - timeStart;
        cout << "Time to complete operation with " << threadArray[i] << " threads(ms): " << totalTime * 1000 << '\n';
    }

    int *deviceArray1, *deviceArray2, *deviceArray3;
    cudaMallocManaged(&deviceArray1, arraySize * sizeof(int));
    cudaMallocManaged(&deviceArray2, arraySize * sizeof(int));
    cudaMallocManaged(&deviceArray3, arraySize * sizeof(int));

    cudaMemcpy(deviceArray1, hostArray1, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    //allocate CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int blockSize = BLOCKSIZE;
    int numBlocks = NUMBLOCKS ? (arraySize + blockSize - 1) / blockSize : NUMBLOCKS;

    cudaEventRecord(start, NULL);
    cudaMultiply<<<numBlocks, blockSize>>>(deviceArray1, deviceArray2, deviceArray3, arraySize);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);

//    cudaDeviceSynchronize();

    float totalTimeGPU = 0.0f;
    cudaEventElapsedTime(&totalTimeGPU, start, stop);
    cout << "Time to complete operation with the GPU using blocksize: " << blockSize << " and numBlocks: " << numBlocks << " was(ms): " << totalTimeGPU << '\n';

    // Free memory
    delete [] hostArray1;
    delete [] hostArray2;
    delete [] hostArray3;
    cudaFree(deviceArray1);
    cudaFree(deviceArray2);
    cudaFree(deviceArray3);

    return 0;
}
