#include <omp.h>
#include <cstdlib>
#include <iostream>
#include <math.h>

#ifndef ARRAYSIZE
#define ARRAYSIZE 1000000
#endif

#ifndef BLOCKSIZE
#define BLOCKSIZE 256
#endif

#ifndef NUMBLOCKS
#define NUMBLOCKS NULL
#endif

#ifndef NUMTRIES
#define NUMTRIES 5
#endif

using std::cout;
using std::cerr;
using std:: endl;

__global__
void cudaMultiply(int *array1, int *array2, int *array3, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < size; i += stride)
        array3[i] = array1[i] * array2[i];
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

        double totalTime = 0;
        for(int j = 0; j < NUMTRIES; j++) {
            double timeStart = omp_get_wtime();
            #pragma omp parallel for
            for(int k = 0; k < arraySize; k++) {
                hostArray3[k] = hostArray1[k] * hostArray2[k];
            }
            double timeComplete = omp_get_wtime();
            totalTime = totalTime + timeComplete - timeStart;
        }
        double averageTime = totalTime / NUMTRIES;

        cout << "Time to complete operation with " << threadArray[i] << " threads(ms): " << averageTime * 1000 << '\n';
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
    int numBlocks = NUMBLOCKS ? NUMBLOCKS : (arraySize + blockSize - 1) / blockSize;

    cout << "Block Size: " << blockSize << endl;
    cout << "Number of Blocks: " << numBlocks << endl;

    double totalTimeGPU = 0.0;
    for(int i = 0; i < NUMTRIES; i++) {
        cudaEventRecord(start, NULL);
        cudaMultiply<<<numBlocks, blockSize>>>(deviceArray1, deviceArray2, deviceArray3, arraySize);
        cudaEventRecord(stop, NULL);
        cudaEventSynchronize(stop);

        float runTimeGPU = 0.0f;
        cudaEventElapsedTime(&runTimeGPU, start, stop);

        totalTimeGPU = totalTimeGPU + runTimeGPU;
    }

    double averageTimeGPU = totalTimeGPU / NUMTRIES;

    cout << "Time to complete operation with the GPU using blocksize: " << blockSize << " and numBlocks: " << numBlocks << " was(ms): " << averageTimeGPU << '\n';

    // Free memory
    delete [] hostArray1;
    delete [] hostArray2;
    delete [] hostArray3;
    cudaFree(deviceArray1);
    cudaFree(deviceArray2);
    cudaFree(deviceArray3);

    return 0;
}

