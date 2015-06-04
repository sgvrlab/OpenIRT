/********************************************************************
	created:	2011/08/03
	file path:	d:\Projects\Redering\OpenIRT\src\CUDA
	file base:	CUDACommonDefines
	file ext:	cuh
	author:		Tae-Joon Kim (tjkim.kaist@gmail.com)
	
	comment:	Common defines for CUDA
*********************************************************************/

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "CUDADataStructures.cuh"

/*
#include "cuPrintf.cu"
//The macro CUPRINTF is defined for architectures
//with different compute capabilities.
#ifndef CUPRINTF
#if __CUDA_ARCH__ < 200 	//Compute capability 1.x architectures
#define CUPRINTF cuPrintf 
#else						//Compute capability 2.x architectures
#define CUPRINTF(fmt, ...) printf("[%d, %d]:\t" fmt, \
								blockIdx.y*gridDim.x+blockIdx.x,\
								threadIdx.z*blockDim.x*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x,\
								__VA_ARGS__)
#endif
#endif
*/

#ifndef checkCudaErrors
#define checkCudaErrors(err)           __checkCudaErrors (err, __FILE__, __LINE__)
inline void __checkCudaErrors(cudaError err, const char *file, const int line)
{
    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
                file, line, (int)err, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
#endif

#define TIMER_START \
     cudaEvent_t start, stop ;\
     cudaEventCreate(&start) ; cudaEventCreate(&stop) ; cudaEventRecord(start, 0) ;

#define TIMER_STOP(elapsedTime) \
     cudaEventRecord(stop,0) ; cudaEventSynchronize(stop) ; \
     cudaEventElapsedTime(&elapsedTime, start, stop) ; \
     cudaEventDestroy(start) ; cudaEventDestroy(stop) ;