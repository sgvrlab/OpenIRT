#ifndef COMMON_GPU_H
#define COMMON_GPU_H

#include <stdio.h>
#include <stdlib.h>
#include <io.h>
#include <cuda.h>
//#include <cuda_runtime_api.h>
#include <cutil.h>
//#include <cutil_gl_error.h>

#define CUDA_INLINE inline

//#define USE_HCCMESH

#include "Vector_GPU.cuh"
#include "DataType_GPU.cuh"

#define AXIS(node) ((node)->left & 3)
#define ISLEAF(node) (((node)->left & 3) == 3)
#define GETLEFTCHILD(node) ((node)->left >> 2)
#define GETRIGHTCHILD(node) ((node)->right >> 2)
#define GETIDXOFFSET(node) ((node)->right)
#define GETCHILDCOUNT(node) ((node)->left >> 2)
#define BSP_EPSILON 0.001f
#define INTERSECT_EPSILON 0.01f
#define TRI_INTERSECT_EPSILON 0.0001f
#define FLT_MAX 3.402823466e+38F

#define MAX_NUM_GPU 4

#define MASK_USE_SHADOW_RAYS	0x1
#define MASK_USE_REFLECTION		0x2
#define MASK_USE_REFRACTION		0x4
#define MASK_USE_FAKE_BASEPLANE 0x8
#define MASK_USE_VERTEX_NORMAL	0x10
#define MASK_USE_VERTEX_TEXTURE	0x20

#include "cuPrintf.cu"
//The macro CUPRINTF is defined for architectures
//with different compute capabilities.
#if __CUDA_ARCH__ < 200 	//Compute capability 1.x architectures
#define CUPRINTF cuPrintf 
#else						//Compute capability 2.x architectures
#define CUPRINTF(fmt, ...) printf("[%d, %d]:\t" fmt, \
								blockIdx.y*gridDim.x+blockIdx.x,\
								threadIdx.z*blockDim.x*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x,\
								__VA_ARGS__)
#endif

#endif // #ifndef COMMON_GPU_H
