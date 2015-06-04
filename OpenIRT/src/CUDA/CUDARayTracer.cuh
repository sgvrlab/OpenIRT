/********************************************************************
	created:	2011/08/03
	file path:	d:\Projects\Redering\OpenIRT\src\CUDA
	file base:	CUDARayTracer
	file ext:	cu
	author:		Tae-Joon Kim (tjkim.kaist@gmail.com)
	
	comment:	Ray tracer using CUDA
*********************************************************************/
#pragma once

// includes
#include <cutil_inline.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "CUDADataStructures.cuh"

//#include "cuPrintf.cu"
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

using namespace CUDA;

namespace CUDARayTracer
{
__constant__ Camera c_camera;
__constant__ Scene c_scene;
__constant__ Emitter c_emitterList[MAX_NUM_EMITTERS];
__constant__ int c_imgWidth;
__constant__ int c_imgHeight;

Scene h_scene;
Image h_image;
unsigned char *d_cacheMem[MAX_NUM_MODELS];
unsigned char *d_imageData;

texture<float4, cudaTextureType1D, cudaReadModeElementType> t_model0;
texture<float4, cudaTextureType1D, cudaReadModeElementType> t_model1;
texture<float4, cudaTextureType1D, cudaReadModeElementType> t_model2;
texture<float4, cudaTextureType1D, cudaReadModeElementType> t_model3;
texture<float4, cudaTextureType1D, cudaReadModeElementType> t_model4;
texture<float4, cudaTextureType1D, cudaReadModeElementType> t_model5;
texture<float4, cudaTextureType1D, cudaReadModeElementType> t_model6;
texture<float4, cudaTextureType1D, cudaReadModeElementType> t_model7;
texture<float4, cudaTextureType1D, cudaReadModeElementType> t_model8;
texture<float4, cudaTextureType1D, cudaReadModeElementType> t_model9;
texture<float4, cudaTextureType1D, cudaReadModeElementType> t_model10;
texture<float4, cudaTextureType1D, cudaReadModeElementType> t_model11;
texture<float4, cudaTextureType1D, cudaReadModeElementType> t_model12;
texture<float4, cudaTextureType1D, cudaReadModeElementType> t_model13;
texture<float4, cudaTextureType1D, cudaReadModeElementType> t_model14;
texture<float4, cudaTextureType1D, cudaReadModeElementType> t_model15;
texture<float4, cudaTextureType1D, cudaReadModeElementType> *t_models[MAX_NUM_MODELS];

__constant__ size_t c_offsetVerts[MAX_NUM_MODELS];
__constant__ size_t c_offsetTris[MAX_NUM_MODELS];
__constant__ size_t c_offsetNodes[MAX_NUM_MODELS];
__constant__ size_t c_offsetMats[MAX_NUM_MODELS];

__constant__ Matrix c_transfMatrix[MAX_NUM_MODELS];
__constant__ Matrix c_invTransfMatrix[MAX_NUM_MODELS];

#define VERT_FETCH(model, id, offset, ret) \
	switch(model) { \
	case 0 : (ret) = tex1Dfetch(t_model0, c_offsetVerts[model]+(id)*sizeof(Vertex)/16+(offset)); break; \
	case 1 : (ret) = tex1Dfetch(t_model1, c_offsetVerts[model]+(id)*sizeof(Vertex)/16+(offset)); break; \
	}
#define TRI_FETCH(model, id, offset, ret) \
	switch(model) { \
	case 0 : (ret) = tex1Dfetch(t_model0, c_offsetTris[model]+(id)*sizeof(Triangle)/16+(offset)); break; \
	case 1 : (ret) = tex1Dfetch(t_model1, c_offsetTris[model]+(id)*sizeof(Triangle)/16+(offset)); break; \
	}
#define NODE_FETCH(model, id, offset, ret) \
	switch(model) { \
	case 0 : (ret) = tex1Dfetch(t_model0, c_offsetNodes[model]+(id)*sizeof(BVHNode)/16+(offset)); break; \
	case 1 : (ret) = tex1Dfetch(t_model1, c_offsetNodes[model]+(id)*sizeof(BVHNode)/16+(offset)); break; \
	}
#define TEX_FETCH(model, id, offset, ret) \
	switch(model) { \
	case 0 : (ret) = tex1Dfetch(t_model0, c_offsetMats[model]+(id)*sizeof(Material)/16+(offset)); break; \
	case 1 : (ret) = tex1Dfetch(t_model1, c_offsetMats[model]+(id)*sizeof(Material)/16+(offset)); break; \
	}
#define SCENE_NODE_FETCH(id, offset, ret) (ret) = tex1Dfetch(t_model0, (id)*sizeof(SceneNode)/16+(offset));

__device__ bool RayTriIntersect(const Ray &ray, int model, unsigned int triID, HitPoint &hit, float tmax);
__device__ bool RayModelBVHIntersect(const Ray &oriRay, HitPoint &hit, int model);
__device__ bool RaySceneBVHIntersect(const Ray &ray, HitPoint &hit);
__device__ bool RaySceneIntersect(const Ray &ray, HitPoint &hit);
__device__ void trace(const Ray &ray, Vector3 &color);
}
