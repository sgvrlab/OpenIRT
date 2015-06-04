#include "GBufferFilter.h"
#include <stdio.h>
#include <stdarg.h>
#include "Vector3.h"

#ifndef fminf
#define fminf(a,b) (((a) < (b)) ? (a) : (b))
#endif

#ifndef fmaxf
#define fmaxf(a,b) (((a) > (b)) ? (a) : (b))
#endif

#include "CUDA/CUDADataStructures.cuh"

extern "C" void gBufferFilter(CUDA::Image *source, CUDA::Image *target, int startX, int startY, int width, int height, int filterSize, CUDA::HitPoint *hitCache);
extern "C" void gBufferFilterMulti(CUDA::Image *source, CUDA::Image *target, int count, int *startX, int *startY, int width, int height, int filterSize, CUDA::HitPoint *hitCache);
extern "C" void gBufferFilterMulti1(CUDA::Image *source, CUDA::Image *target, int count, int *startX, int *startY, int width, int height, int filterSize, CUDA::HitPoint *hitCache);
extern "C" void gBufferFilterMulti2(CUDA::Image *source, CUDA::Image *target, int count, int *startX, int *startY, int width, int height, int filterSize, CUDA::HitPoint *hitCache);
extern "C" void gBufferFilterMulti3(CUDA::Image *source, CUDA::Image *target, int count, int *startX, int *startY, int width, int height, int filterSize, CUDA::HitPoint *hitCache);

#define USE_CUDA

using namespace irt;

void GBufferFilter::filter(Image *source, Image *target, int count, int *startX, int *startY, int width, int height, int filterSize, ...)
{
	va_list vaList;
	va_start(vaList, filterSize);
	CUDA::HitPoint *hit = va_arg(vaList, CUDA::HitPoint*);
	va_end(vaList);

	gBufferFilterMulti((CUDA::Image*)&source->width, (CUDA::Image*)target, count, startX, startY, width, height, filterSize, hit);
	/*
	DWORD tic1 = GetTickCount();
	gBufferFilterMulti1((CUDA::Image*)&source->width, (CUDA::Image*)target, count, startX, startY, width, height, filterSize, hit);
	DWORD tic2 = GetTickCount();
	gBufferFilterMulti2((CUDA::Image*)&source->width, (CUDA::Image*)target, count, startX, startY, width, height, filterSize, hit);
	DWORD tic3 = GetTickCount();
	gBufferFilterMulti3((CUDA::Image*)&source->width, (CUDA::Image*)target, count, startX, startY, width, height, filterSize, hit);
	DWORD tic4 = GetTickCount();
	printf("%dms %dms %dms\n", tic2-tic1, tic3-tic2, tic4-tic3);
	*/
}

void GBufferFilter::filter(Image *source, Image *target, int startX, int startY, int width, int height, int filterSize, ...)
{
	va_list vaList;
	va_start(vaList, filterSize);
	CUDA::HitPoint *hit = va_arg(vaList, CUDA::HitPoint*);
	va_end(vaList);

#	ifdef USE_CUDA
	gBufferFilter((CUDA::Image*)&source->width, (CUDA::Image*)target, startX, startY, width, height, filterSize, hit);
#	else

	int halfWindowSize = filterSize / 2;

	int gci, gi, lci, li;	// gci : global index of center, gi : glocal index of sample point, lci : local index of center, local index of sample point

	float dist;
	float cdepth, depth;
	float weight, sumWeight;

	float stdD2 = 50.0f;
	float stdDepth = 0.1f;
	float stdNormal = 0.1f;

	float color[3];

	for(int cy=startY;cy<startY+height;cy++) 
	{
		for(int cx=startX;cx<startX+width;cx++) 
		{
			sumWeight = 0.0f;
			gci = (cy*source->width)+cx;
			lci = (cy-startY)*width + (cx-startX);
			cdepth = hit[lci].t;

			color[0] = color[1] = color[2] = 0.0f;

			for(int y=cy-halfWindowSize;y<=cy+halfWindowSize;y++)
			{
				for(int x=cx-halfWindowSize;x<=cx+halfWindowSize;x++)
				{
					if(x-startX < 0 || x-startX >= width || y-startY < 0 || y-startY >= height) continue;

					gi = (y*source->width)+x;

					if(gi < 0 || gi >= source->width*source->height) continue;

					li = (y-startY)*width + (x-startX);
					//if(li < 0 || li >= width*height) continue;

					// distance
					dist = (float)((x - cx)*(x - cx) + (y - cy)*(y - cy));
					weight = expf(-1.0f * dist / stdD2);

					// depth
					depth = hit[li].t;
					dist = (depth - cdepth)*(depth - cdepth);
					weight *= expf((-1.0f * dist) / (2.0f * stdDepth * stdDepth));

					// normal
					dist = hit[li].n.e[0] - hit[lci].n.e[0];
					weight *= expf((-1.0f * dist * dist) / (2.0f * stdNormal * stdNormal));
					dist = hit[li].n.e[1] - hit[lci].n.e[1];
					weight *= expf((-1.0f * dist * dist) / (2.0f * stdNormal * stdNormal));
					dist = hit[li].n.e[2] - hit[lci].n.e[2];
					weight *= expf((-1.0f * dist * dist) / (2.0f * stdNormal * stdNormal));

					sumWeight += weight;
					color[0] += weight * source->data[gi*3+0];
					color[1] += weight * source->data[gi*3+1];
					color[2] += weight * source->data[gi*3+2];
				}
			}

			target->data[gci*3+0] = (unsigned char)(color[0] / sumWeight);
			target->data[gci*3+1] = (unsigned char)(color[1] / sumWeight);
			target->data[gci*3+2] = (unsigned char)(color[2] / sumWeight);
		}
	}
#	endif
}

void GBufferFilter::simpleFilter(Image *source, Image *target, int startX, int startY, int width, int height, int filterSize, ...)
{
	va_list vaList;
	va_start(vaList, filterSize);
	CUDA::HitPoint *hit = va_arg(vaList, CUDA::HitPoint*);
	float depthRange = *(va_arg(vaList, float*));
	va_end(vaList);

	static float kernel3[] = {1, 1, 1, 1, 2, 1, 1, 1, 1};
	static float kernel5[] = {
		 1,  4,  7,  4,  1,
		 4, 16, 26, 16,  4,
		 7, 26, 41, 26,  7,
		 4, 16, 26, 16,  4,
		 1,  4,  7,  4,  1};
	static const float nLimit = 0.8f;
	static const float dLimit = 1000.0f;
	float denominator = 0.0f;
	float red, green, blue;
	int ired, igreen, iblue, indexOffset;

	int indices3[] = {
		-(source->width + 1),  -source->width,     -(source->width - 1), 
		-1,                0,           +1, 
		source->width - 1,      source->width,      source->width + 1
	};
	int indices5[] = {
		-(source->width*2 + 2), -(source->width*2 + 1), -source->width*2, -(source->width*2 - 1), -(source->width*2 - 2),
		-(source->width + 2), -(source->width + 1), -source->width, -(source->width - 1), -(source->width - 2),
		-2, -1, 0, +1, +2,
		source->width - 2, source->width - 1, source->width, source->width + 1, source->width + 2,
		source->width*2 - 2, source->width*2 - 1, source->width*2, source->width + 1, source->width*2 + 2
	};

	float *kernel = filterSize == 3 ? kernel3 : kernel5;
	int *indices = filterSize == 3 ? indices3 : indices5;

	int filterArea = filterSize*filterSize;

	float *newKernel = new float[filterArea];

	for(int i=0;i<filterArea;i++)
		denominator += kernel[i];
	if (denominator==0.0f) denominator = 1.0f;
	for(int i=startY;i<startY+height;i++) 
	{
		for(int j=startX;j<startX+width;j++) 
		{
			red = green = blue = 0.0f;
			indexOffset = (i*source->width)+j;

			if(hit[indexOffset].n.e[0] == 0.0f && hit[indexOffset].n.e[1] == 0.0f && hit[indexOffset].n.e[2] == 0.0f)
			{
				ired = source->data[indexOffset*3];
				igreen = source->data[indexOffset*3+1];
				iblue = source->data[indexOffset*3+2];
			}
			else
			{
				denominator = 0.0f;

				Vector3 norm1, norm2;
				Vector3 pos1, pos2;

				for(int k=0;k<filterArea;k++)
				{
					int cIndex = indexOffset;
					int nIndex = indexOffset+indices[k];
					if(nIndex < 0 || nIndex >= source->width*source->height) continue;

					norm1 = *((Vector3*)&hit[nIndex].n);
					norm2 = *((Vector3*)&hit[cIndex].n);
					pos1 = *((Vector3*)&hit[nIndex].x);
					pos2 = *((Vector3*)&hit[cIndex].x);
					//newKernel[k] = (dot(norm1, norm2) > nLimit) && (fabs(hit[nIndex].t - hit[cIndex].t)*dLimit < depthRange) ? kernel[k] : 0.0f;
					//newKernel[k] = (dot(norm1, norm2) > nLimit) && ((pos1-pos2).length()*dLimit < depthRange) ? kernel[k] : 0.0f;
					newKernel[k] = dot(norm1, norm2) > nLimit ? kernel[k] : 0.0f;
					//newKernel[k] = (pos1-pos2).length()*dLimit < depthRange ? kernel[k] : 0.0f;

					denominator += newKernel[k];
				}
				if (denominator==0.0f) denominator = 1.0f;

				for(int k=0;k<filterArea;k++) 
				{
					if(indexOffset+indices[k] < 0 || indexOffset+indices[k] >= source->width*source->height) continue;

					red += source->data[(indexOffset+indices[k])*3]*newKernel[k];
					green += source->data[(indexOffset+indices[k])*3+1]*newKernel[k];
					blue += source->data[(indexOffset+indices[k])*3+2]*newKernel[k];
				} 
				ired = (int)(red / denominator);
				igreen = (int)(green / denominator);
				iblue = (int)(blue / denominator);
				if (ired>0xff) ired = 0xff;
				else if (ired<0) ired = 0;
				if (igreen>0xff) igreen = 0xff;
				else if (igreen<0) igreen = 0;
				if (iblue>0xff) iblue = 0xff;
				else if (iblue<0) iblue = 0;
			}
			target->data[indexOffset*3] = ired;
			target->data[indexOffset*3+1] = igreen;
			target->data[indexOffset*3+2] = iblue;
		}
	}

	delete[] newKernel;
}