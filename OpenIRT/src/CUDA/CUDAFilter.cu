
#include "CUDACommonDefines.cuh"

using namespace CUDA;

namespace Filter
{

unsigned char *d_source;
unsigned char *d_target;
HitPoint *d_hitCache;
int *d_startXList;
int *d_startYList;
__constant__ int c_scrWidth;
__constant__ int c_scrHeight;
__constant__ int c_startX;
__constant__ int c_startY;
__constant__ int c_width;
__constant__ int c_height;
__constant__ int c_filterSize;

__global__ void
GBufferFilter(unsigned char *source, unsigned char *target, HitPoint *hitCache)
{
	const unsigned int lx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int ly = blockIdx.y * blockDim.y + threadIdx.y;
	int cx = lx + c_startX;
	int cy = ly + c_startY;

	float sumWeight = 0.0f;
	int gci = (cy*c_scrWidth)+cx;
	int lci = ly*c_width + lx;
	//float cdepth = hitCache[lci].t;

	float color[3];
	color[0] = color[1] = color[2] = 0.0f;

	int halfWindowSize = c_filterSize / 2;

	const float stdD2 = 1.0f;
	//const float stdDepth = 0.1f;
	const float stdNormal = 0.1f;

	for(int y=cy-halfWindowSize;y<=cy+halfWindowSize;y++)
	{
		for(int x=cx-halfWindowSize;x<=cx+halfWindowSize;x++)
		{
			if(x-c_startX < 0 || x-c_startX >= c_width || y-c_startY < 0 || y-c_startY >= c_height) continue;

			int gi = (y*c_scrWidth)+x;

			if(gi < 0 || gi >= c_scrWidth*c_scrHeight) continue;

			int li = (y-c_startY)*c_width + (x-c_startX);
			//if(li < 0 || li >= width*height) continue;

			// distance
			float dist = (float)((x - cx)*(x - cx) + (y - cy)*(y - cy));
			float weight = expf(-1.0f * dist / stdD2);

			/*
			// depth
			float depth = hitCache[li].t;
			dist = (depth - cdepth)*(depth - cdepth);
			weight *= expf((-1.0f * dist) / (2.0f * stdDepth * stdDepth));
			*/

			// normal
			dist = hitCache[li].n.e[0] - hitCache[lci].n.e[0];
			weight *= expf((-1.0f * dist * dist) / (2.0f * stdNormal * stdNormal));
			dist = hitCache[li].n.e[1] - hitCache[lci].n.e[1];
			weight *= expf((-1.0f * dist * dist) / (2.0f * stdNormal * stdNormal));
			dist = hitCache[li].n.e[2] - hitCache[lci].n.e[2];
			weight *= expf((-1.0f * dist * dist) / (2.0f * stdNormal * stdNormal));

			sumWeight += weight;
			color[0] += weight * source[gi*3+0];
			color[1] += weight * source[gi*3+1];
			color[2] += weight * source[gi*3+2];
		}
	}

	target[gci*3+0] = (unsigned char)(color[0] / sumWeight);
	target[gci*3+1] = (unsigned char)(color[1] / sumWeight);
	target[gci*3+2] = (unsigned char)(color[2] / sumWeight);
}

__global__ void
GBufferFilterMulti(unsigned char *source, unsigned char *target, HitPoint *hitCache, int *startX, int *startY)
{
	const unsigned int lx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int ly = blockIdx.y * blockDim.y + threadIdx.y;
	int cx = lx + startX[blockIdx.z];
	int cy = ly + startY[blockIdx.z];

	float sumWeight = 0.0f;
	int gci = (cy*c_scrWidth)+cx;
	float cdepth = hitCache[gci].t;

	float color[3];
	color[0] = color[1] = color[2] = 0.0f;

	int halfWindowSize = c_filterSize / 2;

	const float stdD2 = 1.0f;
	const float stdDepth = 0.1f;
	const float stdNormal = 0.1f;

	for(int y=cy-halfWindowSize;y<=cy+halfWindowSize;y++)
	{
		for(int x=cx-halfWindowSize;x<=cx+halfWindowSize;x++)
		{
			int gi = (y*c_scrWidth)+x;

			if(gi < 0 || gi >= c_scrWidth*c_scrHeight) continue;

			// distance
			float dist = (float)((x - cx)*(x - cx) + (y - cy)*(y - cy));
			float weight = expf(-1.0f * dist / stdD2);

			// depth
			float depth = hitCache[gi].t;
			dist = (depth - cdepth)*(depth - cdepth);
			weight *= expf((-1.0f * dist) / (2.0f * stdDepth * stdDepth));

			// normal
			dist = hitCache[gi].n.e[0] - hitCache[gci].n.e[0];
			weight *= expf((-1.0f * dist * dist) / (2.0f * stdNormal * stdNormal));
			dist = hitCache[gi].n.e[1] - hitCache[gci].n.e[1];
			weight *= expf((-1.0f * dist * dist) / (2.0f * stdNormal * stdNormal));
			dist = hitCache[gi].n.e[2] - hitCache[gci].n.e[2];
			weight *= expf((-1.0f * dist * dist) / (2.0f * stdNormal * stdNormal));

			sumWeight += weight;
			color[0] += weight * source[gi*3+0];
			color[1] += weight * source[gi*3+1];
			color[2] += weight * source[gi*3+2];
		}
	}

	target[gci*3+0] = (unsigned char)(color[0] / sumWeight);
	target[gci*3+1] = (unsigned char)(color[1] / sumWeight);
	target[gci*3+2] = (unsigned char)(color[2] / sumWeight);
}

extern "C" void gBufferFilter(Image *source, Image *target, int startX, int startY, int width, int height, int filterSize, HitPoint *hitCache)
{
	int device = 0;
	cudaSetDevice(device);

	checkCudaErrors(cudaMalloc((void **)&d_source, source->width*source->height*source->bpp));
	checkCudaErrors(cudaMalloc((void **)&d_target, source->width*source->height*source->bpp));
	checkCudaErrors(cudaMalloc((void **)&d_hitCache, width*height*sizeof(HitPoint)));
	checkCudaErrors(cudaMemcpy(d_source, source->data, source->width*source->height*source->bpp, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_hitCache, hitCache, width*height*sizeof(HitPoint), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpyToSymbol(c_scrWidth, &source->width, sizeof(int), 0, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(c_scrHeight, &source->height, sizeof(int), 0, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(c_startX, &startX, sizeof(int), 0, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(c_startY, &startY, sizeof(int), 0, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(c_width, &width, sizeof(int), 0, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(c_height, &height, sizeof(int), 0, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(c_filterSize, &filterSize, sizeof(int), 0, cudaMemcpyHostToDevice));

	int tileWidth = 16;
	int tileHeight = 16;

	dim3 dimBlock(tileWidth, tileHeight);
	dim3 dimGrid(width / tileWidth, height / tileHeight);

	GBufferFilter<<<dimGrid, dimBlock>>>(d_source, d_target, d_hitCache);

	//checkCudaErrors(cudaMemcpy(target->data, d_target, source->width*source->height*source->bpp, cudaMemcpyDeviceToHost));
	int offset = (startY*source->width+startX)*source->bpp;
	for(int y=startY;y<startY+height;y++)
	{
		checkCudaErrors(cudaMemcpy( ((unsigned char*)target->data)+offset, ((unsigned char*)d_target)+offset, width*source->bpp, cudaMemcpyDeviceToHost));
		offset += (source->width)*source->bpp;
	}

	checkCudaErrors(cudaFree(d_source));
	checkCudaErrors(cudaFree(d_target));
	checkCudaErrors(cudaFree(d_hitCache));
}

Image h_image;
extern "C" void gBufferFilterMulti(Image *source, Image *target, int count, int *startX, int *startY, int width, int height, int filterSize, HitPoint *hitCache)
{
	int device = 0;
	cudaSetDevice(device);

	if(source->width != h_image.width || source->height != h_image.height || source->bpp != h_image.bpp)
	{
		checkCudaErrors(cudaFree(d_source));
		checkCudaErrors(cudaFree(d_target));
		checkCudaErrors(cudaFree(d_hitCache));

		h_image = *source;

		checkCudaErrors(cudaMalloc((void **)&d_source, source->width*source->height*source->bpp));
		checkCudaErrors(cudaMalloc((void **)&d_target, source->width*source->height*source->bpp));
		checkCudaErrors(cudaMalloc((void **)&d_hitCache, source->width*source->height*sizeof(HitPoint)));

		checkCudaErrors(cudaMemcpyToSymbol(c_scrWidth, &source->width, sizeof(int), 0, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpyToSymbol(c_scrHeight, &source->height, sizeof(int), 0, cudaMemcpyHostToDevice));
	}

	for(int i=0;i<count;i++)
	{
		int offset = startY[i]*source->width+startX[i];
		for(int y=startY[i];y<startY[i]+height;y++)
		{
			int size;
			size = source->bpp;
			checkCudaErrors(cudaMemcpy(((unsigned char*)d_source)+offset*size, ((unsigned char*)target->data)+offset*size, width*size, cudaMemcpyHostToDevice));
			size = sizeof(HitPoint);
			checkCudaErrors(cudaMemcpy(((unsigned char*)d_hitCache)+offset*size, ((unsigned char*)hitCache)+offset*size, width*size, cudaMemcpyHostToDevice));
			offset += source->width;
		}
	}
	checkCudaErrors(cudaMalloc((void **)&d_startXList, count*sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&d_startYList, count*sizeof(int)));
	checkCudaErrors(cudaMemcpy(d_startXList, startX, count*sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_startYList, startY, count*sizeof(int), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpyToSymbol(c_width, &width, sizeof(int), 0, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(c_height, &height, sizeof(int), 0, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(c_filterSize, &filterSize, sizeof(int), 0, cudaMemcpyHostToDevice));

	int tileWidth = 16;
	int tileHeight = 16;

	dim3 dimBlock(tileWidth, tileHeight);
	dim3 dimGrid(width/tileWidth, height/tileHeight, count);

	GBufferFilterMulti<<<dimGrid, dimBlock>>>(d_source, d_target, d_hitCache, d_startXList, d_startYList);

	for(int i=0;i<count;i++)
	{
		int offset = startY[i]*source->width+startX[i];
		for(int y=startY[i];y<startY[i]+height;y++)
		{
			int size;
			size = source->bpp;
			checkCudaErrors(cudaMemcpy(((unsigned char*)target->data)+offset*size, ((unsigned char*)d_target)+offset*size, width*size, cudaMemcpyDeviceToHost));
			offset += source->width;
		}
	}

	checkCudaErrors(cudaFree(d_startXList));
	checkCudaErrors(cudaFree(d_startYList));
}

}