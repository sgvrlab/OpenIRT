#include "CUDACommonDefines.cuh"

__device__ unsigned int boxFilter(SampleData *sampleData, int cx, int cy, int width, int height, int filterSize, float *kernel)
{
	float sumWeight = 0.0f;

	Vector3 color(0.0f);

	int halfWindowSize = filterSize / 2;

	int localOffset = 0;
	for(int y=cy-halfWindowSize;y<=cy+halfWindowSize;y++)
	{
		for(int x=cx-halfWindowSize;x<=cx+halfWindowSize;x++)
		{
			if(x < 0 || x >= width || y < 0 || y >= height) continue;

			int offset = (y*width)+x;

			Vector3 neighbor = sampleData[offset].getFinalColor();
			
			float weight = kernel[localOffset];
			sumWeight += weight; 
			color += neighbor * weight * 255.0f;
			localOffset++;
		}
	}
	unsigned int r = (unsigned int)fminf(color.e[0] / sumWeight, 255.0f);
	unsigned int g = (unsigned int)fminf(color.e[1] / sumWeight, 255.0f);
	unsigned int b = (unsigned int)fminf(color.e[2] / sumWeight, 255.0f);

	return (r << 16) | (g << 8) | b;
}

__device__ unsigned int gBufferFilter(SampleData *sampleData, int cx, int cy, int width, int height, int filterSize, const Vector3 &sceneSize, float stdD2, float stdDepth, float stdNormal)
{
	float sumWeight = 0.0f;
	int coffset = (cy*width)+cx;

	Vector3 color(0.0f);

	int halfWindowSize = filterSize / 2;

	Vector3 cHitPoint = sampleData[coffset].summedHitPoint / sampleData[coffset].numIntergration;
	Vector3 cHitNormal = sampleData[coffset].summedHitNormal / sampleData[coffset].numIntergration;

	Vector3 invalid(0.0f);

	if(!sampleData[coffset].hasHit)
	{
		color = sampleData[coffset].getFinalColor() * 255.0f;
		sumWeight = 1.0f;
	}
	else
	{
		for(int y=cy-halfWindowSize;y<=cy+halfWindowSize;y++)
		{
			for(int x=cx-halfWindowSize;x<=cx+halfWindowSize;x++)
			{
				if(x < 0 || x >= width || y < 0 || y >= height) continue;

				int offset = (y*width)+x;

				Vector3 neighbor = sampleData[offset].getFinalColor();

				// distance
				float dist = (float)((x - cx)*(x - cx) + (y - cy)*(y - cy));
				float weight = expf(-1.0f * dist / stdD2);

				// depth
				Vector3 hitPoint = sampleData[offset].numIntergration > 0 ? sampleData[offset].summedHitPoint / sampleData[offset].numIntergration : invalid;
				dist = ((hitPoint - cHitPoint)/sceneSize).magsqr();
				weight *= expf((-1.0f * dist) / (2.0f * stdDepth * stdDepth));

				// normal
				Vector3 hitNormal = sampleData[offset].numIntergration > 0 ? sampleData[offset].summedHitNormal / sampleData[offset].numIntergration : invalid;
				dist = hitNormal.e[0] - cHitNormal.e[0];
				weight *= expf((-1.0f * dist * dist) / (2.0f * stdNormal * stdNormal));
				dist = hitNormal.e[1] - cHitNormal.e[1];
				weight *= expf((-1.0f * dist * dist) / (2.0f * stdNormal * stdNormal));
				dist = hitNormal.e[2] - cHitNormal.e[2];
				weight *= expf((-1.0f * dist * dist) / (2.0f * stdNormal * stdNormal));

				sumWeight += weight;
				color += neighbor * weight * 255.0f;
			}
		}
	}
	unsigned int r = (unsigned int)fminf(color.e[0] / sumWeight, 255.0f);
	unsigned int g = (unsigned int)fminf(color.e[1] / sumWeight, 255.0f);
	unsigned int b = (unsigned int)fminf(color.e[2] / sumWeight, 255.0f);

	return (r << 16) | (g << 8) | b;
}

__device__ unsigned int gBufferFilter2(SampleData *sampleData, int cx, int cy, int xo, int yo, int width, int height, int filterSize, const Vector3 &sceneSize, float stdD2, float stdDepth, float stdNormal)
{
	float sumWeight = 0.0f;
	int coffset = (cy*width)+cx;

	int halfWindowSize = filterSize / 2;

	Vector3 cHitPoint = sampleData[coffset].summedHitPoint / sampleData[coffset].numIntergration;
	Vector3 cHitNormal = sampleData[coffset].summedHitNormal / sampleData[coffset].numIntergration;

	Vector3 invalid(0.0f);

	Vector3 color = sampleData[coffset].getFinalColor(1) * 255.0f;

	if(sampleData[coffset].hasHit)
	{
		Vector3 color2(0.0f);
		for(int y=cy-halfWindowSize;y<=cy+halfWindowSize;y++)
		{
			for(int x=cx-halfWindowSize;x<=cx+halfWindowSize;x++)
			{
				if(x < 0 || x >= width || y < 0 || y >= height) continue;

				int offset = (y*width)+x;

				/*
				int offseto = (yo*width)+xo;

				Vector3 neighbor = sampleData[offseto].getFinalColor(2);
				*/
				int x0 = xo;
				int y0 = yo;
				int x1 = xo+TILE_SIZE;
				int y1 = yo+TILE_SIZE;
				if(x1 >= width) x1 = x0;
				if(y1 >= height) y1 = y0;

				Vector3 c00 = sampleData[(y0*width)+x0].getFinalColor(2);
				Vector3 c01 = sampleData[(y1*width)+x0].getFinalColor(2);
				Vector3 c10 = sampleData[(y0*width)+x1].getFinalColor(2);
				Vector3 c11 = sampleData[(y1*width)+x1].getFinalColor(2);
				float subPixelPosX = (x % TILE_SIZE)/((float)(TILE_SIZE));
				float subPixelPosY = (y % TILE_SIZE)/((float)(TILE_SIZE));
				float w00 = (1.0f - subPixelPosX) * (1.0f - subPixelPosY);
				float w01 = (1.0f - subPixelPosX) * subPixelPosY;
				float w10 = subPixelPosX * (1.0f - subPixelPosY);
				float w11 = subPixelPosX * subPixelPosY;
				float norm = 1.0f / (w00 + w01 + w10 + w11);
				Vector3 neighbor = (c00 * w00 + c01 * w01 + c10 * w10 + c11 * w11)*norm;

				// distance
				float dist = (float)((x - cx)*(x - cx) + (y - cy)*(y - cy));
				float weight = expf(-1.0f * dist / stdD2);

				// depth
				Vector3 hitPoint = sampleData[offset].numIntergration > 0 ? sampleData[offset].summedHitPoint / sampleData[offset].numIntergration : invalid;
				dist = ((hitPoint - cHitPoint)/sceneSize).magsqr()*10000.0f;
				weight *= expf((-1.0f * dist) / (2.0f * stdDepth * stdDepth));

				// normal
				Vector3 hitNormal = sampleData[offset].numIntergration > 0 ? sampleData[offset].summedHitNormal / sampleData[offset].numIntergration : invalid;
				dist = hitNormal.e[0] - cHitNormal.e[0];
				weight *= expf((-1.0f * dist * dist) / (2.0f * stdNormal * stdNormal));
				dist = hitNormal.e[1] - cHitNormal.e[1];
				weight *= expf((-1.0f * dist * dist) / (2.0f * stdNormal * stdNormal));
				dist = hitNormal.e[2] - cHitNormal.e[2];
				weight *= expf((-1.0f * dist * dist) / (2.0f * stdNormal * stdNormal));

				sumWeight += weight;
				color2 += neighbor * weight * 255.0f;
			}
		}

		color.e[0] += color2.e[0] / sumWeight;
		color.e[1] += color2.e[1] / sumWeight;
		color.e[2] += color2.e[2] / sumWeight;
	}
	unsigned int r = (unsigned int)fminf(color.e[0], 255.0f);
	unsigned int g = (unsigned int)fminf(color.e[1], 255.0f);
	unsigned int b = (unsigned int)fminf(color.e[2], 255.0f);

	return (r << 16) | (g << 8) | b;
}

__device__ unsigned int noneFilter3(float *color, float *depth, float *normal, int cx, int cy, int width, int height, int bpp, int filterSize, float stdD2, float stdDepth, float stdNormal)
{
	int cOffset = ((cy * width) + cx) * bpp;

	float4 outColor = make_float4(color[cOffset+0], color[cOffset+1], color[cOffset+2], 1.0f);
	if(bpp == 4) outColor.w = color[cOffset+3];

	color[cOffset+0] = outColor.x;
	color[cOffset+1] = outColor.y;
	color[cOffset+2] = outColor.z;
	if(bpp == 4) color[cOffset+3] = outColor.w;

	unsigned int r = (unsigned int)fminf(outColor.x * 255.0f, 255.0f);
	unsigned int g = (unsigned int)fminf(outColor.y * 255.0f, 255.0f);
	unsigned int b = (unsigned int)fminf(outColor.z * 255.0f, 255.0f);
	unsigned int a = (unsigned int)fminf(outColor.w * 255.0f, 255.0f);

	return (r << 24) | (g << 16) | (b << 8) | a;
}

// prefix 'c' means center, 'n' means neighbor
__device__ unsigned int gBufferFilter3(float *color, float *depth, float *normal, int cx, int cy, int width, int height, int bpp, int filterSize, float stdD2, float stdDepth, float stdNormal)
{
	float sumWeight = 0.0f;
	int cOffset = ((cy * width) + cx) * bpp;

	float4 outColor = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

	int halfWindowSize = filterSize / 2;

	float4 cColor = make_float4(color[cOffset+0], color[cOffset+1], color[cOffset+2], 1.0f);
	if(bpp == 4) cColor.w = color[cOffset+3];

	Vector3 cDepth(depth[cOffset+0], depth[cOffset+1], depth[cOffset+2]);
	Vector3 cNormal(normal[cOffset+0], normal[cOffset+1], normal[cOffset+2]);

	Vector3 invalid(0.0f);

	if(cNormal.x == 0.0f && cNormal.y == 0.0f && cNormal.z == 0.0f)
	{
		outColor = cColor;
		sumWeight = 1.0f;
	}
	else
	{
		for(int y=cy-halfWindowSize;y<=cy+halfWindowSize;y++)
		{
			for(int x=cx-halfWindowSize;x<=cx+halfWindowSize;x++)
			{
				if(x < 0 || x >= width || y < 0 || y >= height) continue;

				int nOffset = ((y * width) + x) * bpp;

				float4 nColor = make_float4(color[nOffset+0], color[nOffset+1], color[nOffset+2], 1.0f);
				if(bpp == 4) nColor.w = color[nOffset+3];

				// distance
				float dist = (float)((x - cx)*(x - cx) + (y - cy)*(y - cy));
				float weight = expf(-1.0f * dist / stdD2);

				// depth
				Vector3 nDepth(depth[nOffset+0], depth[nOffset+1], depth[nOffset+2]);
				dist = (nDepth - cDepth).magsqr();
				weight *= expf((-1.0f * dist) / (2.0f * stdDepth * stdDepth));

				// normal
				Vector3 nNormal(normal[nOffset+0], normal[nOffset+1], normal[nOffset+2]);
				dist = nNormal.e[0] - cNormal.e[0];
				weight *= expf((-1.0f * dist * dist) / (2.0f * stdNormal * stdNormal));
				dist = nNormal.e[1] - cNormal.e[1];
				weight *= expf((-1.0f * dist * dist) / (2.0f * stdNormal * stdNormal));
				dist = nNormal.e[2] - cNormal.e[2];
				weight *= expf((-1.0f * dist * dist) / (2.0f * stdNormal * stdNormal));

				sumWeight += weight;
				outColor += nColor * weight;
			}
		}
	}

	color[cOffset+0] = outColor.x / sumWeight;
	color[cOffset+1] = outColor.y / sumWeight;
	color[cOffset+2] = outColor.z / sumWeight;
	if(bpp == 4) color[cOffset+3] = outColor.w / sumWeight;

	unsigned int r = (unsigned int)fminf(outColor.x / sumWeight * 255.0f, 255.0f);
	unsigned int g = (unsigned int)fminf(outColor.y / sumWeight * 255.0f, 255.0f);
	unsigned int b = (unsigned int)fminf(outColor.z / sumWeight * 255.0f, 255.0f);
	unsigned int a = (unsigned int)fminf(outColor.w / sumWeight * 255.0f, 255.0f);

	return (r << 24) | (g << 16) | (b << 8) | a;

}