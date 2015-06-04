/********************************************************************
	created:	2011/08/03
	file path:	d:\Projects\Redering\OpenIRT\src\CUDA
	file base:	CUDAPathTracer
	file ext:	cu
	author:		Tae-Joon Kim (tjkim.kaist@gmail.com)
	
	comment:	Path tracer using CUDA
*********************************************************************/

#include "CUDACommonDefines.cuh"

using namespace CUDA;

namespace CUDAPathTracer
{
	
#include "CUDARayTracerCommon.cuh"
#include "CUDAFilter.cuh"
	
__device__ __inline__ bool
trace(const Ray &ray, float4 &color, HitPoint &hit, Material &hitMat, float4 &attenuation, unsigned int prevRnd, int depth)
{
	hit.t = FLT_MAX;
	
	if(!RaySceneBVHIntersect(ray, hit, hitMat, 0.0f, prevRnd))
	{
		float4 envMapColor = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
		if(c_hasEnvMap)
		{
			shadeEnvironmentMap(ray.dir, envMapColor);
		}
		
		if(depth == 0)
		{
			if(c_hasEnvMap) color = envMapColor;
			else color = make_float4(c_envCol.x, c_envCol.y, c_envCol.z, 1.0f);

			if(!c_controller.drawBackground) color.w = 0.0f;
		}
		else
		{
			color.x += envMapColor.x * c_controller.envMapWeight + c_envCol.x * c_controller.envColWeight;
			color.y += envMapColor.y * c_controller.envMapWeight + c_envCol.y * c_controller.envColWeight;
			color.z += envMapColor.z * c_controller.envMapWeight + c_envCol.z * c_controller.envColWeight;
			color = color * attenuation;
		}

		return false;
	}

	//if(depth == 1)
	//{ 
	//	color = Vector3(0.0f);
	//	return false;
	//}
	
	//getMaterial(hitMat, hit);
	
	for(int i=0;i<c_scene.numEmitters;i++)
	{
		const Emitter &emitter = c_emitterList[i];
		
		Vector3 hitPoint = ray.ori + hit.t*ray.dir;
		Vector3 shadowDir;
		Vector3 samplePos = emitter.sample(prevRnd);

		shadowDir = samplePos - hitPoint;	
		shadowDir = shadowDir.normalize();
		
		const Vector3 &lightAmbient = emitter.color_Ka;
		const Vector3 &lightDiffuse = emitter.color_Kd;

		float cosFac = shadowDir.dot(hit.n);

		if(cosFac > 0.0f)
		{
			// cast shadow ray
			Ray shadowRay;
			int idx = shadowDir.indexOfMaxComponent();
			float tLimit = (samplePos.e[idx] - hitPoint.e[idx]) / shadowDir.e[idx];

			HitPoint shadowHit;
			Material tempMat;
			shadowHit.t = tLimit;

			shadowRay.set(hitPoint, shadowDir);

			if(!RaySceneBVHIntersect(shadowRay, shadowHit, tempMat, tLimit, prevRnd))
			{
				color += (lightAmbient * hitMat.mat_Ka + lightDiffuse * hitMat.mat_Kd * cosFac) * attenuation;
			}
		}
	}
	return true;
}

__global__ void
Render(unsigned char *image, float *summedImage, float *summedImageHit, float *summedImageNormal, int frame, int globalSeed)
{
	Ray ray;
	float4 outColor;
	
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = (y*c_imgWidth + x)*c_imgBpp;

	unsigned int seed = tea<16>((y*c_imgWidth + x), globalSeed);

	c_camera.getRayWithOrigin(ray, 
		(x + rnd(seed))/c_imgWidth, 
		(y + rnd(seed))/c_imgHeight);

	outColor.x = outColor.y = outColor.z = 0.0f;
	outColor.w = 1.0f;
	if(frame == 1)
	{
		summedImage[offset + 2] = summedImage[offset + 1] = summedImage[offset + 0] = 0.0f;
		if(c_imgBpp == 4) summedImage[offset + 3] = 0.0f;
	
#		ifdef EXTRACT_IMAGE_DEPTH
		summedImageHit[offset + 2] = summedImageHit[offset + 1] = summedImageHit[offset + 0] = 0.0f;
#		endif

#		ifdef EXTRACT_IMAGE_NORMAL
		summedImageNormal[offset + 2] = summedImageNormal[offset + 1] = summedImageNormal[offset + 0] = 0.0f;
#		endif
	}

	bool hasBounce = false;
	Ray curRay;
	curRay = ray;
	HitPoint hit;
	Material material;
	
	float4 attenuation = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
	//bool isDiffuse = false;
	hasBounce = trace(curRay, outColor, hit, material, attenuation, seed, 0);
	
	bool hasHit = hasBounce;
	HitPoint firstHit = hit;

	attenuation = attenuation*material.mat_Kd;

#	ifdef USE_PHONG_HIGHLIGHTING
	Material matPhong = material;
	bool bouncePhong = hasBounce;
#	endif
	
	for(int depth=1;depth<3;depth++)
	{
		if(hasBounce)
		{
			Vector3 prevHitN = hit.n;
			float4 color = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
			//isDiffuse = material.isDiffuse(seed);
			
			// no reflection for RealFit midterm check
			curRay.set(hit.t * curRay.dir + curRay.ori, material.sampleDiffuseDirection(hit.n, seed));
			//curRay.set(hit.t * curRay.dir + curRay.ori, material.sampleDirection(hit.n, curRay.dir, seed));
			//attenuation*(material.mat_Kd.maxComponent() + material.mat_Ks.maxComponent());
			lcg(seed);
			hasBounce = trace(curRay, color, hit, material, attenuation, seed, depth);
			attenuation = attenuation*material.mat_Kd;
			outColor += color * prevHitN.dot(curRay.dir);
		}
	}

	// Phong highlighting
#	ifdef USE_PHONG_HIGHLIGHTING
	if(bouncePhong)
	{
		Vector3 Rm = (firstHit.x - c_emitterList[0].pos).normalize();
		Rm = Material::sampleDirection(0.0f, 1.0f, 1.0f, 2048.0f, firstHit.n, Rm, 0);
		outColor += powf(Rm.dot(-ray.dir), matPhong.mat_Ns)*matPhong.mat_Ks;
	}
#	endif

	// clamp
	outColor.x = fminf(outColor.x, 1.0f);
	outColor.y = fminf(outColor.y, 1.0f);
	outColor.z = fminf(outColor.z, 1.0f);
	outColor.w = fminf(outColor.w, 1.0f);

	summedImage[offset + 0] += outColor.x;
	summedImage[offset + 1] += outColor.y;
	summedImage[offset + 2] += outColor.z;
	if(c_imgBpp == 4) summedImage[offset + 3] += outColor.w;

#	ifdef EXTRACT_IMAGE_DEPTH
	if(hasHit)
	{
		summedImageHit[offset + 0] += firstHit.x.e[0];
		summedImageHit[offset + 1] += firstHit.x.e[1];
		summedImageHit[offset + 2] += firstHit.x.e[2];
	}
#	endif

#	ifdef EXTRACT_IMAGE_NORMAL
	if(hasHit)
	{
		summedImageNormal[offset + 0] += firstHit.n.e[0];
		summedImageNormal[offset + 1] += firstHit.n.e[1];
		summedImageNormal[offset + 2] += firstHit.n.e[2];
	}
#	endif
	
	//int offset = ((height - (start_y + y) - 1)*width + (start_x + x))*3;

	/*
	image[offset + 2] = (unsigned char)(outColor.x * 255);
	image[offset + 1] = (unsigned char)(outColor.y * 255);
	image[offset + 0] = (unsigned char)(outColor.z * 255);
	*/
	int offset2 = ((c_imgHeight-y-1)*c_imgWidth + x)*c_imgBpp;
	image[offset2 + 0] = (unsigned char)(summedImage[offset + 0] / frame * 255);
	image[offset2 + 1] = (unsigned char)(summedImage[offset + 1] / frame * 255);
	image[offset2 + 2] = (unsigned char)(summedImage[offset + 2] / frame * 255);
	if(c_imgBpp == 4) image[offset2 + 3] = (unsigned char)(summedImage[offset + 3] / frame * 255);
}


__global__ void
ConvertHitToDepth(float *summedImageHit, float *summedImageDepth)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = (y*c_imgWidth + x)*c_imgBpp;

	float minDepth, maxDepth;

	float distCenter = (0.5f*(c_scene.bbMin + c_scene.bbMax) - c_camera.eye).length();
	float radius = 0.5f*(c_scene.bbMax - c_scene.bbMin).length();
	minDepth = distCenter - radius;
	maxDepth = distCenter + radius;

	float range = maxDepth - minDepth;

	Vector3 pos(summedImageHit[offset+0], summedImageHit[offset+1], summedImageHit[offset+2]);
	float depth = (pos - c_camera.eye).length();

	summedImageDepth[offset + 0] = (depth - minDepth) / range;
	summedImageDepth[offset + 1] = (depth - minDepth) / range;
	summedImageDepth[offset + 2] = (depth - minDepth) / range;
}

__host__ void convertHitToDepthCUDA()
{
	int tileWidth = 16;
	int tileHeight = 16;

	dim3 dimBlock(tileWidth, tileHeight);
	dim3 dimGrid(h_image.width / tileWidth, h_image.height / tileHeight);

	ConvertHitToDepth<<<dimGrid, dimBlock>>>(d_summedImageHitData, d_summedImageDepthData);
}

extern "C" void unloadSceneCUDAPathTracer()
{
	unloadSceneCUDA();
}

extern "C" void loadSceneCUDAPathTracer(Scene *scene)
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	int deviceCount;
	cudaError_t err = cudaGetDeviceCount(&deviceCount);
	cudaDeviceReset();
	err = cudaSetDevice(deviceCount-1);
	loadSceneCUDA(scene);
}

extern "C" void lightChangedCUDAPathTracer(Scene *scene)
{
	lightChangedCUDA(scene);
}

extern "C" void materialChangedCUDAPathTracer(Scene *scene)
{
	materialChangedCUDA(scene);
}

extern "C" void renderCUDAPathTracer(Camera *camera, Image *image, Controller *controller, int frame, int seed)
{
	renderBeginCUDA(camera, image, controller, frame);

	int tileWidth = 16;
	int tileHeight = 16;

	dim3 dimBlock(tileWidth, tileHeight);
	dim3 dimGrid(h_image.width / tileWidth, h_image.height / tileHeight);

	// execute the kernel
	Render<<<dimGrid, dimBlock>>>(d_imageData, d_summedImageData, d_summedImageHitData, d_summedImageNormalData, frame, seed);
	
	renderEndCUDA(image);
}

extern "C" void getImageCUDAPathTracer(Image *image)
{
	float *summedImageData = new float[h_image.width*h_image.height*h_image.bpp];

	checkCudaErrors(cudaMemcpy(summedImageData, d_summedImageData, h_image.width*h_image.height*h_image.bpp*sizeof(float), cudaMemcpyDeviceToHost));

	for(int i=0;i<image->height;i++)
	{
		for(int j=0;j<image->width;j++)
		{
			int offset = (i*image->width + j)*image->bpp;

			for(int k=0;k<image->bpp;k++)
				image->data[offset + k] = summedImageData[offset + k] / h_frame;
		}
	}
	delete[] summedImageData;
}

extern "C" void getDepthImageCUDAPathTracer(Image *image)
{
	convertHitToDepthCUDA();

	float *summedImageDepthData = new float[h_image.width*h_image.height*h_image.bpp];
	unsigned char *data = (unsigned char*)image->data;

	checkCudaErrors(cudaMemcpy(summedImageDepthData, d_summedImageDepthData, h_image.width*h_image.height*h_image.bpp*sizeof(float), cudaMemcpyDeviceToHost));

	for(int i=0;i<image->height;i++)
	{
		for(int j=0;j<image->width;j++)
		{
			int offset = (i*image->width + j)*image->bpp;
			Vector3 depth(summedImageDepthData[offset+0] / h_frame, summedImageDepthData[offset+1] / h_frame, summedImageDepthData[offset+2] / h_frame);

			data[offset + 0] = (unsigned char)(depth.x * 255.0f);
			data[offset + 1] = (unsigned char)(depth.y * 255.0f);
			data[offset + 2] = (unsigned char)(depth.z * 255.0f);
		}
	}
	delete[] summedImageDepthData;
}

extern "C" void getNormalImageCUDAPathTracer(Image *image)
{
	float *summedImageNormalData = new float[h_image.width*h_image.height*h_image.bpp];
	unsigned char *data = (unsigned char*)image->data;

	checkCudaErrors(cudaMemcpy(summedImageNormalData, d_summedImageNormalData, h_image.width*h_image.height*h_image.bpp*sizeof(float), cudaMemcpyDeviceToHost));

	for(int i=0;i<image->height;i++)
	{
		for(int j=0;j<image->width;j++)
		{
			int offset = (i*image->width + j)*image->bpp;
			Vector3 normal(summedImageNormalData[offset+0] / h_frame, summedImageNormalData[offset+1] / h_frame, summedImageNormalData[offset+2] / h_frame);

			if(normal.length() != 0)
			{
				normal = normal.normalize();

				data[offset + 0] = (unsigned char)((normal.x + 1.0f) * 0.5f * 255.0f);
				data[offset + 1] = (unsigned char)((normal.y + 1.0f) * 0.5f * 255.0f);
				data[offset + 2] = (unsigned char)((normal.z + 1.0f) * 0.5f * 255.0f);
			}
		}
	}
	delete[] summedImageNormalData;
}


__global__ void
FilterPass1(float *summedImage, float *summedImageHit, float *summedImageNormal, int frame, float *color, float *depth, float *normal)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = (y*c_imgWidth + x)*c_imgBpp;
	int offset2 = ((c_imgHeight - y - 1)*c_imgWidth + x)*c_imgBpp;

	// color
	color[offset2+0] = summedImage[offset+0] / frame;
	color[offset2+1] = summedImage[offset+1] / frame;
	color[offset2+2] = summedImage[offset+2] / frame;
	if(c_imgBpp == 4) color[offset2+3] = summedImage[offset+3] / frame;

	// depth
	float minDepth, maxDepth;

	float distCenter = (0.5f*(c_scene.bbMin + c_scene.bbMax) - c_camera.eye).length();
	float radius = 0.5f*(c_scene.bbMax - c_scene.bbMin).length();
	minDepth = distCenter - radius;
	maxDepth = distCenter + radius;

	float range = maxDepth - minDepth;

	Vector3 pos(summedImageHit[offset+0] / frame, summedImageHit[offset+1] / frame, summedImageHit[offset+2] / frame);
	float d = (pos - c_camera.eye).length();

	depth[offset2 + 0] = (d - minDepth) / range;
	depth[offset2 + 1] = (d - minDepth) / range;
	depth[offset2 + 2] = (d - minDepth) / range;

	// normal
	normal[offset2+0] = summedImageNormal[offset+0] / frame;
	normal[offset2+1] = summedImageNormal[offset+1] / frame;
	normal[offset2+2] = summedImageNormal[offset+2] / frame;
}

__global__ void
FilterPass2(unsigned char *result, float *color, float *depth, float *normal)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = (y*c_imgWidth + x)*c_imgBpp;

	unsigned int rgba = gBufferFilter3(color, depth, normal, x, y, c_imgWidth, c_imgHeight, c_imgBpp, c_controller.filterWindowSize, c_controller.filterParam1, c_controller.filterParam2, c_controller.filterParam3);
	//unsigned int rgba = noneFilter3(color, depth, normal, x, y, c_imgWidth, c_imgHeight, c_imgBpp, c_controller.filterWindowSize, c_controller.filterParam1, c_controller.filterParam2, c_controller.filterParam3);
	
	result[offset + 0] = (rgba >> 24) & 0xFF;
	result[offset + 1] = (rgba >> 16) & 0xFF;
	result[offset + 2] = (rgba >>  8) & 0xFF;
	if(c_imgBpp == 4) result[offset + 3] = (rgba >>  0) & 0xFF;
}

extern "C" void gBufferFilterCUDAPathTracer(Image *image)
{
	checkCudaErrors(cudaMemset(d_imageData, 0, h_image.width*h_image.height*h_image.bpp));

	int tileWidth = 16;
	int tileHeight = 16;

	dim3 dimBlock(tileWidth, tileHeight);
	dim3 dimGrid(h_image.width / tileWidth, h_image.height / tileHeight);

	float *d_color, *d_depth, *d_normal;
	checkCudaErrors(cudaMalloc((void**)&d_color, h_image.width*h_image.height*h_image.bpp*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_depth, h_image.width*h_image.height*h_image.bpp*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_normal, h_image.width*h_image.height*h_image.bpp*sizeof(float)));

	FilterPass1<<<dimGrid, dimBlock>>>(d_summedImageData, d_summedImageHitData, d_summedImageNormalData, h_frame, d_color, d_depth, d_normal);

	for(int i=0;i<h_controller.filterIteration;i++)
	{
		FilterPass2<<<dimGrid, dimBlock>>>(d_imageData, d_color, d_depth, d_normal);
	}

	checkCudaErrors(cudaFree(d_color));
	checkCudaErrors(cudaFree(d_depth));
	checkCudaErrors(cudaFree(d_normal));

	checkCudaErrors(cudaMemcpy(image->data, d_imageData, h_image.width*h_image.height*h_image.bpp, cudaMemcpyDeviceToHost));
}

extern "C" void updateControllerCUDAPathTracer(Controller *controller)
{
	updateController(controller);
}

}