/********************************************************************
	created:	2011/08/03
	file path:	d:\Projects\Redering\OpenIRT\src\CUDA
	file base:	CUDARayTracer
	file ext:	cu
	author:		Tae-Joon Kim (tjkim.kaist@gmail.com)
	
	comment:	Ray tracer using CUDA
*********************************************************************/

#include "CUDACommonDefines.cuh"

using namespace CUDA;

namespace CUDARayTracer
{

#include "CUDARayTracerCommon.cuh"

__device__ void
trace(const Ray &ray, float4 &color)
{
	HitPoint hit;
	Material material;
	hit.t = FLT_MAX;

	color.x = color.y = color.z = 0.0f;

	//bool hasHit = false;

	if(!RaySceneBVHIntersect(ray, hit, material, 0.0f, 0))
	{
		if(c_hasEnvMap) shadeEnvironmentMap(ray.dir, color);
		else color = make_float4(c_envCol.x, c_envCol.y, c_envCol.z, 1.0f);

		if(!c_controller.drawBackground) color.w = 0.0f;

		return;
	}

	for(int i=0;i<1;i++)
	{
		Vector3 hitPoint = ray.ori + hit.t*ray.dir;
		Vector3 shadowDir;
		shadowDir = c_emitterList[i].pos - hitPoint;	
		shadowDir = shadowDir.normalize();

		const Vector3 &lightAmbient = c_emitterList[i].color_Ka;
		const Vector3 &lightDiffuse = c_emitterList[i].color_Kd;

		float cosFac = shadowDir.dot(hit.n);

		//float4 mat0;
		//float4 mat1;
		//MAT_FETCH(hit.model, hit.material, 0, mat0);
		//MAT_FETCH(hit.model, hit.material, 1, mat1);
		
		//Vector3 matAmbient(mat0.x, mat0.y, mat0.z);
		//Vector3 matDiffuse(mat0.w, mat1.x, mat1.y);

		color += lightAmbient * material.mat_Ka + lightDiffuse * material.mat_Kd * cosFac;
	}
}

__global__ void
Render(unsigned char *image)
{
	Ray ray;
	float4 outColor = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
	
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	c_camera.getRayWithOrigin(ray, 
		(x + 0.5f)/c_imgWidth, 
		(y + 0.5f)/c_imgHeight);

	trace(ray, outColor);

	// clamp
	outColor.x = fminf(outColor.x, 1.0f);
	outColor.y = fminf(outColor.y, 1.0f);
	outColor.z = fminf(outColor.z, 1.0f);
	outColor.w = fminf(outColor.w, 1.0f);

	//int offset = (y*c_imgWidth + x)*c_imgBpp;
	int offset2 = ((c_imgHeight-y-1)*c_imgWidth + x)*c_imgBpp;

	image[offset2 + 0] = (unsigned char)(outColor.x * 255);
	image[offset2 + 1] = (unsigned char)(outColor.y * 255);
	image[offset2 + 2] = (unsigned char)(outColor.z * 255);
	if(c_imgBpp == 4) image[offset2 + 3] = (unsigned char)(outColor.w * 255);
}

extern "C" void unloadSceneCUDARayTracer()
{
	unloadSceneCUDA();
}

extern "C" void loadSceneCUDARayTracer(Scene *scene)
{
	loadSceneCUDA(scene);
}

extern "C" void materialChangedRayTracer(Scene *scene)
{
	materialChangedCUDA(scene);
}

extern "C" void renderCUDARayTracer(Camera *camera, Image *image, Controller *controller)
{
	int frame;
	renderBeginCUDA(camera, image, controller, frame);

	int tileWidth = 16;
	int tileHeight = 16;

	dim3 dimBlock(tileWidth, tileHeight);
	dim3 dimGrid(h_image.width / tileWidth, h_image.height / tileHeight);

	// execute the kernel
	Render<<<dimGrid, dimBlock>>>(d_imageData);

	renderEndCUDA(image);
}

extern "C" void clearResultCUDARayTracer(int &frame)
{
	clearResult(frame);
}


}
