/********************************************************************
	created:	2011/08/28
	file path:	d:\Projects\Redering\OpenIRT\src\CUDA
	file base:	CUDAPhotonMapping
	file ext:	cu
	author:		Tae-Joon Kim (tjkim.kaist@gmail.com)
	
	comment:	Photon Mapping using CUDA
*********************************************************************/

#include "CUDACommonDefines.cuh"

using namespace CUDA;

namespace CUDAPhotonMapping
{

#include "CUDARayTracerCommon.cuh"

#define alpha 0.7f

Photon *d_PhotonList;

texture<float4, cudaTextureType1D, cudaReadModeElementType> t_Photons;
__constant__ float c_cosTheta[256];
__constant__ float c_sinTheta[256];
__constant__ float c_cosPhi[256];
__constant__ float c_sinPhi[256];

#define PHOTON_DIRECTION(dir, photon) \
	(dir).x = c_sinTheta[(photon).theta]*cosPhi[(photon).phi]; \
	(dir).y = c_sinTheta[(photon).theta]*sinPhi[(photon).phi]; \
	(dir).z = c_cosTheta[(photon).theta]

#define PHOTON_FETCH(id, offset, ret) (ret) = tex1Dfetch(t_Photons, (id)*sizeof(Photon)/16+(offset));

__device__ inline void
gatherPhotons(Vector3 &color, const Vector3 &hitPosition, const Vector3 &hitNormal, float radius2, int numPhotons)
{
	color = Vector3(0.0f);

	unsigned int stack[100];
	unsigned int stackPtr = 0;
	unsigned int node = 0; // 0 is the start

#define push(N) stack[stackPtr++] = (N)
#define pop()   stack[--stackPtr]

	push(node);

	unsigned int numGatheredPhotons = 0u;
	Vector3 flux = Vector3(0.0f);

	Photon photon;
	unsigned short axis;
	Vector3 diff;

	do 
	{
		PHOTON_FETCH(node, 0, photon.low);
		PHOTON_FETCH(node, 1, photon.high);

		axis = photon.axis;

		if(axis != Photon::SPLIT_AXIS_NULL)
		{
			diff = hitPosition - photon.pos;
			float distance2 = diff.dot(diff);

			if(distance2 <= radius2) 
			{
				// hit normal X photon dir?
				flux += photon.power;
				numGatheredPhotons++;
			}

			if(axis != Photon::SPLIT_AXIS_LEAF) 
			{
				float d = diff.e[axis];

				// Calculate the next child selector. 0 is left, 1 is right.
				int selector = d < 0.0f ? 0 : 1;
				if( d*d < radius2 ) {
					push((node<<1) + 2 - selector);
				}

				node = (node<<1) + 1 + selector;
			} else
			{
				node = pop();
			}
		}
		else
		{
			node = pop();
		}
	} while (node);

	/*
	float newN = numPhotons + alpha*numGatheredPhotons;

	float reductionFactor2 = 1.0f;
	if( numGatheredPhotons != 0 ) {
		reductionFactor2 = newN / (numPhotons + numGatheredPhotons);
		radius2 *= reductionFactor2; 
	}
	numPhotons = (int)newN;

	// Compute indirectflux
	float3 new_flux = ( rec_flux + flux_M ) * reductionFactor2;
	rec.d = make_float4( new_flux ); // set rec.flux
	float3 indirect_flux = 1.0f / ( M_PIf * new_R2 ) * new_flux / total_emitted;
	*/

	if(numGatheredPhotons > 0)
	{
		//printf("numGatheredPhotons = %d, flux = %f %f %f\n", numGatheredPhotons, flux.x, flux.y, flux.z);
		color = flux / (radius2 * PI);// / numGatheredPhotons;
	}
}

__device__ inline bool
trace(const Ray &ray, float4 &color, HitPoint &hit, Material &hitMat, const Vector3 &brdf, float4 &attenuation, unsigned int prevRnd, bool gather, int depth)
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

	//getMaterial(hitMat, hit);

	Vector3 hitPoint = ray.ori + hit.t*ray.dir;

	Vector3 shadowDir;
	Vector3 samplePos;
	float cosFac;

	if(gather)
	{
		Vector3 colorFromPhotons(0.0f);
		gatherPhotons(colorFromPhotons, hitPoint, hit.n, 10.0f, 100);
		color += (colorFromPhotons * brdf);
	}
	else
	{
		for(int i=0;i<c_scene.numEmitters;i++)
		{
			const Emitter &emitter = c_emitterList[i];

			samplePos = emitter.sample(prevRnd);

			shadowDir = samplePos - hitPoint;	
			shadowDir = shadowDir.normalize();

			const Vector3 &lightAmbient = emitter.color_Ka;
			const Vector3 &lightDiffuse = emitter.color_Kd;

			cosFac = shadowDir.dot(hit.n);

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
	}
	return true;
}

__global__ void
TracePhotons(int emitterIndex, Photon *outPhotons, int offset, int pass)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int numPhotons = blockDim.x * gridDim.x;

	unsigned int seed = tea<16>(idx, 2*pass+0);
	unsigned int seed2 = tea<16>(idx, 2*pass+1);

	const Emitter &emitter = c_emitterList[emitterIndex];
	//const Parallelogram &target = emitter.spotTarget;

	Ray ray;
	Vector3 ori = emitter.sample(seed);
	//Vector3 dir = (target.sample(seed) - ori).normalize();
	Vector3 dir = emitter.sampleEmitDirection(ori, seed);
	ray.set(ori, dir);

	HitPoint hit;
	hit.t = FLT_MAX;

	Photon photon;
	photon.pos = Vector3(0.0f, 0.0f, 0.0f);
	photon.power = Vector3(0.0f, 0.0f, 0.0f);

	// bounce once when hit diffuse material
	// bounce when hit specular material
	int skip = 1;
	int maxDepth = 5;
	int depth = 0;
	//bool isDiffuse = false;
	Material material;

	while(++depth < maxDepth)
	{
		if(!RaySceneBVHIntersect(ray, hit, material, 0.0f, seed)) return;
		//getMaterial(material, hit);
		if(material.hasDiffuse())
			if(skip-- == 0) break;
		//isDiffuse = material.isDiffuse(seed);
		ray.set(hit.t * ray.dir + ray.ori, material.sampleDirection(hit.n, ray.dir, seed2));
		ray.set(ray.ori + ray.dir, ray.dir);	// offseting
		hit.t = FLT_MAX;
	}

	photon.pos = ray.ori + hit.t * ray.dir;

	float cosFac = -ray.dir.dot(hit.n);

	if(cosFac > 0.0f)
	{
		photon.power = (emitter.color_Ka * material.mat_Ka + emitter.color_Kd * material.mat_Kd * cosFac) * (emitter.intensity / numPhotons);
		photon.setDirection(ray.dir);
	}

	outPhotons[offset+idx] = photon;
}

__global__ void
Render(unsigned char *image, float *summedImage, int frame, int globalSeed)
{
	Ray ray;
	float4 outColor = make_float4(0.0f, 0.0f, 0.0f, 1.0f);

	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = (y*c_imgWidth + x)*c_imgBpp;

	unsigned int seed = tea<16>((y*c_imgWidth + x), globalSeed);

	c_camera.getRayWithOrigin(ray, 
		(x + rnd(seed))/c_imgWidth, 
		(y + rnd(seed))/c_imgHeight);

	if(frame == 1)
	{
		summedImage[offset + 2] = summedImage[offset + 1] = summedImage[offset + 0] = 0.0f;
		if(c_imgBpp == 4) summedImage[offset + 3] = 0.0f;
	}

	bool hasBounce = false;
	Ray curRay;
	curRay = ray;
	HitPoint hit;
	Material material;
	float4 attenuation = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
	bool isDiffuse = false;
	int maxDepth = 5;
	int depth = 0;

	hasBounce = trace(curRay, outColor, hit, material, Vector3(1.0f), attenuation, seed, isDiffuse, depth);
	while(++depth < maxDepth && hasBounce && !isDiffuse)
	{
		Vector3 prevHitN = hit.n;
		float4 color = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
		isDiffuse = material.isDiffuse(seed);
		curRay.set(hit.t * curRay.dir + curRay.ori, material.sampleDirection(hit.n, curRay.dir, seed));
		attenuation = attenuation*material.mat_Kd.maxComponent();
		hasBounce = trace(curRay, color, hit, material, material.brdf(seed, isDiffuse), attenuation, seed, isDiffuse, depth);
		outColor += color * prevHitN.dot(curRay.dir);
	}

	// clamp
	outColor.x = fminf(outColor.x, 1.0f);
	outColor.y = fminf(outColor.y, 1.0f);
	outColor.z = fminf(outColor.z, 1.0f);
	outColor.w = fminf(outColor.w, 1.0f);

	summedImage[offset + 0] += outColor.x;
	summedImage[offset + 1] += outColor.y;
	summedImage[offset + 2] += outColor.z;
	if(c_imgBpp == 4) summedImage[offset + 3] += outColor.w;

	int offset2 = ((c_imgHeight-y-1)*c_imgWidth + x)*c_imgBpp;
	image[offset2 + 0] = (unsigned char)(summedImage[offset + 0] / frame * 255);
	image[offset2 + 1] = (unsigned char)(summedImage[offset + 1] / frame * 255);
	image[offset2 + 2] = (unsigned char)(summedImage[offset + 2] / frame * 255);
	if(c_imgBpp == 4) image[offset2 + 3] = (unsigned char)(summedImage[offset + 3] / frame * 255);
}

extern "C" void unloadSceneCUDAPhotonMapping()
{
	unloadSceneCUDA();

	checkCudaErrors(cudaFree(d_PhotonList));
}

extern "C" void loadSceneCUDAPhotonMapping(Scene *scene)
{
	loadSceneCUDA(scene);

	// precompute angular conversions
	float cosTheta[256];
	float sinTheta[256];
	float cosPhi[256];
	float sinPhi[256];

	for(int i=0;i<256;i++)
	{
		float angle = i / 256.0f * PI;
		cosTheta[i] = cosf(angle);
		sinTheta[i] = sinf(angle);
		cosPhi[i] = cosf(2.0f*angle);
		sinPhi[i] = sinf(2.0f*angle);
	}

	checkCudaErrors(cudaMemcpyToSymbol(c_cosTheta, cosTheta, sizeof(float)*256, 0, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(c_sinTheta, sinTheta, sizeof(float)*256, 0, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(c_cosPhi, cosPhi, sizeof(float)*256, 0, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(c_sinPhi, sinPhi, sizeof(float)*256, 0, cudaMemcpyHostToDevice));
}

extern "C" int tracePhotons(int size, void *outPhotons)
{
	int processingBlockSize = 1024*1024;
	int numRemainPhotons;

	int numTotalPhotons = 0;

	for(int i=0;i<h_scene.numEmitters;i++)
		numTotalPhotons += h_scene.emitters[i].numScatteringPhotons;

	checkCudaErrors(cudaFree(d_PhotonList));
	checkCudaErrors(cudaMalloc((void **)&d_PhotonList, numTotalPhotons*sizeof(Photon)));

	numTotalPhotons = 0;
	for(int i=0;i<h_scene.numEmitters;i++)
	{
		int numPhotons = h_scene.emitters[i].numScatteringPhotons;
		numRemainPhotons = numPhotons;
		
		for(int pass=0;numRemainPhotons > 0;pass++)
		{
			int numPhotonsCurPass = min(processingBlockSize, numRemainPhotons);

			dim3 dimBlock(min(numPhotonsCurPass, 512), 1);
			dim3 dimGrid(numPhotonsCurPass / dimBlock.x, 1);

			TracePhotons<<<dimGrid, dimBlock>>>(i, d_PhotonList, numTotalPhotons, pass);
			numTotalPhotons += numPhotonsCurPass;
			numRemainPhotons -= numPhotonsCurPass;
		}
	}

	Photon *h_photons = (Photon *)outPhotons;
	if(!outPhotons)
		h_photons = new Photon[numTotalPhotons];

	checkCudaErrors(cudaMemcpy(h_photons, d_PhotonList, numTotalPhotons*sizeof(Photon), cudaMemcpyDeviceToHost));

	int numValidPhotons = 0;
	// remove invalid photons
#	define IS_VALID_PHOTON(photon) (((photon).power.x != 0) || ((photon).power.y != 0) || ((photon).power.z != 0))
	int left = 0, right = numTotalPhotons-1;
	while(true)
	{
		while(IS_VALID_PHOTON(h_photons[left]))
		{
			left++;
			if(left >= right) break;
		}
		while(!IS_VALID_PHOTON(h_photons[right]))
		{
			right--;
			if(left >= right) break;
		}
		if(left >= right) break;

		Photon temp = h_photons[left];
		h_photons[left] = h_photons[right];
		h_photons[right] = temp;
	}

	numValidPhotons = left+1;
	//int numValidPhotons = numTotalPhotons;

	checkCudaErrors(cudaMemcpy(d_PhotonList, h_photons, numValidPhotons*sizeof(Photon), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaBindTexture((size_t *)0, t_Photons, d_PhotonList, numValidPhotons*sizeof(Photon)));

	if(!outPhotons)
		delete[] h_photons;

	return numValidPhotons;
}

extern "C" int buildPhotonKDTree(int size, void *kdtree)
{
	int sizeKDTree = size;
	if(kdtree)
	{
		// if kd tree is already given, just replace
		checkCudaErrors(cudaFree(d_PhotonList));
		checkCudaErrors(cudaMalloc((void **)&d_PhotonList, size*sizeof(Photon)));

		checkCudaErrors(cudaMemcpy(d_PhotonList, kdtree, size*sizeof(Photon), cudaMemcpyHostToDevice));

		checkCudaErrors(cudaBindTexture((size_t *)0, t_Photons, d_PhotonList, size*sizeof(Photon)));
	}
	return sizeKDTree;
}

extern "C" void renderCUDAPhotonMapping(Camera *camera, Image *image, Controller *controller, int frame, int seed)
{
	renderBeginCUDA(camera, image, controller, frame);
	int tileWidth = 16;
	int tileHeight = 16;

	dim3 dimBlock(tileWidth, tileHeight);
	dim3 dimGrid(h_image.width / tileWidth, h_image.height / tileHeight);

	cudaFuncSetCacheConfig(Render, cudaFuncCachePreferL1);
	// execute the kernel
	Render<<<dimGrid, dimBlock>>>(d_imageData, d_summedImageData, frame, seed);

	renderEndCUDA(image);
}

}