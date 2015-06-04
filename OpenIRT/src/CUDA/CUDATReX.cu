/********************************************************************
	created:	2011/09/28
	file path:	d:\Projects\Redering\OpenIRT\src\CUDA
	file base:	TReX
	file ext:	cu
	author:		Tae-Joon Kim (tjkim.kaist@gmail.com)
	
	comment:	GPU side of hybrid renderer
*********************************************************************/

#include "CommonOptions.h"
#include "CUDACommonDefines.cuh"

template <class T>
void reduce(int size, int threads, int blocks, 
                 int whichKernel, T *d_idata, T *d_odata);

using namespace CUDA;

extern "C"
bool isPow2(unsigned int x)
{
    return ((x&(x-1))==0);
}

namespace TReX
{
	
#include "CUDARayTracerCommon.cuh"
#include "CUDAFilter.cuh"

#define EMITTER_SAMPLE_RESOLUTION 101

int h_numVoxels;
int h_sizeVoxelBuffer;
OctreeHeader h_octreeHeader;
Voxel *h_octree;

int h_numOOCVoxels;
OOCVoxel *h_oocVoxels;
OOCVoxel *d_oocVoxels;

Photon *d_PhotonList;
//Ray *d_rayCache;
HitPoint *d_hitCache;
ExtraRayInfo *d_extRayCache;
//HColor *d_colorCache;
Voxel *d_octree;
PhotonVoxel *d_photonOctree;
PhotonVoxel *d_photonOctree2;
Material *d_material;

SampleData *d_sampleData;

float *d_requestCountList;

int h_frame2;

StatData *d_statData;

bool h_isReset = false;

#define NUM_MAX_THREADS (128*1024)

texture<float4, cudaTextureType1D, cudaReadModeElementType> t_Photons;
texture<float4, cudaTextureType1D, cudaReadModeElementType> t_octree;
texture<float4, cudaTextureType1D, cudaReadModeElementType> t_photonOctree;
texture<float4, cudaTextureType1D, cudaReadModeElementType> t_photonOctree2;
texture<float4, cudaTextureType1D, cudaReadModeElementType> t_material;

__constant__ float c_cosTheta[256];
__constant__ float c_sinTheta[256];
__constant__ float c_cosPhi[256];
__constant__ float c_sinPhi[256];
__constant__ OctreeHeader c_octreeHeader;
__constant__ int c_numVoxels;
__constant__ float c_leafVoxelSize;
//__constant__ char c_childOrder[8*8];

__constant__ int c_frame2;

__constant__ float c_boxFilterKernel3[9];
__constant__ float c_boxFilterKernel5[25];

__constant__ float c_photonMorphingRatio;

__constant__ float c_emitterUsageDistribution[MAX_NUM_EMITTERS];
__constant__ int c_emitterUsageHistogram[EMITTER_SAMPLE_RESOLUTION];

__constant__ int c_manualEmitterSelection[MAX_NUM_EMITTERS];
__constant__ int c_numSelectedEmitters;

#define PHOTON_DIRECTION(dir, photon) \
	(dir).x = c_sinTheta[(photon).theta]*cosPhi[(photon).phi]; \
	(dir).y = c_sinTheta[(photon).theta]*sinPhi[(photon).phi]; \
	(dir).z = c_cosTheta[(photon).theta]

#define PHOTON_FETCH(id, offset, ret) (ret) = tex1Dfetch(t_Photons, (id)*sizeof(Photon)/16+(offset));

#define VOXEL_FETCH(id, offset, ret) (ret) = tex1Dfetch(t_octree, (id)*sizeof(Voxel)/16+(offset));
#define PHOTON_VOXEL_FETCH(id, offset, ret) (ret) = tex1Dfetch(t_photonOctree, (id)*sizeof(PhotonVoxel)/16+(offset));
#define PHOTON_VOXEL2_FETCH(id, offset, ret) (ret) = tex1Dfetch(t_photonOctree2, (id)*sizeof(PhotonVoxel)/16+(offset));

#ifdef MAT_FETCH
#undef MAT_FETCH
#define MAT_FETCH(id, offset, ret) (ret) = tex1Dfetch(t_material, (id)*sizeof(Material)/16+(offset))
#endif

#define CONVERT_DIRECTION_2TO3(theta, phi, dir) \
	(dir).x = c_sinTheta[theta]*c_cosPhi[phi]; \
	(dir).y = c_sinTheta[theta]*c_sinPhi[phi]; \
	(dir).z = c_cosTheta[theta]

/*
__device__ inline int getChildOrder(const Ray &ray, int order)
{
	order = order + (ray.posNeg[0] ? 0 : (order / 4 ? -4 : 4));
	order = order + (ray.posNeg[1] ? 0 : ((order / 2) % 2 ? -2 : 2));
	order = order + (ray.posNeg[2] ? 0 : (order % 2 ? -1 : 1));
	return order;
}

__device__ inline int getChildOrder2(const Ray &ray, int order)
{
	return c_childOrder[8*ray.posNeg[3] + order];
}
*/

__device__ inline void computeSubBox(int childIndex, 
	float &minx, float &miny, float &minz, 
	float &maxx, float &maxy, float &maxz,
	const float pminx, const float pminy, const float pminz, 
	const float pmaxx, const float pmaxy, const float pmaxz
	)
{
	/*
	*minx = pminy; *miny = pminy; *minz = pminz;
	*maxx = pmaxy; *maxy = pmaxy; *maxz = pmaxz;
	*(childIndex & 0x4 ? minx : maxx) = 0.5f*(pminx + pmaxx);
	*(childIndex & 0x2 ? miny : maxy) = 0.5f*(pminy + pmaxy);
	*(childIndex & 0x1 ? minz : maxz) = 0.5f*(pminz + pmaxz);
	*/
	if(childIndex & 0x4)
	{
		minx = 0.5f*(pminx + pmaxx);
		maxx = pmaxx;
	}
	else
	{
		maxx = 0.5f*(pminx + pmaxx);
		minx = pminx;
	}
	if(childIndex & 0x2)
	{
		miny = 0.5f*(pminy + pmaxy);
		maxy = pmaxy;
	}
	else
	{
		maxy = 0.5f*(pminy + pmaxy);
		miny = pminy;
	}
	if(childIndex & 0x1)
	{
		minz = 0.5f*(pminz + pmaxz);
		maxz = pmaxz;
	}
	else
	{
		maxz = 0.5f*(pminz + pmaxz);
		minz = pminz;
	}
}

__device__ inline void computeSubBox(int childIndex, Vector3 &minBB, Vector3 &maxBB, const Vector3 &pminBB, const Vector3 &pmaxBB)
{
	Vector3 mid = 0.5f*(pminBB + pmaxBB);
	minBB.x = childIndex & 0x4 ? mid.x : pminBB.x;
	maxBB.x = childIndex & 0x4 ? pmaxBB.x : mid.x;
	minBB.y = childIndex & 0x2 ? mid.y : pminBB.y;
	maxBB.y = childIndex & 0x2 ? pmaxBB.y : mid.y;
	minBB.z = childIndex & 0x1 ? mid.z : pminBB.z;
	maxBB.z = childIndex & 0x1 ? pmaxBB.z : mid.z;
}

__device__ inline bool RayLeafVoxelIntersect(const Ray &ray, const Ray &transRay, const AABB &aabb, const Voxel &voxel, HitPoint &hit, Material &material, float tmax, unsigned int seed, int x, int y)
{
	//CUDA::Vector3 mM = aabb.max * transRay.dir + transRay.ori;
	//CUDA::Vector3 mm = aabb.min * transRay.dir + transRay.ori;
	CUDA::Vector3 mM = aabb.max * transRay.dir + ray.ori;
	CUDA::Vector3 mm = aabb.min * transRay.dir + ray.ori;
	float t0 = 0, t1 = 0.0f;
	ray.BoxIntersect(mm, mM, t0, t1);
	//CUDA::Vector3 mM = c_octreeHeader.max - (aabb.max * transRay.dir);
	//CUDA::Vector3 mm = c_octreeHeader.min - (aabb.min * transRay.dir);
	//if(x == 246 && y == 250)
	//{
	//	printf("t0 = %f, t1 = %f\n", t0, t1);
	//	printf("sceneB %f %f %f - %f %f %f\n", c_octreeHeader.min.x, c_octreeHeader.min.y, c_octreeHeader.min.z, c_octreeHeader.max.x, c_octreeHeader.max.y, c_octreeHeader.max.z);
	//	printf("before %f %f %f - %f %f %f\n", mm.e[0], mm.e[1], mm.e[2], mM.e[0], mM.e[1], mM.e[2]);
	//}
	//if(ray.dir.x < 0) {mM.e[0] *= -1.0f; mm.e[0] *= -1.0f;}
	//if(ray.dir.y < 0) {mM.e[1] *= -1.0f; mm.e[1] *= -1.0f;}
	//if(ray.dir.z < 0) {mM.e[2] *= -1.0f; mm.e[2] *= -1.0f;}
	float hitBBDiag2 = (mM - mm).magsqr() * (1 + 10.0f*abs(hit.n.dot(ray.dir)));
	//float hitBBDist2 = (0.5f*(mM + mm) - ray.ori).magsqr();
	float hitBBDist2 = t0*t0;

	//if(x == 246 && y == 250)
	//{
	//	printf("after  %f %f %f - %f %f %f\n", mm.e[0], mm.e[1], mm.e[2], mM.e[0], mM.e[1], mM.e[2]);
	//	printf("ray %d %d %d - %f %f %f - %f %f %f\n", ray.dir.e[0] > 0, ray.dir.e[1] > 0, ray.dir.e[2] > 0, ray.ori.e[0], ray.ori.e[1], ray.ori.e[2], ray.dir.e[0], ray.dir.e[1], ray.dir.e[2]);
	//	printf("%d hitBBRadi2 = %f, hitBBDist2 = %f, tmax2 = %f\n", hitBBDist2 < hitBBDiag2 || hitBBDist2 > tmax*tmax, hitBBDiag2, hitBBDist2, tmax*tmax);
	//}
	if(hitBBDist2 < hitBBDiag2 || hitBBDist2 > tmax*tmax) return false;

	/*
	if(x == 256 && y == 256)
	{
		printf("t0 = %f, t1 = %f\n", t0, t1);
		printf("sceneB %f %f %f - %f %f %f\n", c_octreeHeader.min.x, c_octreeHeader.min.y, c_octreeHeader.min.z, c_octreeHeader.max.x, c_octreeHeader.max.y, c_octreeHeader.max.z);
		printf("before %f %f %f - %f %f %f\n", mm.e[0], mm.e[1], mm.e[2], mM.e[0], mM.e[1], mM.e[2]);
		printf("after  %f %f %f - %f %f %f\n", mm.e[0], mm.e[1], mm.e[2], mM.e[0], mM.e[1], mM.e[2]);
		printf("ray %d %d %d - %f %f %f - %f %f %f\n", ray.dir.e[0] > 0, ray.dir.e[1] > 0, ray.dir.e[2] > 0, ray.ori.e[0], ray.ori.e[1], ray.ori.e[2], ray.dir.e[0], ray.dir.e[1], ray.dir.e[2]);
		printf("%d hitBBRadi2 = %f, hitBBDist2 = %f, tmax2 = %f\n", hitBBDist2 < hitBBDiag2 || hitBBDist2 > tmax*tmax, hitBBDiag2, hitBBDist2, tmax*tmax);
	}
	*/

	material.mat_Kd = voxel.getKd();
	material.mat_Ks = voxel.getKs();
	material.mat_d = voxel.getD();
	material.mat_Ns = voxel.getNs();
	material.recalculateRanges();
	if(material.isRefraction(seed)) return false;

	CUDA::Vector3 norm;
	CONVERT_DIRECTION_2TO3(voxel.theta, voxel.phi, norm);

	// we have a hit:
	// fill hitpoint structure:
	//
	hit.material = 0;
	hit.t = sqrtf(hitBBDist2);
	//hit.n = vdot > 0 ? -norm : norm;
	hit.n = norm;
	return true;	
}

__device__ inline bool RayLODIntersect(const Ray &ray, const Ray &transRay, const AABB &aabb, const Voxel &voxel, HitPoint &hit, Material *material, float tmax, float tLimit, unsigned int seed, int x, int y)
{
	// ignore invalid voxel
	if(!(voxel.d < FLT_MAX && voxel.d > -FLT_MAX)) return false;

	Vector3 norm;
	CONVERT_DIRECTION_2TO3(voxel.theta, voxel.phi, norm);
	
	float vdot = norm.x*ray.dir.x + norm.y*ray.dir.y + norm.z*ray.dir.z;
	float vdot2 = norm.x*ray.ori.x + norm.y*ray.ori.y + norm.z*ray.ori.z;
	float t = (voxel.d - vdot2) / vdot;

#	ifdef USE_PLANE_D
	if(t < aabb.min.x || t < aabb.min.y || t < aabb.min.z || t > aabb.max.x  || t > aabb.max.y || t > aabb.max.z) return false;
#	endif

	// if either too near or further away than a previous hit, we stop
	//if (t < (bbSize*1.8f-INTERSECT_EPSILON*10) || t > (tmax + INTERSECT_EPSILON*10) || t > tLimit - bbSize*1.8f)
	if (t < (0.0f-INTERSECT_EPSILON*10) || t > (tmax + INTERSECT_EPSILON*10))
	//if (t < (c_leafVoxelSize*OOCVOXEL_SUPER_RESOLUTION-INTERSECT_EPSILON*10) || t > (tmax + INTERSECT_EPSILON*10))

		return false;

#	ifndef USE_PLANE_D
	CUDA::Vector3 mM = aabb.max * transRay.dir + ray.ori;
	CUDA::Vector3 mm = aabb.min * transRay.dir + ray.ori;
	//float t0 = 0, t1 = 0.0f;
	//ray.BoxIntersect(mm, mM, t0, t1);
	float hitBBDiag2 = (mM - mm).magsqr();// * (1 + 10.0f*abs(hit.n.dot(ray.dir)));
	float hitBBDist2 = (0.5f*(mM + mm) - ray.ori).magsqr();
	//float hitBBDist2 = t0*t0;
	if(hitBBDist2 < hitBBDiag2 || hitBBDist2 > tmax*tmax) return false;
#	endif
	

	if(material)
	{
		Material &mat = *material;
#		ifdef USE_VOXEL_LOD
		mat.mat_Kd = voxel.getKd();
		mat.mat_Ks = voxel.getKs();
		mat.mat_d = voxel.getD();
		mat.mat_Ns = voxel.getNs();
		mat.recalculateRanges();
#		else
		MAT_FETCH(voxel.m, 0, mat._0);
		MAT_FETCH(voxel.m, 1, mat._1);
		MAT_FETCH(voxel.m, 2, mat._2);
		MAT_FETCH(voxel.m, 3, mat._3);
		mat.recalculateRanges();
		if(mat.isRefraction(seed)) return false;
#		endif
	}

	// we have a hit:
	// fill hitpoint structure:
	//
	hit.material = voxel.m;
	hit.t = t;
	hit.n = vdot > 0 ? -norm : norm;
	return true;	
}

__device__ inline bool RayOctreeIntersect(const Ray &ray, HitPoint &hit, Material *material, int *hitIndex, float *hitBBSize, float tLimit, unsigned int seed, float *requestCountList, int x, int y, int frame2)
{
	float t0, t1;
	if(!ray.BoxIntersect(c_octreeHeader.min, c_octreeHeader.max, t0, t1)) return false;

	char flag = 0;

	Vector3 ori = ray.ori, dir = ray.dir;
	Vector3 temp = c_octreeHeader.max + c_octreeHeader.min;

	if(ray.dir.x < 0.0f)
	{
		ori.e[0] = temp.e[0] - ori.e[0];
		dir.e[0] = -dir.e[0];
		flag |= 4;
	}

	if(ray.dir.y < 0.0f)
	{
		ori.e[1] = temp.e[1] - ori.e[1];
		dir.e[1] = -dir.e[1];
		flag |= 2;
	}

	if(ray.dir.z < 0.0f)
	{
		ori.e[2] = temp.e[2] - ori.e[2];
		dir.e[2] = -dir.e[2];
		flag |= 1;
	}

	Ray transRay;
	transRay.set(ori, dir);

	AABB currentBB;
	currentBB.min = (c_octreeHeader.min - transRay.ori) * transRay.invDir;
	currentBB.max = (c_octreeHeader.max - transRay.ori) * transRay.invDir;

	int N = c_octreeHeader.dim;

	typedef struct tempStack_t
	{
		int childIndex;
		char child;
	} tempStack;

	Vector3 mid;
	Voxel currentVoxel;
	currentVoxel.setChildIndex(0);
	tempStack stack[100];
//	__shared__ tempStack stack[256][16];
//#	define _stack stack[threadIdx.x]
#	define _stack stack

	int stackPtr;

	mid = 0.5f*(currentBB.min + currentBB.max);

	stackPtr = 0;

	char child = 0;
	int childIndex = 0;

	//float bbSize;// = c_octreeHeader.max.x - c_octreeHeader.min.x;

	bool isEmpty;
	while(true)
	{
		if(c_frame2 != frame2) return false;

		isEmpty = currentVoxel.isEmpty();

		if(!isEmpty && 
			currentBB.max.x > 0.0f && currentBB.max.y > 0.0f && currentBB.max.z > 0.0f &&
			currentBB.min.x < hit.t && currentBB.min.y < hit.t && currentBB.min.y < hit.t)
		{
#			ifdef USE_VOXEL_LOD
			Vector3 mM = currentBB.max * transRay.dir + ray.ori;
			Vector3 mm = currentBB.min * transRay.dir + ray.ori;
			//float t0 = 0, t1 = 0.0f;
			//ray.BoxIntersect(mm, mM, t0, t1);
			float hitBBDiag = (mM - mm).length();
			float hitBBDist = (0.5f*(mM + mm) - ray.ori).length();
			if(currentVoxel.isLeaf() || hitBBDiag / hitBBDist < limit)
#			else
			if(currentVoxel.isLeaf())
#			endif
			{
				//printf("currentBB.max.x - currentBB.min.x = %f, BBSize = %f, min = %f, max = %f\n", currentBB.max.x - currentBB.min.x, bbSize, c_octreeHeader.min.x, c_octreeHeader.max.x);
				if(hitBBSize) *hitBBSize = (currentBB.max.x - currentBB.min.x) * transRay.dir.x;
				//if(RayLeafVoxelIntersect(ray, transRay, currentBB, currentVoxel, hit, material, hit.t, seed, x, y))
				if(RayLODIntersect(ray, transRay, currentBB, currentVoxel, hit, material, hit.t, tLimit, seed, x, y))
				{
					if(hitIndex) *hitIndex = childIndex + (child^flag);
					//hitBBSize = bbSize;
					return true;
				}
				goto NEXT_SIBILING;
			}
			// push down

			//FIRST_NODE(currentBB, mid, child);
			// get first child
			child = 0;
			if (currentBB.min.y < currentBB.min.x && currentBB.min.z < currentBB.min.x)
			{
				// YZ plane
				if (mid.y < currentBB.min.x) child |= 2;
				if (mid.z < currentBB.min.x) child |= 1;
			}
			else if (currentBB.min.z < currentBB.min.y)
			{
				// XZ plane
				if (mid.x < currentBB.min.y) child |= 4;
				if (mid.z < currentBB.min.y) child |= 1;
			}
			else
			{
				// XY plane
				if (mid.x < currentBB.min.z) child |= 4;
				if (mid.y < currentBB.min.z) child |= 2;
			}

			childIndex = currentVoxel.getChildIndex() * N * N * N;

			_stack[stackPtr].childIndex = childIndex;
			_stack[stackPtr++].child = child;

			currentBB.min.e[0] = (child & 0x4) ? mid.e[0] : currentBB.min.e[0];
			currentBB.min.e[1] = (child & 0x2) ? mid.e[1] : currentBB.min.e[1];
			currentBB.min.e[2] = (child & 0x1) ? mid.e[2] : currentBB.min.e[2];
			currentBB.max.e[0] = (child & 0x4) ? currentBB.max.e[0] : mid.e[0];
			currentBB.max.e[1] = (child & 0x2) ? currentBB.max.e[1] : mid.e[1];
			currentBB.max.e[2] = (child & 0x1) ? currentBB.max.e[2] : mid.e[2];
			mid = 0.5f*(currentBB.min + currentBB.max);
			
			VOXEL_FETCH(childIndex + (child^flag), 0, currentVoxel.low);
			VOXEL_FETCH(childIndex + (child^flag), 1, currentVoxel.high);
			if(currentVoxel.hasLink2Low()) requestCountList[currentVoxel.getLink2Low()]++;
			//bbSize *= 0.5f;

			continue;
		}

		// move to next sibiling
NEXT_SIBILING : 

		if(stackPtr < 1) return false;

		// get parent bb
		Vector3 size = currentBB.max - currentBB.min;
		currentBB.min.e[0] -= (child & 0x4) ? size.e[0] : 0.0f;
		currentBB.min.e[1] -= (child & 0x2) ? size.e[1] : 0.0f;
		currentBB.min.e[2] -= (child & 0x1) ? size.e[2] : 0.0f;
		currentBB.max.e[0] += (child & 0x4) ? 0.0f : size.e[0];
		currentBB.max.e[1] += (child & 0x2) ? 0.0f : size.e[1];
		currentBB.max.e[2] += (child & 0x1) ? 0.0f : size.e[2];

		// get stack top
		childIndex = _stack[stackPtr-1].childIndex;
		child = _stack[stackPtr-1].child;
		mid = 0.5f*(currentBB.min + currentBB.max);

#		define NEW_NODE(x, y, z, a, b, c) (((x) < (y) && (x) < (z)) ? (a) : (((y) < (z)) ? (b) : (c)))

		switch(child)
		{
		case 0 : child = NEW_NODE(mid.x, mid.y, mid.z, 4, 2, 1); break;
		case 1 : child = NEW_NODE(mid.x, mid.y, currentBB.max.z, 5, 3, 8); break;
		case 2 : child = NEW_NODE(mid.x, currentBB.max.y, mid.z, 6, 8, 3); break;
		case 3 : child = NEW_NODE(mid.x, currentBB.max.y, currentBB.max.z, 7, 8, 8); break;
		case 4 : child = NEW_NODE(currentBB.max.x, mid.y, mid.z, 8, 6, 5); break;
		case 5 : child = NEW_NODE(currentBB.max.x, mid.y, currentBB.max.z, 8, 7, 8); break;
		case 6 : child = NEW_NODE(currentBB.max.x, currentBB.max.y, mid.z, 8, 8, 7); break;
		case 7 : child = 8;	break;
		}
		_stack[stackPtr-1].child = child;

		if(child == 8)
		{
			if (--stackPtr == 0) break;

			child = _stack[stackPtr-1].child;

			goto NEXT_SIBILING;
		}
		
		VOXEL_FETCH(childIndex + (child^flag), 0, currentVoxel.low);
		VOXEL_FETCH(childIndex + (child^flag), 1, currentVoxel.high);
		if(currentVoxel.hasLink2Low()) requestCountList[currentVoxel.getLink2Low()]++;
		currentBB.min.e[0] = (child & 0x4) ? mid.e[0] : currentBB.min.e[0];
		currentBB.min.e[1] = (child & 0x2) ? mid.e[1] : currentBB.min.e[1];
		currentBB.min.e[2] = (child & 0x1) ? mid.e[2] : currentBB.min.e[2];
		currentBB.max.e[0] = (child & 0x4) ? currentBB.max.e[0] : mid.e[0];
		currentBB.max.e[1] = (child & 0x2) ? currentBB.max.e[1] : mid.e[1];
		currentBB.max.e[2] = (child & 0x1) ? currentBB.max.e[2] : mid.e[2];
		mid = 0.5f*(currentBB.min + currentBB.max);
	}
	return false;
}

__device__ inline bool RayOctreeIntersect2(const Ray &ray, HitPoint &hit, Material *material, int *hitIndex, float *hitBBSize, float limit, float tLimit, unsigned int seed, float *requestCountList, int x, int y, int frame2)
{
	float t0, t1;
	if(!ray.BoxIntersect(c_octreeHeader.min, c_octreeHeader.max, t0, t1)) return false;

	char flag = 0;

	Vector3 ori = ray.ori, dir = ray.dir;
	Vector3 temp = c_octreeHeader.max + c_octreeHeader.min;

	if(ray.dir.x < 0.0f)
	{
		ori.e[0] = temp.e[0] - ori.e[0];
		dir.e[0] = -dir.e[0];
		flag |= 4;
	}

	if(ray.dir.y < 0.0f)
	{
		ori.e[1] = temp.e[1] - ori.e[1];
		dir.e[1] = -dir.e[1];
		flag |= 2;
	}

	if(ray.dir.z < 0.0f)
	{
		ori.e[2] = temp.e[2] - ori.e[2];
		dir.e[2] = -dir.e[2];
		flag |= 1;
	}

	Ray transRay;
	transRay.set(ori, dir);

	AABB currentBB;
	currentBB.min = (c_octreeHeader.min - transRay.ori) * transRay.invDir;
	currentBB.max = (c_octreeHeader.max - transRay.ori) * transRay.invDir;

	int N = c_octreeHeader.dim;

	typedef struct tempStack_t
	{
		int childIndex;
		char child;
	} tempStack;

	Vector3 mid;
	Voxel currentVoxel;
	currentVoxel.setChildIndex(0);
	tempStack stack[100];
//	__shared__ tempStack stack[256][16];
//#	define _stack stack[threadIdx.x]
#	define _stack stack

	int stackPtr;

	mid = 0.5f*(currentBB.min + currentBB.max);

	stackPtr = 0;

	char currentIsLeaf = 0;
	char currentGeomBitmapDepth = 0;
	unsigned char currentGeomBitmap = 0;

	char child = 0;
	int childIndex = 0;

	//float bbSize;// = c_octreeHeader.max.x - c_octreeHeader.min.x;

	bool isEmpty;
	while(true)
	{
		if(c_frame2 != frame2) return false;
		
		switch(currentGeomBitmapDepth)
		{
		case 0 : isEmpty = currentVoxel.isEmpty(); break;
		case 1 : isEmpty = currentGeomBitmap == 0; break;
		case 2 : isEmpty = (currentGeomBitmap & (1u << (child^flag))) == 0; break;
		}

		if(!isEmpty && 
			currentBB.max.x > 0.0f && currentBB.max.y > 0.0f && currentBB.max.z > 0.0f &&
			currentBB.min.x < hit.t && currentBB.min.y < hit.t && currentBB.min.y < hit.t)
		{
			currentIsLeaf = currentVoxel.isLeaf();
			if(currentIsLeaf && currentGeomBitmapDepth == 0)
				if(hitIndex) *hitIndex = childIndex + (child^flag);

#			ifdef USE_VOXEL_LOD
			Vector3 mM = currentBB.max * transRay.dir + ray.ori;
			Vector3 mm = currentBB.min * transRay.dir + ray.ori;
			//float t0 = 0, t1 = 0.0f;
			//ray.BoxIntersect(mm, mM, t0, t1);
			float hitBBDiag = (mM - mm).length();
			float hitBBDist = (0.5f*(mM + mm) - ray.ori).length();
			/*
			if((currentGeomBitmapDepth == 2 || hitBBDiag / hitBBDist < limit) && x == 256 && y == 256)
			{
				printf("currentGeomBitmapDepth = %d, hitBBDiag = %f, hitBBDist = %f, val = %f (< %f)\n", currentGeomBitmapDepth, hitBBDiag, hitBBDist, hitBBDiag / hitBBDist, limit);
			}
			*/
			if(currentGeomBitmapDepth == 2 || hitBBDiag / hitBBDist < limit)
#			else
			if(currentGeomBitmapDepth == 2)
#			endif
			{
				if(hitBBSize) *hitBBSize = (currentBB.max.x - currentBB.min.x) * transRay.dir.x;
				//if(RayLeafVoxelIntersect(ray, transRay, currentBB, currentVoxel, hit, material, hit.t, seed, x, y))
				if(RayLODIntersect(ray, transRay, currentBB, currentVoxel, hit, material, hit.t, tLimit, seed, x, y))
				{
					//printf("[%d] size = %f %f %f\n", stackPtr, currentBB.max.x - currentBB.min.x, currentBB.max.y - currentBB.min.y, currentBB.max.z - currentBB.min.z);
					//hitBBSize = bbSize;
					/*
					if(x == 256 && y == 256)
					{
						Vector3 leafBB = (c_octreeHeader.max - c_octreeHeader.min) / (1 << stackPtr);
						printf("leafsize = %f %f %f -> %f\n", leafBB.e[0], leafBB.e[1], leafBB.e[2], leafBB.length());
					}
					*/
					return true;
				}
				goto NEXT_SIBILING;
			}
			// push down

			//FIRST_NODE(currentBB, mid, child);
			// get first child
			child = 0;
			if (currentBB.min.y < currentBB.min.x && currentBB.min.z < currentBB.min.x)
			{
				// YZ plane
				if (mid.y < currentBB.min.x) child |= 2;
				if (mid.z < currentBB.min.x) child |= 1;
			}
			else if (currentBB.min.z < currentBB.min.y)
			{
				// XZ plane
				if (mid.x < currentBB.min.y) child |= 4;
				if (mid.z < currentBB.min.y) child |= 1;
			}
			else
			{
				// XY plane
				if (mid.x < currentBB.min.z) child |= 4;
				if (mid.y < currentBB.min.z) child |= 2;
			}

			if(!currentIsLeaf)
				childIndex = currentVoxel.getChildIndex() * N * N * N;

			_stack[stackPtr].childIndex = childIndex;
			_stack[stackPtr++].child = child;

			if(currentIsLeaf) currentGeomBitmapDepth++;
			if(currentGeomBitmapDepth == 1) currentGeomBitmap = currentVoxel.geomBitmap[child^flag]; 

			currentBB.min.e[0] = (child & 0x4) ? mid.e[0] : currentBB.min.e[0];
			currentBB.min.e[1] = (child & 0x2) ? mid.e[1] : currentBB.min.e[1];
			currentBB.min.e[2] = (child & 0x1) ? mid.e[2] : currentBB.min.e[2];
			currentBB.max.e[0] = (child & 0x4) ? currentBB.max.e[0] : mid.e[0];
			currentBB.max.e[1] = (child & 0x2) ? currentBB.max.e[1] : mid.e[1];
			currentBB.max.e[2] = (child & 0x1) ? currentBB.max.e[2] : mid.e[2];
			mid = 0.5f*(currentBB.min + currentBB.max);

			if(!currentIsLeaf)
			{
				VOXEL_FETCH(childIndex + (child^flag), 0, currentVoxel.low);
				VOXEL_FETCH(childIndex + (child^flag), 1, currentVoxel.high);
				if(currentVoxel.hasLink2Low()) requestCountList[currentVoxel.getLink2Low()]++;
				//if(currentVoxel.hasLink2Low()) printf("%d -- %d %d\n", currentVoxel.childIndex, childIndex + (child^flag), currentVoxel.getLink2Low());
				//bbSize *= 0.5f;
			}

			continue;
		}

		// move to next sibiling
NEXT_SIBILING : 

		if(stackPtr < 1) return false;

		// get parent bb
		Vector3 size = currentBB.max - currentBB.min;
		currentBB.min.e[0] -= (child & 0x4) ? size.e[0] : 0.0f;
		currentBB.min.e[1] -= (child & 0x2) ? size.e[1] : 0.0f;
		currentBB.min.e[2] -= (child & 0x1) ? size.e[2] : 0.0f;
		currentBB.max.e[0] += (child & 0x4) ? 0.0f : size.e[0];
		currentBB.max.e[1] += (child & 0x2) ? 0.0f : size.e[1];
		currentBB.max.e[2] += (child & 0x1) ? 0.0f : size.e[2];

		// get stack top
		childIndex = _stack[stackPtr-1].childIndex;
		child = _stack[stackPtr-1].child;
		mid = 0.5f*(currentBB.min + currentBB.max);

#		define NEW_NODE(x, y, z, a, b, c) (((x) < (y) && (x) < (z)) ? (a) : (((y) < (z)) ? (b) : (c)))

		switch(child)
		{
		case 0 : child = NEW_NODE(mid.x, mid.y, mid.z, 4, 2, 1); break;
		case 1 : child = NEW_NODE(mid.x, mid.y, currentBB.max.z, 5, 3, 8); break;
		case 2 : child = NEW_NODE(mid.x, currentBB.max.y, mid.z, 6, 8, 3); break;
		case 3 : child = NEW_NODE(mid.x, currentBB.max.y, currentBB.max.z, 7, 8, 8); break;
		case 4 : child = NEW_NODE(currentBB.max.x, mid.y, mid.z, 8, 6, 5); break;
		case 5 : child = NEW_NODE(currentBB.max.x, mid.y, currentBB.max.z, 8, 7, 8); break;
		case 6 : child = NEW_NODE(currentBB.max.x, currentBB.max.y, mid.z, 8, 8, 7); break;
		case 7 : child = 8;	break;
		}
		_stack[stackPtr-1].child = child;

		if(child == 8)
		{
			if (--stackPtr == 0) break;

			if(--currentGeomBitmapDepth <= 0)
			{
				currentGeomBitmapDepth = 0;
				currentIsLeaf = false;
			}

			//if(!currentIsLeaf)
			//	bbSize *= 2.0f;
			
			child = _stack[stackPtr-1].child;
			
			goto NEXT_SIBILING;
		}

		if(!currentIsLeaf)
		{
			VOXEL_FETCH(childIndex + (child^flag), 0, currentVoxel.low);
			VOXEL_FETCH(childIndex + (child^flag), 1, currentVoxel.high);
			if(currentVoxel.hasLink2Low()) requestCountList[currentVoxel.getLink2Low()]++;
		}
		else
		{
			if(currentGeomBitmapDepth == 1) currentGeomBitmap = currentVoxel.geomBitmap[child^flag]; 
		}
		currentBB.min.e[0] = (child & 0x4) ? mid.e[0] : currentBB.min.e[0];
		currentBB.min.e[1] = (child & 0x2) ? mid.e[1] : currentBB.min.e[1];
		currentBB.min.e[2] = (child & 0x1) ? mid.e[2] : currentBB.min.e[2];
		currentBB.max.e[0] = (child & 0x4) ? currentBB.max.e[0] : mid.e[0];
		currentBB.max.e[1] = (child & 0x2) ? currentBB.max.e[1] : mid.e[1];
		currentBB.max.e[2] = (child & 0x1) ? currentBB.max.e[2] : mid.e[2];
		mid = 0.5f*(currentBB.min + currentBB.max);
	}
	return false;
}


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

	if(numGatheredPhotons > 0)
	{
		color = flux / (radius2 * PI);// / numGatheredPhotons;
	}
}

#define USE_OCTREE
__device__ inline bool
trace(const Ray &ray, const Vector3 &prevHitN, float4 &color, HitPoint &hit, Material &hitMat, const Vector3 &brdf, float4 &attenuation, unsigned int seed, unsigned int seed2, int frame2, bool computed, int depth, float *requestCountList, int x, int y)
{
	bool hasHit = computed;
	//int hitIndex;
	//float hitBBSize;
#	ifndef TEST_VOXEL
	if(!computed)
#	endif
	{
#		ifndef TEST_VOXEL
		if(depth > 0)
#		endif
		{
			hit.t = FLT_MAX;
#			ifdef USE_OCTREE
			hasHit = RayOctreeIntersect(ray, hit, NULL, NULL, NULL, FLT_MAX, seed, requestCountList, x, y, frame2);
#			else
			hasHit = RaySceneBVHIntersect(ray, hit, 0.0f);
			if(hasHit) getMaterial(hitMat, hit);
#			endif
		}
	}
#	ifndef TEST_VOXEL
	else
	{
		// ray offseting
		//hit.t -= 0.1f;
		//printf("model = %d, offset = %d, id = %d\n", hit.model, c_offsetMats[hit.model], hit.material);
		getMaterial(hitMat, hit);

	}
#	endif
	//hit.t = FLT_MAX;
	//hasHit = RayOctreeIntersect(ray, NULL, 0, hit, 0.0f, 0.0f);

	if(!hasHit && !computed)
	{
		if(c_hasEnvMap)
		{
			float4 envMapColor;
			shadeEnvironmentMap(ray.dir, envMapColor);
			color.x += envMapColor.x;
			color.y += envMapColor.y;
			color.z += envMapColor.z;
	
		}
		else
		{
			color.x += c_envCol.x;
			color.y += c_envCol.y;
			color.z += c_envCol.z;
		}
		color = color * attenuation;
		return false;
	}

	//getMaterial(hitMat, hit);
	//if(!hitMat.isDiffuse(seed)) return true;

	Vector3 hitPoint = ray.ori + hit.t*ray.dir;

	Vector3 shadowDir;
	Vector3 samplePos;
	float cosFac;
	
//#	define USE_REVERSE_SHADOW_RAY

	for(int i=0;i<c_scene.numEmitters;i++)
	//int i = c_emitterUsageHistogram[seed % EMITTER_SAMPLE_RESOLUTION];
	//for(int i=0;i<c_controller.numShadowRays;i++)
	{
		const Emitter &emitter = c_emitterList[i];
		/*
		int e = 0;
		if(i == 1)
		{
			e = c_manualEmitterSelection[seed % c_numSelectedEmitters];
		}
		const Emitter &emitter = c_emitterList[e];
		*/

		//float limit = (emitter.planar.v1 + emitter.planar.v2).length() / (0.5f*(emitter.planar.v1 + emitter.planar.v2) + emitter.planar.corner - hitPoint).length() * 0.5f;

		//for(int j=0;j<c_controller.numShadowRays;j++)
		{
			samplePos = emitter.sample(seed2);
			lcg(seed2);

#			ifdef USE_REVERSE_SHADOW_RAY
			shadowDir = hitPoint- samplePos;	
#			else
			shadowDir = samplePos - hitPoint;
#			endif
			shadowDir = shadowDir.normalize();

			const Vector3 &lightAmbient = emitter.color_Ka;
			const Vector3 &lightDiffuse = emitter.color_Kd;
			
			cosFac = shadowDir.dot(hit.n);
#			ifdef USE_REVERSE_SHADOW_RAY
			cosFac *= -1.0f;
#			endif

			float cosFac2 = 1.0f;
			if(emitter.isCosLight)
				cosFac2 = -shadowDir.dot(emitter.planar.normal);

			if(cosFac > 0.0f && cosFac2 > 0.0f)
			{
				// cast shadow rayz
				Ray shadowRay;
				int idx = shadowDir.indexOfMaxComponent();
#				ifdef USE_REVERSE_SHADOW_RAY
				float tLimit = (hitPoint.e[idx] - samplePos.e[idx]) / shadowDir.e[idx];
#				else
				float tLimit = (samplePos.e[idx] - hitPoint.e[idx]) / shadowDir.e[idx];
#				endif
				
				HitPoint shadowHit;
				shadowHit.t = tLimit;
				shadowHit.n = prevHitN;
				
				
				//float emitterLength = (emitter.planar.v1 + emitter.planar.v2).length();

#				ifdef USE_REVERSE_SHADOW_RAY
				shadowRay.set(samplePos, shadowDir);
#				else
				// manuall setting for boeing scene
				shadowRay.set(hitPoint + shadowDir*(0.5f*c_leafVoxelSize*OOCVOXEL_SUPER_RESOLUTION), shadowDir);
#				endif

				Material tempMat;
#				ifdef USE_OCTREE
				if(!c_controller.useShadowRays || !RayOctreeIntersect2(shadowRay, shadowHit, &tempMat, NULL, NULL, 0.0f, tLimit, seed, requestCountList, x, y, frame2))
#				else
				if(!c_controller.useShadowRays || !RaySceneBVHIntersect(shadowRay, shadowHit, NULL, tLimit))
#				endif
				{
					color += ((lightAmbient * hitMat.mat_Ka + lightDiffuse * hitMat.mat_Kd * cosFac * cosFac2) * brdf * attenuation);// / c_controller.numShadowRays;
				}
			}
			lcg(seed);
		}
	}
#	ifdef LOCAL_GATHERING
	Ray gatheringRay;
	bool isDiffuse;
	Voxel hitVoxel;
#	define NUM_GATHER (c_limit*20+1)
	Vector3 gatherColor(0.0f);
	for(int i=0;i<NUM_GATHER;i++)
	{
	isDiffuse = material.isDiffuse(seed2);
	gatheringRay.set(hitPoint, hitMat.sampleDirection(hit.n, ray.dir, seed2));
	lcg(seed);
	lcg(seed2);
	if(RayOctreeIntersect(gatheringRay, NULL, 0, hitVoxel, c_limit, 0.0f))
	{
		gatherColor += hitVoxel.col * hit.n.dot(gatheringRay.dir) * attenuation;
	}
	}
	color += gatherColor / NUM_GATHER;
#	else
#	endif

	return true;
}

__global__ void
Render(int iter, unsigned char *image, int frame, int frame2, HitPoint *hitCache, ExtraRayInfo *extRayCache, SampleData *sampleData, float *requestCountList)
{
	float4 outColor = make_float4(0.0f, 0.0f, 0.0f, 1.0f);

	int threadID = iter*NUM_MAX_THREADS + blockIdx.x * blockDim.x + threadIdx.x;
#	ifdef USE_WARP_OPTIMIZATION
	int offsetRay = threadID / c_controller.warpSizeS;
#	else
	int offsetRay = threadID / c_controller.numShadowRays;
#	endif
	const int x = (int)extRayCache[offsetRay].pixelX;
	const int y = (int)extRayCache[offsetRay].pixelY;
	int offset2 = y*c_imgWidth + x;

	//unsigned int seed = tea<16>(offsetRay, frame);
	unsigned int seed = extRayCache[offsetRay].seed;
	unsigned int seed2 = tea<16>(seed, threadID % c_controller.numShadowRays);

	SampleData &sample = sampleData[offset2];

	/*
	c_camera.getRayWithOrigin(ray, 
		(x + rnd(seed))/c_imgWidth, 
		(y + rnd(seed))/c_imgHeight);
	*/

	if(frame <= 1)
	{
		//sample.reset();
	}
	
	Ray curRay;
#	ifdef USE_CPU_GLOSSY
	curRay = rayCache[offsetRay];
#	else
	curRay.set(c_camera.eye, hitCache[offsetRay].x - c_camera.eye);
#	endif
	bool hasBounce = false;
	HitPoint hit;
	Material material;
	float4 attenuation = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
	Vector3 brdf;
	int maxDepth = c_controller.pathLength;
	int depth = 0;
	Vector3 prevHitN;
	float4 color = make_float4(0.0f, 0.0f, 0.0f, 1.0f);

	// trace with emitters
	// purpose: visualize emitters

	// intersection test with two triangles
	// this may be inefficient but can reuse codes from ray-triangle intersection test
	Triangle tris[2];
	Vector3 p[4];
	float t, vdot, vdot2;
	float tmax = hitCache[offsetRay].t;
	float u0, u1, u2, v0, v1, v2;
	float point[2];
	float alpha, beta;
	for(int i=0;c_controller.showLights && i<c_scene.numEmitters;i++)
	{
		const Emitter &emitter = c_emitterList[i];
		if(emitter.type != Emitter::PARALLELOGRAM_LIGHT) continue;

		// setup triangles
		p[0] = emitter.planar.corner + emitter.planar.v1;
		p[1] = emitter.planar.corner;
		p[2] = emitter.planar.corner + emitter.planar.v2;
		p[3] = emitter.planar.corner + emitter.planar.v1 + emitter.planar.v2;
		tris[0].p[0] = 0;
		tris[0].p[1] = 1;
		tris[0].p[2] = 2;
		tris[1].p[0] = 2;
		tris[1].p[1] = 3;
		tris[1].p[2] = 0;

		for(int j=0;j<2;j++)
		{
			Triangle &tri = tris[j];

			int oriP[3] = {tri.p[0], tri.p[1], tri.p[2]};

			tri.n = (p[1] - p[0]).cross(p[2] - p[0]);
			tri.n.normalize();
 
			tri.d = p[0].dot(tri.n);

			// find best projection plane (YZ, XZ, XY)
			if (fabs(tri.n[0]) > fabs(tri.n[1]) && fabs(tri.n[0]) > fabs(tri.n[2])) {								
				tri.i1 = 1;
				tri.i2 = 2;
			}
			else if (fabs(tri.n[1]) > fabs(tri.n[2])) {								
				tri.i1 = 0;
				tri.i2 = 2;
			}
			else {								
				tri.i1 = 0;
				tri.i2 = 1;
			}

			int firstIdx;
			float u1list[3];
			u1list[0] = fabs(p[1].e[tri.i1] - p[0].e[tri.i1]);
			u1list[1] = fabs(p[2].e[tri.i1] - p[1].e[tri.i1]);
			u1list[2] = fabs(p[0].e[tri.i1] - p[2].e[tri.i1]);

			if (u1list[0] >= u1list[1] && u1list[0] >= u1list[2])
				firstIdx = 0;
			else if (u1list[1] >= u1list[2])
				firstIdx = 1;
			else
				firstIdx = 2;

			int secondIdx = (firstIdx + 1) % 3;
			int thirdIdx = (firstIdx + 2) % 3;

			// apply coordinate order to tri structure:
			tri.p[0] = oriP[firstIdx];
			tri.p[1] = oriP[secondIdx];
			tri.p[2] = oriP[thirdIdx];		

			// is ray parallel to plane or a back face ?
			vdot = curRay.dir.dot(tri.n);

			if(vdot == 0.0f) continue;

			// find parameter t of ray -> intersection point
			vdot2 = curRay.ori.dot(tri.n);
			t = (tri.d - vdot2) / vdot;

			// if either too near or further away than a previous hit, we stop
			if (t < INTERSECT_EPSILON || t > tmax + INTERSECT_EPSILON)
				continue;

			// intersection point with plane
			point[0] = curRay.ori.e[tri.i1] + curRay.dir.e[tri.i1] * t;
			point[1] = curRay.ori.e[tri.i2] + curRay.dir.e[tri.i2] * t;

			// begin barycentric intersection algorithm 
			const Vector3 &tri_p0 = p[tri.p[0]];
			const Vector3 &tri_p1 = p[tri.p[1]]; 
			const Vector3 &tri_p2 = p[tri.p[2]];

			float p0_1 = tri_p0.e[tri.i1], p0_2 = tri_p0.e[tri.i2]; 
			u0 = point[0] - p0_1; 
			v0 = point[1] - p0_2; 
			u1 = tri_p1[tri.i1] - p0_1; 
			v1 = tri_p1[tri.i2] - p0_2; 
			u2 = tri_p2[tri.i1] - p0_1; 
			v2 = tri_p2[tri.i2] - p0_2;

			beta = (v0 * u1 - u0 * v1) / (v2 * u1 - u2 * v1);
			//if (beta < 0 || beta > 1)
			if (beta < 0.0f || beta > 1.0f)
				continue;
			alpha = (u0 - beta * u2) / u1;	
			
			// not in triangle ?	
			if (alpha < 0.0f || (alpha + beta) > 1.0f)
				continue;
			
			outColor = make_float4(emitter.color_Kd.x, emitter.color_Kd.y, emitter.color_Kd.z, 1.0f);

			goto SHADE;
		}
	}
	
	if(extRayCache[offsetRay].hasHit()) hit = hitCache[offsetRay];

	prevHitN = hit.n;
	hasBounce = trace(curRay, prevHitN, color, hit, material, Vector3(1.0f), attenuation, seed, seed2, frame2, extRayCache[offsetRay].hasHit(), depth, requestCountList, x, y);

	if(extRayCache[offsetRay].wasBounced() || !extRayCache[offsetRay].isSpecular()) 
		outColor = color;

	while(++depth < maxDepth && hasBounce)
	{
		prevHitN = hit.n;
		curRay.set(hit.t * curRay.dir + curRay.ori, material.sampleDirection(hit.n, curRay.dir, seed));
		//attenuation = attenuation*material.mat_Kd.maxComponent();
		brdf = material.brdf(seed);
		hasBounce = trace(curRay, prevHitN, color, hit, material, brdf, attenuation, seed, seed2, frame2, false, depth, requestCountList, x, y);
		outColor += (color * brdf) * prevHitN.dot(curRay.dir);
	}

	if(c_frame2 != frame2) return;

SHADE:

	/*
	// clamp
	outColor.x = fminf(outColor.x, 1.0f);
	outColor.y = fminf(outColor.y, 1.0f);
	outColor.z = fminf(outColor.z, 1.0f);
	*/

#	ifndef VIS_PHOTONS
#	ifdef USE_GAUSSIAN_RECONSTRUCTION
	Vector3 wColor;
	for(unsigned int y2=y;y2<y+2;y2++)
		for(unsigned int x2=x;x2<x+2;x2++)
		{
			if(y2 >= c_imgHeight || x2 >= c_imgWidth) continue;
			unsigned int offset = y2*c_imgWidth + x2;

			float dist2 = (extRayCache[offsetRay].pixelX - (float)x2)*(extRayCache[offsetRay].pixelX - (float)x2);
			dist2 += (extRayCache[offsetRay].pixelY - (float)y2)*(extRayCache[offsetRay].pixelY - (float)y2);
			
			float weight = exp(-1.0f*dist2*4);

			SampleData &sample = sampleData[offset];
			//sample.color1 += outColor * weight;
			//sample.numIntergration += weight;
			wColor = outColor * weight;
			atomicAdd(&sample.color1.e[0], wColor.e[0]);
			atomicAdd(&sample.color1.e[1], wColor.e[1]);
			atomicAdd(&sample.color1.e[2], wColor.e[2]);
			atomicAdd(&sample.numIntergration, weight);
		}
#	else
#	ifdef USE_ATOMIC_OPERATIONS
	atomicAdd(&sample.color1.e[0], outColor.x);
	atomicAdd(&sample.color1.e[1], outColor.y);
	atomicAdd(&sample.color1.e[2], outColor.z);
	atomicAdd(&sample.numIntergration, 1.0f);// / c_controller.numShadowRays);
#	else
	sample.color1 += outColor;
	sample.numIntergration++;
#	endif
#	endif
#	endif

	sample.isHit = false;

	if(hasBounce)
	{
		sample.isHit = true;
		sample.hitPoint = curRay.ori + hit.t*curRay.dir;
		sample.hitNormal = hit.n;
		sample.hitMat.getSampler(material);
		sample.inDir = curRay.dir;
		sample.brdf = material.brdf(seed, material.isDiffuse(seed));

		sample.hasHit = true;
		sample.summedHitPoint += sample.hitPoint;
		sample.summedHitNormal += sample.hitNormal;
	}
}

__global__ void
AmbientOcclusion(unsigned char *image, int frame2, HitPoint *hitCache, ExtraRayInfo *extRayCache, SampleData *sampleData, float *requestCountList)
{
	Vector3 outColor(0.0f);

	int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	int offsetRay = threadID / c_controller.warpSizeS;
	const int x = (int)extRayCache[offsetRay].pixelX;
	const int y = (int)extRayCache[offsetRay].pixelY;
	int offset2 = y*c_imgWidth + x;

	//unsigned int seed = tea<16>(offsetRay, frame);
	unsigned int seed = extRayCache[offsetRay].seed;

	SampleData &sample = sampleData[offset2];

	if(extRayCache[offsetRay].hasHit())
	{
		HitPoint hit = hitCache[offsetRay];

		for(int i=0;i<2;i++)
		{
			Vector3 dir = Material::sampleDiffuseDirection(hit.n, seed);
			Vector3 ori = hit.x + dir*(0.5f*c_leafVoxelSize*OOCVOXEL_SUPER_RESOLUTION);
		
			Ray AORay;
			AORay.set(ori, dir);

			hit.t = c_controller.AODistance;

			sample.numAOSample++;
			if(RayOctreeIntersect2(AORay, hit, NULL, NULL, NULL, hit.t, 0.0f, seed, requestCountList, x, y, frame2))
			{
				sample.numAOHit++;
			}
			lcg(seed);
		}
	}

	
	/*

	bool hasBounce = false;
	HitPoint hit;
	Material material;
	Vector3 attenuation(1.0f, 1.0f, 1.0f);
	Vector3 brdf;
	bool isDiffuse = false;
	int maxDepth = c_controller.pathLength;
	int depth = 0;
	Vector3 prevHitN;

	if(extRayCache[offsetRay].hasHit()) hit = hitCache[offsetRay];

	Vector3 color(0.0f);
	prevHitN = hit.n;
	hasBounce = trace(curRay, prevHitN, color, hit, material, Vector3(1.0f), attenuation, seed, seed2, extRayCache[offsetRay].hasHit(), depth, requestCountList, x, y);

	if(extRayCache[offsetRay].wasBounced() || !extRayCache[offsetRay].isSpecular()) 
		outColor = color;

	while(++depth < maxDepth && hasBounce)
	{
		prevHitN = hit.n;
		curRay.set(hit.t * curRay.dir + curRay.ori, material.sampleDirection(hit.n, curRay.dir, seed));
		//attenuation = attenuation*material.mat_Kd.maxComponent();
		brdf = material.brdf(seed);
		hasBounce = trace(curRay, prevHitN, color, hit, material, brdf, attenuation, seed, seed2, false, depth, requestCountList, x, y);
		outColor += (color * brdf) * prevHitN.dot(curRay.dir);
	}

	if(c_frame2 != frame2) return;

SHADE:

#	ifdef USE_ATOMIC_OPERATIONS
	atomicAdd(&sample.color1.e[0], outColor.e[0]);
	atomicAdd(&sample.color1.e[1], outColor.e[1]);
	atomicAdd(&sample.color1.e[2], outColor.e[2]);
	atomicAdd(&sample.numIntergration, 1);
#	else
	sample.color1 += outColor;
	sample.numIntergration++;
#	endif

	sample.isHit = false;

	if(hasBounce)
	{
		sample.isHit = true;
		sample.hitPoint = curRay.ori + hit.t*curRay.dir;
		sample.hitNormal = hit.n;
		sample.hitMat.getSampler(material);
		sample.inDir = curRay.dir;
		sample.brdf = material.brdf(seed, material.isDiffuse(seed));

		sample.hasHit = true;
		sample.summedHitPoint += sample.hitPoint;
		sample.summedHitNormal += sample.hitNormal;
	}
	*/
}

/*
__global__ void
GatherPhotonsWithJittering(int iteration, int totalIteration, SampleData *sampleData, unsigned int seed)
{	
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int jx = (x + lcg(seed) * 2) % c_imgWidth;
	const unsigned int jy = (y + lcg(seed) * 2) % c_imgHeight;
	int offsetRay = jy*c_imgWidth + jx;

	Ray gatheringRay;
	Voxel hitVoxel;
	bool isDiffuse;
	
	int warp = x / 8 + (y / 4) * (c_imgWidth / 8);

	if(sampleData[offsetRay].isHit)
	{
		//unsigned int seed = tea<16>(warp, totalIteration);
		unsigned int seed = tea<16>(offsetRay, totalIteration);

		const MaterialSampler &mat = sampleData[offsetRay].hitMat;

		gatheringRay.set(sampleData[offsetRay].hitPoint, Material::sampleDirection(mat.rangeKd, mat.rangeKs, mat.mat_d, mat.mat_Ns, sampleData[offsetRay].hitNormal, sampleData[offsetRay].inDir, seed));
		//gatheringRay.set(sampleData[offsetRay].hitPoint, Material::sampleDiffuseDirection(sampleData[offsetRay].hitNormal, seed));
		//gatheringRay.set(sampleData[offsetRay].hitPoint, Material::sampleDeterminedDiffuseDirection(sampleData[offsetRay].hitNormal, iteration));
		//if(RayOctreeIntersect(gatheringRay, NULL, 0, hitVoxel, c_limit, 0.0f))
		HitPoint hit;
		hit.t = FLT_MAX;
		//Material material;
		int hitIndex;
		//float hitBBSize;
		if(RayOctreeIntersect(gatheringRay, hit, NULL, &hitIndex, NULL, 0.0f, FLT_MAX, seed, 0, 0, 0))
		{
			PhotonVoxel pv;
			PHOTON_VOXEL_FETCH(hitIndex, 0, pv.data);
			sampleData[offsetRay].color2 += pv.power * sampleData[offsetRay].brdf * sampleData[offsetRay].hitNormal.dot(gatheringRay.dir);
			//sampleData[offsetRay].color2 += hitVoxel.getKd() * sampleData[offsetRay].brdf * sampleData[offsetRay].hitNormal.dot(gatheringRay.dir);
		}
	}
}
*/

__global__ void
GatherPhotons(int iter, int totalIteration, int frame2, ExtraRayInfo *extRayCache, SampleData *sampleData, float *requestCountList)
{	
	int threadID = iter * NUM_MAX_THREADS + blockIdx.x * blockDim.x + threadIdx.x;

#	ifdef USE_WARP_OPTIMIZATION
	int offsetRay = threadID/c_controller.warpSizeG;
#	else
	int offsetRay = threadID/c_controller.numGatheringRays;
#	endif
	//int gran = blockDim.x / c_controller.warpSizeG;
	//int offsetRay = (threadID % gran) + (gran * blockIdx.x);
#	ifdef USE_2ND_RAYS_FILTER
	const int x = (int)(extRayCache[offsetRay].pixelX / TILE_SIZE) * TILE_SIZE;
	const int y = (int)(extRayCache[offsetRay].pixelY / TILE_SIZE) * TILE_SIZE;
#	else
	const int x = (int)extRayCache[offsetRay].pixelX;
	const int y = (int)extRayCache[offsetRay].pixelY;
#	endif
	int offsetRay2 = y*c_imgWidth + x;

	Ray gatheringRay;

	//totalIteration = iteration;
	float div = 2.0f;
	Vector3 outColor(0.0f);

	if(sampleData[offsetRay2].isHit)
	{
#		ifdef USE_2ND_RAYS_FILTER
		unsigned int seed = tea<16>(offsetRay, totalIteration);
#		else
		//unsigned int seed = tea<16>(offsetRay2, totalIteration);
		unsigned int seed = tea<16>(extRayCache[offsetRay].seed, threadID % c_controller.numGatheringRays);
#		endif

		const MaterialSampler &mat = sampleData[offsetRay2].hitMat;

		Vector3 ori, dir;
		if(extRayCache[offsetRay].wasBounced()) 
			dir = Material::sampleDiffuseDirection(sampleData[offsetRay2].hitNormal, seed);
		else
		{
			
			if(c_controller.warpSizeG > 1)
				dir = Material::sampleDirectionWithJitter(mat.rangeKd, mat.rangeKs, mat.mat_d, mat.mat_Ns, sampleData[offsetRay2].hitNormal, sampleData[offsetRay2].inDir, seed, tea<16>(threadID, seed));
			else
				dir = Material::sampleDirection(mat.rangeKd, mat.rangeKs, mat.mat_d, mat.mat_Ns, sampleData[offsetRay2].hitNormal, sampleData[offsetRay2].inDir, seed);
		}
		seed = lcg(seed);
		
		if(c_controller.warpSizeG > 1)
		{
			unsigned int seed2 = tea<16>(threadID, totalIteration);
			Vector3 randDir((rnd(seed2)-0.5f)/div, (rnd(seed2)-0.5f)/div, (rnd(seed2)-0.5f)/div);
			dir = dir + randDir;
			dir = dir.normalize();
		}
		
		//ori = sampleData[offsetRay2].hitPoint + dir*(0.5f*c_leafVoxelSize*0.5f);
		ori = sampleData[offsetRay2].hitPoint + dir*(0.5f*c_leafVoxelSize*OOCVOXEL_SUPER_RESOLUTION);
		//ori = sampleData[offsetRay2].hitPoint;
		gatheringRay.set(ori, dir);
		//gatheringRay.set(sampleData[offsetRay2].hitPoint, Material::sampleDirection(mat.rangeKd, mat.rangeKs, mat.mat_d, mat.mat_Ns, sampleData[offsetRay2].hitNormal, sampleData[offsetRay2].inDir, seed));
#		ifdef VIS_PHOTONS
		gatheringRay.set(c_camera.eye, sampleData[offsetRay2].hitPoint - c_camera.eye);
#		endif
		HitPoint hit;
		hit.t = FLT_MAX;
		Material material;
		int hitIndex;
		Vector3 cf = sampleData[offsetRay2].brdf * sampleData[offsetRay2].hitNormal.dot(gatheringRay.dir);

		float hitBBSize;
		if(RayOctreeIntersect(gatheringRay, hit, &material, &hitIndex, &hitBBSize, FLT_MAX, seed, requestCountList, x+1000, y, frame2))
		{
			PhotonVoxel pv, pv2;
			PHOTON_VOXEL_FETCH(hitIndex, 0, pv.data);
			PHOTON_VOXEL2_FETCH(hitIndex, 0, pv2.data);
			pv.power = c_photonMorphingRatio*pv.power + (1.0f - c_photonMorphingRatio)*pv2.power;
			pv.power = pv.power * PHOTON_INTENSITY_SCALING_FACTOR;

#			ifdef USE_VOXEL_LOD
			Vector3 sceneBB = c_octreeHeader.max - c_octreeHeader.min;
			float leafBBSize = sceneBB.x / (c_octreeHeader.dim << (c_octreeHeader.maxDepth - 1));
			pv.power = pv.power / (hitBBSize*hitBBSize/leafBBSize/leafBBSize);
#			endif
#			ifdef VIS_PHOTONS
			outColor = pv.power;
#			else
			outColor = pv.power * material.mat_Kd * cf;

			//if(outColor.x > 0.0f || outColor.y > 0.0f || outColor.z > 0.0f)
			//printf("%f %f %f - %f %f %f - %f %f %f\n", outColor.x, outColor.y, outColor.z, pv.power.x, pv.power.y, pv.power.z, cf.x, cf.y, cf.z);
			//printf("%f %f %f\n", outColor.x, outColor.y, outColor.z);
#			endif
			//outColor = pv.power;
			//sampleData[offsetRay2].color2 += pv.power * cf;
		}
		else
		{
			float4 envMapColor = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
			if(c_hasEnvMap)
			{
				shadeEnvironmentMap(gatheringRay.dir, envMapColor);
			}
			outColor.x = cf.x * (envMapColor.x * c_controller.envMapWeight + c_envCol.x * c_controller.envColWeight);
			outColor.y = cf.y * (envMapColor.y * c_controller.envMapWeight + c_envCol.y * c_controller.envColWeight);
			outColor.z = cf.z * (envMapColor.z * c_controller.envMapWeight + c_envCol.z * c_controller.envColWeight);
		}
	}

	// clamp
	//outColor.x = fminf(outColor.x, 1.0f);
	//outColor.y = fminf(outColor.y, 1.0f);
	//outColor.z = fminf(outColor.z, 1.0f);

#	if 0//ifdef USE_GAUSSIAN_RECONSTRUCTION
	Vector3 wColor;
	for(int y2=y;y2<y+2;y2++)
		for(int x2=x;x2<x+2;x2++)
		{
			if(y2 >= c_imgHeight || x2 >= c_imgWidth) continue;
			int offset = y2*c_imgWidth + x2;

			float dist2 = (extRayCache[offsetRay].pixelX - (float)x2)*(extRayCache[offsetRay].pixelX - (float)x2);
			dist2 += (extRayCache[offsetRay].pixelY - (float)y2)*(extRayCache[offsetRay].pixelY - (float)y2);
			
			float weight = exp(-1.0f*dist2*4);

			SampleData &sample = sampleData[offset];
			//sample.color1 += outColor * weight;
			//sample.numIntergration += weight;
			wColor = outColor * weight;
#				ifdef USE_ATOMIC_OPERATIONS
			atomicAdd(&sample.color2.e[0], wColor.e[0]);
			atomicAdd(&sample.color2.e[1], wColor.e[1]);
			atomicAdd(&sample.color2.e[2], wColor.e[2]);
			atomicAdd(&sample.numIteration, weight);
#				else
			sample.color2 += wColor;
			sample.numIteration += weight;
#				endif
		}
#	else
#	ifdef USE_ATOMIC_OPERATIONS
	atomicAdd(&sampleData[offsetRay2].color2.e[0], outColor.e[0]);
	atomicAdd(&sampleData[offsetRay2].color2.e[1], outColor.e[1]);
	atomicAdd(&sampleData[offsetRay2].color2.e[2], outColor.e[2]);
	atomicAdd(&sampleData[offsetRay2].numIteration, 1.0f);
#	else
	sampleData[offsetRay2].color2 += outColor;
	sampleData[offsetRay2].numIteration++;
#	endif
#	endif
}

__global__ void
ApplyToImage(unsigned char *image, SampleData *sampleData, ExtraRayInfo *extRayCache, int frame2)
{
	int offsetRay = blockIdx.x * blockDim.x + threadIdx.x;
	const int x = (int)extRayCache[offsetRay].pixelX;
	const int y = (int)extRayCache[offsetRay].pixelY;
	
	int offsetRay2 = y*c_imgWidth + x;

#	ifdef USE_2ND_RAYS_FILTER
	const int xo = (int)(extRayCache[offsetRay].pixelX / TILE_SIZE) * TILE_SIZE;
	const int yo = (int)(extRayCache[offsetRay].pixelY / TILE_SIZE) * TILE_SIZE;
	int offsetRay2o = yo*c_imgWidth + xo;
#	endif

	Vector3 color = sampleData[offsetRay2].getFinalColor((c_controller.gatherPhotons << 1) | c_controller.shadeLocalIllumination);

	color.e[0] = fminf(color.e[0], 1.0f);
	color.e[1] = fminf(color.e[1], 1.0f);
	color.e[2] = fminf(color.e[2], 1.0f);
	color.e[0] = fmaxf(color.e[0], 0.0f);
	color.e[1] = fmaxf(color.e[1], 0.0f);
	color.e[2] = fmaxf(color.e[2], 0.0f);

	image[offsetRay2*c_imgBpp + 0] = (unsigned char)(color.e[0] * 255);
	image[offsetRay2*c_imgBpp + 1] = (unsigned char)(color.e[1] * 255);
	image[offsetRay2*c_imgBpp + 2] = (unsigned char)(color.e[2] * 255);
	if(c_imgBpp == 4) image[offsetRay2*c_imgBpp+ 3] = (unsigned char)(1.0f * 255);
}

extern "C" void unloadSceneTReX()
{
	unloadSceneCUDA();
	
	checkCudaErrors(cudaFree(d_PhotonList));
	checkCudaErrors(cudaFree(d_octree));
	checkCudaErrors(cudaFree(d_photonOctree));
	checkCudaErrors(cudaFree(d_photonOctree2));
	checkCudaErrors(cudaFree(d_oocVoxels));
	checkCudaErrors(cudaFree(d_material));

	if(h_oocVoxels) delete[] h_oocVoxels;
}

extern "C" void materialChangedTReX(Scene *scene)
{
	materialChangedCUDA(scene);
	checkCudaErrors(cudaMemcpy(d_material, scene->models[0].mats, scene->models[0].numMats*sizeof(Material), cudaMemcpyHostToDevice));
}

extern "C" void lightChangedTReX(Scene *scene)
{
	lightChangedCUDA(scene);
}

extern "C" void loadSceneTReX(Scene *scene, OctreeHeader *octreeHeader, int numVoxels, Voxel *octree, int numVoxelsForOOC)
{
	loadSceneCUDA(scene);

	h_numVoxels = numVoxels;
	h_sizeVoxelBuffer = numVoxels+numVoxelsForOOC;
	h_octree = octree;
	h_octreeHeader = *octreeHeader;
	
	checkCudaErrors(cudaMalloc((void **)&d_octree, h_sizeVoxelBuffer*sizeof(Voxel)));
	checkCudaErrors(cudaMemcpy(d_octree, octree, numVoxels*sizeof(Voxel), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaBindTexture((size_t *)0, t_octree, d_octree, h_sizeVoxelBuffer*sizeof(Voxel)));
	checkCudaErrors(cudaMemcpyToSymbol(c_octreeHeader, octreeHeader, sizeof(OctreeHeader), 0, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(c_numVoxels, &numVoxels, sizeof(int), 0, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void **)&d_photonOctree, h_sizeVoxelBuffer*sizeof(PhotonVoxel)));
	checkCudaErrors(cudaMalloc((void **)&d_photonOctree2, h_sizeVoxelBuffer*sizeof(PhotonVoxel)));
	checkCudaErrors(cudaMemset(d_photonOctree, 0, h_sizeVoxelBuffer*sizeof(PhotonVoxel)));
	checkCudaErrors(cudaMemset(d_photonOctree2, 0, h_sizeVoxelBuffer*sizeof(PhotonVoxel)));
	checkCudaErrors(cudaBindTexture((size_t *)0, t_photonOctree, d_photonOctree, h_sizeVoxelBuffer*sizeof(PhotonVoxel)));
	checkCudaErrors(cudaBindTexture((size_t *)0, t_photonOctree2, d_photonOctree2, h_sizeVoxelBuffer*sizeof(PhotonVoxel)));

	checkCudaErrors(cudaMalloc((void **)&d_material, scene->models[0].numMats*sizeof(Material)));
	checkCudaErrors(cudaMemcpy(d_material, scene->models[0].mats, scene->models[0].numMats*sizeof(Material), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaBindTexture((size_t *)0, t_material, d_material, scene->models[0].numMats*sizeof(Material)));

	float len[3] = {octreeHeader->max.x - octreeHeader->min.x, octreeHeader->max.y - octreeHeader->min.y, octreeHeader->max.z - octreeHeader->min.z};
	len[0] /= (octreeHeader->dim << (octreeHeader->maxDepth - 1)) / 2;
	len[1] /= (octreeHeader->dim << (octreeHeader->maxDepth - 1)) / 2;
	len[2] /= (octreeHeader->dim << (octreeHeader->maxDepth - 1)) / 2;
	float leafVoxelSize = sqrtf(len[0]*len[0] + len[1]*len[1] + len[2]*len[2]);
	checkCudaErrors(cudaMemcpyToSymbol(c_leafVoxelSize, &leafVoxelSize, sizeof(float), 0, cudaMemcpyHostToDevice));

	/*
	char childOrder[8*8] = {
		7, 6, 5, 4, 3, 2, 1, 0,
		6, 7, 4, 5, 2, 3, 0, 1, 
		5, 4, 7, 6, 1, 0, 3, 2, 
		4, 5, 6, 7, 0, 1, 2, 3, 
		3, 2, 1, 0, 7, 6, 5, 4, 
		2, 3, 0, 1, 6, 7, 4, 5, 
		1, 0, 3, 2, 5, 4, 7, 6, 
		0, 1, 2, 3, 4, 5, 6, 7};
	checkCudaErrors(cudaMemcpyToSymbol(c_childOrder, &childOrder, sizeof(char)*8*8, 0, cudaMemcpyHostToDevice));
	*/

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

/*
extern "C" int tracePhotonsTReX(int size, void *outPhotons)
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

	checkCudaErrors(cudaMemcpy(d_PhotonList, h_photons, numValidPhotons*sizeof(Photon), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaBindTexture((size_t *)0, t_Photons, d_PhotonList, numValidPhotons*sizeof(Photon)));

	if(!outPhotons)
		delete[] h_photons;

	return numValidPhotons;
}

extern "C" int buildPhotonKDTreeTReX(int size, void *kdtree)
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
*/

__global__ void
TracePhotons(int emitterIndex, int numPhotons, int offset, PhotonVoxel *photonOctree, float *requestCountList)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//const unsigned int numPhotons = blockDim.x * gridDim.x;
	
	unsigned int seed = tea<16>(idx+offset, emitterIndex);

	const Emitter &emitter = c_emitterList[emitterIndex];
	//const Parallelogram &target = emitter.spotTarget;
	
	Ray ray;
	Vector3 ori = emitter.sample(seed);
	/*
	//Vector3 dir = (target.sample(seed) - ori).normalize();
	Vector3 target;
	switch(emitter.type)
	{
	case Emitter::POINT_LIGHT : target = Vector3(rnd(seed) - 0.5f, rnd(seed) - 0.5f, rnd(seed) - 0.5f) + ori; break;
	case Emitter::PARALLELOGRAM_LIGHT :	target = emitter.spotTarget.sample(seed); break;
	}
	Vector3 dir = (target - ori).normalize();
	*/
	Vector3 dir = emitter.sampleEmitDirection(ori, seed);
	ray.set(ori, dir);

	HitPoint hit;
	hit.t = FLT_MAX;
	
	//bool hasHit = false;
	
	// bounce once when hit diffuse material
	// bounce when hit specular material
	int skip = seed % 6;
	skip = skip < 3 ? 0 : (skip == 5 ? 2 : 1);
	//int skip = 1;
	int maxDepth = 5;
	int depth = 0;
	Material material;
	int hitIndex;
	float hitBBSize;
	while(++depth < maxDepth)
	{
		if(!RayOctreeIntersect(ray, hit, &material, &hitIndex, &hitBBSize, FLT_MAX, seed, requestCountList, 0, 0, c_frame2)) return;
		if(material.hasDiffuse())
			if(skip-- == 0) break;
		lcg(seed);
		ray.set(hit.t * ray.dir + ray.ori, material.sampleDirection(hit.n, ray.dir, seed));
		ray.set(ray.ori + ray.dir*(0.5f*c_leafVoxelSize*OOCVOXEL_SUPER_RESOLUTION), ray.dir);	// offseting
		hit.t = FLT_MAX;
	}

	float cosFac = -ray.dir.dot(hit.n);

	if(cosFac > 0.0f)
	{
		//Vector3 power = (emitter.color_Ka * material.mat_Ka + emitter.color_Kd * material.mat_Kd * cosFac) * (emitter.intensity / numPhotons / (hitBBSize*hitBBSize*0.5f));//(hitBBSize*1.73205f));
		Vector3 power = (emitter.color_Ka + emitter.color_Kd * cosFac) * (emitter.intensity / numPhotons / (hitBBSize*hitBBSize));

		atomicAdd(&photonOctree[hitIndex].power.x, power.x);
		atomicAdd(&photonOctree[hitIndex].power.y, power.y);
		atomicAdd(&photonOctree[hitIndex].power.z, power.z);
	}
}

/*
extern "C" Vector3 buildPhotonLODRec(PhotonVoxel *octree, int index)
{
	if(h_octree[index].isEmpty()) return Vector3(0.0f);

	if(h_octree[index].isLeaf()) return octree[index].power;

	int N = h_octreeHeader.dim;
	int childIndex = h_octree[index].getChildIndex() * N * N * N;
	octree[index].power = Vector3(0.0f);
	for(int i=0;i<N*N*N;i++)
		octree[index].power += buildPhotonLODRec(octree, childIndex+i);
	return octree[index].power;
}

extern "C" void buildPhotonLOD(PhotonVoxel *octree)
{
	int N = h_octreeHeader.dim;
	octree[0].power = Vector3(0.0f);
	for(int i=0;i<N*N*N;i++)
		octree[0].power += buildPhotonLODRec(octree, i);
}
*/

extern "C" int tracePhotonsTReX()
{
	//checkCudaErrors(cudaMemset(d_photonOctree, 0, h_numVoxels*sizeof(PhotonVoxel)));
	checkCudaErrors(cudaMemset(d_photonOctree, 0, h_sizeVoxelBuffer*sizeof(PhotonVoxel)));
	
	for(int i=0;i<h_scene.numEmitters;i++)
	{
		int numPhotons = h_scene.emitters[i].numScatteringPhotons;

		if(numPhotons == 0) continue;
		
		dim3 dimBlock(min(numPhotons, 256), 1);
		dim3 dimGrid(numPhotons / dimBlock.x, 1);

		TracePhotons<<<dimGrid, dimBlock>>>(i, numPhotons, 0, d_photonOctree, d_requestCountList);
	}

#	ifdef USE_VOXEL_LOD
	PhotonVoxel *photonOctree = new PhotonVoxel[h_sizeVoxelBuffer];
	checkCudaErrors(cudaMemcpy(photonOctree, d_photonOctree, h_sizeVoxelBuffer*sizeof(PhotonVoxel), cudaMemcpyDeviceToHost));
	buildPhotonLOD(photonOctree);
	checkCudaErrors(cudaMemcpy(d_photonOctree, photonOctree, h_sizeVoxelBuffer*sizeof(PhotonVoxel), cudaMemcpyHostToDevice));
	delete[] photonOctree;
#	endif

	float ratio = 1.0f;
	checkCudaErrors(cudaMemcpyToSymbol(c_photonMorphingRatio, &ratio, sizeof(float), 0, cudaMemcpyHostToDevice));
	
	return 0;
}

extern "C" int traceSubPhotonsToBackBufferTReX(int pos, int numPhotonsPerEmitter, int frame3)
{
	if(pos == 0)
		checkCudaErrors(cudaMemset(d_photonOctree2, 0, h_sizeVoxelBuffer*sizeof(PhotonVoxel)));
	
	int maxPhotons = 0;
	for(int i=0;i<h_scene.numEmitters;i++)
	{
		int numPhotons = min(h_scene.emitters[i].numScatteringPhotons - pos, numPhotonsPerEmitter);
		maxPhotons = max(maxPhotons, h_scene.emitters[i].numScatteringPhotons);

		if(numPhotons <= 0) continue;
		
		dim3 dimBlock(min(numPhotons, 256), 1);
		dim3 dimGrid(numPhotons / dimBlock.x, 1);

		TracePhotons<<<dimGrid, dimBlock>>>(i, h_scene.emitters[i].numScatteringPhotons, pos, d_photonOctree2, d_requestCountList);
	}
	
	int blockPerFrame = (h_image.width*h_image.height/h_controller.threadBlockSize);
	int accumulatedFrame = pos / numPhotonsPerEmitter / blockPerFrame;
	//float ratio = (maxPhotons-pos)/(float)maxPhotons;
	float ratio = numPhotonsPerEmitter / (float)maxPhotons * blockPerFrame;
	ratio *= accumulatedFrame+h_frame + 1;
	ratio = 1.0f - ratio;
	checkCudaErrors(cudaMemcpyToSymbol(c_photonMorphingRatio, &ratio, sizeof(float), 0, cudaMemcpyHostToDevice));
	return 0;
}

extern "C" int swapPhotonBufferTReX()
{
	PhotonVoxel *temp = d_photonOctree;
	d_photonOctree = d_photonOctree2;
	d_photonOctree2 = temp;
	checkCudaErrors(cudaBindTexture((size_t *)0, t_photonOctree, d_photonOctree, h_sizeVoxelBuffer*sizeof(PhotonVoxel)));
	checkCudaErrors(cudaBindTexture((size_t *)0, t_photonOctree2, d_photonOctree2, h_sizeVoxelBuffer*sizeof(PhotonVoxel)));
	float ratio = 1.0f;
	checkCudaErrors(cudaMemcpyToSymbol(c_photonMorphingRatio, &ratio, sizeof(float), 0, cudaMemcpyHostToDevice));
	return 0;
}

/*
extern "C" int tracePhotonsToOOCVoxelTReX(int oocVoxelIdx)
{
	OOCVoxel &oocVoxel = h_oocVoxels[oocVoxelIdx];

	checkCudaErrors(cudaMemset(&d_photonOctree[oocVoxel.offset], 0, oocVoxel.numVoxels*sizeof(PhotonVoxel)));
	for(int i=0;i<h_scene.numEmitters;i++)
	{
		int numPhotons = h_scene.emitters[i].numScatteringPhotons;

		if(numPhotons == 0) continue;
		
		dim3 dimBlock(min(numPhotons, 256), 1);
		dim3 dimGrid(numPhotons / dimBlock.x, 1);

		TracePhotons<<<dimGrid, dimBlock>>>(i, &d_oocVoxels[oocVoxelIdx].rootBB, oocVoxel.offset/8, d_photonOctree);
	}
	
	return 0;
}
*/

extern "C" void updateControllerTReX(Controller *controller)
{
	updateController(controller);
}

extern "C" void renderBeginTReX(Camera *camera, Image *image, Controller *controller, int frame)
{
	int device = 0;
	int oldDevice;
	cudaGetDevice(&oldDevice);
	if(device != oldDevice)
		cudaSetDevice(device);

	if(image->width != h_image.width || image->height != h_image.height || image->bpp != h_image.bpp)
	{
		checkCudaErrors(cudaFree(d_hitCache));
		checkCudaErrors(cudaFree(d_extRayCache));
		checkCudaErrors(cudaMalloc((void**)&d_hitCache, image->width*image->height*sizeof(HitPoint)));
		checkCudaErrors(cudaMalloc((void**)&d_extRayCache, image->width*image->height*sizeof(ExtraRayInfo)));

		checkCudaErrors(cudaFree(d_sampleData));
		checkCudaErrors(cudaMalloc((void**)&d_sampleData, (image->width*image->height)*sizeof(SampleData)));
	}

	renderBeginCUDA(camera, image, controller, frame);

	cudaFuncSetCacheConfig(Render, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(GatherPhotons, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(ApplyToImage, cudaFuncCachePreferL1);
	
	/*
	static float kernel3[] = {1, 1, 1, 1, 2, 1, 1, 1, 1};
	static float kernel5[] = {
			1,  4,  7,  4,  1,
			4, 16, 26, 16,  4,
			7, 26, 41, 26,  7,
			4, 16, 26, 16,  4,
			1,  4,  7,  4,  1};

	checkCudaErrors(cudaMemcpyToSymbol(c_boxFilterKernel3, kernel3, sizeof(float)*9, 0, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(c_boxFilterKernel5, kernel5, sizeof(float)*25, 0, cudaMemcpyHostToDevice));
	*/
	int emitterUsageHistogram[EMITTER_SAMPLE_RESOLUTION];

	double *emitterStat = new double[MAX_NUM_EMITTERS];
	double sum = 0.0;
	
	for(int i=0;i<h_scene.numEmitters;i++)
	{
#		if 0	// uniform distribution
		emitterStat[i] = 1.0;
#		else	// compute distance from camera
		emitterStat[i] = 1.0 / (camera->eye - h_scene.emitters[i].pos).length();
#		endif
		sum += emitterStat[i];
	}

	if(h_scene.numEmitters > 1)
	{
		// special care of first emitter (50% weight)
		double temp = emitterStat[0];
		emitterStat[0] = sum - temp;
		sum += emitterStat[0] - temp;
	}

	// normalize
	for(int i=0;i<h_scene.numEmitters;i++)
		emitterStat[i] /= sum;

	float emitterUsageDistribution[MAX_NUM_EMITTERS];
	for(int i=0;i<MAX_NUM_EMITTERS;i++)
	{
		emitterUsageDistribution[i] = (float)emitterStat[i];
	}

	// scaling
	for(int i=0;i<h_scene.numEmitters;i++)
		emitterStat[i] *= EMITTER_SAMPLE_RESOLUTION;

	// accumulate
	double curValue = 0.0;
	for(int i=0;i<h_scene.numEmitters;i++)
	{
		curValue += emitterStat[i];
		emitterStat[i] = curValue;
	}

	for(int i=0, j=0;i<EMITTER_SAMPLE_RESOLUTION && j<h_scene.numEmitters;i++)
	{
		while(i > emitterStat[j]) j++;
		emitterUsageHistogram[i] = j;
	}
	
	delete[] emitterStat;

	checkCudaErrors(cudaMemcpyToSymbolAsync(c_emitterUsageDistribution, emitterUsageDistribution, sizeof(float)*MAX_NUM_EMITTERS, 0, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbolAsync(c_emitterUsageHistogram, emitterUsageHistogram, sizeof(int)*EMITTER_SAMPLE_RESOLUTION, 0, cudaMemcpyHostToDevice));

	int manualEmitterSelection[MAX_NUM_EMITTERS];
	int numSelectedEmitters;

#	define STR_EQ(a, b, ret)  (ret) = true; for(int _i=0;(a)[_i]!=0;_i++) if((a)[_i] != (b)[_i]) (ret) = false;

	bool equalStr;
	// for boeing test viewpoints
	STR_EQ(camera->name, "0_overview", equalStr);
	if(equalStr)
	{
		numSelectedEmitters = 1;
		manualEmitterSelection[0] = 1;
	}
	STR_EQ(camera->name, "0_cockpit", equalStr);
	if(equalStr)
	{
		numSelectedEmitters = 1;
		manualEmitterSelection[0] = 1;
	}
	STR_EQ(camera->name, "Camera1", equalStr);
	if(equalStr)
	{
		numSelectedEmitters = 1;
		manualEmitterSelection[0] = 1;
	}
	STR_EQ(camera->name, "0_cabin", equalStr);
	if(equalStr)
	{
		numSelectedEmitters = 8;
		manualEmitterSelection[0] = 2;
		manualEmitterSelection[1] = 3;
		manualEmitterSelection[2] = 4;
		manualEmitterSelection[3] = 5;
		manualEmitterSelection[4] = 6;
		manualEmitterSelection[5] = 7;
		manualEmitterSelection[6] = 8;
		manualEmitterSelection[7] = 9;
	}
	STR_EQ(camera->name, "0_engine", equalStr);
	if(equalStr)
	{
		numSelectedEmitters = 1;
		manualEmitterSelection[0] = 10;
	}

	checkCudaErrors(cudaMemcpyToSymbolAsync(c_manualEmitterSelection, manualEmitterSelection, sizeof(int)*MAX_NUM_EMITTERS, 0, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbolAsync(c_numSelectedEmitters, &numSelectedEmitters, sizeof(int), 0, cudaMemcpyHostToDevice));
	
}

extern "C" void resetSyncTReX()
{
	if(h_isReset)
	{
		checkCudaErrors(cudaMemcpyToSymbolAsync(c_frame2, &h_frame2, sizeof(int), 0, cudaMemcpyHostToDevice));
		//checkCudaErrors(cudaMemset(d_imageData, 0, h_image.width*h_image.height*h_image.bpp));
		checkCudaErrors(cudaMemsetAsync(d_sampleData, 0, sizeof(SampleData)*h_image.width*h_image.height));
		h_isReset = false;
	}
}

//extern "C" void renderPartTReX(int frame, int frame2, float &time, Ray *rayCache, HitPoint *hitCache, ExtraRayInfo *extRayCache)//, HColor *colorCache)
extern "C" void renderPartTReX(int frame, int frame2, float &time, HitPoint *hitCache, ExtraRayInfo *extRayCache, int numRays, int offset)
{
	TIMER_START;
	h_frame = frame;

	checkCudaErrors(cudaMemcpyAsync(&d_hitCache[offset], &hitCache[offset], numRays*sizeof(HitPoint), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyAsync(&d_extRayCache[offset], &extRayCache[offset], numRays*sizeof(ExtraRayInfo), cudaMemcpyHostToDevice));
	
	dim3 dimBlock(256, 1);
	dim3 dimGrid(numRays/dimBlock.x, 1);

	dim3 dimGrid2(numRays/dimBlock.x*h_controller.numShadowRays, 1);
	dim3 dimGrid3(numRays/dimBlock.x*h_controller.warpSizeG, 1);
	dim3 dimGrid4(numRays/dimBlock.x*h_controller.numGatheringRays, 1);
	// execute the kernel

	int numIter = (int)(ceil((float)dimGrid2.x*dimBlock.x / NUM_MAX_THREADS));
	for(int i=0;i<numIter;i++)
	{
		if(h_frame2 != frame2) break;
		dim3 dimGrid0(min(NUM_MAX_THREADS/dimBlock.x, dimGrid2.x), 1);
		resetSyncTReX();
		Render<<<dimGrid0, dimBlock>>>(i, d_imageData, frame, frame2, &d_hitCache[offset], &d_extRayCache[offset], d_sampleData, d_requestCountList);
	}

	// gather photons
//#	define NUM_GATHER 1//(limit*20+1)
	
	//if(h_frame2 != frame2)
	//	return;

	/*
	if(h_controller.gatherPhotons)
	{
		for(int i=0;i<h_controller.numGatheringRays;i++)
		{
			GatherPhotons<<<dimGrid3, dimBlock3>>>(i, i+frame*h_controller.numGatheringRays, &d_extRayCache[offset], d_sampleData, d_requestCountList);
		}
	}
	*/
	if(h_controller.gatherPhotons && h_controller.numGatheringRays > 0)
	{
		//GatherPhotons<<<dimGrid4, dimBlock>>>(0, frame, frame2, &d_extRayCache[offset], d_sampleData, d_requestCountList);
		int numIter = (int)(ceil((float)dimGrid4.x*dimBlock.x / NUM_MAX_THREADS));
		for(int i=0;i<numIter;i++)
		{
			if(h_frame2 != frame2) break;
			dim3 dimGrid0(min(NUM_MAX_THREADS/dimBlock.x, dimGrid4.x), 1);

			resetSyncTReX();
			GatherPhotons<<<dimGrid0, dimBlock>>>(i, frame, frame2, &d_extRayCache[offset], d_sampleData, d_requestCountList);
		}
	}

	if(h_controller.useAmbientOcclusion)
		AmbientOcclusion<<<dimGrid2, dimBlock>>>(d_imageData, frame2, &d_hitCache[offset], &d_extRayCache[offset], d_sampleData, d_requestCountList);
	
	if(h_frame2 != frame2) 
		return;

	ApplyToImage<<<dimGrid, dimBlock>>>(d_imageData, d_sampleData, &d_extRayCache[offset], frame2);

	//cudaDeviceSynchronize();
	TIMER_STOP(time);
}

extern "C" void renderEndTReX(Image *image)
{
	renderEndCUDA(image);
}

extern "C" void resetTReX(int frame2)
{
	h_frame2 = frame2;
	h_isReset = true;
}

extern "C" void initWithImageTReX(Image *image)
{
	checkCudaErrors(cudaMemcpyAsync(d_imageData, image->data, image->width*image->height*image->bpp, cudaMemcpyHostToDevice));
}

extern "C" void loadVoxelsTReX(int count, const Voxel *voxels, int oocVoxelIdx, int offset, int ttt)
{
	offset += h_numVoxels;
	//checkCudaErrors(cudaMemcpy(&h_oocVoxels[oocVoxelIdx].offset, &offset, sizeof(int), cudaMemcpyHostToHost));
	h_oocVoxels[oocVoxelIdx].offset = offset;
	checkCudaErrors(cudaMemcpy(&d_oocVoxels[oocVoxelIdx].offset, &offset, sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(&d_octree[offset], voxels, sizeof(Voxel)*count, cudaMemcpyHostToDevice));
	int newIdx = ((offset/8) << 2) | 0x2;
	checkCudaErrors(cudaMemcpy(&d_octree[ttt].childIndex, &newIdx, sizeof(int), cudaMemcpyHostToDevice));
}

extern "C" void loadPhotonVoxelsTReX(int count, const PhotonVoxel *voxels, int oocVoxelIdx, int offset)
{
	offset += h_numVoxels;
	checkCudaErrors(cudaMemcpy(&d_photonOctree[offset], voxels, sizeof(PhotonVoxel)*count, cudaMemcpyHostToDevice));
}

extern "C" void loadOOCVoxelInfoTReX(int count, const OOCVoxel *oocVoxels)
{
	h_numOOCVoxels = count;
	h_oocVoxels = new OOCVoxel[count];
	checkCudaErrors(cudaMalloc((void **)&d_oocVoxels, sizeof(OOCVoxel)*count));
	checkCudaErrors(cudaMemcpy(h_oocVoxels, oocVoxels, sizeof(OOCVoxel)*count, cudaMemcpyHostToHost));
	checkCudaErrors(cudaMemcpy(d_oocVoxels, oocVoxels, sizeof(OOCVoxel)*count, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void **)&d_requestCountList, sizeof(float)*count));
}

extern "C" void OnVoxelChangedTReX(int numIn, int *in, int numOut, int *out)
{
	for(int i=0;i<numIn;i++)
	{
		//int rootIndex = in[i*2+0];
		//int offset = (((in[i*2+1] + h_numVoxels)/8) << 2) | 0x2;
		/*
		//if(rootIndex == 45480)
		printf("!r = %d, c = %d(%d) -> %d(%d)\n", rootIndex, h_octree[rootIndex].childIndex, (h_octree[rootIndex].childIndex >> 2)*8, offset, (offset >> 2)*8);
		Voxel bVoxel = h_octree[rootIndex];
		Voxel bcVoxel = h_octree[(h_octree[rootIndex].childIndex >> 2) * 8];
		*/
		//checkCudaErrors(cudaMemcpy(&d_octree[rootIndex].childIndex, &offset, sizeof(int), cudaMemcpyHostToDevice));
		/*
		offset = (offset >> 2)*8;
		Voxel voxel;
		checkCudaErrors(cudaMemcpy(&voxel, &d_octree[offset], sizeof(Voxel), cudaMemcpyDeviceToHost));
		*/
	}

	for(int i=0;i<numOut;i++)
	{
		int rootIndex = out[i*2+0];
		checkCudaErrors(cudaMemcpy(&d_octree[rootIndex].childIndex, &h_octree[rootIndex].childIndex, sizeof(int), cudaMemcpyHostToDevice));
	}
}


extern "C" void beginGatheringRequestCountTReX()
{
	//checkCudaErrors(cudaMemset(d_requestCountList, 0, sizeof(float)*h_numOOCVoxels));
	float *temp = new float[h_numOOCVoxels];
	for(int i=0;i<h_numOOCVoxels;i++)
	{
		temp[i] = i/(float)h_numOOCVoxels;
	}
	checkCudaErrors(cudaMemcpy(d_requestCountList, temp, sizeof(float)*h_numOOCVoxels, cudaMemcpyHostToDevice));
	delete[] temp;
}

extern "C" void endGatheringRequestCountTReX(float *requestCountList)
{
	checkCudaErrors(cudaMemcpy(requestCountList, d_requestCountList, sizeof(float)*h_numOOCVoxels, cudaMemcpyDeviceToHost));
}

/*
typedef struct {
	int width;
	int height;
	float* elem;
} Matrix;

__global__ void
testKernel(Matrix A, Matrix B, Matrix C)
{
	float c = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	for(int e=0;e<A.width;e++)
		c += A.elem[row * A.width + e] * B.elem[e * B.width + col];
	C.elem[row * C.width + col] = c;
}

int g_mWidth = 1024, g_mHeight = 1024;
Matrix d_A, d_B, d_C;
extern "C" void testset()
{
	size_t size = g_mWidth * g_mHeight * sizeof(float);
	float *a = new float[g_mWidth * g_mHeight];
	float *b = new float[g_mWidth * g_mHeight];
	float *c = new float[g_mWidth * g_mHeight];
	for(int i=0;i<g_mWidth * g_mHeight;i++)
	{
		a[i] = (rand() / (float)RAND_MAX);
		b[i] = (rand() / (float)RAND_MAX);
	}
	checkCudaErrors(cudaMalloc((void **)&d_A.elem, size));
	checkCudaErrors(cudaMemcpy(d_A.elem, a, size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void **)&d_B.elem, size));
	checkCudaErrors(cudaMemcpy(d_B.elem, b, size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void **)&d_C.elem, size));
	checkCudaErrors(cudaMemcpy(d_C.elem, c, size, cudaMemcpyHostToDevice));
	d_A.width = d_B.width = d_C.width = g_mWidth;
	d_A.height = d_B.height = d_C.height = g_mHeight;
}

extern "C" void testtest(int i)
{
	dim3 dimBlock(16, 16);
	dim3 dimGrid(g_mWidth / dimBlock.x, g_mHeight / dimBlock.y);
	//switch(i % 3)
	//{
	//case 0 : testKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C); break;
	//case 1 : testKernel<<<dimGrid, dimBlock>>>(d_C, d_A, d_B); break;
	//case 2 : testKernel<<<dimGrid, dimBlock>>>(d_B, d_C, d_A); break;
	//}
	testKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
}

extern "C" void testfinish(float *result)
{
	checkCudaErrors(cudaMemcpy(result, d_C.elem, sizeof(float), cudaMemcpyDeviceToHost));
}
*/

}