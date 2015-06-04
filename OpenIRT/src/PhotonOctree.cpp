#include "CommonHeaders.h"
#include "PhotonOctree.h"
#include <io.h>
#include "Scene.h"
#include "handler.h"
#include "OpenIRT.h"
#include "rgb.h"

using namespace irt;

PhotonOctree *g_photonOctree;

PhotonOctree::PhotonOctree(void)
{
	g_photonOctree = this;
}

void PhotonOctree::addPhoton(const Photon& photon)
{
	//g_photonOctree->m_octree[g_photonOctree->m_hash[g_photonOctree->getPosition(photon.pos)]].getColorRef() += photon.power;
	printf("Wrong!!\n");
}

RGBf PhotonOctree::makeLOD(int index)
{
	int N = m_header.dim;
	int childIndex = index * N * N * N;

	RGBf power;
	for(int x=0;x<N;x++)
		for(int y=0;y<N;y++)
			for(int z=0;z<N;z++)
			{
				if(m_octree[childIndex].hasChild())
				{
					//m_octree[childIndex].getColorRef() = makeLOD(m_octree[childIndex].getChildIndex());
					printf("Wrong!!\n");
					power += m_octree[childIndex].getColorRef();
				}
				else if(m_octree[childIndex].isLeaf())
				{
					power += m_octree[childIndex].getColorRef();
				}
				childIndex++;
			}
	return power / (float)(N*N*N);
}

#define BSP_EPSILON 0.001f
#define INTERSECT_EPSILON 0.01f
#define TRI_INTERSECT_EPSILON 0.0001f
bool PhotonOctree::RayLODIntersect(const Ray &ray, const Voxel &voxel, HitPointInfo &hit, Material &material, float tmax, unsigned int seed)
{
	// ignore invalid voxel
	if(!(voxel.d < FLT_MAX && voxel.d > -FLT_MAX)) return false;

	Vector3 norm;
	norm = voxel.getNorm();

	float vdot = norm.x()*ray.direction().x() + norm.y()*ray.direction().y() + norm.z()*ray.direction().z();
	float vdot2 = norm.x()*ray.origin().x() + norm.y()*ray.origin().y() + norm.z()*ray.origin().z();
	float t = (voxel.d - vdot2) / vdot;

	// if either too near or further away than a previous hit, we stop
	if (t < (0-INTERSECT_EPSILON*10) || t > (tmax + INTERSECT_EPSILON*10))
		return false;	

	material.mat_Kd = voxel.mat.getKd();
	material.mat_Ks = voxel.mat.getKs();
	material.mat_d = voxel.mat.getD();
	material.mat_Ns = voxel.mat.getNs();
	material.recalculateRanges();
	if(material.isRefraction(voxel.mat.getD(), seed)) return false;

	// we have a hit:
	// fill hitpoint structure:
	//
	hit.m = 0;
	hit.t = t;
	hit.n = vdot > 0 ? -norm : norm;
	return true;	
}

#define USE_GEOM_BITMAP
#if 1
bool PhotonOctree::RayOctreeIntersect(const Ray &ray, HitPointInfo &hit, Material &material, int &hitIndex, float &hitBBSize, float limit, float tLimit, unsigned int seed, int x, int y)
{
	const OctreeHeader &c_octreeHeader = m_header;

	float t0, t1;
	if(!ray.boxIntersect(c_octreeHeader.min, c_octreeHeader.max, t0, t1)) return false;

	int flag = 0;

	Vector3 ori = ray.origin(), dir = ray.direction();
	Vector3 temp = c_octreeHeader.max + c_octreeHeader.min;

	if(ray.direction().x() < 0.0f)
	{
		ori.e[0] = temp.e[0] - ori.e[0];
		dir.e[0] = -dir.e[0];
		flag |= 4;
	}

	if(ray.direction().y() < 0.0f)
	{
		ori.e[1] = temp.e[1] - ori.e[1];
		dir.e[1] = -dir.e[1];
		flag |= 2;
	}

	if(ray.direction().z() < 0.0f)
	{
		ori.e[2] = temp.e[2] - ori.e[2];
		dir.e[2] = -dir.e[2];
		flag |= 1;
	}

	Ray transRay;
	transRay.set(ori, dir);

	AABB newBB;
	newBB.min = (c_octreeHeader.min - transRay.origin()) * transRay.invDirection();
	newBB.max = (c_octreeHeader.max - transRay.origin()) * transRay.invDirection();

	int N = c_octreeHeader.dim;

	typedef struct tempStack_t
	{
		int childIndex;
		int child;
		AABB bb;
	} tempStack;

	Vector3 mid;
	Voxel root;
	root.setChildIndex(0);
	tempStack stack[100];

	int stackPtr;

	mid = 0.5f*(newBB.min + newBB.max);

	stackPtr = 0;

	AABB currentBB = newBB;
	Voxel currentVoxel = root;

	int child = -1, childIndex = 0;

	float bbSize = c_octreeHeader.max.x() - c_octreeHeader.min.x();

	while(true)
	{
		if(!currentVoxel.isEmpty() && 
			currentBB.max.x() > 0.0f && currentBB.max.y() > 0.0f && currentBB.max.z() > 0.0f &&
			currentBB.min.x() < hit.t && currentBB.min.y() < hit.t && currentBB.min.y() < hit.t)
		{
			if(currentVoxel.isLeaf())
			{
				//if(currentBB.max.x() < bbSize*3 || currentBB.max.y() < bbSize*3 || currentBB.max.z() < bbSize*3) goto NEXT_SIBILING;

				//if(RayLODIntersect(ray, currentVoxel, hit, material, fminf(currentBB.max.minComponent(), hit.t), seed))
				if(RayLODIntersect(ray, currentVoxel, hit, material, hit.t, seed))
				{
					hitIndex = childIndex + (child^flag);
					hitBBSize = bbSize;
					return true;
				}
				goto NEXT_SIBILING;
			}

			// push down

			//FIRST_NODE(currentBB, mid, child);
			// get first child
			child = 0;
			if (currentBB.min.y() < currentBB.min.x() && currentBB.min.z() < currentBB.min.x())
			{
				// YZ plane
				if (mid.y() < currentBB.min.x()) child |= 2;
				if (mid.z() < currentBB.min.x()) child |= 1;
			}
			else if (currentBB.min.z() < currentBB.min.y())
			{
				// XZ plane
				if (mid.x() < currentBB.min.y()) child |= 4;
				if (mid.z() < currentBB.min.y()) child |= 1;
			}
			else
			{
				// XY plane
				if (mid.x() < currentBB.min.z()) child |= 4;
				if (mid.y() < currentBB.min.z()) child |= 2;
			}

			childIndex = currentVoxel.getChildIndex() * N * N * N;

			stack[stackPtr].bb = currentBB;
			stack[stackPtr].childIndex = childIndex;
			stack[stackPtr++].child = child;

			currentBB.min.e[0] = (child & 0x4) ? mid.e[0] : currentBB.min.e[0];
			currentBB.min.e[1] = (child & 0x2) ? mid.e[1] : currentBB.min.e[1];
			currentBB.min.e[2] = (child & 0x1) ? mid.e[2] : currentBB.min.e[2];
			currentBB.max.e[0] = (child & 0x4) ? currentBB.max.e[0] : mid.e[0];
			currentBB.max.e[1] = (child & 0x2) ? currentBB.max.e[1] : mid.e[1];
			currentBB.max.e[2] = (child & 0x1) ? currentBB.max.e[2] : mid.e[2];
			mid = 0.5f*(currentBB.min + currentBB.max);

			/*
			VOXEL_FETCH(childIndex + (child^flag), 0, currentVoxel.low);
			VOXEL_FETCH(childIndex + (child^flag), 1, currentVoxel.high);
			*/
			currentVoxel = m_octree[childIndex + (child^flag)];

			bbSize *= 0.5f;

			continue;
		}

		// move to next sibiling
NEXT_SIBILING : 

		if(stackPtr < 1) return false;
		// get stack top
		childIndex = stack[stackPtr-1].childIndex;
		currentBB = stack[stackPtr-1].bb;
		child = stack[stackPtr-1].child;
		mid = 0.5f*(currentBB.min + currentBB.max);

#		define NEW_NODE(x, y, z, a, b, c) (((x) < (y) && (x) < (z)) ? (a) : (((y) < (z)) ? (b) : (c)))

		switch(child)
		{
		case 0 : child = NEW_NODE(mid.x(), mid.y(), mid.z(), 4, 2, 1); break;
		case 1 : child = NEW_NODE(mid.x(), mid.y(), currentBB.max.z(), 5, 3, 8); break;
		case 2 : child = NEW_NODE(mid.x(), currentBB.max.y(), mid.z(), 6, 8, 3); break;
		case 3 : child = NEW_NODE(mid.x(), currentBB.max.y(), currentBB.max.z(), 7, 8, 8); break;
		case 4 : child = NEW_NODE(currentBB.max.x(), mid.y(), mid.z(), 8, 6, 5); break;
		case 5 : child = NEW_NODE(currentBB.max.x(), mid.y(), currentBB.max.z(), 8, 7, 8); break;
		case 6 : child = NEW_NODE(currentBB.max.x(), currentBB.max.y(), mid.z(), 8, 8, 7); break;
		case 7 : child = 8;	break;
		}
		stack[stackPtr-1].child = child;

		if(child == 8)
		{
			if (--stackPtr == 0) break;

			bbSize *= 2.0f;

			goto NEXT_SIBILING;
		}

		/*
		VOXEL_FETCH(childIndex + (child^flag), 0, currentVoxel.low);
		VOXEL_FETCH(childIndex + (child^flag), 1, currentVoxel.high);
		*/
		currentVoxel = m_octree[childIndex + (child^flag)];
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
#else
bool PhotonOctree::RayOctreeIntersect(const Ray &ray, HitPointInfo &hit, Material &material, int &hitIndex, float &hitBBSize, float limit, float tLimit, unsigned int seed, int x, int y)
{
	const OctreeHeader &c_octreeHeader = m_header;

	float t0, t1;
	if(!ray.boxIntersect(c_octreeHeader.min, c_octreeHeader.max, t0, t1)) return false;

	int flag = 0;

	Vector3 ori = ray.origin(), dir = ray.direction();
	Vector3 temp = c_octreeHeader.max + c_octreeHeader.min;

	if(ray.direction().x() < 0.0f)
	{
		ori.e[0] = temp.e[0] - ori.e[0];
		dir.e[0] = -dir.e[0];
		flag |= 4;
	}

	if(ray.direction().y() < 0.0f)
	{
		ori.e[1] = temp.e[1] - ori.e[1];
		dir.e[1] = -dir.e[1];
		flag |= 2;
	}

	if(ray.direction().z() < 0.0f)
	{
		ori.e[2] = temp.e[2] - ori.e[2];
		dir.e[2] = -dir.e[2];
		flag |= 1;
	}

	Ray transRay;
	transRay.set(ori, dir);

	AABB newBB;
	newBB.min = (c_octreeHeader.min - transRay.origin()) * transRay.invDirection();
	newBB.max = (c_octreeHeader.max - transRay.origin()) * transRay.invDirection();

	int N = c_octreeHeader.dim;

	typedef struct tempStack_t
	{
		int childIndex;
		int child;
	} tempStack;

	Vector3 mid;
	Voxel root;
	root.setChildIndex(0);
	tempStack stack[100];

	int stackPtr;

	mid = 0.5f*(newBB.min + newBB.max);

	stackPtr = 0;

#	ifdef USE_GEOM_BITMAP
	char currentIsLeaf = 0;
	char currentGeomBitmapDepth = 0;
	unsigned char currentGeomBitmap = 0;
#	endif

	AABB currentBB = newBB;
	Voxel currentVoxel = root;

	int child = -1, childIndex = 0;

	float bbSize = c_octreeHeader.max.x() - c_octreeHeader.min.x();

	bool isEmpty;
	while(true)
	{
#		ifdef USE_GEOM_BITMAP
		switch(currentGeomBitmapDepth)
		{
		case 0 : isEmpty = currentVoxel.isEmpty(); break;
		case 1 : isEmpty = currentGeomBitmap == 0; break;
		case 2 : isEmpty = (currentGeomBitmap & (1u << (child^flag))) == 0; break;
		}
#		else
		isEmpty = currentVoxel.isEmpty();
#		endif

		if(!isEmpty && 
			currentBB.max.x() > 0.0f && currentBB.max.y() > 0.0f && currentBB.max.z() > 0.0f &&
			currentBB.min.x() < hit.t && currentBB.min.y() < hit.t && currentBB.min.y() < hit.t)
		{
#			ifdef USE_GEOM_BITMAP
			currentIsLeaf = currentVoxel.isLeaf();
			if(currentIsLeaf && currentGeomBitmapDepth == 0)
				hitIndex = childIndex + (child^flag);
			if(currentGeomBitmapDepth == 2)
			{
				if(RayLODIntersect(ray, currentVoxel, hit, material, hit.t, seed))
				{
					hitBBSize = bbSize;
					return true;
				}
				goto NEXT_SIBILING;
			}
#			else
			if(currentVoxel.isLeaf())
			{
				//if(currentBB.max.x() < bbSize*3 || currentBB.max.y() < bbSize*3 || currentBB.max.z() < bbSize*3) goto NEXT_SIBILING;

				//if(RayLODIntersect(ray, currentVoxel, hit, material, fminf(currentBB.max.minComponent(), hit.t), seed))
				if(RayLODIntersect(ray, currentVoxel, hit, material, hit.t, seed))
				{
					hitIndex = childIndex + (child^flag);
					hitBBSize = bbSize;
					return true;
				}
				goto NEXT_SIBILING;
			}
#endif

			// push down

			//FIRST_NODE(currentBB, mid, child);
			// get first child
			child = 0;
			if (currentBB.min.y() < currentBB.min.x() && currentBB.min.z() < currentBB.min.x())
			{
				// YZ plane
				if (mid.y() < currentBB.min.x()) child |= 2;
				if (mid.z() < currentBB.min.x()) child |= 1;
			}
			else if (currentBB.min.z() < currentBB.min.y())
			{
				// XZ plane
				if (mid.x() < currentBB.min.y()) child |= 4;
				if (mid.z() < currentBB.min.y()) child |= 1;
			}
			else
			{
				// XY plane
				if (mid.x() < currentBB.min.z()) child |= 4;
				if (mid.y() < currentBB.min.z()) child |= 2;
			}

#			ifdef USE_GEOM_BITMAP
			if(!currentIsLeaf)
#			endif
				childIndex = currentVoxel.getChildIndex() * N * N * N;
	
			stack[stackPtr].childIndex = childIndex;
			stack[stackPtr++].child = child;

#			ifdef USE_GEOM_BITMAP
			if(currentIsLeaf) 
				currentGeomBitmapDepth++;
			if(currentGeomBitmapDepth == 1) currentGeomBitmap = currentVoxel.geomBitmap[child^flag]; 
#			endif

			currentBB.min.e[0] = (child & 0x4) ? mid.e[0] : currentBB.min.e[0];
			currentBB.min.e[1] = (child & 0x2) ? mid.e[1] : currentBB.min.e[1];
			currentBB.min.e[2] = (child & 0x1) ? mid.e[2] : currentBB.min.e[2];
			currentBB.max.e[0] = (child & 0x4) ? currentBB.max.e[0] : mid.e[0];
			currentBB.max.e[1] = (child & 0x2) ? currentBB.max.e[1] : mid.e[1];
			currentBB.max.e[2] = (child & 0x1) ? currentBB.max.e[2] : mid.e[2];
			mid = 0.5f*(currentBB.min + currentBB.max);

			/*
			VOXEL_FETCH(childIndex + (child^flag), 0, currentVoxel.low);
			VOXEL_FETCH(childIndex + (child^flag), 1, currentVoxel.high);
			*/

#			ifdef USE_GEOM_BITMAP
			if(!currentIsLeaf)
#			endif
			{
				currentVoxel = m_octree[childIndex + (child^flag)];

				bbSize *= 0.5f;
			}


			continue;
		}

		// move to next sibiling
NEXT_SIBILING : 

		if(stackPtr < 1) return false;

		// get parent
		Vector3 size = currentBB.max - currentBB.min;
		currentBB.min.e[0] -= (child & 0x4) ? size.e[0] : 0.0f;
		currentBB.min.e[1] -= (child & 0x2) ? size.e[1] : 0.0f;
		currentBB.min.e[2] -= (child & 0x1) ? size.e[2] : 0.0f;
		currentBB.max.e[0] += (child & 0x4) ? 0.0f : size.e[0];
		currentBB.max.e[1] += (child & 0x2) ? 0.0f : size.e[1];
		currentBB.max.e[2] += (child & 0x1) ? 0.0f : size.e[2];

		// get stack top
		childIndex = stack[stackPtr-1].childIndex;
		child = stack[stackPtr-1].child;
		mid = 0.5f*(currentBB.min + currentBB.max);

#		define NEW_NODE(x, y, z, a, b, c) (((x) < (y) && (x) < (z)) ? (a) : (((y) < (z)) ? (b) : (c)))

		switch(child)
		{
		case 0 : child = NEW_NODE(mid.x(), mid.y(), mid.z(), 4, 2, 1); break;
		case 1 : child = NEW_NODE(mid.x(), mid.y(), currentBB.max.z(), 5, 3, 8); break;
		case 2 : child = NEW_NODE(mid.x(), currentBB.max.y(), mid.z(), 6, 8, 3); break;
		case 3 : child = NEW_NODE(mid.x(), currentBB.max.y(), currentBB.max.z(), 7, 8, 8); break;
		case 4 : child = NEW_NODE(currentBB.max.x(), mid.y(), mid.z(), 8, 6, 5); break;
		case 5 : child = NEW_NODE(currentBB.max.x(), mid.y(), currentBB.max.z(), 8, 7, 8); break;
		case 6 : child = NEW_NODE(currentBB.max.x(), currentBB.max.y(), mid.z(), 8, 8, 7); break;
		case 7 : child = 8;	break;
		}
		stack[stackPtr-1].child = child;

		if(child == 8)
		{
			if (--stackPtr == 0) break;

#			ifdef USE_GEOM_BITMAP
			if(--currentGeomBitmapDepth <= 0)
			{
				currentGeomBitmapDepth = 0;
				currentIsLeaf = false;
			}
#			endif

#			ifdef USE_GEOM_BITMAP
			if(!currentIsLeaf)
#			endif
				bbSize *= 2.0f;

			child = stack[stackPtr-1].child;

			goto NEXT_SIBILING;
		}

		/*
		VOXEL_FETCH(childIndex + (child^flag), 0, currentVoxel.low);
		VOXEL_FETCH(childIndex + (child^flag), 1, currentVoxel.high);
		*/
#		ifdef USE_GEOM_BITMAP
		if(!currentIsLeaf)
#		endif
			currentVoxel = m_octree[childIndex + (child^flag)];
#		ifdef USE_GEOM_BITMAP
		else
		{
			//currentGeomBitmapDepth++;
			if(currentGeomBitmapDepth == 1) currentGeomBitmap = currentVoxel.geomBitmap[child^flag]; 
		}
#		endif
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
#endif

void PhotonOctree::tracePhotons(int emitterIndex, PhotonVoxel *photonOctree, int idx)
{
	const Scene *scene = OpenIRT::getSingletonPtr()->getCurrentScene();
	const Emitter &emitter = scene->getConstEmitter(emitterIndex);
	const Parallelogram &target = emitter.spotTarget;

	const unsigned int numPhotons = emitter.numScatteringPhotons;

	unsigned int seed = tea<16>(idx, emitterIndex);

	Ray ray;
	Vector3 ori = emitter.sample(seed);
	//Vector3 dir = (target.sample(seed) - ori);
	Vector3 dir = emitter.sampleEmitDirection(ori, seed);
	dir.makeUnitVector();
	ray.set(ori, dir);

	HitPointInfo hit;
	hit.t = FLT_MAX;

	bool hasHit = false;

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
		if(!RayOctreeIntersect(ray, hit, material, hitIndex, hitBBSize, 0.0f, 0.0f, seed, 0, 0)) return;
		if(material.hasDiffuse())
			if(skip-- == 0) break;
		lcg(seed);
		ray.set(hit.t * ray.direction() + ray.origin(), material.sampleDirection(hit.n, ray.direction(), seed));
		ray.set(ray.origin() + ray.direction(), ray.direction());	// offseting
		hit.t = FLT_MAX;
	}

	float cosFac = -dot(ray.direction(), hit.n);
	if(cosFac > 0.0f)
	{
		//Vector3 power = (emitter.color_Ka * material.mat_Ka + emitter.color_Kd * material.mat_Kd * cosFac) * (emitter.intensity / numPhotons / (hitBBSize*hitBBSize*0.5f));//(hitBBSize*1.73205f));
		RGBf power = (emitter.color_Ka + emitter.color_Kd * cosFac) * (emitter.intensity / numPhotons / (hitBBSize*hitBBSize));
#		define USE_ATOMIC

#		ifdef USE_ATOMIC
#		pragma omp atomic
		photonOctree[hitIndex].power.e[0] += power.e[0];
#		pragma omp atomic
		photonOctree[hitIndex].power.e[1] += power.e[1];
#		pragma omp atomic
		photonOctree[hitIndex].power.e[2] += power.e[2];
#		else
		photonOctree[hitIndex].power.e[0] += power.e[0];
		photonOctree[hitIndex].power.e[1] += power.e[1];
		photonOctree[hitIndex].power.e[2] += power.e[2];
#		endif
	}
}

void PhotonOctree::tracePhotonsWithVoxels(const char *fileBase)
{
	FILE *fpVoxel, *fpPhoton;
	char fileName[MAX_PATH];

	sprintf_s(fileName, MAX_PATH, "%s_voxel.ooc", fileBase);
	fopen_s(&fpVoxel, fileName, "rb");
	sprintf_s(fileName, MAX_PATH, "%s_photonVoxel.ooc", fileBase);
	fopen_s(&fpPhoton, fileName, "wb");

	__int64 numVoxels = (_filelengthi64(_fileno(fpVoxel)) - sizeof(OctreeHeader)) / sizeof(Voxel);

	Voxel *oldOctree = m_octree;
	printf("Allocating %I64d bytes...\n", numVoxels*sizeof(Voxel));
	if(!(m_octree = new Voxel[numVoxels]))
	{
		printf("Memory allocating error! (%I64d bytes required)\n", numVoxels*sizeof(Voxel));
		fclose(fpVoxel);
		fclose(fpPhoton);
		m_octree = oldOctree;
		return;
	}

	printf("Read voxels...\n");
	
	// read voxels
	fread(&m_header, sizeof(OctreeHeader), 1, fpVoxel);
	fread(m_octree, sizeof(Voxel), numVoxels, fpVoxel);
	fclose(fpVoxel);

	printf("Allocating %I64d bytes...\n", numVoxels*sizeof(PhotonVoxel));
	PhotonVoxel *photonVoxel;
	if(!(photonVoxel = new PhotonVoxel[numVoxels]))
	{
		printf("Memory allocating error! (%I64d bytes required)\n", numVoxels*sizeof(PhotonVoxel));

		delete[] m_octree;
		fclose(fpPhoton);
		m_octree = oldOctree;
		return;
	}

	printf("Trace photons...\n");
	Scene *scene = OpenIRT::getSingletonPtr()->getCurrentScene();
	for(int i=0;i<scene->getNumEmitters();i++)
	{
		int numPhotons = scene->getEmitter(i).numScatteringPhotons;

		if(numPhotons == 0) continue;

#		pragma omp parallel for
		for(int j=0;j<numPhotons;j++)
			tracePhotons(i, photonVoxel, j);
	}

	printf("Write photon voxels...\n");
	fwrite(photonVoxel, sizeof(PhotonVoxel), numVoxels, fpPhoton);

	delete[] m_octree;

	fclose(fpPhoton);

	m_octree = oldOctree;
}

void PhotonOctree::tracePhotonsWithFullDetailedVoxels(const char *fileBase)
{
	typedef struct OOCVoxel_t
	{
		int rootChildIndex;
		int startDepth;
		int offset;
		int numVoxels;
		AABB rootBB;
	} OOCVoxel;

	FILE *fpHigh, *fpLow, *fpHeader, *fpPhoton;
	char fileName[MAX_PATH];

	sprintf_s(fileName, MAX_PATH, "%s_voxel.ooc", fileBase);
	fopen_s(&fpHigh, fileName, "rb");
	sprintf_s(fileName, MAX_PATH, "%s_OOCVoxel.ooc", fileBase);
	fopen_s(&fpLow, fileName, "rb");
	sprintf_s(fileName, MAX_PATH, "%s_OOCVoxel.hdr", fileBase);
	fopen_s(&fpHeader, fileName, "rb");
	sprintf_s(fileName, MAX_PATH, "%s_photonVoxel.ooc", fileBase);
	fopen_s(&fpPhoton, fileName, "wb");

	__int64 numHighVoxels = (_filelengthi64(_fileno(fpHigh)) - sizeof(OctreeHeader)) / sizeof(Voxel);
	__int64 numLowVoxels = _filelengthi64(_fileno(fpLow)) / sizeof(Voxel);
	__int64 numOVoxels = _filelengthi64(_fileno(fpHeader)) / sizeof(OOCVoxel);

	Voxel *oldOctree = m_octree;
	printf("Allocating %I64d bytes...\n", (numHighVoxels + numLowVoxels)*sizeof(Voxel));
	if(!(m_octree = new Voxel[numHighVoxels + numLowVoxels]))
	{
		printf("Memory allocating error! (%I64d bytes required)\n", (numHighVoxels + numLowVoxels)*sizeof(Voxel));
		fclose(fpHigh);
		fclose(fpLow);
		fclose(fpHeader);
		fclose(fpPhoton);
		m_octree = oldOctree;
		return;
	}

	printf("Read voxels...\n");
	// read OOC voxel header
	OOCVoxel *oocVoxelList = new OOCVoxel[numOVoxels];
	fread(oocVoxelList, sizeof(OOCVoxel), numOVoxels, fpHeader);
	fclose(fpHeader);
	
	__int64 offset = 0;
	// read high level voxels
	fread(&m_header, sizeof(OctreeHeader), 1, fpHigh);
	offset += fread(&m_octree[offset], sizeof(Voxel), numHighVoxels, fpHigh);
	fclose(fpHigh);

	// read low level voxels
	fread(&m_octree[offset], sizeof(Voxel), numLowVoxels, fpLow);
	fclose(fpLow);

	printf("Link voxels...\n");
	// link all
	for(int i=0;i<numOVoxels;i++)
	{
		const OOCVoxel &oocVoxel = oocVoxelList[i];

		m_octree[oocVoxel.rootChildIndex].setChildIndex((int)(offset/8));
		//m_octree[oocVoxel.rootChildIndex].setLink2Low(i);
		for(int j=0;j<oocVoxel.numVoxels;j++)
		{
			Voxel &voxel = m_octree[offset+j];
			if(voxel.hasChild())
				voxel.setChildIndex(voxel.getChildIndex() + (int)(offset/8));
		}

		offset += oocVoxel.numVoxels;
	}

	printf("Allocating %I64d bytes...\n", (numHighVoxels + numLowVoxels)*sizeof(PhotonVoxel));
	PhotonVoxel *photonVoxel;
	if(!(photonVoxel = new PhotonVoxel[numHighVoxels + numLowVoxels]))
	{
		printf("Memory allocating error! (%I64d bytes required)\n", (numHighVoxels + numLowVoxels)*sizeof(PhotonVoxel));

		delete[] m_octree;
		delete[] oocVoxelList;
		fclose(fpPhoton);
		m_octree = oldOctree;
		return;
	}
	/*
	Ray ray;
	//Vector3 ori(252.517776f, 119.610329f, 0.000000f);
	//Vector3 dir(253.448547f, 119.848061f, -0.277751f);
	//dir = dir - ori;
	//dir.makeUnitVector();
	Vector3 ori(428.910797f, 167.063599f, -52.698574f);
	Vector3 dir(-0.558171, 0.814202f, 0.159747f);
	ray.set(ori, dir);
	HitPointInfo hit;
	hit.t = FLT_MAX;
	Material material;
	int hitIndex;
	float hitBBSize;
	bool isHit = RayOctreeIntersect(ray, hit, material, hitIndex, hitBBSize, 0.0f, 0.0f, 0, 0, 0);
	*/
	printf("Trace photons...\n");
	Scene *scene = OpenIRT::getSingletonPtr()->getCurrentScene();
	for(int i=0;i<scene->getNumEmitters();i++)
	{
		int numPhotons = scene->getEmitter(i).numScatteringPhotons;

		if(numPhotons == 0) continue;

#		pragma omp parallel for
		for(int j=0;j<numPhotons;j++)
			tracePhotons(i, photonVoxel, j);
	}

	printf("Write photon voxels...\n");
	int curWrittenVoxels = 0;
	int curPos = (int)numHighVoxels;
	int blockSize = 1024*1024;
	while(curWrittenVoxels < numLowVoxels)
	{
		int curSize = (int)(blockSize < numLowVoxels - curWrittenVoxels ? blockSize : numLowVoxels - curWrittenVoxels);//min(blockSize, numLowVoxels - curWrittenVoxels);
		fwrite(&photonVoxel[curPos], sizeof(PhotonVoxel), curSize, fpPhoton);
		curWrittenVoxels += curSize;
		curPos += curSize;
		//fwrite(&photonVoxel[numHighVoxels], sizeof(PhotonVoxel), numLowVoxels, fpPhoton);
	}

	/*
	Ray ray;
	Vector3 ori(108.576164f, 330.403473f, -90.860794f);
	Vector3 target(109.013214f, 329.716797f, -90.279877f);
	Vector3 dir = target - ori;
	dir.makeUnitVector();
	ray.set(ori, dir);

	HitPointInfo hit;
	hit.t = FLT_MAX;

	Material material;
	int hitIndex;
	float hitBBSize;
	bool hashit = RayOctreeIntersect(ray, hit, material, hitIndex, hitBBSize, 0.0f, 0.0f, 0, 0, 0);
	*/

	delete[] m_octree;
	delete[] oocVoxelList;

	fclose(fpPhoton);

	m_octree = oldOctree;
}