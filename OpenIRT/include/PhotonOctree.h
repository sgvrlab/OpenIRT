#pragma once

#include "Octree.h"
#include "Photon.h"
#include "Ray.h"
#include "HitPointInfo.h"
#include "Material.h"

namespace irt
{

class PhotonOctree : public Octree
{
public:
	PhotonOctree(void);

	static void addPhoton(const Photon& photon);

	RGBf makeLOD(int index);

	bool RayLODIntersect(const Ray &ray, const Voxel &voxel, HitPointInfo &hit, Material &material, float tmax, unsigned int seed);
	bool RayOctreeIntersect(const Ray &ray, HitPointInfo &hit, Material &material, int &hitIndex, float &hitBBSize, float limit, float tLimit, unsigned int seed, int x, int y);
	void tracePhotons(int emitterIndex, PhotonVoxel *photonOctree, int idx);
	void tracePhotonsWithVoxels(const char *fileBase);
	void tracePhotonsWithFullDetailedVoxels(const char *fileBase);
};

};