/********************************************************************
	created:	2011/08/28
	file path:	d:\Projects\Redering\OpenIRT\include
	file base:	GLDebugging
	file ext:	h
	author:		Tae-Joon Kim (tjkim.kaist@gmail.com)
	
	comment:	Rasterizer for debugging using OpenGL
*********************************************************************/

#pragma once

#include "Renderer.h"
#include "Photon.h"
#include "PhotonOctree.h"
#include "Plane.h"
#include "GLAdapter.h"

namespace irt
{

class GLDebugging :
	public Renderer, public GLAdapter
{
protected:
	Photon *m_photons;
	int m_numPhotons;
	AABB m_bbPhotons;
	PhotonOctree m_octree;
	PhotonOctree *m_octreeList;
	std::vector<AABB> m_bbList;

public:
	GLDebugging(void);
	virtual ~GLDebugging(void);

	virtual void init(Scene *scene);
	virtual void done();

	virtual void resized(int width, int height);

	virtual void sceneChanged();

	// renderer
	virtual void render(Camera *camera, Image *image, unsigned int seed = UINT_MAX);

	void render(SceneNode *sceneNode);
	void render(Camera *camera, int voxel, const AABB &bb, int index, bool drawGeomBit, int depth = 0);
	void visVoxels(Camera *camera, int voxel, const AABB &bb, int index, bool drawGeomBit, int depth = 0);
	void renderCut(const AABB &bb, int index, const Vector3 &from);

	int frustumBoxIntersect(int numPlane, const Plane plane[], const AABB &bb);
	void renderCulling();
	void renderCulling(const Plane plane[], int index, int depth = 0);

	AABB computeSubBox(int x, int y, int z, const AABB &box);
	void tracePhotons();
	void buildPhotonKDTree();
	void renderPhotonBB(int curNode, const AABB &curBB, int depth);

	bool RayLODIntersect(const Ray &ray, const AABB &bb, const Voxel &voxel, HitPointInfo &hit, Material &material, float tmax, unsigned int seed);
	bool RayOctreeIntersect(const Ray &ray, AABB *startBB, int startIdx, HitPointInfo &hit, AABB &hitBB, Material &material, int &hitIndex, float limit, float tLimit, unsigned int seed, int x, int y);
};

};