#pragma once

#include "Voxel.h"
#include "BV.h"
#include "Octree.h"
#include "Material.h"
#include "Model.h"

#define N 2
#define MAX_DEPTH 6
#define MAX_DEPTH2 8

namespace irt
{

class VoxelMaterialExtra
{
public:
	Vector3 summedKd;
	Vector3 summedKs;
	float summedD;
	float summedNs;

	int numElements;

	VoxelMaterialExtra() : summedKd(Vector3(0.0f)), summedKs(Vector3(0.0f)), summedD(0), summedNs(0), numElements(0) {}

	void addMaterial(const Material &material)
	{
		summedKd += material.getMatKd();
		summedKs += material.getMatKs();
		summedD += material.getMat_d();
		summedNs += material.getMat_Ns();
		numElements++;
	}

	static VoxelMaterialExtra getTransparentMaterial()
	{
		VoxelMaterialExtra mat;
		mat.summedKs = Vector3(1.0f);
		mat.summedNs = 2048.0f;
		mat.numElements = 1;
		return mat;
	}

	Vector3 getKd() {return summedKd / (float)numElements;}
	Vector3 getKs() {return summedKs / (float)numElements;}
	float getD() {return summedD / numElements;}
	float getNs() {return summedNs / numElements;}
	void normailize()
	{
		summedKd = getKd();
		summedKs = getKs();
		summedD = getD();
		summedNs = getNs();
		numElements = 1;
	}

	VoxelMaterialExtra operator + (const VoxelMaterialExtra &x) const
	{
		VoxelMaterialExtra sum;

		sum.summedKd = summedKd + x.summedKd;
		sum.summedKs = summedKs + x.summedKs;
		sum.summedD = summedD + x.summedD;
		sum.summedNs = summedNs + x.summedNs;
		sum.numElements = numElements + x.numElements;

		return sum;
	}
};

class Voxelize
{
protected:
	typedef struct OOCVoxel_t
	{
		int rootChildIndex;
		int startDepth;
		int offset;
		int numVoxels;
		AABB rootBB;

		OOCVoxel_t(int rootChildIndex, int startDepth, const AABB &rootBB) :
			rootChildIndex(rootChildIndex), startDepth(startDepth), rootBB(rootBB), offset(0), numVoxels(0) {}
	} OOCVoxel;

protected:
	Model *m_model;
	AABB m_BB;
	Vector3 m_voxelDelta;
	int m_lastIndex;

	Octree m_octree;
	VoxelMaterialExtra *m_leafVoxelMat;
	MaterialList m_matList;
public:
	Voxelize(Model *model);
	~Voxelize();

	/*
	bool isIntersect(const AABB &box, const Triangle &tri);
	bool isIntersect(const AABB &box, Vector3 *vert);
	bool isIntersect(const AABB &a, const BVHNode *b);
	bool isIntersect(const AABB &a);
	*/
	AABB computeSubBox(int x, int y, int z, const AABB &box);
	void setGeomBitmap(Voxel &voxel, const AABB &box);
	int createOctreeNode();
	int createOctreeNode(FILE *fp, const OOCVoxel &voxel);
	int createOctreeNode(FILE *fp, int parentIndex, int myIndex, int &lastIndex, const AABB &box, int depth);

	float triArea(std::vector<Vector3> &verts, int pos);
	void applyTri(int index, const AABB &bb, Vector3 *vert, const Vector3 &norm, const Material &material);
	VoxelMaterialExtra computeMaterialLOD(int index);
	void computeLOD(const char *fileName, int startDepth = 1);
	void setGeomLOD(int index, const AABB &bb);

	//int Do(const char *filepath);

	//bool depthTest(int depth, const AABB &bb);
};

};