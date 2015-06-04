#ifndef VOXELIZE_H
#define VOXELIZE_H
#include "BVHNodeDefine.h"
#include "Vertex.h"
#include "Triangle.h"
#include "Photon.h"
#include "Voxel.h"
#include "BV.h"
#include "OOCFile6464.h"
#include "Progression.h"
#include "Octree.h"
#include "OOC_PCA.h"
#include "NewMaterial.h"

//#define BUILD_BOEING
#define BUILD_SPONZA

#define BVHNode BSPArrayTreeNode
#define OOC_FILE_CLASS OOCFile6464
#define N 2

#ifdef BUILD_BOEING
#define MAX_DEPTH 10
#define MAX_DEPTH2 12
#define MAX_LOW_DEPTH 5
#endif

#ifdef BUILD_SPONZA
#define MAX_DEPTH 8
#define MAX_DEPTH2 10
#define MAX_LOW_DEPTH 5
#endif

class VoxelMaterialExtra
{
public:
	Vector3 summedKd;
	Vector3 summedKs;
	float summedD;
	float summedNs;

	int numElements;

	VoxelMaterialExtra() : summedKd(Vector3(0.0f)), summedKs(Vector3(0.0f)), summedD(0), summedNs(0), numElements(0) {}

	void addMaterial(const NewMaterial &material)
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

	Vector3 getKd() {return summedKd / numElements;}
	Vector3 getKs() {return summedKs / numElements;}
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

	typedef std::vector<OOCVoxel> OOCVoxelList;

protected:
	FILE *m_fp;

	OOC_FILE_CLASS<Vertex> *m_vertPtr;
	OOC_FILE_CLASS<Triangle> *m_triPtr;
	OOC_FILE_CLASS<BVHNode> *m_nodePtr;
	
	OOC_FILE_CLASS<Vertex> m_vert;
	OOC_FILE_CLASS<Triangle> m_tri;
	OOC_FILE_CLASS<BVHNode> m_node;

	AABB m_BB;
	Vector3 m_voxelDelta;
	float m_intersectEpsilon;
	int m_lastIndex;
	unsigned char *m_indexMap;

	Octree m_octree;
	COOCPCAwoExtent *m_PCAOctree;
	VoxelMaterialExtra *m_leafVoxelMat;
	NewMaterialList m_matList;
	OOCVoxelList m_oocVoxelList;

	BVHNode *m_oocVoxelBSPtree;
public:
	Voxelize();
	~Voxelize();

	bool isIntersect(const AABB &box, const Triangle &tri);
	bool isIntersect(const AABB &box, Vector3 *vert);
	bool isIntersect(const AABB &a, const BVHNode *b);
	bool isIntersect(const AABB &a);
	void getIntersectVoxels(const BVHNode *tree, const Triangle &tri, std::vector<int> &list);
	AABB computeSubBox(int x, int y, int z, const AABB &box);
	void setGeomBitmap(Voxel &voxel, const AABB &box);
	int createOctreeNode();
	int createOctreeNode(FILE *fp, const OOCVoxel &voxel);
	int createOctreeNode(FILE *fp, int parentIndex, int myIndex, int &lastIndex, const AABB &box, int depth, bool generateOOC = false, bool *isOOCPart = NULL);

	float triArea(std::vector<Vector3> &verts, int pos);
	void applyTri(int index, const AABB &bb, Vector3 *vert, const Vector3 &norm, const NewMaterial &material);
	COOCPCAwoExtent computeGeomLOD(const AABB &bb, int index);
	VoxelMaterialExtra computeMaterialLOD(int index);
	void computeLOD(const char *fileName, int startDepth = 1);
	void setGeomLOD(int index, const AABB &bb);

	int Do(const char *filepath, int maxDepth = 0, int maxDepth2 = 0, int maxLowDepth = 0);

	bool depthTest(int depth, const AABB &bb);

	bool loadMaterialFromMTL(const char *fileName, NewMaterialList &matList);

	typedef std::vector<unsigned int> TriangleIndexList;
	typedef TriangleIndexList::iterator TriangleIndexListIterator;
	FORCEINLINE void updateBB(Vector3 &min, Vector3 &max, const Vector3 &vec)
	{
		min.e[0] = ( min.e[0] < vec.e[0] ) ? min.e[0] : vec.e[0];
		min.e[1] = ( min.e[1] < vec.e[1] ) ? min.e[1] : vec.e[1];
		min.e[2] = ( min.e[2] < vec.e[2] ) ? min.e[2] : vec.e[2];

		max.e[0] = ( max.e[0] > vec.e[0] ) ? max.e[0] : vec.e[0];
		max.e[1] = ( max.e[1] > vec.e[1] ) ? max.e[1] : vec.e[1];
		max.e[2] = ( max.e[2] > vec.e[2] ) ? max.e[2] : vec.e[2];
	}
	bool Subdivide(BVHNode *tree, const AABB &rootBB, int &numNodes, TriangleIndexList *triIDs, unsigned int left, unsigned int right, unsigned int myIndex, unsigned int nextIndex, int depth);
	void buildBVH(int numTris, const AABB &bb, const char *fileName);
	void buildVoxelBVH(const char *fileName, const AABB &rootBB);
	void oocVoxelize(const char *filePath);

	void buildOOCVoxelBSPTree();
	bool SubdivideBSPTree(BVHNode *tree, const AABB &rootBB, int &numNodes, std::vector<int> *voxelIDs, unsigned int left, unsigned int right, unsigned int myIndex, unsigned int nextIndex, int depth);
};

#endif