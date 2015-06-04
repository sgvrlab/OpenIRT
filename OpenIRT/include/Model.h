/********************************************************************
	created:	2009/06/07
	file path:	d:\Projects\Redering\OpenIRT\include
	file base:	Model
	file ext:	h
	author:		Tae-Joon Kim (tjkim@tclab.kaist.ac.kr)
	
	comment:	Model class, has model instance. Provide functions for
				ray casting into a model.
*********************************************************************/

#pragma once

typedef unsigned int Index_t;
#define MAX_NUM_THREADS 32
#define MAX_NUM_INTERSECTION_STREAM 4

#include "Vertex.h"
#include "Triangle.h"
#include "Face.h"
#include "BVHNode.h"
#include "Material.h"
#include "HitPointInfo.h"
#include "Ray.h"
#include "RayPacket.h"
#include "Matrix.h"
#include <map>

namespace irt
{

class BVHBuilder;

class Model
{
public:
	friend BVHBuilder;

	enum ModelType
	{
		NONE,
		OOC_FILE,
		HCCMESH,
		HCCMESH2,
		SMALL_MODEL
	};
// Member variables
protected:
	char m_fileName[256];
	char m_name[256];

	// geometry
	Vertex *m_vertList;
	Triangle *m_triList;
	BVHNode *m_nodeList;
	int m_numVerts;
	int m_numTris;
	int m_numNodes;

	bool m_visible;
	bool m_enabled;

	// array of stacks (one for each thread when using parallel threads, otherwise just one)
	typedef struct {
		Index_t index;
		BVHNode *node;
		Index_t firstNonHit;		
	} StackElem;
	__declspec(align(16)) StackElem **stacks;

	// material
	bool m_useMTL;	// use material template library (MTL)
	MaterialList m_matList;

	//Vector3 m_BBMin, m_BBMax; // bounding box of this model
	AABB m_BB;
	Matrix m_transfMatrix;
	Matrix m_invTransfMatrix;

// Member functions
public:
	Model(void);
	virtual ~Model(void);
	virtual ModelType getType() {return OOC_FILE;}

	virtual bool load(const char *fileName);
	virtual void unload();

	bool isVisible() {return m_visible;}
	bool isEnabled() {return m_enabled;}
	void setVisibility(bool visible) {m_visible = visible;}
	void enableModel(bool enable) {m_enabled = enable;}

	bool load(Vertex *vertList, int numVerts, Face *faceList, int numFaces, const Material &mat);

	const char *getFileName() {return m_fileName;}

	void setName(const char *name) {strcpy_s(m_name, 256, name);}
	const char *getName() {return m_name;}
	void saveMaterials(const char *fileName);

	int getNumVertexs() {return m_numVerts;}
	int getNumTriangles() {return m_numTris;}
	int getNumNodes() {return m_numNodes;}
	int getNumMaterials() {return (int)m_matList.size();}
	virtual int getNumIndices() {return 0;}

	// APIs for accessing vertices and triangles
	Vertex *getVertex(const Index_t n);
	Triangle *getTriangle(const Index_t n);

	// APIs for accessing BVH
	Index_t getRootIdx();
	BVHNode *getBV(const Index_t n);
	bool isLeaf(const Index_t n);
	Index_t getLeftChildIdx(const Index_t n);
	Index_t getRightChildIdx(const Index_t n);
	Index_t getTriangleIdx(const Index_t n);
	int getNumTriangles(const Index_t n);
	int getAxis(const Index_t n);

	// overloaded APIs for efficiency
	bool isLeaf(const BVHNode *n);
	Index_t getLeftChildIdx(const BVHNode *n);
	Index_t getRightChildIdx(const BVHNode *n);
	Index_t getTriangleIdx(const BVHNode *n);
	int getNumTriangles(const BVHNode *n);
	int getAxis(const BVHNode *n);

	void setModelBB(const AABB &bb) {m_BB = bb;}
	const AABB &getModelBB() {return m_BB;}

	Material &getMaterial(int idx) {return m_matList[idx >= 0 && idx < (int)m_matList.size() ? idx : 0];}
	Material getMaterial(const HitPointInfo &hit)
	{
		Material mat = getMaterial(hit.m);
		RGBf texValue;

		BitmapTexture *map = NULL;

		map = mat.getMapKa();
		if(map)
		{
			map->getTexValue(texValue, hit.uv.e[0], hit.uv.e[1]);
			mat.setMatKa(mat.getMatKa() * texValue);
		}

		map = mat.getMapKd();
		if(map)
		{
			map->getTexValue(texValue, hit.uv.e[0], hit.uv.e[1]);
			mat.setMatKd(mat.getMatKd() * texValue);
		}

		return mat;
	}
	void setMaterial(Material &mat, int idx) {m_matList[idx >= 0 && idx < (int)m_matList.size() ? idx : 0] = mat;}

	const Matrix &getTransfMatrix() {return m_transfMatrix;}
	const Matrix &getInvTransfMatrix() {return m_invTransfMatrix;}
	void setTransfMatrix(const Matrix &mat) {m_transfMatrix = mat;}
	void setInvTransfMatrix(const Matrix &mat) {m_invTransfMatrix = mat;}

	virtual bool getIntersection(const Ray &ray, HitPointInfo &hitPointInfo, float tLimit = 0.0f, int stream = 0);
	bool getIntersection(const Ray &ray, Vector3 *box, float &interval_min, float &interval_max);
	bool getIntersection(const Ray &ray, BVHNode *node, HitPointInfo &hitPointInfo, float tmax);

	virtual bool isIntersect(const AABB &a) {return false;}
	bool isOverlap(const AABB &a, const BVHNode *b);	// implemented in Voxelize.cpp
	bool isOverlap(const AABB &a, const AABB &b);		// implemented in Voxelize.cpp
	bool isOverlap(const AABB &box, Vector3 *vert);	// implemented in Voxelize.cpp

	virtual void updateTransformedBB(AABB &bb, const Matrix &mat);

	// an example for building a model
	Triangle makeTriangle(const Vector3 &v0, const Vector3 &v1, const Vector3 &v2, Index_t indexList[], unsigned short material = 0);
	void makeCornellBoxModel(const char *filePath);

	RayPacketTemplate
	bool getIntersectionWithTri(RayPacketT &rayPacket, int triID, int firstActiveRay);
	RayPacketTemplate 
	bool getIntersection(RayPacketT &rayPacket, int stream = 0);
};

#include "updateSIMDHitpoints.h"

RayPacketTemplate 
bool Model::getIntersectionWithTri(RayPacketT &rayPacket, int triID, int firstActiveRay) 
{
	Model *modelPtr = this;

	const Triangle &tri = *getTriangle(triID);
	
	Vertex *verts[3] = {getVertex(tri.p[0]), getVertex(tri.p[1]), getVertex(tri.p[2])};

	Vector3 &tri_p0 = verts[0]->v;
	Vector3 &tri_p1 = verts[1]->v;
	Vector3 &tri_p2 = verts[2]->v;

	Vector3 vertNormals[3] = {verts[0]->n, verts[1]->n, verts[2]->n};

#	ifdef USE_TEXTURING
	Vector2 vertTextures[3] = {verts[0]->uv, verts[1]->uv, verts[2]->uv};
#	endif

	Index_t matID = tri.material;
	if(getMaterial(matID).getMat_d() < 0.1f) return false;

	const __m128 origin4 = _mm_load_ps(rayPacket.origin.e);
	const __m128 aminusO = _mm_sub_ps(_mm_load_ps(tri_p0.e), origin4);
	const __m128 bminusO = _mm_sub_ps(_mm_load_ps(tri_p1.e), origin4);
	const __m128 cminusO = _mm_sub_ps(_mm_load_ps(tri_p2.e), origin4);

	// compute cross(cminusO, bminusO)
	const __m128 v0cross =  _mm_sub_ps(_mm_mul_ps( cminusO, _mm_shuffle_ps(bminusO, bminusO, _MM_SHUFFLE(3, 0, 2, 1)) ),
									   _mm_mul_ps( _mm_shuffle_ps(cminusO, cminusO, _MM_SHUFFLE(3, 0, 2, 1)), bminusO ));
	const __m128 v0cross_x = _mm_shuffle_ps(v0cross, v0cross, _MM_SHUFFLE(1, 1, 1, 1));
	const __m128 v0cross_y = _mm_shuffle_ps(v0cross, v0cross, _MM_SHUFFLE(2, 2, 2, 2));
	const __m128 v0cross_z = _mm_shuffle_ps(v0cross, v0cross, _MM_SHUFFLE(0, 0, 0, 0));
	
	// compute cross(bminusO, aminusO)
	const __m128 v1cross =  _mm_sub_ps(_mm_mul_ps( bminusO, _mm_shuffle_ps(aminusO, aminusO, _MM_SHUFFLE(3, 0, 2, 1)) ),
									   _mm_mul_ps( _mm_shuffle_ps(bminusO, bminusO, _MM_SHUFFLE(3, 0, 2, 1)), aminusO ));
	const __m128 v1cross_x = _mm_shuffle_ps(v1cross, v1cross, _MM_SHUFFLE(1, 1, 1, 1));
	const __m128 v1cross_y = _mm_shuffle_ps(v1cross, v1cross, _MM_SHUFFLE(2, 2, 2, 2));
	const __m128 v1cross_z = _mm_shuffle_ps(v1cross, v1cross, _MM_SHUFFLE(0, 0, 0, 0));

	// compute cross(aminusO, cminusO)
	const __m128 v2cross =  _mm_sub_ps(_mm_mul_ps( aminusO, _mm_shuffle_ps(cminusO, cminusO, _MM_SHUFFLE(3, 0, 2, 1)) ),
									   _mm_mul_ps( _mm_shuffle_ps(aminusO, aminusO, _MM_SHUFFLE(3, 0, 2, 1)), cminusO ));
	const __m128 v2cross_x = _mm_shuffle_ps(v2cross, v2cross, _MM_SHUFFLE(1, 1, 1, 1));
	const __m128 v2cross_y = _mm_shuffle_ps(v2cross, v2cross, _MM_SHUFFLE(2, 2, 2, 2));
	const __m128 v2cross_z = _mm_shuffle_ps(v2cross, v2cross, _MM_SHUFFLE(0, 0, 0, 0));	

	if (hasCornerRays) {	
		//
		// try early rejection or accept of ray frustum:
		//

		const __m128 dx = _mm_load_ps(rayPacket.cornerRays.direction[0]);
		const __m128 dy = _mm_load_ps(rayPacket.cornerRays.direction[1]);
		const __m128 dz = _mm_load_ps(rayPacket.cornerRays.direction[2]);

		register const __m128 zero = _mm_setzero_ps();
		const __m128 v0d = _mm_cmpgt_ps(_mm_dot3_ps(v0cross_x, v0cross_y, v0cross_z, dx, dy, dz), zero);
		const __m128 v1d = _mm_cmpgt_ps(_mm_dot3_ps(v1cross_x, v1cross_y, v1cross_z, dx, dy, dz), zero);
		const __m128 v2d = _mm_cmpgt_ps(_mm_dot3_ps(v2cross_x, v2cross_y, v2cross_z, dx, dy, dz), zero);

		#ifdef BACKFACE_CULL
		const __m128 veq = _mm_and_ps(v0d, _mm_and_ps(v1d, v2d));
		#else
		const __m128 veq = *(__m128 *)&_mm_and_si128(_mm_cmpeq_epi32(*(__m128i *)&v0d, *(__m128i *)&v1d), _mm_cmpeq_epi32(*(__m128i *)&v1d, *(__m128i *)&v2d));
		#endif

		// generate hitmask of rays that are inside the triangle
		// (which is if the signs of the three edge tests match)
		if (_mm_movemask_ps(veq) == 15) {
			// early accept: full hit of triangle
			const __m128 nominator = _mm_set1_ps(dot(tri.n, tri_p0 - rayPacket.origin));		

			const __m128 trin_x = _mm_set1_ps(tri.n.e[0]);
			const __m128 trin_y = _mm_set1_ps(tri.n.e[1]);
			const __m128 trin_z = _mm_set1_ps(tri.n.e[2]);	
			// for each ray in ray packet:
			for (int r = firstActiveRay; r < nRays; r++) {
				const SIMDRay &rays = rayPacket.rays[r];
				SIMDHitpoint *hit = &rayPacket.hitpoints[r];
				
				#define USE_DIRECT_MAT_ID
				#define SKIP_INSIDE_TEST			
				#include "asm_intersect_onetri_pluecker.h"
				#undef SKIP_INSIDE_TEST
				#undef USE_DIRECT_MAT_ID
						
				rayPacket.rayHasHit[r] |= newHitMask;		
			}
			return 1;				
		}			
		else if (_mm_movemask_ps(v0d) == 0 && _mm_movemask_ps(v1d) == 0 && _mm_movemask_ps(v2d) == 0) {
			// early reject: frustum misses triangle
			return 0;
		}		
	}

	const __m128 nominator = _mm_set1_ps(dot(tri.n, tri_p0 - rayPacket.origin));		

	const __m128 trin_x = _mm_set1_ps(tri.n.e[0]);
	const __m128 trin_y = _mm_set1_ps(tri.n.e[1]);
	const __m128 trin_z = _mm_set1_ps(tri.n.e[2]);
	
	// for each ray in ray packet:
	for (int r = firstActiveRay; r < nRays; r++) {
		const SIMDRay &rays = rayPacket.rays[r];
		SIMDHitpoint *hit = &rayPacket.hitpoints[r];
		
		#define USE_DIRECT_MAT_ID
		#include "asm_intersect_onetri_pluecker.h"
		#undef USE_DIRECT_MAT_ID
				
		rayPacket.rayHasHit[r] |= newHitMask;		
	}

	return 1;
}

RayPacketTemplate 
bool Model::getIntersection(RayPacketT &rayPacket, int stream) 
{
	if(!m_nodeList) return false;

	int threadID = omp_get_thread_num();
	StackElem *stack = stacks[threadID+MAX_NUM_THREADS*stream];
	BVHNode * currentNode;
	int stackPtr;		
	int firstNonHit = 0;	

	Index_t rootIndex = getRootIdx();
	currentNode = getBV(rootIndex);
	int curIdx = rootIndex;

	stack[0].index = rootIndex;
	stackPtr = 1;	

	Index_t lChild, rChild;
	int axis;

	// traverse BVH tree:
	while (1) {
		// is current node intersected and also closer than previous hit?
		firstNonHit = rayPacket.intersectWithBox(&currentNode->min, firstNonHit); // does intersect?

		if (firstNonHit < nRays) { // yes, at least one ray intersects
			// is inner node?
			if (!isLeaf(currentNode)) {				
				axis = getAxis(currentNode);
				
				if(rayPacket.rays[firstNonHit].rayChildOffsets[axis])
				{
					lChild = getLeftChildIdx(currentNode);
					rChild = getRightChildIdx(currentNode);

					currentNode = getBV(lChild);
					curIdx = lChild;

					stack[stackPtr].index = rChild;
				}
				else
				{
					lChild = getLeftChildIdx(currentNode);
					rChild = getRightChildIdx(currentNode);

					currentNode = getBV(rChild);
					curIdx = rChild;

					stack[stackPtr].index = lChild;
				}

				stack[stackPtr++].firstNonHit = firstNonHit;
				continue;
			}
			else {				
				// is leaf node:
				// intersect with current node's members
				Model::getIntersectionWithTri(rayPacket, getTriangleIdx(currentNode), firstNonHit);
			}
		}

		// traversal ends when stack empty
		if(--stackPtr == 0) break;

		// fetch next node from stack
		currentNode = getBV(stack[stackPtr].index);
		curIdx = stack[stackPtr].index;

		firstNonHit = stack[stackPtr].firstNonHit;
		
	}

	// return hit status
	return 1;

}

typedef std::vector<Model*> ModelList;
typedef std::map<Model*, int> ModelListMap;

};