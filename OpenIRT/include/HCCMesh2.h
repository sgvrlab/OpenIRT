/********************************************************************
	created:	2011/10/31
	file path:	d:\Projects\Redering\OpenIRT\include
	file base:	HCCMesh22
	file ext:	h
	author:		Tae-Joon Kim (tjkim@tclab.kaist.ac.kr)
	
	comment:	HCCMesh22 class, has model instance with HCCMesh22 representation. 
				Provide functions for ray casting into a model.
*********************************************************************/

#pragma once

#include "CommonOptions.h"
#include "Model.h"

#define NUM_QUANTIZE_NORMALS 104

namespace irt
{

class HCCMesh2 : public Model
{
	typedef struct Vert16_t
	{
		Vector3 v;
		unsigned int m;
	} Vert16;

// Member variables
protected:
	// geometry
	Vert16 *m_vertList;
	unsigned int *m_triList;
	unsigned int *m_clusterVertOffset;
	unsigned int m_numClusters;

	Vector3 m_quantizedNormals[NUM_QUANTIZE_NORMALS*NUM_QUANTIZE_NORMALS*6];
// Member functions
public:
	HCCMesh2(void);
	~HCCMesh2(void);
	virtual ModelType getType() {return HCCMESH2;}

	virtual bool load(const char *fileName);

	inline Vert16 *getVertex(const int clusterID, const Index_t n);
	inline unsigned int getTriangle(const Index_t n);

	int getNumTriangles(const Index_t n);
	int getNumTriangles(const BVHNode *n);
	inline int getClusterID(const Index_t n);
	inline int getClusterID(const BVHNode *n);

	virtual bool getIntersection(const Ray &ray, HitPointInfo &hitPointInfo, float tLimit = 0.0f, int stream = 0);
	static inline bool getIntersection(const Ray &ray, Vector3 *box, float &interval_min, float &interval_max);
	inline bool getIntersection(const Ray &ray, BVHNode *node, HitPointInfo &hitPointInfo, float tmax);
	virtual void updateTransformedBB(AABB &bb, const Matrix &mat);

	FORCEINLINE void storeOrderedChildren(BVHNode *node, SIMDRay &rays, BVHNode **nearNode, BVHNode **farNode) {
		BVHNode *child = getBV(getLeftChildIdx(node));
		int axis = getAxis(node);
		*farNode = child + rays.rayChildOffsets[axis];
		*nearNode = child + (rays.rayChildOffsets[axis] ^ 1);
	}
	RayPacketTemplate
	bool getIntersectionWithTri(RayPacketT &rayPacket, int clusterID, int triID, int firstActiveRay);
	RayPacketTemplate 
	bool getIntersection(RayPacketT &rayPacket, int stream = 0);
};

#include "updateSIMDHitpoints.h"

RayPacketTemplate 
bool HCCMesh2::getIntersectionWithTri(RayPacketT &rayPacket, int clusterID, int triID, int firstActiveRay) 
{
	unsigned int tri = m_triList[triID];
	HCCMesh2 *modelPtr = this;
	int p[3];
	p[0] = (tri >>  2) & 0x3FF;
	p[1] = (tri >> 12) & 0x3FF;
	p[2] = (tri >> 22) & 0x3FF;
	/*
	const Vector3 &tri_p0 = getVertex(clusterID, p[0])->v; 
	const Vector3 &tri_p1 = getVertex(clusterID, p[1])->v; 
	const Vector3 &tri_p2 = getVertex(clusterID, p[2])->v;
	*/

#	ifdef USE_VERTEX_NORMALS
	Vert16 *verts[3] = {getVertex(clusterID, p[0]), getVertex(clusterID, p[1]), getVertex(clusterID, p[2])};

	Vector3 &tri_p0 = verts[0]->v;
	Vector3 &tri_p1 = verts[1]->v;
	Vector3 &tri_p2 = verts[2]->v;

	Vector3 vertNormals[3] = {Vector3(0, 1, 0), Vector3(0, 1, 0), Vector3(0, 1, 0)};
#	else
	const Vector3 &tri_p0 = getVertex(clusterID, p[0])->v;
	const Vector3 &tri_p1 = getVertex(clusterID, p[1])->v;
	const Vector3 &tri_p2 = getVertex(clusterID, p[2])->v;
#	endif

#	ifdef USE_TEXTURING
	Vector2 vertTextures[3];	// do not support texture for HCCMesh yet
#	endif

	Vector3 triN;
	triN = cross(tri_p1-tri_p0, tri_p2-tri_p0);
	triN.makeUnitVector();
	unsigned int matID = getVertex(clusterID, p[0])->m;

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
			const __m128 nominator = _mm_set1_ps(dot(triN, tri_p0 - rayPacket.origin));		

			const __m128 trin_x = _mm_set1_ps(triN.e[0]);
			const __m128 trin_y = _mm_set1_ps(triN.e[1]);
			const __m128 trin_z = _mm_set1_ps(triN.e[2]);	
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

	const __m128 nominator = _mm_set1_ps(dot(triN, tri_p0 - rayPacket.origin));		

	const __m128 trin_x = _mm_set1_ps(triN.e[0]);
	const __m128 trin_y = _mm_set1_ps(triN.e[1]);
	const __m128 trin_z = _mm_set1_ps(triN.e[2]);
	
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
bool HCCMesh2::getIntersection(RayPacketT &rayPacket, int stream) 
{
	int threadID = omp_get_thread_num();
	StackElem *stack = stacks[threadID+MAX_NUM_THREADS*stream];
	BVHNode *currentNode;
	int stackPtr;		
	int firstNonHit = 0;	
	stack[0].node = 0;
	stackPtr = 1;	
	currentNode = &m_nodeList[0];

	// traverse BVH tree:
	while (1) {
		// is current node intersected and also closer than previous hit?
		firstNonHit = rayPacket.intersectWithBox(&currentNode->min, firstNonHit); // does intersect?

		if (firstNonHit < nRays) { // yes, at least one ray intersects

			// is inner node?
			if (!isLeaf(currentNode)) {				
				storeOrderedChildren(currentNode, rayPacket.rays[firstNonHit], &currentNode, &stack[stackPtr].node);				
				stack[stackPtr++].firstNonHit = firstNonHit;
				continue;
			}
			else {				
				// is leaf node:
				// intersect with current node's members
				int clusterID = getClusterID(currentNode);
				int count = getNumTriangles(currentNode);
				int idxList = getTriangleIdx(currentNode);
				for(int i=0;i<count;i++, idxList++)
					HCCMesh2::getIntersectionWithTri(rayPacket, clusterID, idxList, firstNonHit);
			}
		}

		// fetch next node from stack
		currentNode = stack[--stackPtr].node;

		firstNonHit = stack[stackPtr].firstNonHit;
		
		// traversal ends when stack empty
		if (currentNode == NULL)
			break;
	}

	// return hit status
	return 1;

}

};