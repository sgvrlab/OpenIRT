/********************************************************************
	created:	2011/09/31
	file path:	d:\Projects\Redering\OpenIRT\include
	file base:	HCCMesh
	file ext:	h
	author:		Tae-Joon Kim (tjkim@tclab.kaist.ac.kr)
	
	comment:	HCCMesh class, has model instance with HCCMesh representation. 
				Provide functions for ray casting into a model.
*********************************************************************/

#pragma once

#include <vector>
#include "CommonOptions.h"
#include "Model.h"

#define MAX_SIZE_TEMPLATE 15
#define DEPTH_TEMPLATE 4
#define MAX_NUM_TEMPLATES 26
#define NUM_QUANTIZE_NORMALS 104

#define BIT_MASK_16 0xFFFF
#define BIT_MASK_14 0x3FFF
#define BIT_MASK_9 0x1FF
#define BIT_MASK_5 0x1F
#define BIT_MASK_3 0x7
#define BIT_MASK_2 0x3
#define CISLEAF(node) (((node)->left & 1) == 1)
#define CISLEAFOFPATCH(node) (((node)->left & 0x3) == 2)
#define CISINCOMPLETETREE(node) (((node)->left & 8) == 8)

namespace irt
{

typedef struct TemplateTable_t {
	int numNodes;
	int numLeafs;
	BVHNode tree[MAX_SIZE_TEMPLATE];
	unsigned int listLeaf[(MAX_SIZE_TEMPLATE+1)/2];
} TemplateTable, *TemplateTablePtr;

class HCCMesh : public Model
{
	////////////////////////////////////////////////////////////////////
	// Defines for compact in-core representations
	////////////////////////////////////////////////////////////////////
	typedef union CompTreeNode_t
	{
		unsigned int data;
		unsigned int left;
	} CompTreeNode, *CompTreeNodePtr;

	typedef struct CompTreeSupp_t
	{
		unsigned short leftIndex;
		unsigned short rightIndex;
		unsigned int data;
	} CompTreeSupp, *CompTreeSuppPtr;

#ifdef USE_VERTEX_QUANTIZE

#pragma pack(push, 1)
	typedef struct CompTreeVert_t
	{
		unsigned short qV[3];
		unsigned int data;
	} CompTreeVert, *CompTreeVertPtr;
#pragma pack(pop)

#else
#pragma pack(push, 1)
	typedef struct QCompTreeVert_t
	{
		unsigned short qV[3];
		unsigned int data;
	} QCompTreeVert, *QCompTreeVertPtr;
#pragma pack(pop)

	typedef struct CompTreeVert_t
	{
		_Vector4 vert;
	} CompTreeVert, *CompTreeVertPtr;
#endif

	typedef struct CompClusterHeader_t
	{
		unsigned int fileSize;
		unsigned int numNode;
		unsigned int numSupp;
		unsigned int numVert;
#ifdef USE_HCCMESH_MT
		unsigned int sizeTris;
#endif
		unsigned int rootType;
		Vector3 BBMin;
		Vector3 BBMax;
	} CompClusterHeader, *CompClusterHeaderPtr;

	typedef struct CompCluster_t
	{
		CompClusterHeader header;
		CompTreeNodePtr node;
		CompTreeSuppPtr supp;
		CompTreeVertPtr vert;
#ifdef USE_HCCMESH_MT
		unsigned char *tris;
#endif
	} CompCluster, *CompClusterPtr;

	typedef struct QCompCluster_t
	{
		CompClusterHeader header;
		CompTreeNodePtr node;
		CompTreeSuppPtr supp;
		QCompTreeVertPtr vert;
#ifdef USE_HCCMESH_MT
		unsigned char *tris;
#endif
	} QCompCluster, *QCompClusterPtr;

#ifdef USE_HCCMESH_LOD
	typedef struct CompLOD_t
	{
		LODNode lod;
	} CompLOD, *CompLODPtr;
#endif

	typedef struct TempTri_t
	{
		unsigned short p[3];
#ifdef USE_HCCMESH_MT
		unsigned char i1, i2;
#endif
	} TempTri, *TempTriPtr;

	typedef struct TriList_t
	{
		unsigned short offsetOrBackup;
		unsigned char numTris;
		std::vector<TempTri> tris;
	} TriList, *TriListPtr;


	typedef struct TravStat_t
	{
		int cluster;
		unsigned int rootTemplate;
		int type;
		Index_t index;
		int isLeft;
		int axis;
		BVHNode node;
		/*
		Vector3 BBMin;
		Vector3 BBMax;
		*/
	} TravStat, *TravStatPtr;

	typedef struct TravStat2_t
	{
		int cluster;
		unsigned int rootTemplate;
		int type;
		int index;
	} TravStat2, *TravStatPtr2;


	typedef struct {
		unsigned int index;
		BVHNode node;
		TravStat ts;
		unsigned int minBB;
		unsigned int maxBB;
		unsigned int firstNonHit;
#ifdef USE_LOD
		char m_MinDim [4];
		char m_MaxDim [4];
#endif // USE_LOD
	} StackElem;

	/*
	class LRUEntry {
	public:
		unsigned int m_clusterID;
		LRUEntry* m_pNext, *m_pPrev;

		LRUEntry()
		{
			m_clusterID = 0;
			m_pNext = m_pPrev = NULL;
		}
	};

	typedef struct ClusterEntry_t {
		int Loaded;
		unsigned int fileSizeOut;
		unsigned int geomFileSizeOut;
		unsigned int fileSizeIn;
		__int64 fileStartOffset;
		__int64 geomFileStartOffset;
		LRUEntry *entryLRU;
	} ClusterEntry;
	*/

// Member variables
protected:
	// geometry
	unsigned char *m_compFile;
	unsigned int m_numClusters;

	__declspec(align(16)) StackElem **stacks;

	BVHNode *m_compHighTree;
#	ifdef USE_HCCMESH_QUANTIZATION
	QCompCluster *m_compCluster;
#	else
	CompCluster *m_compCluster;
#	endif

	int m_numTemplates;
	TemplateTable m_templates[MAX_NUM_TEMPLATES];
	Vector3 m_quantizedNormals[NUM_QUANTIZE_NORMALS*NUM_QUANTIZE_NORMALS*6];

	double m_qEnMult, m_qDeMult;
	const static unsigned int m_qStep = 0xFFFF;
// Member functions
public:
	HCCMesh(void);
	~HCCMesh(void);
	virtual ModelType getType() {return HCCMESH;}

	virtual bool load(const char *fileName);

	virtual bool getIntersection(const Ray &ray, HitPointInfo &hitPointInfo, float tLimit = 0.0f, int stream = 0);
	bool getIntersection(const Ray &ray, Vector3 *box, float &interval_min, float &interval_max);
	bool getIntersection(const Ray &ray, BVHNode *node, HitPointInfo &hitPointInfo, float tmax, TravStat &ts);

	//virtual bool isIntersect(const AABB &a);		// implemented in Voxelize.cpp

	virtual void updateTransformedBB(AABB &bb, const Matrix &mat);

	void calculateQuantizedNormals();

	BVHNode *getBV(unsigned int index, TravStat &ts, unsigned int minBB, unsigned int maxBB);
	Index_t getRootIdx(TravStat &ts);
	Index_t getLeftChildIdx(BVHNode* node, TravStat &ts, unsigned int &minBB);
	Index_t getRightChildIdx(BVHNode* node, TravStat &ts, unsigned int &maxBB);
#	ifdef USE_HCCMESH_QUANTIZATION
	_Vector4 getVertexC(unsigned int idx, TravStat &ts);
#	else
	_Vector4 &getVertexC(unsigned int idx, TravStat &ts);
#	endif
	Vertex getVertex(unsigned int idx, TravStat &ts);

	// read header and high level tree. return read bytes.
	void readHeader(FILE *fp);

	RayPacketTemplate
	bool getIntersectionWithTri(RayPacketT &rayPacket, int triID, int firstActiveRay, TravStat &ts);
	RayPacketTemplate 
	bool getIntersection(RayPacketT &rayPacket, int stream = 0);
};


#include "updateSIMDHitpoints.h"

RayPacketTemplate 
bool HCCMesh::getIntersectionWithTri(RayPacketT &rayPacket, int triID, int firstActiveRay, TravStat &ts) 
{
	HCCMesh *modelPtr = this;
	int p[3];
	p[0] = (triID >> 23) & 0x1FF;
	p[1] = (triID >> 14) & 0x1FF;
	p[2] = (triID >>  5) & 0x1FF;

#	ifdef USE_VERTEX_NORMALS
	Vertex verts[3] = {getVertex(p[0], ts), getVertex(p[1], ts), getVertex(p[2], ts)};

	Vector3 &tri_p0 = verts[0].v;
	Vector3 &tri_p1 = verts[1].v;
	Vector3 &tri_p2 = verts[2].v;

	Vector3 vertNormals[3] = {verts[0].n, verts[1].n, verts[2].n};

	/*
	if(dot(verts[0].n, rayPacket.cornerRays.getDirection(0)) > 0.0f)
		vertNormals[0] *= -1.0f;
	if(dot(verts[1].n, rayPacket.cornerRays.getDirection(0)) > 0.0f)
		vertNormals[1] *= -1.0f;
	if(dot(verts[2].n, rayPacket.cornerRays.getDirection(0)) > 0.0f)
		vertNormals[2] *= -1.0f;
	*/
#	else
	_Vector4 &tri_p0 = getVertexC(p[0], ts);
	_Vector4 &tri_p1 = getVertexC(p[1], ts);
	_Vector4 &tri_p2 = getVertexC(p[2], ts);
#	endif

#	ifdef USE_TEXTURING
	Vector2 vertTextures[3];	// do not support texture for HCCMesh yet
#	endif

	Vector3 triN;
	triN = cross(tri_p1-tri_p0, tri_p2-tri_p0);
	triN.makeUnitVector();
#	ifdef USE_VERTEX_NORMALS
	unsigned int matID = (*((unsigned int*)&verts[0].c[0])) & 0xFFFF;
#	else
	unsigned int matID = (*((unsigned int*)&tri_p0.m_alpha)) & 0xFFFF;
#	endif
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
bool HCCMesh::getIntersection(RayPacketT &rayPacket, int stream) 
{
	int threadID = omp_get_thread_num();
	StackElem *stack = stacks[threadID+MAX_NUM_THREADS*stream];
	TravStat currentTS;
	BVHNode * currentNode;
	int stackPtr;		
	int firstNonHit = 0;	

	Index_t rootIndex = getRootIdx(currentTS);
	currentNode = getBV(rootIndex, currentTS, 0, 0);

	stack[0].index = Model::getRootIdx();
	stack[0].ts = currentTS;
	stackPtr = 1;	

	unsigned int lChild, rChild;
	TravStat leftTS, rightTS;
	int axis;
	unsigned int minBB, maxBB;

	int curCluster = 0;
	unsigned int curIdx = rootIndex;
	// traverse BVH tree:
	while (1) {
		// is current node intersected and also closer than previous hit?
		firstNonHit = rayPacket.intersectWithBox(&currentNode->min, firstNonHit); // does intersect?

		if (firstNonHit < nRays) { // yes, at least one ray intersects

			// is inner node?
			if (!isLeaf(currentNode)) {				
				axis = currentTS.axis;
				
				if(rayPacket.rays[firstNonHit].rayChildOffsets[axis])
				{
					stack[stackPtr].ts = currentTS;
					lChild = getLeftChildIdx(currentNode, currentTS, minBB);
					rChild = getRightChildIdx(currentNode, stack[stackPtr].ts, maxBB);

					currentNode = getBV(lChild, currentTS, minBB, maxBB);

					stack[stackPtr].index = rChild;
					stack[stackPtr].minBB = minBB;
					stack[stackPtr].maxBB = maxBB;

					curIdx = lChild;
					curCluster = currentTS.cluster;
				}
				else
				{
					stack[stackPtr].ts = currentTS;
					lChild = getLeftChildIdx(currentNode, stack[stackPtr].ts, minBB);
					rChild = getRightChildIdx(currentNode, currentTS, maxBB);

					currentNode = getBV(rChild, currentTS, minBB, maxBB);

					stack[stackPtr].index = lChild;
					stack[stackPtr].minBB = minBB;
					stack[stackPtr].maxBB = maxBB;

					curIdx = rChild;
					curCluster = currentTS.cluster;
				}
				stack[stackPtr++].firstNonHit = firstNonHit;
				continue;
			}
			else {				
				// is leaf node:
				// intersect with current node's members
				HCCMesh::getIntersectionWithTri(rayPacket, currentNode->left, firstNonHit, currentTS);
			}
		}

		// traversal ends when stack empty
		if(--stackPtr == 0) break;

		// fetch next node from stack
		currentTS = stack[stackPtr].ts;
		currentNode = getBV(stack[stackPtr].index, currentTS, stack[stackPtr].minBB, stack[stackPtr].maxBB);

		firstNonHit = stack[stackPtr].firstNonHit;

		curIdx = stack[stackPtr].index;
		curCluster = currentTS.cluster;
		
	}

	// return hit status
	return 1;

}

};