#ifndef COMMON_SIMDBSPTREEDEFINES_H
#define COMMON_SIMDBSPTREEDEFINES_H

#ifdef BUNNY
#define BSP_EPSILON 0.001f
//#define BSP_EPSILON 0.01f
#define INTERSECT_EPSILON 0.01f 
#endif

#ifdef DRAGON
#define BSP_EPSILON 0.1f
#define INTERSECT_EPSILON 0.01f 
#endif

#ifdef ST_MATTHEW
#define BSP_EPSILON 0.005f
#define INTERSECT_EPSILON 0.4f 
#endif

#ifdef FOREST
#define BSP_EPSILON 0.002f
#define INTERSECT_EPSILON 0.8f 
#endif



//#define TRI_INTERSECT_EPSILON 0.0001f
//#define SIMDTRI_INTERSECT_EPSILON 0.0001f
#ifndef DRAGON
#define TRI_INTERSECT_EPSILON 0.001f
#define SIMDTRI_INTERSECT_EPSILON 0.001f
#else
#define TRI_INTERSECT_EPSILON 0.01f
#define SIMDTRI_INTERSECT_EPSILON 0.001f
#endif

#define MAXBSPSIZE  100

#ifdef _SIMD_SHOW_STATISTICS
#define GATHER_STATISTICS
#endif

//#ifdef KDTREENODE_16BYTES
#ifdef BVHNODE_16BYTES

#if HIERARCHY_TYPE == TYPE_KD_TREE
// 16-byte node:
typedef union BSPArrayTreeNode_t {
	struct { // inner node
		unsigned int children;
		unsigned int children2;
		float splitcoord;
		unsigned int lodIndex;
	};
	struct { // leaf node
#ifdef FOUR_BYTE_FOR_KD_NODE
		unsigned int indexOffset;
		unsigned int indexCount;
#else
		unsigned int indexCount;
		unsigned int indexOffset;
#endif
		unsigned int dummy1;
		unsigned int dummy2;
	};
	struct { // access for SIMD traversal
		unsigned int childOffset[2];		
		float dummy3;
		unsigned int dummy4;
	};
} BSPArrayTreeNode, *BSPArrayTreeNodePtr;
#define BSPTREENODESIZE 16
#endif

#if HIERARCHY_TYPE == TYPE_BVH
#ifndef BSP_ARRAY_TREE_NODE_DEF
#define
#define BVHNODE_BYTES 32
typedef union BSPArrayTreeNode_t {
	struct { // inner node
		unsigned int children;
		unsigned int children2;
		Vector3 min;
		Vector3 max;
	};
	struct { // leaf node
#ifdef FOUR_BYTE_FOR_KD_NODE
		unsigned int indexOffset;
		unsigned int indexCount;
#else
		unsigned int indexCount;
		unsigned int indexOffset;
#endif
		Vector3 min;
		Vector3 max;
	};
	struct { // access for SIMD traversal
		unsigned int childOffset[2];		
		Vector3 min;
		Vector3 max;
	};
} BSPArrayTreeNode, *BSPArrayTreeNodePtr;
#define BSPTREENODESIZE 32
#endif
#endif

#define BSPTREENODESIZEPOWER 5
#else

/**
* Condensed structure for a BSP tree node.
*
* children points to the first child node for inner nodes, the second node 
* will be assumed to be (children+8), since nodes have a fixed
* size in memory (if pointers are 32 bit, this won't work on 
* 64 bit CPUs !). For leaf nodes, this points to the triangle list.
* Additionally, the first two low-order bits 
* of the pointer are used for determining the type of the node:
*  3 - Leaf
*  0 - Split by X plane
*  1 - Split by Y plane
*  2 - Split by Z plane
* This is possible because the structure should be aligned to
* 32bit boundaries in memory, so the respective bits would be
* zero anyway.
*
* The member split is the coordinate of the split plane or the
* number of triangles for leaf nodes.
*/

// 8-byte node:
typedef union BSPArrayTreeNode_t {
	struct { // inner node
		unsigned int children;		
		float splitcoord;
	};
	struct { // leaf node
		unsigned int indexCount;
		unsigned int indexOffset;
	};
} BSPArrayTreeNode, *BSPArrayTreeNodePtr;
#define BSPTREENODESIZE 8
#define BSPTREENODESIZEPOWER 3
#endif

#include "LOD_header.h"

/**
* Information structure, containing information about tree.
**/
typedef struct BSPTreeInfo_t {
	int numTris;		// (real, not counting duplicates) number of triangles in scene 
	int numNodes;		// number of nodes in tree
	int numLeafs;		// number of leaf nodes
	int maxLeafDepth;	// largest depth in tree	
	unsigned int sumDepth;		// sum of leaf depth, for calculating avgDepth
	unsigned int sumTris;		// effective number of tri(-indices) in tree
	unsigned int maxTriCountPerLeaf; 
	int maxDepth;		// maximal allowed depth when generating
	int maxListLength;  // target number of tris per leaf
	float emptySubdivideRatio; // ratio of free to used surface area to generate an empty tree node

	float timeBuild;	// time to build, in seconds
	Vector3 min, max;	// bounding box

	// error quantization
	float m_MinErr, m_ErrRatio;
	int m_NumAllowedBit;

} BSPTreeInfo, *BSPTreeInfoPtr;

// expected version in BSP tree file (increase when format changes so that
// no files from old incompatible versions are loaded)
#define BSP_FILEVERSION 7

#define BSP_FILEIDSTRING "BSPTREE"
#define BSP_FILEIDSTRINGLEN strlen(BSP_FILEIDSTRING)

#define BSP_STACKPADDING (64 - 8*sizeof(float) - sizeof(BSPArrayTreeNodePtr) - sizeof(unsigned int))


// cost constants for the SAH subdivision algorithm
#define BSP_COST_TRAVERSAL 0.3f
#define BSP_COST_INTERSECTION 1.0f

// easy macros for working with the compressed BSP tree
// structure that is a bit hard to understand otherwise
// (and access may vary on whether OOC mode is on or not)
#ifdef FOUR_BYTE_FOR_KD_NODE	
# define AXIS(node) ((node)->children2 & 3)
# define ISLEAF(node) (((node)->children2 & 3) == 3)
# define ISNOLEAF(node) (((node)->children2 & 3) != 3)
#else
# define AXIS(node) ((node)->children & 3)
# define ISLEAF(node) (((node)->children & 3) == 3)
# define ISNOLEAF(node) (((node)->children & 3) != 3)
#endif

#define GETIDXOFFSET(node) ((node)->indexOffset)
#define MAKECHILDCOUNT(count) ((count << 2) | 3)
#define GETCHILDCOUNT(node) ((node)->indexCount >> 2)

#if defined(_USE_OOC) && !defined(_USE_OOC_DIRECTMM)
# define BSPNEXTNODE 1

# ifdef KDTREENODE_16BYTES
#  ifdef FOUR_BYTE_FOR_KD_NODE	
#   define GETCHILDNUM(node,num) (node->children + (num))
#   define GETLEFTCHILD(node) (node->children)
#   define GETRIGHTCHILD(node) (node->children + 1)		
#  else
#   define GETCHILDNUM(node,num) (node->childOffset[num] >> 2)
#   define GETLEFTCHILD(node) (node->children >> 2)
#   define GETRIGHTCHILD(node) (node->children2 >> 2)
#  endif
# else
#  define GETCHILDNUM(node,num) ((node->children >> 2) + num)
#  define GETLEFTCHILD(node) (node->children >> 2)
#  define GETRIGHTCHILD(node) (node->children >> 2) + 1
#endif

#define GETNODE(object,offset) ((BSPArrayTreeNodePtr)&((*object)[offset]))
#define MAKEIDX_PTR(object,offset) ((unsigned int *)&((*object)[offset]))

#else // !_USE_OOC

# define BSPNEXTNODE BSPTREENODESIZE

# ifdef KDTREENODE_16BYTES

#  ifdef FOUR_BYTE_FOR_KD_NODE	
#   define GETCHILDNUM(node,num) (node->children + (num))
#   define GETLEFTCHILD(node) (node->children)
#   define GETRIGHTCHILD(node) (node->children + 1)
#   define GETNODE(object,offset) (&object[offset])
#   define MAKEIDX_PTR(object,offset) (&object[offset])
#  else
#   define GETCHILDNUM(node,num) ((node->childOffset[num] & ~3) << 1)
#   define GETLEFTCHILD(node) ((node->children & ~3) << 1)
#   define GETRIGHTCHILD(node) (node->children2 << 1)
#   define GETNODE(object,offset) ((BSPArrayTreeNodePtr)((char *)object + offset))
#   define MAKEIDX_PTR(object,offset) (&object[offset])
#  endif
# else
#  define GETCHILDNUM(node,num) (((node->children & ~3) << 1) + num*BSPTREENODESIZE) 
#  define GETLEFTCHILD(node) ((node->children & ~3) << 1)
#  define GETRIGHTCHILD(node) (((node->children & ~3) << 1) + BSPTREENODESIZE)
#  define GETNODE(object,offset) ((BSPArrayTreeNodePtr)((char *)object + offset))
#  define MAKEIDX_PTR(object,offset) (&object[offset])
# endif

#endif

#define GETHLNODE(object,offset) (&object[offset])
#define MAKEHLIDX_PTR(object,offset) (&object[offset])

/*
// macros for accessing single triangles and vertices, depending on
// whether we're using OOC or normal mode
#if defined(_USE_OOC) && !defined(_USE_OOC_DIRECTMM)
#define GETTRI(idx) (*trianglelist)[idx]
#define GETVERTEX(idx) (*vertexList)[idx*sizeof(_Vector4)]
#else
#define GETTRI(idx) trianglelist[idx]
#define GETVERTEX(idx) vertexList[idx]
#endif*/

#ifndef COMMON_SCENE_H
// macros for accessing single triangles and vertices, depending on
// whether we're using OOC or normal mode
#if defined(_USE_OOC) && !defined(_USE_OOC_DIRECTMM)
#define GETTRI(object,idx) (*((object)->trilist))[idx]
//#define GETVERTEX(object,idx) (*((object)->vertices))[idx*sizeof(_Vector4)]
#define GETVERTEX(object,idx) (*((object)->vertices))[idx]
#else
#define GETTRI(object,idx) (object)->trilist[idx]
#define GETVERTEX(object,idx) (object)->vertices[idx]
#endif
#endif

#define GETLODERROR(object,idx) object->g_QuanErrs[idx];

__declspec(align(16)) static const unsigned int maskLUTable[16][4] = { 
	{ 0x00000000, 0x00000000, 0x00000000, 0x00000000 },	// 0
	{ 0xffffffff, 0x00000000, 0x00000000, 0x00000000 }, // 1
	{ 0x00000000, 0xffffffff, 0x00000000, 0x00000000 }, // 2
	{ 0xffffffff, 0xffffffff, 0x00000000, 0x00000000 }, // 3
	{ 0x00000000, 0x00000000, 0xffffffff, 0x00000000 }, // 4
	{ 0xffffffff, 0x00000000, 0xffffffff, 0x00000000 }, // 5
	{ 0x00000000, 0xffffffff, 0xffffffff, 0x00000000 }, // 6
	{ 0xffffffff, 0xffffffff, 0xffffffff, 0x00000000 }, // 7
	{ 0x00000000, 0x00000000, 0x00000000, 0xffffffff },	// 8
	{ 0xffffffff, 0x00000000, 0x00000000, 0xffffffff }, // 9
	{ 0x00000000, 0xffffffff, 0x00000000, 0xffffffff }, // 10
	{ 0xffffffff, 0xffffffff, 0x00000000, 0xffffffff }, // 11
	{ 0x00000000, 0x00000000, 0xffffffff, 0xffffffff }, // 12
	{ 0xffffffff, 0x00000000, 0xffffffff, 0xffffffff }, // 13
	{ 0x00000000, 0xffffffff, 0xffffffff, 0xffffffff }, // 14
	{ 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff }, // 15
};

#if COMMON_COMPILER == COMPILER_MSVC
// disable portability warnings generated by 
// pointer arithmetic in code
#pragma warning( disable : 4311 4312 4102 )
#endif 

// sungeui start ------------------------------------
// mininum volume reduction between two nodes of LOD
const float MIN_VOLUME_REDUCTION_BW_LODS = pow ((float)2, (int)5);
const unsigned int MAX_NUM_LODs = (unsigned int) 1 << 31;
const unsigned int LOD_BIT = 1;
//#define HAS_LOD(idx) (idx & LOD_BIT)
#define HAS_LOD(idx) (idx != 0)
const unsigned int ERR_BITs = 5;
const unsigned int _QUAN_ERR_BIT_MASK = ((1 << ERR_BITs) - 1);
#define GET_REAL_IDX(idx) (idx >> ERR_BITs)
//#define GET_REAL_IDX(idx) (idx >> 1)
#define GET_ERR_QUANTIZATION_IDX(idx) (idx & _QUAN_ERR_BIT_MASK)

#define GET_FIRST_TRI_IDX(Idx, i1, i2, i3) {i1 = (Idx & 3); i2 = ((Idx & (3 << 2)) >> 2); i3 = ((Idx & (3 << 4)) >> 4);}
#define GET_SECOND_TRI_IDX(Idx, i1, i2, i3) {i1 = (Idx & (3 << 6)) >> 6; i2 = ((Idx & (3 << 8)) >> 8); i3 = ((Idx & (3 << 10)) >> 10);}

#if defined(_USE_OOC) && !defined(_USE_OOC_DIRECTMM)
#define GET_LOD(object,Idx) (*object->lodlist)[Idx*sizeof(LODNode)]
#else
#define GET_LOD(object,Idx) object->lodlist[Idx]
#endif


// surface area of a voxel (for SAH construction) 
FORCEINLINE float surfaceArea(float dim1, float dim2, float dim3) {
	return 2.0f * ((dim1 * dim2) + (dim2 * dim3) + (dim1 * dim3));
}

#endif