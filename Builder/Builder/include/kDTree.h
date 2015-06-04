#ifndef KDTREE_H
#define KDTREE_H

#include "common.h"
#include <hash_map>
#include <vector>
#include "helpers.h"
#include "OptionManager.h"
#include "Logger.h"
#include "Vector3.h"
#include "Ray.h"
#include "Triangle.h"

/********************************************************************
created:	2004/10/12
created:	12:10:2004   19:19
filename: 	c:\MSDev\MyProjects\Renderer\Common\BSPTree.h
file path:	c:\MSDev\MyProjects\Renderer\Common
file base:	BSPTree
file ext:	h
author:		Christian Lauterbach (lauterb@informatik.uni-bremen.de)

purpose:	Highly optimized BSP tree scene graph class (used by Scene
class. A BSP node uses only 8 bytes (see below) and separates
along one axis.
*********************************************************************/

// max depth of kD tree
#define MAXBSPSIZE  100

// cost constants for the SAH subdivision algorithm
#define BSP_COST_TRAVERSAL 0.3f
#define BSP_COST_INTERSECTION 1.0f

#ifdef KDTREENODE_16BYTES

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
} BSPArrayTreeNode, *BSPArrayTreeNodePtr;
#define BSPTREENODESIZE 16
#define BSPTREENODESIZEPOWER 4
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

// sungeui start ----------------------------------------------
#include "OOCFile.h"
#include "OOCFile64.h"

typedef OOC_TRI_FILECLASS<Triangle>*			 BSPTriList;
typedef OOC_VERTEX_FILECLASS<_Vector4>*           BSPVertexList;
typedef OOC_BSPNODE_FILECLASS<BSPArrayTreeNode>* BSPNodeList;
typedef OOC_BSPIDX_FILECLASS<unsigned int>*      BSPIndexList;
// sungeui end ------------------------------------------------

#ifdef FOUR_BYTE_FOR_KD_NODE
	#define AXIS(node) ((node)->children2 & 3)
	#define ISLEAF(node) (((node)->children2 & 3) == 3)
	#define ISNOLEAF(node) (((node)->children2 & 3) != 3)
	#define GETLEFTCHILD(node) ((node)->children)
	#define GETRIGHTCHILD(node) ((node)->children + 1)
	#define GETNODE(offset) ((BSPArrayTreeNodePtr)((char *)tree + (offset<<4)))
#else
	#define AXIS(node) ((node)->children & 3)
	#define ISLEAF(node) (((node)->children & 3) == 3)
	#define ISNOLEAF(node) (((node)->children & 3) != 3)
	#define GETLEFTCHILD(node) (((node)->children & ~3) << 1)
	#define GETRIGHTCHILD(node) (((node)->children & ~3) << 1) + BSPTREENODESIZE
	#define GETNODE(offset) ((BSPArrayTreeNodePtr)((char *)tree + offset))
#endif

#define GETIDXOFFSET(node) ((node)->indexOffset)
#define MAKECHILDCOUNT(count) ((count << 2) | 3)
#define GETCHILDCOUNT(node) ((node)->indexCount >> 2)

#define MAKEIDX_PTR(offset) (&indexlists[offset])

// index list via STL vector (auto-resizable)
typedef std::vector<unsigned int> TriangleIndexList;
typedef TriangleIndexList::iterator TriangleIndexListIterator;

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

#define BSP_STACKPADDING (48 - 8*sizeof(float) - sizeof(BSPArrayTreeNodePtr))


/**
 * kD-tree class.
 * 
 */
class kDTree {
public:

	/**
	* Constructor, sets parameters and gets pointer to triangle list to build
	* the tree for.
	*/
	kDTree(const Triangle *tris, const unsigned int* triIdx, unsigned int numTris, stdext::hash_map<unsigned int, _Vector4> *vertexBuffer, Vector3 &min, Vector3 &max, unsigned int treeID = 0) {
		OptionManager *opt = OptionManager::getSingletonPtr();

		// reset statistics
		memset(&treeStats, 0, sizeof(BSPTreeInfo));

		// store pointers to geometry information
		trianglelist = tris;	
		triIndexList = triIdx;
		m_pVertexCache = vertexBuffer;

		// scene info:
		treeStats.numTris = numTris;
		this->min = min;
		this->max = max;
		treeStats.min = min;
		treeStats.max = max;

		this->treeID = treeID;

		// load variable options
		treeStats.maxDepth			  = 95;
		treeStats.maxListLength		  = 1;
		treeStats.emptySubdivideRatio = 0.15f;

		if (treeStats.maxDepth >= MAXBSPSIZE)
			treeStats.maxDepth = MAXBSPSIZE-1;
		if (treeStats.maxDepth < 0)
			treeStats.maxDepth = 0;
		if (treeStats.maxListLength < 2)
			treeStats.maxListLength = 2;

		tree = NULL;
		indexlists = NULL;
		_debugEnable = false;

		subdivisionMode = BSP_SUBDIVISIONMODE_NORMAL;
		minvals = NULL;
		maxvals = NULL;
	}

	/**
	 * Destructor, frees memory of nodes and triangle lists
	 */
	~kDTree() {	
		// deallocate node array
		if (tree)
			delete [] tree;
		// deallocate tri index array for leaves
		if (indexlists)
			delete [] indexlists;

		// delete min/max values of triangles
		if (minvals)
			delete minvals;
		if (maxvals)
			delete maxvals;
	}

	/**
	 * Initialize the tree and build the structure. Only use the
	 * tree after calling this function once, or you will use an
	 * empty scene.
	 */	
	void buildTree();
	void makeFlatRepresentation(FILE *indexFP, FILE *nodeFP);
	
	/**
	 *	Load BSP tree from file dump or save current tree to file
	 */
	bool loadFromFile(const char* filename);
	bool loadFromFiles(const char* filename);
	bool saveToFile(const char* filename);

	/**
	 * Prints statistical information on the BSP tree (dump of tree structure, too, 
	 * if dumpTree = true)
	 */
	void printTree(bool dumpTree = false, const char *LoggerName = NULL);

	/**
	 * Get number of triangles in this tree.
	 * This counts the real number of triangle and counts triangles contained
	 * in multiple nodes only once.
	 */
	int getNumTris();

	Vector3			min, max;        // bounding box of the scene/tree

	// different subdivision modes, see SubdivideXYZ() methods below
	const static int BSP_SUBDIVISIONMODE_SIMPLE   = 1;
	const static int BSP_SUBDIVISIONMODE_NORMAL   = 2;
	const static int BSP_SUBDIVISIONMODE_BALANCED = 3;

	int	subdivisionMode; // which subdivision function to use
	bool _debugEnable;

protected:

	/**
	 * Builds the BSP tree by subdividing along the center of x, y, or z bounds, one
	 * each time this function is called. This function calls itself recursively until
	 * either the tree is deeper than MaxDepth or all of the tree leaves contains less
	 * than MaxListLength of objects.
	 */
	bool Subdivide(long myOffset, TriangleIndexList *trilist, int depth, int axis, Vector3 &min, Vector3 &max, FILE *indexFP, FILE *nodeFP);
	bool SubdivideSAH(long myOffset, TriangleIndexList *trilist, int depth,  Vector3 &min, Vector3 &max, FILE *indexFP, FILE *nodeFP);

	/**
	 * Recursively walks the tree and outputs the nodes
	 */
	void printNodeInArray(const char *LoggerName = NULL, BSPArrayTreeNodePtr current = NULL, int depth = 0);
	void drawNode(BSPArrayTreeNodePtr node, Vector3 min, Vector3 max, int depth);
	void renderSplit(int axis, float splitCoord, Vector3 min, Vector3 max,TriangleIndexList *newlists[2]);

	
	/////////////////////////////////////////////////////
	// Variables:

	const Triangle			*trianglelist;      // triangle list	
	const unsigned int		*triIndexList;
	stdext::hash_map<unsigned int, _Vector4> *m_pVertexCache; // cached ooc list of vertices	

	// Index lists used during construction of the tree. We only need 2
	// lists for the left branches since they can be overwritten directly after
	// being used. For the right branch, we need as many lists as the maximal depth
	// of the tree. The lists will be deleted immediately after the tree is built.
	TriangleIndexList        *leftlist[2];
	TriangleIndexList		 *rightlist[MAXBSPSIZE]; 

	BSPArrayTreeNodePtr  tree;       // array containing all nodes after construction
	int            *indexlists;		 // array containing all triangle index lists after construction	
	Vector3		   *minvals,		 // list of min/max values of all tris, used for balancing
				   *maxvals;
	unsigned int	treeID;			 // unique ID for this tree, used for filenames


	unsigned int curIndex;	     // for tree construction

	// Statistics:
	BSPTreeInfo treeStats;

	// Time stats:
	TimerValue timeBuildStart, timeBuildEnd;
};

#endif