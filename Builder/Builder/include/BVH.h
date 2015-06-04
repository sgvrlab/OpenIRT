#ifndef BVH_H
#define BVH_H

#include "common.h"
#include <hash_map>
#include <vector>
#include "helpers.h"
#include "OptionManager.h"
#include "Logger.h"
#include "Vector3.h"
#include "Ray.h"
#include "Vertex.h"
#include "Triangle.h"
#include "Progression.h"

// max depth of BVH
#define MAXBSPSIZE  100

// cost constants for the SAH subdivision algorithm
#define BSP_COST_TRAVERSAL 0.3f
#define BSP_COST_INTERSECTION 1.0f
#define GETTRI(idx) (trianglelist[idx])
//#define GETVERTEX(idx) ((*m_pVertexCache)[idx])
#define GETVERTEX(idx) (m_useOOC ? (*m_pVertexCache)[idx] : m_pVertexFile[idx])

#ifdef BVHNODE_16BYTES

#if HIERARCHY_TYPE == TYPE_BVH 
#include "BVHNodeDefine.h"
#endif

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
typedef OOC_VERTEX_FILECLASS<Vertex>*           BSPVertexList;
typedef OOC_BSPNODE_FILECLASS<BSPArrayTreeNode>* BSPNodeList;
typedef OOC_BSPIDX_FILECLASS<unsigned int>*      BSPIndexList;
// sungeui end ------------------------------------------------

#ifdef FOUR_BYTE_FOR_BV_NODE
	#define AXIS(node) ((node)->children2 & 3)
	#define ISLEAF(node) (((node)->children2 & 3) == 3)
	#define ISNOLEAF(node) (((node)->children2 & 3) != 3)
	#define GETLEFTCHILD(node) ((node)->children)
	#define GETRIGHTCHILD(node) ((node)->children + 1)
	#define GETNODE(offset) ((BSPArrayTreeNodePtr)((char *)tree + (offset<<4)))
#else
	#define GETROOT() (0)
	#define AXIS(node) ((node)->children & 3)
	#define ISLEAF(node) (((node)->children & 3) == 3)
	#define ISNOLEAF(node) (((node)->children & 3) != 3)
	#define GETLEFTCHILD(node) (((node)->children & ~3) << 3)
	#define GETRIGHTCHILD(node) (((node)->children & ~3) << 3) + BVHNODE_BYTES
	#define GETNODE(offset) ((BSPArrayTreeNodePtr)((char *)tree + offset))
#endif

#define MAKECHILDCOUNT(count) ((count << 2) | 3)
#ifdef _USE_ONE_TRI_PER_LEAF
#define GETIDXOFFSET(node) ((node)->triIndex >> 2)
#define GETCHILDCOUNT(node) (1)
#else
#define GETIDXOFFSET(node) ((node)->indexOffset)
#define GETCHILDCOUNT(node) ((node)->indexCount >> 2)
#endif

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

#define TEST_TYPE 7

/**
 * BVH class.
 * 
 */
class BVH {
public:

	/**
	* Constructor, sets parameters and gets pointer to triangle list to build
	* the tree for.
	*/
	BVH(const Triangle *tris, const unsigned int* triIdx, unsigned int numTris, stdext::hash_map<unsigned int, Vertex> *vertexBuffer, Vector3 &min, Vector3 &max, unsigned int treeID = 0, const char *testFileName = NULL) {
		OptionManager *opt = OptionManager::getSingletonPtr();

		m_useOOC = true;

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
		maxNumTrisPerLeaf = 1;

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

//		testFile = fopen(testFileName, "w");
		numCase1 = 0;
		numCase2 = 0;
		
		curNodeIdx = 1;
	}

	BVH(const Triangle *tris, const unsigned int* triIdx, unsigned int numTris, Vertex *vertexBuffer, Vector3 &min, Vector3 &max, unsigned int treeID = 0, const char *testFileName = NULL) {
		OptionManager *opt = OptionManager::getSingletonPtr();

		m_useOOC = false;

		// reset statistics
		memset(&treeStats, 0, sizeof(BSPTreeInfo));

		// store pointers to geometry information
		trianglelist = tris;	
		triIndexList = triIdx;
		m_pVertexFile = vertexBuffer;

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
		maxNumTrisPerLeaf = 1;

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

//		testFile = fopen(testFileName, "w");
		numCase1 = 0;
		numCase2 = 0;
		
		curNodeIdx = 1;
	}
	/**
	 * Destructor, frees memory of nodes and triangle lists
	 */
	~BVH() {	
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

		#if TEST_TYPE == 7
//		fprintf(testFile, "%d %d %f\n", numCase1, numCase2, ((float)numCase1)/numCase2);
		#endif
//		fclose(testFile);
	}

	/**
	 * Initialize the tree and build the structure. Only use the
	 * tree after calling this function once, or you will use an
	 * empty scene.
	 */	
	void buildTree();
	void buildTreeSAH();
	void makeFlatRepresentation(FILE *indexFP, FILE *nodeFP);
	
	/**
	 *	Load BSP tree from file dump or save current tree to file
	 */
	bool loadFromFile(const char* filename);
	bool loadFromFiles(const char* filename);
	bool saveToFile(const char* filename);
	bool saveToFile(const char* fileNameHeader, const char* fileNameTree, const char* fileNameIndex);

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

public:

	FORCEINLINE void setBB(Vector3 &min, Vector3 &max, Vector3 &init)
	{
		min.e[0] = init.e[0];
		min.e[1] = init.e[1];
		min.e[2] = init.e[2];
		max.e[0] = init.e[0];
		max.e[1] = init.e[1];
		max.e[2] = init.e[2];
	}

	FORCEINLINE void updateBB(Vector3 &min, Vector3 &max, Vector3 &vec)
	{
		min.e[0] = ( min.e[0] < vec.e[0] ) ? min.e[0] : vec.e[0];
		min.e[1] = ( min.e[1] < vec.e[1] ) ? min.e[1] : vec.e[1];
		min.e[2] = ( min.e[2] < vec.e[2] ) ? min.e[2] : vec.e[2];

		max.e[0] = ( max.e[0] > vec.e[0] ) ? max.e[0] : vec.e[0];
		max.e[1] = ( max.e[1] > vec.e[1] ) ? max.e[1] : vec.e[1];
		max.e[2] = ( max.e[2] > vec.e[2] ) ? max.e[2] : vec.e[2];
	}

#if 0
	/**
	 * Builds the BSP tree by subdividing along the center of x, y, or z bounds, one
	 * each time this function is called. This function calls itself recursively until
	 * either the tree is deeper than MaxDepth or all of the tree leaves contains less
	 * than MaxListLength of objects.
	 */
	bool Subdivide(long myOffset, TriangleIndexList *trilist, int depth, int axis, Vector3 &min, Vector3 &max, FILE *indexFP, FILE *nodeFP);
	bool SubdivideSAH(long myOffset, TriangleIndexList *trilist, int depth,  Vector3 &min, Vector3 &max, FILE *indexFP, FILE *nodeFP);
#endif
	bool Subdivide(TriangleIndexList *triIDs, unsigned int left, unsigned int right, unsigned int myIndex = 0, unsigned int nextIndex = 1, int depth = 0);
	int maxDEPTH;

	class FeelEvent
	{
	public :
		int triIndex;
		float coor;

		bool operator < (const FeelEvent &T) const 
		{
			return ( this->coor < T.coor );
		}
	};
	bool SubdivideSAH(TriangleIndexList *triIDs, unsigned int left, unsigned int right, unsigned int myIndex, unsigned int nextIndex, int depth);
	bool SubdivideSAHSingle(TriangleIndexList *triIDs, unsigned int left, unsigned int right, unsigned int myIndex, unsigned int nextIndex, int depth);
//	bool Subdivide(TriangleIndexList *triIDs, TriangleIndexList *subTriIDs, unsigned int left, unsigned int right, unsigned int myIndex = 0, unsigned int nextIndex = 1);

	/**
	 * Recursively walks the tree and outputs the nodes
	 */
	void printNodeInArray(const char *LoggerName = NULL, BSPArrayTreeNodePtr current = NULL, int depth = 0);
	void drawNode(BSPArrayTreeNodePtr node, Vector3 min, Vector3 max, int depth);
	void renderSplit(int axis, float splitCoord, Vector3 min, Vector3 max,TriangleIndexList *newlists[2]);

	/**
	 * For test. Write BB coordinates.
	 */
	void testWrite(BSPArrayTreeNodePtr curNode, BSPArrayTreeNodePtr parentNode = NULL);

	
	/////////////////////////////////////////////////////
	// Variables:

	const Triangle			*trianglelist;      // triangle list	
	const unsigned int		*triIndexList;
	stdext::hash_map<unsigned int, Vertex> *m_pVertexCache; // cached ooc list of vertices
	Vertex *m_pVertexFile;
	bool m_useOOC;

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


	unsigned int curNodeIdx;	// for tree construction : node index
	unsigned int curIndex;	     // for tree construction : triangle index

	FILE *testFile;
	unsigned int numCase1, numCase2;

	// Statistics:
	BSPTreeInfo treeStats;

	// Time stats:
	TimerValue timeBuildStart, timeBuildEnd;

	int maxNumTrisPerLeaf;

	Progression *progBuildTree;
};

#endif