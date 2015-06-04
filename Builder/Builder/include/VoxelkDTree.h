#ifndef VoxelkDTree_H
#define VoxelkDTree_H

#include "common.h"
#include "kDTree.h"

typedef struct Voxel_t {
	unsigned int index;
	unsigned int numTris;
	Vector3 min, max;
} Voxel;

typedef std::vector<Voxel> VoxelList;
typedef VoxelList::iterator VoxelListIterator;

typedef std::vector<unsigned int> VoxelIndexList;
typedef VoxelIndexList::iterator VoxelIndexListIterator;


/**
 * kD-tree class for high-level tree of voxels
 * 
 */
class VoxelkDTree {
public:

	/**
	* Constructor, sets parameters and gets pointer to triangle list to build
	* the tree for.
	*/
	VoxelkDTree(Voxel *tris, unsigned int numTris, Vector3 &min, Vector3 &max) {
		OptionManager *opt = OptionManager::getSingletonPtr();

		// reset statistics
		memset(&treeStats, 0, sizeof(BSPTreeInfo));

		// store pointers to geometry information
		voxellist = tris;			

		// scene info:
		treeStats.numTris = numTris;
		this->min = min;
		this->max = max;
		treeStats.min = min;
		treeStats.max = max;

		// load variable options
		//treeStats.maxDepth			  = 50;	
		treeStats.maxDepth			  = 100;		// modifed for lucy
		treeStats.maxListLength		  = 1;
		treeStats.emptySubdivideRatio = 0.15f;

		if (treeStats.maxDepth >= MAXBSPSIZE)
			treeStats.maxDepth = MAXBSPSIZE-1;
		if (treeStats.maxDepth < 0)
			treeStats.maxDepth = 0;
		
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
	~VoxelkDTree() {	
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
	void saveNodeInArray(LARGE_INTEGER myOffset, BSPArrayTreeNodePtr current, HANDLE nodePtr, HANDLE idxPtr, BSPTreeInfo *treeStats, 
		    			 LARGE_INTEGER &nodeOffset, LARGE_INTEGER &idxOffset, const char* filename);
	BSPArrayTreeNode writeVoxelkdTree(Voxel &voxel, HANDLE nodePtr, HANDLE idxPtr, LARGE_INTEGER &nodeOffset, LARGE_INTEGER &idxOffset, const char* filename);
	
	/**
	 *	Load BSP tree from file dump or save current tree to file
	 */	
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
	bool SubdivideSAH(long myOffset, VoxelIndexList *trilist, int depth,  Vector3 &min, Vector3 &max, FILE *indexFP, FILE *nodeFP);

	/**
	 * Recursively walks the tree and outputs the nodes
	 */
	void printNodeInArray(const char *LoggerName = NULL, BSPArrayTreeNodePtr current = NULL, int depth = 0);
	void drawNode(BSPArrayTreeNodePtr node, Vector3 min, Vector3 max, int depth);	
	
	/////////////////////////////////////////////////////
	// Variables:

	Voxel			*voxellist;      // triangle list

	// Index lists used during construction of the tree. We only need 2
	// lists for the left branches since they can be overwritten directly after
	// being used. For the right branch, we need as many lists as the maximal depth
	// of the tree. The lists will be deleted immediately after the tree is built.
	VoxelIndexList       *leftlist[2];
	VoxelIndexList		 *rightlist[MAXBSPSIZE]; 

	BSPArrayTreeNodePtr  tree;       // array containing all nodes after construction
	int            *indexlists;		 // array containing all triangle index lists after construction	
	Vector3		   *minvals,		 // list of min/max values of all tris, used for balancing
				   *maxvals;

	// Statistics:
	BSPTreeInfo treeStats;

	// Time stats:
	TimerValue timeBuildStart, timeBuildEnd;
};

#endif