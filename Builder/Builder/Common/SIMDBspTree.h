#ifndef COMMON_SIMDBSPTREE_H
#define COMMON_SIMDBSPTREE_H



#include "common.h"


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

/**
* BSP tree class.
* 
*/
class SIMDBSPTree {
public:

	/**
	 * Constructor, sets parameters and gets pointer to triangle list to build
	 * the tree for.
	 */
	SIMDBSPTree(unsigned int numObjects, ModelInstance *objectList, Material *defaultMaterial) {
		OptionManager *opt = OptionManager::getSingletonPtr();

		this->objectList = objectList;
		nSubObjects = numObjects;

		//subObject = &objectList[0];
		
		memset(&objTreeStats, 0, sizeof(BSPTreeInfo));
		objTreeStats.maxDepth = 99;
		objTreeStats.maxListLength = 1;
		objTreeStats.emptySubdivideRatio = 0.1f;

		
		for (int i = 0; i < nSubObjects; i++) {
		
			// load variable options
			objectList[i].treeStats.maxDepth = opt->getOptionAsInt("raytracing", "maxBSPTreeDepth", 25);
			objectList[i].treeStats.maxListLength = opt->getOptionAsInt("raytracing", "maxBSPLeafTriangles", 7);
			objectList[i].treeStats.emptySubdivideRatio = opt->getOptionAsFloat("raytracing", "BSPemptyCellRatio", 0.2f);

			if ( objectList[i].treeStats.maxDepth >= MAXBSPSIZE)
				 objectList[i].treeStats.maxDepth = MAXBSPSIZE-1;
			if ( objectList[i].treeStats.maxDepth < 0)
				 objectList[i].treeStats.maxDepth = 0;
			if ( objectList[i].treeStats.maxListLength < 2)
				 objectList[i].treeStats.maxListLength = 2;												
		}
		
		_debug_TreeIntersectCount = 0;
		_debug_NodeIntersections = 0;
		_debug_NodeIntersectionsOverhead = 0;
		_debug_LeafIntersectCount = 0;
		_debug_LeafIntersectOverhead = 0;
		_debug_LeafTriIntersectCount = 0;

		subdivisionMode = BSP_SUBDIVISIONMODE_SIMPLE;
		minvals = NULL;
		maxvals = NULL;
		this->defaultMaterial = defaultMaterial;
				
		// allocate array of stacks:
		int s;
		stacks = new StackElem *[omp_get_max_threads()];
		for (s = 0; s < omp_get_max_threads(); s++)
			stacks[s] = (StackElem *)_aligned_malloc(MAXBSPSIZE * sizeof(StackElem), 16);		

		// allocate array of stacks:
		stacksHL = new StackElem *[omp_get_max_threads()];
		for (s = 0; s < omp_get_max_threads(); s++)
			stacksHL[s] = (StackElem *)_aligned_malloc((MAXBSPSIZE / 2) * sizeof(StackElem), 16);		

		stackPtrHL = new int[omp_get_max_threads()];
	}

	/**
	* Destructor, frees memory of nodes and triangle lists
	*/
	~SIMDBSPTree() {	

		LogManager *log = LogManager::getSingletonPtr();
		log->logMessage("Destroying BSP tree...");		
			
		// delete min/max values of triangles
		if (minvals)
			delete minvals;
		if (maxvals)
			delete maxvals;

		// delete array of stacks
		int i;
		if (stacks) {			
			for (i = 0; i < omp_get_max_threads(); i++)
				_aligned_free(stacks[i]);

			delete stacks;
		}
		if (stacksHL) {
			for (i = 0; i < omp_get_max_threads(); i++)
				_aligned_free(stacksHL[i]);

			delete stacksHL;
		}

		if (stackPtrHL)
			delete stackPtrHL;
	}

	// incore tree construction is only available when not in OoC mode
	#ifndef _USE_OOC
	/**
	* Initialize the tree and build the structure. Only use the
	* tree after calling this function once, or you will use an
	* empty scene.
	*/	
	void buildTree(unsigned int subObjectId = 0);
	//void insertNodeIntoArray(BSPTreeNodePtr node, unsigned int myOffset, int myAxis);
	#else
	void buildTree(unsigned int subObjectId = 0) {}
	#endif

	void buildHighLevelTree();

	/**
	 * Traverses ray through SIMDBSPTree and intersects ray with all of the objects along
	 * the way. Returns the hitpoint and true, if something was hit
	 */
	int RayTreeIntersect(SIMDRay &rays, SIMDHitPointPtr hit, float *traveledDistance);
	int RayTreeIntersect(Ray &ray, HitPointPtr hit, float TraveledDist = 0.f);


	int BeamTreeIntersect(Beam &beam, float *traveledDistance, unsigned int startNodeOffset = 0);
	int RayTreeIntersectOffset(SIMDRay &rays, SIMDHitPointPtr hits, float *traveledDistance, unsigned int startOffset);


	/**
	 * Traverses from origin to target and returns true, if nothing was
	 * hit on the way (i.e. origin and target have an unoccluded line of sight)
	 */
	int isVisible(SIMDRay &rays, float *tmax, float *target, int initialMask, float *ErrBndForDirLight = NULL, int *hitLODIdx = NULL);
	int isVisible(const Vector3 &origin, const Vector3 &target, float traveledDist = 0.0f);
	int isVisibleWithLODs(const Vector3 &light_origin, const Vector3 &hit_p, float errBnd = 0.0f, int hitLODIdx = 0);	

	/**
	 * Prints statistical information on the BSP tree (dump of tree structure, too, 
	 * if dumpTree = true)
	 */
	void printTree(bool dumpTree = false, const char *LoggerName = NULL, unsigned int subObjectId = 0);
	void printHighLevelTree(bool dumpTree = false, const char *LoggerName = NULL);

	/**
	 * Get number of triangles in this tree.
	 * This counts the real number of triangle and counts triangles contained
	 * in multiple nodes only once.
	 */
	int getNumTris(unsigned int subObjectId = 0);
	
	void dumpCounts();

	void GLdrawTree(Ray &viewer, unsigned int subObjectId = 0);

	Vector3			objectBB[2];        // bounding box of the scene/tree

	// different subdivision modes, see SubdivideXYZ() methods below
	const static int BSP_SUBDIVISIONMODE_SIMPLE   = 1;
	const static int BSP_SUBDIVISIONMODE_NORMAL   = 2;
	const static int BSP_SUBDIVISIONMODE_BALANCED = 3;

	int				subdivisionMode; // which subdivision function to use

	bool _debugEnable;

protected:

	#ifndef _USE_OOC
	/**
	* Builds the BSP tree by subdividing along the center of x, y, or z bounds, one
	* each time this function is called. This function calls itself recursively until
	* either the tree is deeper than MaxDepth or all of the tree leaves contains less
	* than MaxListLength of objects.
	*/
	bool Subdivide(long myOffset, TriangleIndexList *trilist, int depth, int axis, Vector3 &min, Vector3 &max, FILE *indexFP, FILE *nodeFP, unsigned int subObjectId);
	bool SubdivideSAH(long myOffset, TriangleIndexList *trilist, int depth,  Vector3 &min, Vector3 &max, FILE *indexFP, FILE *nodeFP, unsigned int subObjectId);
	#endif

	bool OBJSubdivide(long myOffset, TriangleIndexList *trilist, int depth,  Vector3 &min, Vector3 &max, FILE *indexFP, FILE *nodeFP);
	
	/**
	 * Recursively walks the tree and outputs the nodes
	 */	
	void printNodeInArray(const char *LoggerName = NULL, unsigned int currentIdx = 0, int depth = 0, unsigned int subObjectId = 0);	
	void printHighLevelNode(const char *LoggerName = NULL, unsigned int currentIdx = 0, int depth = 0);	
	void drawNode(BSPArrayTreeNodePtr node, Vector3 min, Vector3 max, int depth, unsigned int subObjectId = 0);
	void renderSplit(int axis, float splitCoord, Vector3 min, Vector3 max,TriangleIndexList *newlists[2]);

	//
	// Intersection Methods:
	//

	/**
	 * Intersects 4 rays with a list of triangles. Returns the hitmask with the respective lower bit set to 1 if 
	 * a hit was found, with 0 otherwise. Information about the hit is written into the HitPoint object given by
	 * obj.
	 * A fallback version for tracing only one ray is also provided.
	 * 
	 * @param ray		the rays to trace
	 * @param objList	the node containing triangles
	 * @param hitPoint	structure for saving information about the intersections found
	 * @param tmax		Max. t values for ray to hit the triangles. Used for limiting the ray to the node extends
	 * @param hitMask	Bitmask of rays that already hit something and will not generate further hitpoints
	 * @param sign		can be 1.0f oder -1.0f, controls culling of backfacing (1.0) or frontfacing triangles (-1.0f)
	 */

	int RayObjIntersect(const SIMDRay &rays, ModelInstance *subObject, BSPArrayTreeNodePtr objList, SIMDHitPointPtr hitPoint, float *tmax, float *tmins, int hitMask);
	int RayObjIntersect(const Ray &ray, ModelInstance *subObject, BSPArrayTreeNodePtr objList, HitPointPtr obj, float tmax);
	int BeamObjIntersect(Beam &beam, ModelInstance *subObject, BSPArrayTreeNodePtr objList, float *tmaxs, float *tmins);

	/**
	* Tests a ray for intersection with a list of triangles. Returns 1 in the bitmask of the ray if a triangle
	* is hit with a lower t-value than that given by tmaxs. Used by isVisible() to
	* determine if a ray hits a target at a certain position.
	* A fallback version for tracing only one ray is also provided.
	*
	* @param ray		the rays to trace
	* @param objList	the node containing triangles
	* @param hitPoint	structure for saving information about the intersections found
	* @param tmax		Max. t values for ray to hit the triangles. Used for limiting the ray to the node extends
	* @param hitMask	Bitmask of rays that already hit something and will not generate further hitpoints	
	*/	
	int RayObjIntersectTarget(const SIMDRay &rays, ModelInstance *subObject, BSPArrayTreeNodePtr objList, float *tmaxs, float *tmins, float *target_t, int hitMask);
	float RayObjIntersectTarget(const Ray &ray, ModelInstance *subObject, BSPArrayTreeNodePtr objList, float target_t, float ErrorBnd = 0.001f);
	float RayLODIntersectTarget(const Ray &ray, ModelInstance *subObject, unsigned int lodIndex, float tmax, float tmin);
	int RayLODIntersectTarget(const SIMDRay &rays, ModelInstance *subObject, BSPArrayTreeNodePtr currentNode, SIMDHitpoint *hits, SIMDVec4 &tmax, SIMDVec4 &target_t, SIMDVec4 &tmin, int hitValue);


	/**
	 * Intersects a single ray with a bounding box and returns the min and max t values at which
	 * the ray touches the bounding box. Provided for the legacy methods that trace single rays.
	 */
	FORCEINLINE bool RayBoxIntersect(const Ray& r, Vector3 &min, Vector3 &max, float *returnMin, float *returnMax);
	FORCEINLINE int RayBoxIntersect(const SIMDRay& rays, Vector3 *bb, SIMDVec4 &min, SIMDVec4 &max);


	// sungeui start --------------
	// intersect LOD representation with a ray
	FORCEINLINE int RayLODIntersect(const Ray &ray, ModelInstance *subObject, BSPArrayTreeNodePtr objList, HitPointPtr obj, float tmax, float tmin);
	FORCEINLINE int RayLODIntersect(const SIMDRay &rays, ModelInstance *subObject, BSPArrayTreeNodePtr currentNode, SIMDHitpoint *hits, SIMDVec4 &tmax, SIMDVec4 &tmin, int hitValue);	

	// sungeui end ----------------


	/**
	 *	High-level tree intersection functions:
	 */
	FORCEINLINE void SIMDBSPTree::initializeObjectIntersection(int threadID);
	FORCEINLINE ModelInstance *getNextObjectIntersection(Ray &ray, int threadID, float *returnMin, float *returnMax);
	FORCEINLINE ModelInstance *getNextObjectIntersection(SIMDRay &ray, int threadID, SIMDVec4 &returnMin, SIMDVec4 &returnMax);

	/////////////////////////////////////////////////////
	// Variables:
	
	ModelInstance			*objectList;
	//ModelInstance			*subObject;
	int						nSubObjects;

	// High-level kd-tree for storing the bounding boxes of the objects
	// in the scene.
	BSPArrayTreeNode		*objectTree;
	unsigned int			*objectIndexList;
	BSPTreeInfo				objTreeStats;

	// Index lists used during construction of the tree. We only need 2
	// lists for the left branches since they can be overwritten directly after
	// being used. For the right branch, we need as many lists as the maximal depth
	// of the tree. The lists will be deleted immediately after the tree is built.
	TriangleIndexList        *leftlist[2];			
	TriangleIndexList		 *rightlist[MAXBSPSIZE]; 
		
	Vector3		   *minvals,   // list of min/max values of all tris, used for balancing
				   *maxvals;
	
	Material *defaultMaterial;		// default material for tris without one

	// Time stats:
	TimerValue timeBuildStart, timeBuildEnd;

	/**
	 * Data structure for a simple stack (used for recursion unrolling during tracing
	 * rays through the tree). This is padded to 64 bytes for alignment.
	 * Note that there are 4 min and max values each, because for SIMD tracing we need
	 * the individual values for each ray. In single tracing just the first value is 
	 * needed.
	 */
	typedef struct {
		SIMDVec4 min;
		SIMDVec4 max;
		unsigned int node;
		unsigned int mask;
		Vector3 node_min, 
			    node_max;

		// sungeui start
		char m_MinDim [4];
		char m_MaxDim [4];
		// sungeui end
	} StackElem;

	// array of stacks (one for each thread when using parallel threads, otherwise just one)
	__declspec(align(16)) StackElem **stacks;
	__declspec(align(16)) StackElem **stacksHL;
	int *stackPtrHL;

	//
	// Intersection statistics, only used when _SIMD_SHOW_STATISTICS
	// is defined in common.h
	//
	__int64 _debug_TreeIntersectCount;			// total intersections of kd-trees
	__int64 _debug_NodeIntersections;			// total intersections with kd-tree inner nodes
	__int64 _debug_NodeIntersectionsOverhead;	// overhead intersections with kd-tree inner nodes for SIMD
	__int64 _debug_LeafIntersectCount;			// total intersections with leaf nodes
	__int64 _debug_LeafIntersectOverhead;		// overhead intersections with leaf node for SIMD
	__int64 _debug_LeafTriIntersectCount;		// total intersections with triangles
};

	#endif