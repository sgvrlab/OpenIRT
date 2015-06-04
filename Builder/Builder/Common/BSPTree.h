#ifndef COMMON_BSPTREE_H
#define COMMON_BSPTREE_H

#define MAXBSPSIZE  50

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
 * Condensed structure for a BSP tree node.
 *
 * children points to the first child node for inner nodes, the second node 
 * will be assumed to be (children+8), since nodes have a fixed
 * size in memory (if pointers are 32 bit, this won't work on 
 * 64 bit CPUs !). For leaf nodes, this points to the triangle list.
 * Additionally, the first two low-order bits 
 * of the pointer are used for determining the type of the node:
 *  0 - Leaf
 *  1 - Split by X plane
 *  2 - Split by Y plane
 *  3 - Split by Z plane
 * This is possible because the structure should be aligned to
 * 32bit boundaries in memory, so the respective bits would be
 * zero anyway.
 *
 * The member split is the coordinate of the split plane or the
 * number of triangles for leaf nodes.
 */
typedef struct BSPTreeNode_t {
	void *children;
	float splitcoord;
} BSPTreeNode, *BSPTreeNodePtr;

/**
 * BSP tree class.
 * 
 */
class FastBSPTree {
public:

	/**
	* Constructor, sets parameters and gets pointer to triangle list to build
	* the tree for.
	*/
	FastBSPTree(TriangleList &tris, IntersectionTriangleList &tris_intersect, Vector3 &min, Vector3 &max) {
		OptionManager *opt = OptionManager::getSingletonPtr();

		trianglelist = &tris;
		intersectlist = &tris_intersect;
		this->min = min;
		this->max = max;

		// load variable options
		maxDepth = opt->getOptionAsInt("raytracing", "maxBSPTreeDepth", 25);
		maxListLength = opt->getOptionAsInt("raytracing", "maxBSPLeafTriangles", 7);

		stack = new Stack;
		root = NULL;		
		minvals = NULL;
		maxvals = NULL;
		numTris = 0;
		numNodes = 0;
		numLeafs = 0;
		maxLeafDepth = 0;		
		sumDepth = 0;
		sumTris = 0;
		subdivisionMode = BSP_SUBDIVISIONMODE_SIMPLE;

		_debug_TreeIntersectCount = 0;
		_debug_ObjIntersectCount = 0;
		_debug_ObjTriIntersectCount = 0;
		rayNumCounter[0] = 0;
		rayNumCounter[1] = 0;
	}

	/**
	 * Destructor, frees memory of nodes and triangle lists
	 */
	~FastBSPTree() {	

		// recursively free memory
		if (root) {
			destroyNode(root);
			delete root;
		}
		if (stack)
			delete stack;
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
	void buildTree(int subdivisionMode = BSP_SUBDIVISIONMODE_SIMPLE);

	/**
	 * Traverses ray through FastBSPTree and intersects ray with all of the objects along
	 * the way. Returns the hitpoint and true, if something was hit
	 */
	bool RayTreeIntersect(const Ray &ray, HitPointPtr hit, float sign = 1.0f);

	/**
	 * Traverses from origin to target and returns true, if nothing was
	 * hit on the way (i.e. origin and target have an unoccluded line of sight)
	 */
	bool isVisible(const Vector3 &origin, const Vector3 &target);

	/**
	 * Prints statistical information on the BSP tree (dump of tree structure, too, 
	 * of dumpTree = true)
	 */
	void printTree(bool dumpTree = false, const char *LoggerName = NULL);

	/**
	* Get number of triangles in this tree.
	* This counts the real number of triangle and counts triangles contained
	* in multiple nodes only once.
	*/
	int getNumTris();

	void dumpCounts() {
		cout << "BSP Stats after trace:\n";
		cout << "Tree Intersections: " << _debug_TreeIntersectCount << endl;
		cout << "Node Intersections: " << _debug_ObjIntersectCount << endl;

		cout << "Tri Intersections: " << _debug_ObjTriIntersectCount << endl;
		cout << "\tno hits:\t" << rayNumCounter[0] << endl;
		cout << "\thits:\t" << rayNumCounter[1] << endl;

		if (_debug_ObjIntersectCount > 0)
			cout << "Avg. Node Intersects / Ray: " << (float)_debug_ObjIntersectCount/(float)_debug_TreeIntersectCount << endl;
		if (_debug_ObjIntersectCount > 0)
			cout << "Avg. Tri Intersects / Node: " << (float)_debug_ObjTriIntersectCount/(float)_debug_ObjIntersectCount << endl;		
	}


	// different subdivision modes, see SubdivideXYZ() methods below
	const static int BSP_SUBDIVISIONMODE_SIMPLE   = 1;
	const static int BSP_SUBDIVISIONMODE_NORMAL   = 2;
	const static int BSP_SUBDIVISIONMODE_BALANCED = 3;

protected:

	/**
	 * Builds the BSP tree by subdividing along the center of x, y, or z bounds, one
	 * each time this function is called. This function calls itself recursively until
	 * either the tree is deeper than MaxDepth or all of the tree leaves contains less
	 * than MaxListLength of objects.
	 */
	void Subdivide(BSPTreeNodePtr node, TriangleIndexList *trilist, int depth, int axis, Vector3 &min, Vector3 &max, int subdivideFailCount = 0);
	void SubdivideSimple(BSPTreeNodePtr node, TriangleIndexList *trilist, int depth, int axis, Vector3 &min, Vector3 &max);
	void SubdivideBalanced(BSPTreeNodePtr node, TriangleIndexList *trilist, int depth, int axis, Vector3 &min, Vector3 &max);

	/**
	 * Recursively walks the tree and outputs the nodes
	 */
	void printNode(const char *LoggerName = NULL, BSPTreeNodePtr current = NULL, int depth = 0);
	
	/**
	 * Destroys the tree, frees memory. Called by dtor.
	 */	
	void destroyNode(BSPTreeNodePtr root);

	//
	// Intersection Methods:
	//

	/**
	 * Intersects a ray with an axis-aligned bounding-box and returns true, if the box
	 * was hit, false otherwise. On a hit, returnMin and returnMax contain the parametric
	 * values t for the ray on which it intersects the box.
	 */
	bool RayBoxIntersect(const Ray& r, Vector3 &min, Vector3 &max, float *returnMin, float *returnMax);

	/**
	 * Intersects a ray with a list of triangles. Returns true if a hit was found, false
	 * otherwise. Information about the hit is written into the HitPoint object given by
	 * obj.
	 */	
	bool RayObjIntersect(const Ray &ray, BSPTreeNodePtr objList, HitPointPtr obj, float tmax, float sign = 1.0f);  

	/**
	 * Tests a ray for intersection with a list of triangles. Returns true if a triangle
	 * is hit with a lower t-value than that given by target_t. Used by isVisible() to
	 * determine if a ray hits a target at a certain position.
	 */	
	bool RayObjIntersectTarget(const Ray &ray, BSPTreeNodePtr objList, float target_t);
	

	TriangleList	         *trianglelist;      // triangle list
	IntersectionTriangleList *intersectlist;     // triangle list with intersection data
	Vector3					 *minvals,			 // list of min/max values of all tris, used for balancing
		                     *maxvals;

	// Index lists used during construction of the tree. We only need 2
	// lists for the left branches since they can be overwritten directly after
	// being used. For the right branch, we need as many lists as the maximal depth
	// of the tree. The lists will be deleted immediately after the tree is built.
	TriangleIndexList        *leftlist[2];			
	TriangleIndexList		 *rightlist[MAXBSPSIZE]; 

	Vector3			min, max;        // bounding box of the scene/tree
	int				maxDepth;        // max. allowed depth in the tree
	int				maxListLength;   // max. allowed tris per node (will allow more if maxDepth too low)
	BSPTreeNodePtr  root;            // root node of the tree
	int				subdivisionMode; // which subdivision function to use

	// Statistics:
	int numTris;		// (real, not counting duplicates) number of triangles in scene 
	int numNodes;		// number of nodes in tree
	int numLeafs;		// number of leaf nodes
	int maxLeafDepth;	// largest depth in tree	
	int sumDepth;		// sum of leaf depth, for calculating avgDepth
	int sumTris;		// effective number of tri(-indices) in tree

	// Time stats:
	TimerValue timeBuildStart, timeBuildEnd;
	float timeBuild;

	/**
	 * Data structure for a simple stack (used for recursion unrolling during tracing
	 * rays through the tree.
	 */
	typedef struct {
		BSPTreeNodePtr node;
		float     min, max;
	} StackElem;

	typedef struct {
		int       stackPtr;
		StackElem stack[MAXBSPSIZE];
	} Stack, *StackPtr;

	/**
	 * Stack operations.
	 */
	inline void initStack()
	{
		stack->stack[0].node = NULL;
		stack->stackPtr = 1;
	}

	inline void push(BSPTreeNodePtr node, float min, float max)
	{
		stack->stack[stack->stackPtr].node = node;
		stack->stack[stack->stackPtr].min = min;
		stack->stack[stack->stackPtr].max = max;
		(stack->stackPtr)++;
	}

	inline void pop(BSPTreeNodePtr *node, float *min, float *max)
	{
		(stack->stackPtr)--;
		*node = stack->stack[stack->stackPtr].node;
		*min = stack->stack[stack->stackPtr].min;
		*max = stack->stack[stack->stackPtr].max;
	}

	// our stack, used in RayTreeIntersect() and isVisible()
	StackPtr stack;

	unsigned int _debug_TreeIntersectCount;
	unsigned int _debug_ObjIntersectCount;
	unsigned int _debug_ObjTriIntersectCount;
	unsigned int rayNumCounter[2];
};

#endif