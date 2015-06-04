#ifndef COMMON_BVH_H
#define COMMON_BVH_H

#include "common.h"
#include <queue>

#if HIERARCHY_TYPE == TYPE_BVH

#if 0
#define BVHNODE_BYTES 40
typedef union BVHNode_t {
	struct { // inner node
		unsigned int children;
		unsigned int children2;
		float splitcoord;
		unsigned int lodIndex;
		Vector3 min;
		Vector3 max;
	};
	struct { // leaf node
		unsigned int indexCount;
		unsigned int indexOffset;
		unsigned int dummy1;
		unsigned int dummy2;
		Vector3 min;
		Vector3 max;
	};
} BVHNode, *BVHNodePtr;

typedef struct BVHInfo_t {
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

} BVHInfo, *BVHInfoPtr;
#endif

#define MAXBVHDEPTH 150

class BVH
{
public:
	BVH(unsigned int numObjects, ModelInstance *objectList, Material *defaultMaterial)
	{
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
		minvals = NULL;
		maxvals = NULL;
		for(int i=0;i<16;i++) chk[i] = false;
		this->defaultMaterial = defaultMaterial;
	}
	~BVH(void)
	{

		LogManager *log = LogManager::getSingletonPtr();
		log->logMessage("Destroying BSP tree...");		
			
		// delete min/max values of triangles
		if (minvals)
			delete minvals;
		if (maxvals)
			delete maxvals;
	}
	Vector3			objectBB[2];        // bounding box of the scene/tree

	void drawNode(BSPArrayTreeNodePtr node, Vector3 min, Vector3 max, int depth, unsigned int subObjectId = 0);
	void GLdrawTree(Ray &viewer, unsigned int subObjectId = 0);
	void printTree(bool dumpTree = false, const char *LoggerName = NULL, unsigned int subObjectId = 0);

	int RayTriIntersect(const Ray &ray, ModelInstance *object, BSPArrayTreeNodePtr node, HitPointPtr hitPoint, float tmax);  
	FORCEINLINE bool RayBoxIntersect(const Ray& r, Vector3 &min, Vector3 &max, float &interval_min, float &interval_max);
	int RayTreeIntersectRecursive(Ray &ray, HitPointPtr hit, unsigned int index, float TraveledDist = 0.f);
	int RayTreeIntersect(ModelInstance* object, Ray &ray, HitPointPtr hit, float TraveledDist = 0.f);

	void initialize(char *filename);
	void finalize();

	/**
	 * Data structure for a simple stack (used for recursion unrolling during tracing
	 * rays through the tree)..	 
	 */
	typedef struct {
		unsigned int index;
		Vector3 pMin;
		Vector3 pMax;
	} StackElem;

	/////////////////////////////////////////////////////
	// Variables:
	
	bool chk[16];

	ModelInstance			*objectList;
	int						nSubObjects;

	// High-level BVH for storing the bounding boxes of the objects
	// in the scene.
	BSPArrayTreeNode			*objectTree;
	unsigned int	*objectIndexList;
	 BSPTreeInfo			objTreeStats;

	Vector3		   *minvals,   // list of min/max values of all tris, used for balancing
				   *maxvals;
	
	Material *defaultMaterial;		// default material for tris without one

	// Time stats:
	TimerValue timeBuildStart, timeBuildEnd;

	// taejoon
	int curStep;
	typedef struct {
		BSPArrayTreeNodePtr node;
		int depth;
	} queueElem;
	queue<queueElem> nodeQueue;

	/*
	typedef struct
	{
		int index;
		int count;
		int isChildInThis;
		int axis;
	} FrontNode, *FrontNodePtr;
	*/

	/************************************************************************/
	/* ´ö¼ö. Collision Detection                                            */
	/************************************************************************/

	bool isOverlab( BVH* target ) ;				// check overlap between two BVH
};
#endif

#endif