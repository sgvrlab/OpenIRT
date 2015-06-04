#ifndef OUTOFCORETREEACCESS_H
#define OUTOFCORETREEACCESS_H

// sungeui --------------------------------------------------
// this class is created to avoid any conflict between BSPArrayTreeNode of other tree 
// classes.
// This class use runtime out-of-core file access, which is not used in other part of
// preprocessing code.


#include "OutOfCoreTree.h"
#include <hash_map>
#include <vector>
#include "MeshReader.h"
#include "Grid.h"
#include "common.h"

#include "cmdef.h"
#include "helpers.h"
#include "OptionManager.h"
#include "Logger.h"
#include "Vector3.h"
#include "Ray.h"
#include "Triangle.h"
#include "OOC_PCA.h"
#include <xmmintrin.h>
#include "BufferedOutputs.h"
#include "LOD_header.h"
#include "vec3f.hpp"
#include "Varray.h"
#include "Materials.h"
#include "Progression.h"
#include "LoggerImplementationFileout.h"
#include "Statistics.h"
#include "err_quantization.h"


#if HIERARCHY_TYPE == TYPE_BVH
#ifndef BSP_ARRAY_TREE_NODE_DEF
#define BSP_ARRAY_TREE_NODE_DEF
typedef union BSPArrayTreeNode_t {
	struct { // inner node
		unsigned int children;
		unsigned int children2;
		Vector3 min;
		Vector3 max;
	};
	struct { // with LOD
		unsigned int children;
		unsigned int lodIndex;
		Vector3 min;
		Vector3 max;
	};
	struct { // leaf node with LOD
		unsigned int indexCount;
		unsigned int indexOffset;
		Vector3 min;
		Vector3 max;
	};
	struct { // leaf node with LOD
		unsigned int triIndex;
		unsigned int lodIndex;
		Vector3 min;
		Vector3 max;
	};
} BSPArrayTreeNode, *BSPArrayTreeNodePtr;
#endif
#define BSPTREENODESIZE 32
#define BSPTREENODESIZEPOWER 5
#endif


#ifdef _USE_OOC
#include "OOCFile.h"
#include "OOCFile64.h"
#include "TreeLayout.h"
#include "DelayWrite.h"

typedef OOC_TRI_FILECLASS<Triangle>*			 BSPTriList;
typedef OOC_VERTEX_FILECLASS<Vertex>*           BSPVertexList;
typedef OOC_LOD_FILECLASS<LODNode>*				 LODList;
typedef OOC_BSPNODE_FILECLASS<BSPArrayTreeNode>* BSPNodeList;
typedef OOC_BSPIDX_FILECLASS<unsigned int>*      BSPIndexList;
typedef OOC_NodeLayout_FILECLASS<CNodeLayout>*   NodeLayoutList;


#else
#include "TreeLayout.h"

// non-OOC mode:
typedef Triangle*         BSPTriList;
typedef Vertex*		  BSPVertexList;
typedef LODNode*		  LODList;
typedef BSPArrayTreeNode* BSPNodeList;
typedef unsigned int*	  BSPIndexList;
typedef CNodeLayout*		NodeLayoutList;

#endif

#define CONVERT_VEC3F_VECTOR3(s,d) {d.e [0] = s.x;d.e [1] = s.y; d.e[2] = s.z;}
#define CONVERT_VECTOR3_VEC3F(s,d) {d.x = s.e [0];;d.y = s.e [1];d.z = s.e [2];}

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


enum {BFS_LAYOUT, DFS_LAYOUT, VEB_LAYOUT, VEB_LAYOUT_CONTI, CO_LAYOUT, CO_LAYOUT_PRESERVE_ORDER, CO_LAYOUT_CONTI, CA_LAYOUT, COML_LAYOUT};

class OoCTreeAccess: public OutOfCoreTree
{
public:

	// Main caller to compute simplification representation
	bool ComputeSimpRep (void);
	bool ComputeSimpRep (unsigned int NodeIdx, Vector3 & BBMin, Vector3 & BBMax,
						float MinVolume, COOCPCAwoExtent & PCA, Vec3f * RealBB);
	float ComputeRLOD (unsigned int NodeIdx, COOCPCAwoExtent & PCA, 
					  Vector3 & BBMin, Vector3 & BBMax, Vec3f * RealBB);
	bool Check (bool LODCheck);
	bool Check (int Idx, Vec3f BBox [2], bool LODCheck, float PreError,
				int Depth);

	bool PrepareDataForRLODs (char * fileName, bool Material = true);
	void PrintTree(const char *LoggerName = NULL);
	bool PrintError (void);
	bool PrintError (int NodeIdx);
	bool QuantizeErr (void);
	bool QuantizeErr (int NodeIdx);

	bool SaveStatBVH (void);

	// Layout computation -------------------------------------------------------------
	bool ComputeLayout (int Type);
	bool ComputeLayout_CO_CONTI (CTreeCluster & SubCluster);
	bool ComputeLayout_CO (CTreeCluster & SubCluster);
	bool ComputeLayout_CO_PreserveOrder (CTreeCluster & TreeLayout);

	bool ComputeLayout_CA (CTreeCluster & TreeLayout);
	bool ComputeLayout_VEB (
		unsigned int StartIdx, int MaxDepth, VArray <int> & NextFront);
	bool ComputeLayout_VEB_CONTI (
		unsigned int StartIdx, int MaxDepth, VArray <int> & NextFront);
	bool ComputeLayout_BFS (void);
	bool ComputeLayout_DFS (void);



	void ComputeAccessProb (unsigned int ParentIdx, unsigned int CurIdx, 
								Vector3 & BBMin, Vector3 & BBMax,	// BB of the parent node
								Vector3 & Min, Vector3 & Max);		// BB of the current node
	


	void ComputeContainedNodes (CTreeCluster & TreeLayout, bool RootContained = false);

	// --------------------------------------------------------------------------------






protected:	
	char OOCDirName [MAX_PATH];

	BSPTriList				 m_TriangleList;      // triangle list	
	BSPVertexList			 m_VertexList;		// vertex list
	
	BSPNodeList	   m_Tree;       // array containing all nodes after construction
	BSPIndexList   m_Indexlists; // array containing all triangle index lists after construction	
	
	BufferedOutputs<LODNode> * m_pLODBuf;
	LODList			m_LodList;
	NodeLayoutList	m_NodeLayout;


	int m_NumLODs; 
	unsigned int m_NumLODBit;

	VArray <rgb> m_MaterialColor;	// map from material ID to color.

	CErrorQuan m_QuanErr;

	// Statistics:
	BSPTreeInfo m_treeStats; 
	Progression * m_pProg;

	LoggerImplementationFileout * m_pLog;

	Stats<float> m_StatErr, m_StatErrRatio;	// m_Ratio is ratio of error between parent and its children		
	Stats<double> m_DepthStats;		// for depth of leaf

	// layout ----------------------------------------------
	CDelayWrite * m_pLayoutWrite;
	//CTreeCluster	m_TreeLayout;	
	BSPNodeList		m_TreeNew;	// new layout of kd-tree
	LODList			m_LodListNew;	// new layout of LODs

	int m_SavedKDNodes;
	int m_SavedLODNodes;

	// -----------------------------------------------------

};



#endif