#include "common.h"
//#include "kDTree.h"
//#include "OutOfCoreTree.h"
#include "OoCTreeAccess.h"

#include <math.h>
#include <direct.h>
#include "BufferedOutputs.h"
#include "CashedBoxFiles.h"
#include "Triangle.h"
#include "helpers.h"

//#include "VoxelkDTree.h"
#include "OptionManager.h"
#include "Progression.h"
#include "Materials.h"

#include <sys/types.h>
#include <sys/stat.h>
//#include <unistd.h>
#include <time.h>
#include "Statistics.h"
#include "queue"
#include "stack"
#include "ANN.h"
#include "LayoutGraph.h"
#include "Stopwatch.hpp"
#include "limits.h"

using namespace std;

// mininum volume reduction between two nodes of LOD
const int LOD_DEPTH_GRANUALITY = 3; 
const float MIN_VOLUME_REDUCTION_BW_LODS = pow ((float)2, (int)LOD_DEPTH_GRANUALITY);
const int MIN_NUM_TRIANGLES_LOD = pow ((float) 2, (int)LOD_DEPTH_GRANUALITY);
//const unsigned int MAX_NUM_LODs = (1 << 31) - 1;
const unsigned int MAX_NUM_LODs = UINT_MAX;
const unsigned int LOD_BIT = 1;
//#define HAS_LOD(idx) (idx & LOD_BIT)
#define HAS_LOD(idx) (idx != 0)
const unsigned int ERR_BITs = 5;
#define GET_REAL_IDX(idx) (idx >> ERR_BITs)
//#define GET_REAL_IDX(idx) (idx >> LOD_BIT)

#ifdef _USE_OOC
#define GET_LOD(Idx) (*m_LodList)[Idx*sizeof(LODNode)]
#else
#endif

// easy macros for working with the compressed BSP tree
// structure that is a bit hard to understand otherwise
// (and access may vary on whether OOC mode is on or not)
#ifdef FOUR_BYTE_FOR_KD_NODE
	#define AXIS(node) ((node)->children2 & 3)
	#define ISLEAF(node) (((node)->children2 & 3) == 3)
	#define ISNOLEAF(node) (((node)->children2 & 3) != 3)
#else
	#define AXIS(node) ((node)->children & 3)
	#define ISLEAF(node) (((node)->children & 3) == 3)
	#define ISNOLEAF(node) (((node)->children & 3) != 3)
#endif

#define MAKECHILDCOUNT(count) ((count << 2) | 3)
#ifdef _USE_ONE_TRI_PER_LEAF
#define GETIDXOFFSET(node) ((node)->triIndex >> 2)
#define GETCHILDCOUNT(node) (1)
#else
#define GETIDXOFFSET(node) ((node)->indexOffset)
#define GETCHILDCOUNT(node) ((node)->indexCount >> 2)
#endif

#ifdef _USE_OOC
#define BSPNEXTNODE 1

	#ifdef KDTREENODE_16BYTES
		#ifdef FOUR_BYTE_FOR_KD_NODE
			#define GETLEFTCHILD(node) (node->children)
			#define GETRIGHTCHILD(node) (node->children + 1)
		#else
			#define GETLEFTCHILD(node) (node->children >> 2)
			#ifdef _USE_CONTI_NODE
				#define GETRIGHTCHILD(node) ((node->children >> 2) + 1)
			#else
				#define GETRIGHTCHILD(node) (node->children2 >> 2)
			#endif
		#endif
	#else
		#define GETLEFTCHILD(node) (node->children >> 2)
		#define GETRIGHTCHILD(node) (node->children >> 2) + BSPNEXTNODE
	#endif

	#define GETNODE(offset) ((BSPArrayTreeNodePtr)&((*m_Tree)[offset]))
	#define MAKEIDX_PTR(offset) ((unsigned int *)&((*m_Indexlists)[offset]))
#else

#ifdef KDTREENODE_16BYTES
	#ifdef FOUR_BYTE_FOR_KD_NODE
		#define GETLEFTCHILD(node) ((node->children)
		#define GETRIGHTCHILD(node) (node->children + 1)
	#else
		#define GETLEFTCHILD(node) ((node->children & ~3) << 1)
		#define GETRIGHTCHILD(node) (node->children2 << 1)
	#endif
#else
#define BSPNEXTNODE BSPTREENODESIZE
#define GETLEFTCHILD(node) ((node->children & ~3) << 1)
#define GETRIGHTCHILD(node) ((node->children & ~3) << 1) + BSPTREENODESIZE
#endif

#define GETNODE(offset) ((BSPArrayTreeNodePtr)((char *)m_Tree + offset))
#define MAKEIDX_PTR(offset) (&m_Indexlists[offset])
#endif


// macros for accessing single triangles and vertices, depending on
// whether we're using OOC or normal mode
#ifdef _USE_OOC
#define GETTRI(idx) (*m_TriangleList)[idx]
#define GETVERTEX(idx) (*m_VertexList)[idx*sizeof(Vertex)]
#else
#define GETTRI(idx) m_TriangleList[idx]
#define GETVERTEX(idx) m_VertexList[idx]
#endif


#define CONVERT_VEC3F_VECTOR3(s,d) {d.e [0] = s.x;d.e [1] = s.y; d.e[2] = s.z;}
#define CONVERT_VECTOR3_VEC3F(s,d) {d.x = s.e [0];;d.y = s.e [1];d.z = s.e [2];}



// Layout computation --------------------------------
#include "TreeLayout.h"


// for cache-aware layout
int g_NumNodeInBlock = 4 * 1024 / sizeof(BSPArrayTreeNode);
int GetGoodProcessingUnitSize (int NumNode);
// ---------------------------------------------------




bool OoCTreeAccess::ComputeLayout (int Type)
{
	OptionManager *opt = OptionManager::getSingletonPtr();

	char fileNameTri[MAX_PATH], fileNameVertex[MAX_PATH], fileNameMaterial[MAX_PATH];
	char filenameNodes[MAX_PATH], fileNameIndices[MAX_PATH], fileNameLODs[MAX_PATH];
	char fileNameNodeLayouts[MAX_PATH];
	char output[1000];

	sprintf(filenameNodes, "%s.node", getkDTreeFileName ().c_str ());
	sprintf(fileNameIndices, "%s.idx", getkDTreeFileName ().c_str ());
	sprintf(fileNameLODs, "%s.lod", getkDTreeFileName ().c_str ());
	sprintf(fileNameNodeLayouts, "%s.layout_temp", getkDTreeFileName ().c_str ());

	m_Tree = new OOC_BSPNODE_FILECLASS<BSPArrayTreeNode>(filenameNodes, 										 
										 //1024*1024*32,
										 1024*1024*512,	// for BFS layout
										 64*1024);

#ifdef _USE_LOD
	m_LodList = new OOC_LOD_FILECLASS<LODNode>(fileNameLODs, 
										   1024*1024*32,
										   64*1024);  
#endif

	m_NodeLayout = new OOC_NodeLayout_FILECLASS<CNodeLayout>(fileNameNodeLayouts, 
										   1024*1024*32,
										   64*1024, "wc", m_treeStats.numNodes); 

	
	//m_pLog = new LoggerImplementationFileout (getLogFileName ().c_str ());
	m_pLog->logMessage (LOG_INFO, "Layout start.");

	
	
	
	m_pProg = new Progression ("Probability computation", m_treeStats.numNodes, 100);

	m_pLog->logMessage (LOG_INFO, "Probability computation start");

	Vector3 BBMin, BBMax;
	BBMin = grid.p_min;
	BBMax = grid.p_max;

	ComputeAccessProb (NULL_IDX, 0, BBMin, BBMax, BBMin, BBMax);

	m_pLog->logMessage (LOG_INFO, "Probability computation end");
	delete m_pProg;


	

	m_pProg = new Progression ("Layout computation", m_treeStats.numNodes, 100);

	m_pLog->logMessage (LOG_INFO, "Layout computation start");

	// create new layouts	
	char filenameNodes_new[MAX_PATH], fileNameLODs_new[MAX_PATH];

	sprintf(filenameNodes_new, "%s.node_new", getkDTreeFileName ().c_str ());
	sprintf(fileNameLODs_new, "%s.lod_new", getkDTreeFileName ().c_str ());


#ifdef _USE_LOD
	m_NumLODs = m_LodList->m_fileSize.QuadPart / sizeof (LODNode);
#endif
	m_TreeNew = new OOC_BSPNODE_FILECLASS<BSPArrayTreeNode>(filenameNodes_new, 										 
										 //1024*1024*32,
										 1024*1024*512,	// for BFS layout
										 64*1024, "wc", m_treeStats.numNodes);

#ifdef _USE_LOD
	m_LodListNew = new OOC_LOD_FILECLASS<LODNode>(fileNameLODs_new, 
										   1024*1024*32,
										   64*1024, "wc", m_NumLODs); 
#endif


	m_SavedKDNodes = 0;
	m_SavedLODNodes = 0;

	if (Type == CA_LAYOUT) {
		// for cache-aware layout of kd-nodes, not R-LODs
		m_pLayoutWrite = new CDelayWrite (* m_Tree, * m_TreeNew, 
#ifdef _USE_LOD
										* m_LodList, * m_LodListNew, 
#endif
										* m_NodeLayout,
										g_NumNodeInBlock);
	}
	else {
		m_pLayoutWrite = new CDelayWrite (* m_Tree, * m_TreeNew, 
#ifdef _USE_LOD
										* m_LodList, * m_LodListNew, 
#endif
										* m_NodeLayout,
										1);	// no delayed buffer
	
	}

#ifdef _USE_LOD
	m_pLayoutWrite->m_LODCurPos = 1;	// index 0 is non LOD, so we make sure that we do not have it
#endif

	CTreeCluster TreeLayout;
	TreeLayout.m_pParentInDecomposition = NULL;
	TreeLayout.m_StartIdx = 0;
	TreeLayout.m_NumNodes = m_treeStats.numNodes;
	TreeLayout.m_BB [0] = grid.p_min;
	TreeLayout.m_BB [1] = grid.p_max;

	
	
	VArray <int> NextFront;		// this is dummy for the root call


	// NOTE::
		// please do not use "FOUR_BYTE_FOR_KD_NODE" for the moment
	#ifdef FOUR_BYTE_FOR_KD_NODE
	// we store right child right next to left child
	m_pLayoutWrite->Add (0);	// store root first.
								// since we save two child consecutives, saving root first
								// make code simpler.
	#endif

	
	switch (Type) {
		case CO_LAYOUT_PRESERVE_ORDER:
				ComputeLayout_CO_PreserveOrder (TreeLayout);
				break;
		case CO_LAYOUT:
				ComputeLayout_CO (TreeLayout);
				break;

		case CO_LAYOUT_CONTI:
				#ifndef FOUR_BYTE_FOR_KD_NODE
				// we still store right child right next to left child for better performance
				m_pLayoutWrite->Add (0);	// store root first.
											// since we save two child consecutives, saving root first
											// make code simpler.
				#endif


				ComputeLayout_CO_CONTI (TreeLayout);
				break;
		case VEB_LAYOUT:			
				ComputeLayout_VEB (0, m_treeStats.maxDepth, NextFront);
				break;
		case VEB_LAYOUT_CONTI:			
				// we still store right child right next to left child
				m_pLayoutWrite->Add (0);
				ComputeLayout_VEB_CONTI (0, m_treeStats.maxDepth, NextFront);
				break;

				/*
		case VEB_LAYOUT_CONTI:			
				ComputeLayout_VEB _Contiguous(0, m_treeStats.maxDepth, NextFront);
				break;
				*/
		case DFS_LAYOUT:
				ComputeLayout_DFS ();
				break;

		case BFS_LAYOUT:
				ComputeLayout_BFS ();
				break;
		case CA_LAYOUT:
				ComputeLayout_CA (TreeLayout);
				break;
				/*
		case COML_LAYOUT:
				ComputeLayout_COML (TreeLayout);
				break;
				*/
	}
				

	if (Type == CA_LAYOUT) {
		// for cache-aware, we added one dummy value
		if (m_pLayoutWrite->m_CurPos - 1 != m_treeStats.numNodes
#ifdef _USE_LOD
			|| m_pLayoutWrite->m_LODCurPos != m_NumLODs
#endif
			) {
#ifdef _USE_LOD
					printf ("Error during saving data we have %d, %d nodes, but we saved %d %d\n",
					m_pLayoutWrite->m_CurPos, m_pLayoutWrite->m_LODCurPos,
					m_treeStats.numNodes, m_NumLODs);
#else
					printf ("Error during saving data we have %d nodes, but we saved %d\n",
					m_pLayoutWrite->m_CurPos,
					m_treeStats.numNodes);
#endif
			}
	}
	else {
		if (m_pLayoutWrite->m_CurPos != m_treeStats.numNodes
#ifdef _USE_LOD
			|| m_pLayoutWrite->m_LODCurPos != m_NumLODs
#endif
			) {
#ifdef _USE_LOD
				printf ("Error during saving data we have %d, %d nodes, but we saved %d %d\n",
				m_pLayoutWrite->m_CurPos, m_pLayoutWrite->m_LODCurPos,
				m_treeStats.numNodes, m_NumLODs);
#else
				printf ("Error during saving data we have %d nodes, but we saved %d\n",
				m_pLayoutWrite->m_CurPos,
				m_treeStats.numNodes);
#endif
		}
	}

	m_pLog->logMessage (LOG_INFO, "Layout computation  end");

	delete m_pLayoutWrite;


	m_pLog->logMessage (LOG_INFO, "Layout end.");
	delete m_pProg;


#ifdef _USE_LOD
	delete m_LodListNew;
#endif
	delete m_TreeNew;


#ifdef _USE_LOD
	delete m_LodList;
#endif
	delete m_Tree;

	delete m_NodeLayout;


	m_pLog->logMessage (LOG_INFO, "Making a computed layout to a new one.");


	remove(filenameNodes);
	remove(fileNameLODs);

	rename (fileNameLODs_new, fileNameLODs);
	rename (filenameNodes_new, filenameNodes);
	


	
	remove(fileNameNodeLayouts);



	m_pLog->logMessage (LOG_INFO, "Done.");

	//delete m_pLog;

	return true;
}




// for each node in the front, we store its two children.
// So, its probability is to have collision given its parent collision.
// Node of StartIdx is already stored in the parent cluster.
// we layout nodes from the childs of the StartIdx.
bool OoCTreeAccess::ComputeLayout_CO_CONTI (CTreeCluster & TreeLayout)
{
	bool bContained;
	unsigned int StartIdx = TreeLayout.m_StartIdx;

	if (TreeLayout.m_NumNodes == 0)
		return true;

	if (TreeLayout.m_NumNodes <= 2) {
		BSPArrayTreeNodePtr pCurNode = GETNODE(StartIdx);
		unsigned int LeftIdx = GETLEFTCHILD(pCurNode);
		unsigned int RightIdx = GETRIGHTCHILD(pCurNode);

		int NumAdded = 0;
		bContained = TreeLayout.IsBelongToParentDecomposedCluster (LeftIdx);
		
		if (bContained) {
			m_pLayoutWrite->Add (LeftIdx);
			NumAdded++;
			m_pProg->step ();
		}
	
		bContained = TreeLayout.IsBelongToParentDecomposedCluster (RightIdx);
		
		if (bContained) {
			m_pLayoutWrite->Add (RightIdx);
			NumAdded++;
			m_pProg->step ();
		}

		assert (NumAdded == 0 || NumAdded == 2);
		return true;
	}


	//Stopwatch T1 ("collecting root cluster");
	//T1.Start ();
	

	// ----------------------------------------
	// Decomposition the contained tree.	
	// ----------------------------------------
	OOC_NodeLayout_FILECLASS<CNodeLayout> & NodeLayout = * m_NodeLayout;
	int NumNodes = TreeLayout.m_NumNodes;
	int NumNodesInCluster = GetAvgNodesInCluster (NumNodes);
	int MaxNumClusters = NumNodesInCluster + 2;

	// create with the maximum # of clusters
	TreeLayout.SetMaxNumClusters (MaxNumClusters);	

	// compute nodes that belong to the root cluster
	CTreeCluster & RootSubClusters = TreeLayout.m_SubClusters [0];

	// Step1: compute root cluster.
	// store nodes from the local maximum.
	priority_queue <CSortedNodeLayout> LocalHeap;
	CSortedNodeLayout Root (NodeLayout, & RootSubClusters.m_ContainedNodes, StartIdx,
		TreeLayout.m_BB[0], TreeLayout.m_BB[1], false);	// this node is not under the current root cluster


	LocalHeap.push (Root);		// add root
	int NumAddedNodes = 0;

	while (!LocalHeap.empty () && (NumAddedNodes < NumNodesInCluster)) {
		CSortedNodeLayout Node = LocalHeap.top ();
		LocalHeap.pop ();

		// we store two child together since we access them together
		// put two children				
		BSPArrayTreeNodePtr pCurNode = GETNODE(Node.m_Idx);
	
		if (ISNOLEAF(pCurNode)) {	
			Vector3 Min = Node.m_BB [0];
			Vector3 Max = Node.m_BB [1];

			unsigned int LeftIdx = GETLEFTCHILD(pCurNode);
			unsigned int RightIdx = GETRIGHTCHILD(pCurNode);

			Vector3 LBB [2], RBB [2];
			BSPArrayTreeNodePtr lChild = GETNODE(LeftIdx);
			BSPArrayTreeNodePtr rChild = GETNODE(RightIdx);
			LBB [0] = lChild->min; LBB [1] = lChild->max;
			RBB [0] = rChild->min; RBB [1] = rChild->max;

			bContained = TreeLayout.IsBelongToParentDecomposedCluster (LeftIdx);
			
			if (bContained) {
				CSortedNodeLayout Left (NodeLayout, 
					&RootSubClusters.m_ContainedNodes, LeftIdx,
					LBB[0], LBB[1]);
				LocalHeap.push (Left);
				NumAddedNodes++;
			}
			
			bContained = TreeLayout.IsBelongToParentDecomposedCluster (RightIdx);
			
			if (bContained) {
				CSortedNodeLayout Right (NodeLayout,
					&RootSubClusters.m_ContainedNodes, RightIdx,
					RBB[0], RBB[1]);
				LocalHeap.push (Right);
				NumAddedNodes++;
			}
			
		}
	}




	// prepare root cluster of the decomposed clusters
	TreeLayout.m_SubClusters [0].m_pParentInDecomposition = & TreeLayout;
	TreeLayout.m_SubClusters [0].m_StartIdx = StartIdx;
	TreeLayout.m_SubClusters [0].m_NumNodes = NumAddedNodes;
	TreeLayout.m_SubClusters [0].m_BB [0] = TreeLayout.m_BB [0];
	TreeLayout.m_SubClusters [0].m_BB [1] = TreeLayout.m_BB [1];

	assert (
		TreeLayout.m_SubClusters [0].m_ContainedNodes.size () == TreeLayout.m_SubClusters [0].m_NumNodes &&
		TreeLayout.m_SubClusters [0].m_NumNodes != 0);
		
	
	// Step2: compute child clusters.
	//			each node in LocalHeap was already stored in the root cluster
	//			two child nodes of the node will be root node of the child cluster

	// make node in the queue as a root of the child cluster
	int CurrClusterIdx = 1;
	while (!LocalHeap.empty ()) {
		CSortedNodeLayout Node = LocalHeap.top ();
		LocalHeap.pop ();

		TreeLayout.m_SubClusters [CurrClusterIdx].m_pParentInDecomposition = 
			& TreeLayout;
		TreeLayout.m_SubClusters [CurrClusterIdx].m_StartIdx = Node.m_Idx;
		TreeLayout.m_SubClusters [CurrClusterIdx].m_BB [0] = Node.m_BB[0];
		TreeLayout.m_SubClusters [CurrClusterIdx].m_BB [1] = Node.m_BB[1];

		Vector3 Center = (Node.m_BB[0] + Node.m_BB[1])/2.f;
		CONVERT_VECTOR3_VEC3F(Center,TreeLayout.m_SubClusters [CurrClusterIdx].m_Center);

		CurrClusterIdx++;
	}

	TreeLayout.m_NumSubClusters = CurrClusterIdx;
	if (TreeLayout.m_NumSubClusters > 500)
		printf ("%d clusters created.\n", CurrClusterIdx);


	if (CurrClusterIdx > MaxNumClusters)
		printf ("Error: Num of clusters (%d) is bigger than the maximum one (%d)\n", 
			CurrClusterIdx, MaxNumClusters);


	// Compute Cache-Coherent Layout between front nodes ------
	// since we merge several clusters into one block, there should be good spatial locality
	// between successive clusters
	// Here we use cache-coherent method since we do not know any cache-aware method.

	// Compute weight between clusters as a probability to access another cluster.
	// Since we do not compute a probability between clusters if there is no overlap, we only
	// consider clusters whose bonding boxes have some overlap.

	// Note: but, now, we just use minimum distance between cluster as a weight for the simple 
	//			implementation.
	//		--> just use KNN.

	//T1.Stop ();
	//cout << T1 << endl;

	

	int NumNSubClusters = CurrClusterIdx - 1;	// exclude root cluster
    YOON::CANN Ann;

	// add nodes into KNN structure
    Ann.SetNumVertices (NumNSubClusters);
    Vec3f Sum;
	int i;
    for (i = 0;i < NumNSubClusters;i++)
    {
		Ann.SetGeomVertex (i, TreeLayout.m_SubClusters [i + 1].m_Center);
    }

	// create a graph with a KNN
	OpenCCL::CLayoutGraph Graph (NumNSubClusters);

	//Stopwatch T3 ("Ann");
	//T3.Start ();

	int Neighbors [50], NumNeighbors;
	for (i = 0;i < NumNSubClusters;i++)
	{
		Vec3f Center = TreeLayout.m_SubClusters [i + 1].m_Center;
		NumNeighbors = Ann.GetApproximateNNs (Center, Neighbors);

		int k;
        for (k = 0;k < NumNeighbors;k++)
			Graph.AddEdge (i, Neighbors [k]);
	}
	//T3.Stop ();
	//cout << T3 << endl;

	//Stopwatch T2 ("CO layout");
	//T2.Start ();

	Graph.ComputeOrdering (TreeLayout.m_OrderSubClusters);

	//T2.Stop ();
	//cout << T2 << endl;



	// recursively compute ordering
	ComputeLayout_CO_CONTI (TreeLayout.m_SubClusters [0]);
	TreeLayout.m_SubClusters [0].m_ContainedNodes.clear ();

	for (i = 0;i < NumNSubClusters;i++) 
	{
		TreeLayout.m_OrderSubClusters [i]++;		// since root clulster is 0
													// so, all the child has index starting from 1

		int SubClusterIdx = TreeLayout.m_OrderSubClusters [i];


		// compute nodes that belong to the root cluster
		ComputeContainedNodes (TreeLayout.m_SubClusters [SubClusterIdx]);
		ComputeLayout_CO_CONTI (TreeLayout.m_SubClusters [SubClusterIdx]);
		TreeLayout.m_SubClusters [SubClusterIdx].m_ContainedNodes.clear ();
	}

	TreeLayout.ReleaseMemory ();

	return true;
}


// for each node in the front, we store its two children.
// So, its probability is to have collision given its parent collision.
// Node of StartIdx is already stored in the parent cluster.
// we layout nodes from the childs of the StartIdx.
bool OoCTreeAccess::ComputeLayout_CO (CTreeCluster & TreeLayout)
{
	bool bContained;
	unsigned int StartIdx = TreeLayout.m_StartIdx;

	if (TreeLayout.m_NumNodes == 0)
		return true;

	if (TreeLayout.m_NumNodes <= 1) {
		int NumAdded = 0;

		bContained = TreeLayout.IsBelongToParentDecomposedCluster (StartIdx);
		if (bContained) {
			m_pLayoutWrite->Add (StartIdx);
			NumAdded++;
			m_pProg->step ();
		}
		return true;
	}


	//Stopwatch T1 ("collecting root cluster");
	//T1.Start ();
	

	// ----------------------------------------
	// Decomposition the contained tree.	
	// ----------------------------------------
	OOC_NodeLayout_FILECLASS<CNodeLayout> & NodeLayout = * m_NodeLayout;
	int NumNodes = TreeLayout.m_NumNodes;
	int NumNodesInCluster = GetAvgNodesInCluster (NumNodes);
	int MaxNumClusters = (NumNodesInCluster + 2 + 20) * 2;

	// create with the maximum # of clusters
	TreeLayout.SetMaxNumClusters (MaxNumClusters);	

	// compute nodes that belong to the root cluster
	CTreeCluster & RootSubClusters = TreeLayout.m_SubClusters [0];

	// Step1: compute root cluster.
	// store nodes from the local maximum.
	priority_queue <CSortedNodeLayout> LocalHeap;
	CSortedNodeLayout Root (NodeLayout, & RootSubClusters.m_ContainedNodes, StartIdx,
		TreeLayout.m_BB[0], TreeLayout.m_BB[1], true);	// this node is under the current root cluster


	LocalHeap.push (Root);		// add root
	int NumAddedNodes = 1;

	while (!LocalHeap.empty () && (NumAddedNodes < NumNodesInCluster)) {
		CSortedNodeLayout Node = LocalHeap.top ();
		LocalHeap.pop ();

		// we store two child together since we access them together
		// put two children				
		#if HIERARCHY_TYPE == TYPE_BVH
		BSPArrayTreeNodePtr pCurNode = GETNODE(Node.m_Idx);
		#else
		BSPArrayTreeNodePtr pCurNode = GETNODE(Node.m_Idx << 1);
		#endif
	
		if (ISNOLEAF(pCurNode)) {	
			Vector3 Min = Node.m_BB [0];
			Vector3 Max = Node.m_BB [1];

			unsigned int LeftIdx = GETLEFTCHILD(pCurNode);
			unsigned int RightIdx = GETRIGHTCHILD(pCurNode);

			#if HIERARCHY_TYPE == TYPE_BVH
			#else
			LeftIdx = LeftIdx >> 1;
			RightIdx = RightIdx >> 1;
			#endif

			Vector3 LBB [2], RBB [2];
			#if HIERARCHY_TYPE == TYPE_KD_TREE
			LBB [0] = Min; LBB [1] = Max;
			RBB [0] = Min; RBB [1] = Max;

			int currentAxis = AXIS (pCurNode);

			LBB [1].e [currentAxis] = pCurNode->splitcoord;
			RBB [0].e [currentAxis] = pCurNode->splitcoord;
			#endif
			#if HIERARCHY_TYPE == TYPE_BVH
			BSPArrayTreeNodePtr lChild = GETNODE(LeftIdx);
			BSPArrayTreeNodePtr rChild = GETNODE(RightIdx);
			LBB [0] = lChild->min; LBB [1] = lChild->max;
			RBB [0] = rChild->min; RBB [1] = rChild->max;
			#endif

			bContained = TreeLayout.IsBelongToParentDecomposedCluster (LeftIdx);	
			if (bContained) {
				CSortedNodeLayout Left (NodeLayout, 
					&RootSubClusters.m_ContainedNodes, LeftIdx,
					LBB[0], LBB[1]);
				LocalHeap.push (Left);
				NumAddedNodes++;
			}
			bContained = TreeLayout.IsBelongToParentDecomposedCluster (RightIdx);
			
			if (bContained) {
				CSortedNodeLayout Right (NodeLayout,
					&RootSubClusters.m_ContainedNodes, RightIdx,
					RBB[0], RBB[1]);
				LocalHeap.push (Right);
				NumAddedNodes++;
			}
			
		}
	}




	// prepare root cluster of the decomposed clusters
	TreeLayout.m_SubClusters [0].m_pParentInDecomposition = & TreeLayout;
	TreeLayout.m_SubClusters [0].m_StartIdx = StartIdx;
	TreeLayout.m_SubClusters [0].m_NumNodes = NumAddedNodes;
	TreeLayout.m_SubClusters [0].m_BB [0] = TreeLayout.m_BB [0];
	TreeLayout.m_SubClusters [0].m_BB [1] = TreeLayout.m_BB [1];

	assert (
		TreeLayout.m_SubClusters [0].m_ContainedNodes.size () == TreeLayout.m_SubClusters [0].m_NumNodes &&
		TreeLayout.m_SubClusters [0].m_NumNodes != 0);
		
	
	// Step2: compute child clusters.
	//			each node in LocalHeap was already stored in the root cluster
	//			two child nodes of the node will be root node of the child cluster

	// make node in the queue as a root of the child cluster

	VArray <Vec3f> CenterArr;

	int CurrClusterIdx = 1;
	while (!LocalHeap.empty ()) {
		CSortedNodeLayout Node = LocalHeap.top ();
		LocalHeap.pop ();

		#if HIERARCHY_TYPE == TYPE_BVH
		BSPArrayTreeNodePtr pCurNode = GETNODE(Node.m_Idx);
		#else
		BSPArrayTreeNodePtr pCurNode = GETNODE(Node.m_Idx << 1);
		#endif
		if (ISNOLEAF(pCurNode)) {	
			unsigned int LeftIdx = GETLEFTCHILD(pCurNode);
			unsigned int RightIdx = GETRIGHTCHILD(pCurNode);

			#if HIERARCHY_TYPE == TYPE_BVH
			#else
			LeftIdx = LeftIdx >> 1;
			RightIdx = RightIdx >> 1;
			#endif

			Vector3 Min = Node.m_BB [0];
			Vector3 Max = Node.m_BB [1];

			Vector3 LBB [2], RBB [2];
			#if HIERARCHY_TYPE == TYPE_KD_TREE
			LBB [0] = Min; LBB [1] = Max;
			RBB [0] = Min; RBB [1] = Max;

			int currentAxis = AXIS (pCurNode);

			LBB [1].e [currentAxis] = pCurNode->splitcoord;
			RBB [0].e [currentAxis] = pCurNode->splitcoord;
			#endif
			#if HIERARCHY_TYPE == TYPE_BVH
			BSPArrayTreeNodePtr lChild = GETNODE(LeftIdx);
			BSPArrayTreeNodePtr rChild = GETNODE(RightIdx);
			LBB [0] = lChild->min; LBB [1] = lChild->max;
			RBB [0] = rChild->min; RBB [1] = rChild->max;
			#endif

			TreeLayout.m_SubClusters [CurrClusterIdx].m_pParentInDecomposition = 
				& TreeLayout;
			TreeLayout.m_SubClusters [CurrClusterIdx].m_StartIdx = LeftIdx;
			TreeLayout.m_SubClusters [CurrClusterIdx].m_BB [0] = LBB[0];
			TreeLayout.m_SubClusters [CurrClusterIdx].m_BB [1] = LBB[1];

			Vector3 Center = (LBB[0] + LBB[1])/2.f;
			CONVERT_VECTOR3_VEC3F(Center,TreeLayout.m_SubClusters [CurrClusterIdx].m_Center);
			CurrClusterIdx++;

			TreeLayout.m_SubClusters [CurrClusterIdx].m_pParentInDecomposition = 
				& TreeLayout;
			TreeLayout.m_SubClusters [CurrClusterIdx].m_StartIdx = RightIdx;
			TreeLayout.m_SubClusters [CurrClusterIdx].m_BB [0] = RBB[0];
			TreeLayout.m_SubClusters [CurrClusterIdx].m_BB [1] = RBB[1];

			Center = (RBB[0] + RBB[1])/2.f;
			CONVERT_VECTOR3_VEC3F(Center,TreeLayout.m_SubClusters [CurrClusterIdx].m_Center);

			CurrClusterIdx++;


			Vector3 ParentCenter;
			ParentCenter = (Min + Max)/2.f;

			Vec3f _ParentC;
			CONVERT_VECTOR3_VEC3F(ParentCenter,_ParentC);
			CenterArr.Append (_ParentC);	
			

		}
	}

	TreeLayout.m_NumSubClusters = CurrClusterIdx;
	if (TreeLayout.m_NumSubClusters > 500)
		printf ("%d clusters created.\n", CurrClusterIdx);


	if (CurrClusterIdx > MaxNumClusters)
		printf ("Error: Num of clusters (%d) is bigger than the maximum one (%d)\n", 
			CurrClusterIdx, MaxNumClusters);


	// Compute Cache-Coherent Layout between front nodes ------
	// since we merge several clusters into one block, there should be good spatial locality
	// between successive clusters
	// Here we use cache-coherent method since we do not know any cache-aware method.

	// Compute weight between clusters as a probability to access another cluster.
	// Since we do not compute a probability between clusters if there is no overlap, we only
	// consider clusters whose bonding boxes have some overlap.

	// Note: but, now, we just use minimum distance between cluster as a weight for the simple 
	//			implementation.
	//		--> just use KNN.

	//T1.Stop ();
	//cout << T1 << endl;

	

	int NumNSubClusters = CurrClusterIdx - 1;	// exclude root cluster
	NumNSubClusters /= 2;		// since we order parent clusters for left and right cluster
    YOON::CANN Ann;

	// add nodes into KNN structure
    Ann.SetNumVertices (NumNSubClusters);
    Vec3f Sum;
	int i;
    for (i = 0;i < NumNSubClusters;i++)
    {
		//Ann.SetGeomVertex (i, TreeLayout.m_SubClusters [i + 1].m_Center);
		Ann.SetGeomVertex (i, CenterArr [i]);
    }

	// create a graph with a KNN
	OpenCCL::CLayoutGraph Graph (NumNSubClusters);

	//Stopwatch T3 ("Ann");
	//T3.Start ();

	int Neighbors [50], NumNeighbors;
	for (i = 0;i < NumNSubClusters;i++)
	{
		//Vec3f Center = TreeLayout.m_SubClusters [i + 1].m_Center;
		Vec3f Center = CenterArr [i];
		NumNeighbors = Ann.GetApproximateNNs (Center, Neighbors);

		int k;
        for (k = 0;k < NumNeighbors;k++)
			Graph.AddEdge (i, Neighbors [k]);
	}
	//T3.Stop ();
	//cout << T3 << endl;

	//Stopwatch T2 ("CO layout");
	//T2.Start ();

	Graph.ComputeOrdering (TreeLayout.m_OrderSubClusters);

	//T2.Stop ();
	//cout << T2 << endl;



	// recursively compute ordering
	ComputeLayout_CO (TreeLayout.m_SubClusters [0]);
	TreeLayout.m_SubClusters [0].m_ContainedNodes.clear ();

	for (i = 0;i < NumNSubClusters;i++) 
	{
		//TreeLayout.m_OrderSubClusters [i]++;		// since root clulster is 0
													// so, all the child has index starting from 1

		int SubClusterIdx = TreeLayout.m_OrderSubClusters [i];
		SubClusterIdx *= 2;		
		SubClusterIdx++;		// 1 to skip root cluster

		// compute nodes that belong to the root cluster	:: left cluster
		ComputeContainedNodes (TreeLayout.m_SubClusters [SubClusterIdx], true);
		ComputeLayout_CO (TreeLayout.m_SubClusters [SubClusterIdx]);
		TreeLayout.m_SubClusters [SubClusterIdx].m_ContainedNodes.clear ();

		// compute nodes that belong to the root cluster	::  right clcuster
		ComputeContainedNodes (TreeLayout.m_SubClusters [SubClusterIdx + 1], true);
		ComputeLayout_CO (TreeLayout.m_SubClusters [SubClusterIdx + 1]);
		TreeLayout.m_SubClusters [SubClusterIdx + 1].m_ContainedNodes.clear ();

	}

	TreeLayout.ReleaseMemory ();

	return true;
}

// for each node in the front, we store its two children.
// So, its probability is to have collision given its parent collision.
// Node of StartIdx is already stored in the parent cluster.
// we layout nodes from the childs of the StartIdx.
bool OoCTreeAccess::ComputeLayout_CO_PreserveOrder (CTreeCluster & TreeLayout)
{
	bool bContained;
	unsigned int StartIdx = TreeLayout.m_StartIdx;

	if (TreeLayout.m_NumNodes == 0)
		return true;

	if (TreeLayout.m_NumNodes <= 1) {
		int NumAdded = 0;

		bContained = TreeLayout.IsBelongToParentDecomposedCluster (StartIdx);
		if (bContained) {
			m_pLayoutWrite->Add (StartIdx);
			NumAdded++;
			m_pProg->step ();
		}
		return true;
	}


	//Stopwatch T1 ("collecting root cluster");
	//T1.Start ();
	

	// ----------------------------------------
	// Decomposition the contained tree.	
	// ----------------------------------------
	OOC_NodeLayout_FILECLASS<CNodeLayout> & NodeLayout = * m_NodeLayout;
	int NumNodes = TreeLayout.m_NumNodes;
	int NumNodesInCluster = GetAvgNodesInCluster (NumNodes);
	int MaxNumClusters = (NumNodesInCluster + 2 + 20) * 2;

	// create with the maximum # of clusters
	TreeLayout.SetMaxNumClusters (MaxNumClusters);	

	// compute nodes that belong to the root cluster
	CTreeCluster & RootSubClusters = TreeLayout.m_SubClusters [0];

	// Step1: compute root cluster.
	// store nodes from the local maximum.
	priority_queue <CSortedNodeLayout> LocalHeap;

	// init Cache-oblivious layout maintaining tree structure
	CSortedNodeLayout * BVs = new CSortedNodeLayout [MaxNumClusters * 2];
	int IdxBV = 0;

	//CIntHashMap Map;	// BVIdx -> BVs's pointer
	CActiveList <CSortedNodeLayout *> COLayout;
	CSortedNodeLayout * pStart, * pEnd;
	pStart = new CSortedNodeLayout;
	pEnd = new CSortedNodeLayout;
	COLayout.InitList (pStart, pEnd);


	//CSortedNodeLayout Root (NodeLayout, & RootSubClusters.m_ContainedNodes, StartIdx,
	BVs [0].Set (NodeLayout, & RootSubClusters.m_ContainedNodes, StartIdx,
		TreeLayout.m_BB[0], TreeLayout.m_BB[1], true);	// this node is under the current root cluster
	IdxBV++;

	//LocalHeap.push (Root);		// add root
	CSortedNodeLayout ForSort; 
	ForSort.m_Idx = IdxBV - 1; ForSort.m_AccProb = BVs [IdxBV -1 ].m_AccProb;

	LocalHeap.push (ForSort);		// add root
	//NumAddedNodes++;		// root was included

	COLayout.AddatEnd (& BVs [IdxBV - 1]);



	int NumAddedNodes = 1;

	while (!LocalHeap.empty () && (NumAddedNodes < NumNodesInCluster)) {
		CSortedNodeLayout _Node = LocalHeap.top ();
		LocalHeap.pop ();

		int BVIdx = _Node.m_Idx;
		CSortedNodeLayout & Node = BVs [BVIdx];

		assert (& Node == & BVs [BVIdx]);


		// we store two child together since we access them together
		// put two children				
		#if HIERARCHY_TYPE == TYPE_BVH
		BSPArrayTreeNodePtr pCurNode = GETNODE(Node.m_Idx);
		#else
		BSPArrayTreeNodePtr pCurNode = GETNODE(Node.m_Idx << 1);
		#endif
	
		if (ISNOLEAF(pCurNode)) {	
			Vector3 Min = Node.m_BB [0];
			Vector3 Max = Node.m_BB [1];

			unsigned int LeftIdx = GETLEFTCHILD(pCurNode);
			unsigned int RightIdx = GETRIGHTCHILD(pCurNode);

			#if HIERARCHY_TYPE == TYPE_BVH
			#else
			LeftIdx = LeftIdx >> 1;
			RightIdx = RightIdx >> 1;
			#endif

			Vector3 LBB [2], RBB [2];
			#if HIERARCHY_TYPE == TYPE_KD_TREE
			LBB [0] = Min; LBB [1] = Max;
			RBB [0] = Min; RBB [1] = Max;

			int currentAxis = AXIS (pCurNode);

			LBB [1].e [currentAxis] = pCurNode->splitcoord;
			RBB [0].e [currentAxis] = pCurNode->splitcoord;
			#endif
			#if HIERARCHY_TYPE == TYPE_BVH
			BSPArrayTreeNodePtr lChild = GETNODE(LeftIdx);
			BSPArrayTreeNodePtr rChild = GETNODE(RightIdx);
			LBB [0] = lChild->min; LBB [1] = lChild->max;
			RBB [0] = rChild->min; RBB [1] = rChild->max;
			#endif

			bContained = TreeLayout.IsBelongToParentDecomposedCluster (RightIdx);
			
			if (bContained) {
				//CSortedNodeLayout Right (NodeLayout,
				//	&RootSubClusters.m_ContainedNodes, RightIdx,
				//	RBB[0], RBB[1]);
				//LocalHeap.push (Right);
				//NumAddedNodes++;
				BVs [IdxBV++].Set (NodeLayout, 
					&RootSubClusters.m_ContainedNodes, RightIdx,
					RBB[0], RBB[1]);
				CSortedNodeLayout ForSort; 
				ForSort.m_Idx = IdxBV - 1; ForSort.m_AccProb = BVs [IdxBV -1 ].m_AccProb;

				LocalHeap.push (ForSort);
				NumAddedNodes++;

				COLayout.AddNext (& Node, & BVs [IdxBV - 1]);


			}

			bContained = TreeLayout.IsBelongToParentDecomposedCluster (LeftIdx);	
			if (bContained) {
				//CSortedNodeLayout Left (NodeLayout, 
				//	&RootSubClusters.m_ContainedNodes, LeftIdx,
				//	LBB[0], LBB[1]);
				//LocalHeap.push (Left);
				//NumAddedNodes++;

				BVs [IdxBV++].Set (NodeLayout, 
					&RootSubClusters.m_ContainedNodes, LeftIdx,
					LBB[0], LBB[1]);
				CSortedNodeLayout ForSort; 
				ForSort.m_Idx = IdxBV - 1; ForSort.m_AccProb = BVs [IdxBV -1 ].m_AccProb;

				LocalHeap.push (ForSort);
				NumAddedNodes++;

				COLayout.AddNext (& Node, & BVs [IdxBV - 1]);

			}


			
		}

		// delete the Node in the list
		COLayout.Delete (& Node);
			

	}


	if (IdxBV > MaxNumClusters) {
		printf ("Too small reserved BVs. %d %d\n", IdxBV, MaxNumClusters);
		exit (-1);
	}



	// prepare root cluster of the decomposed clusters
	TreeLayout.m_SubClusters [0].m_pParentInDecomposition = & TreeLayout;
	TreeLayout.m_SubClusters [0].m_StartIdx = StartIdx;
	TreeLayout.m_SubClusters [0].m_NumNodes = NumAddedNodes;
	TreeLayout.m_SubClusters [0].m_BB [0] = TreeLayout.m_BB [0];
	TreeLayout.m_SubClusters [0].m_BB [1] = TreeLayout.m_BB [1];

	assert (
		TreeLayout.m_SubClusters [0].m_ContainedNodes.size () == TreeLayout.m_SubClusters [0].m_NumNodes &&
		TreeLayout.m_SubClusters [0].m_NumNodes != 0);
		
	
	// Step2: compute child clusters.
	//			each node in LocalHeap was already stored in the root cluster
	//			two child nodes of the node will be root node of the child cluster

	// make node in the queue as a root of the child cluster


	VArray <Vec3f> CenterArr;

	int CurrClusterIdx = 1;
	assert (COLayout.Size () == LocalHeap.size ());

	COLayout.InitIteration ();
	while (! COLayout.IsEnd ()) {
	//while (!LocalHeap.empty ()) {
	//	CSortedNodeLayout Node = LocalHeap.top ();
	//	LocalHeap.pop ();
		CSortedNodeLayout & Node = * COLayout.GetCurrent ();

		#if HIERARCHY_TYPE == TYPE_BVH
		BSPArrayTreeNodePtr pCurNode = GETNODE(Node.m_Idx);
		#else
		BSPArrayTreeNodePtr pCurNode = GETNODE(Node.m_Idx << 1);
		#endif
		if (ISNOLEAF(pCurNode)) {	
			unsigned int LeftIdx = GETLEFTCHILD(pCurNode);
			unsigned int RightIdx = GETRIGHTCHILD(pCurNode);

			#if HIERARCHY_TYPE == TYPE_BVH
			#else
			LeftIdx = LeftIdx >> 1;
			RightIdx = RightIdx >> 1;
			#endif

			Vector3 Min = Node.m_BB [0];
			Vector3 Max = Node.m_BB [1];

			Vector3 LBB [2], RBB [2];
			#if HIERARCHY_TYPE == TYPE_KD_TREE
			LBB [0] = Min; LBB [1] = Max;
			RBB [0] = Min; RBB [1] = Max;

			int currentAxis = AXIS (pCurNode);

			LBB [1].e [currentAxis] = pCurNode->splitcoord;
			RBB [0].e [currentAxis] = pCurNode->splitcoord;
			#endif
			#if HIERARCHY_TYPE == TYPE_BVH
			BSPArrayTreeNodePtr lChild = GETNODE(LeftIdx);
			BSPArrayTreeNodePtr rChild = GETNODE(RightIdx);
			LBB [0] = lChild->min; LBB [1] = lChild->max;
			RBB [0] = rChild->min; RBB [1] = rChild->max;
			#endif

			TreeLayout.m_SubClusters [CurrClusterIdx].m_pParentInDecomposition = 
				& TreeLayout;
			TreeLayout.m_SubClusters [CurrClusterIdx].m_StartIdx = LeftIdx;
			TreeLayout.m_SubClusters [CurrClusterIdx].m_BB [0] = LBB[0];
			TreeLayout.m_SubClusters [CurrClusterIdx].m_BB [1] = LBB[1];

			Vector3 Center = (LBB[0] + LBB[1])/2.f;
			CONVERT_VECTOR3_VEC3F(Center,TreeLayout.m_SubClusters [CurrClusterIdx].m_Center);
			CurrClusterIdx++;

			TreeLayout.m_SubClusters [CurrClusterIdx].m_pParentInDecomposition = 
				& TreeLayout;
			TreeLayout.m_SubClusters [CurrClusterIdx].m_StartIdx = RightIdx;
			TreeLayout.m_SubClusters [CurrClusterIdx].m_BB [0] = RBB[0];
			TreeLayout.m_SubClusters [CurrClusterIdx].m_BB [1] = RBB[1];

			Center = (RBB[0] + RBB[1])/2.f;
			CONVERT_VECTOR3_VEC3F(Center,TreeLayout.m_SubClusters [CurrClusterIdx].m_Center);

			CurrClusterIdx++;


			Vector3 ParentCenter;
			ParentCenter = (Min + Max)/2.f;

			Vec3f _ParentC;
			CONVERT_VECTOR3_VEC3F(ParentCenter,_ParentC);
			CenterArr.Append (_ParentC);	
		
		}
		COLayout.Advance ();


	}

	delete [] BVs;


	TreeLayout.m_NumSubClusters = CurrClusterIdx;
	if (TreeLayout.m_NumSubClusters > 500)
		printf ("%d clusters created.\n", CurrClusterIdx);


	if (CurrClusterIdx > MaxNumClusters)
		printf ("Error: Num of clusters (%d) is bigger than the maximum one (%d)\n", 
			CurrClusterIdx, MaxNumClusters);


	// Compute Cache-Coherent Layout between front nodes ------
	// since we merge several clusters into one block, there should be good spatial locality
	// between successive clusters
	// Here we use cache-coherent method since we do not know any cache-aware method.

	// Compute weight between clusters as a probability to access another cluster.
	// Since we do not compute a probability between clusters if there is no overlap, we only
	// consider clusters whose bonding boxes have some overlap.

	// Note: but, now, we just use minimum distance between cluster as a weight for the simple 
	//			implementation.
	//		--> just use KNN.

	//T1.Stop ();
	//cout << T1 << endl;

	
	

	int NumNSubClusters = CurrClusterIdx - 1;	// exclude root cluster

	/*

	NumNSubClusters /= 2;		// since we order parent clusters for left and right cluster
    YOON::CANN Ann;

	// add nodes into KNN structure
    Ann.SetNumVertices (NumNSubClusters);
    Vec3f Sum;
	int i;
    for (i = 0;i < NumNSubClusters;i++)
    {
		//Ann.SetGeomVertex (i, TreeLayout.m_SubClusters [i + 1].m_Center);
		Ann.SetGeomVertex (i, CenterArr [i]);
    }

	// create a graph with a KNN
	OpenCCL::CLayoutGraph Graph (NumNSubClusters);

	//Stopwatch T3 ("Ann");
	//T3.Start ();

	int Neighbors [50], NumNeighbors;
	for (i = 0;i < NumNSubClusters;i++)
	{
		//Vec3f Center = TreeLayout.m_SubClusters [i + 1].m_Center;
		Vec3f Center = CenterArr [i];
		NumNeighbors = Ann.GetApproximateNNs (Center, Neighbors);

		int k;
        for (k = 0;k < NumNeighbors;k++)
			Graph.AddEdge (i, Neighbors [k]);
	}
	//T3.Stop ();
	//cout << T3 << endl;

	//Stopwatch T2 ("CO layout");
	//T2.Start ();

	Graph.ComputeOrdering (TreeLayout.m_OrderSubClusters);

	//T2.Stop ();
	//cout << T2 << endl;

	*/

	// recursively compute ordering
	ComputeLayout_CO (TreeLayout.m_SubClusters [0]);
	TreeLayout.m_SubClusters [0].m_ContainedNodes.clear ();
	
	
	int i;
	for (i = 0;i < NumNSubClusters;i++) 
	{
		//TreeLayout.m_OrderSubClusters [i]++;		// since root clulster is 0
													// so, all the child has index starting from 1

	//	int SubClusterIdx = TreeLayout.m_OrderSubClusters [i];
		int SubClusterIdx = i + 1;
		//SubClusterIdx *= 2;		
		//SubClusterIdx++;		// 1 to skip root cluster

		// compute nodes that belong to the root cluster	:: left cluster
		ComputeContainedNodes (TreeLayout.m_SubClusters [SubClusterIdx], true);
		ComputeLayout_CO (TreeLayout.m_SubClusters [SubClusterIdx]);
		TreeLayout.m_SubClusters [SubClusterIdx].m_ContainedNodes.clear ();

		/*
		// compute nodes that belong to the root cluster	::  right clcuster
		ComputeContainedNodes (TreeLayout.m_SubClusters [SubClusterIdx + 1], true);
		ComputeLayout_CO (TreeLayout.m_SubClusters [SubClusterIdx + 1]);
		TreeLayout.m_SubClusters [SubClusterIdx + 1].m_ContainedNodes.clear ();
		*/
	}

	TreeLayout.ReleaseMemory ();

	return true;
}


// compute probabiity that a current node is collided given that its parent node is collided.
// since mm_map is remap, data can be lost. So, before writing new info, get reference.
// Given a prob. of a node that is in priority_queue, if we choose it,
// we store its two child nodes
// Since we access the node, it's likely to access two children, we store it.
// Actually, in ray tracing, we store right node right next to the left node.
// So, it's the case.
void OoCTreeAccess::ComputeAccessProb (unsigned int ParentIdx, unsigned int CurIdx, 
								Vector3 & BBMin, Vector3 & BBMax,	// BB of the parent node
								Vector3 & Min, Vector3 & Max)		// BB of the current node
{
	OOC_NodeLayout_FILECLASS<CNodeLayout> & NodeLayout = * m_NodeLayout;
	BSPArrayTreeNode curNode = *GETNODE(CurIdx);
	BSPArrayTreeNodePtr pCurNode = &curNode;
	int i;
	Vector3 Diff [2];
	float Surfaces [2];
	
	Diff [0] = BBMax - BBMin;
	Diff [1] = Max - Min;
	
	for (i = 0;i < 2;i++)
		Surfaces [i] = Diff[i].x()*Diff[i].y() + Diff[i].x()*Diff[i].z() + Diff[i].y()*Diff[i].z();


	// assign graph parent and parent.
	if (ParentIdx == NULL_IDX) {
		CNodeLayout & CurBV = NodeLayout.GetRef (CurIdx);
		CurBV.m_AccProb = 1;
		CurBV.m_ParentIdx = NULL_IDX;
	}
	else {
		// Since each node is fully contained in the parent node, we can simply compute volume
		// by computing sum of its surface area.	
		float Probability = float (Surfaces [1]) / float (Surfaces [0]);
		//assert (Probability <= 1.);

		CNodeLayout ParentBV = NodeLayout [ParentIdx];
		
		CNodeLayout & CurBV = NodeLayout.GetRef (CurIdx);
		CurBV.m_AccProb = ParentBV.m_AccProb * Probability;
		CurBV.m_ParentIdx = NULL_IDX;
	}

	m_pProg->step ();

	if (ISNOLEAF(pCurNode)) {	

		assert (!ISLEAF(pCurNode));
		unsigned int child_left = GETLEFTCHILD(pCurNode);
		unsigned int child_right = GETRIGHTCHILD(pCurNode);

		

		// compute BB of two child nodes
		Vector3 LBB [2], RBB [2];
		#if HIERARCHY_TYPE == TYPE_KD_TREE
		LBB [0] = Min; LBB [1] = Max;
		RBB [0] = Min; RBB [1] = Max;

		int currentAxis = AXIS (pCurNode);

		LBB [1].e [currentAxis] = pCurNode->splitcoord;
		RBB [0].e [currentAxis] = pCurNode->splitcoord;
		#endif
		#if HIERARCHY_TYPE == TYPE_BVH
		BSPArrayTreeNodePtr lChild = GETNODE(child_left);
		BSPArrayTreeNodePtr rChild = GETNODE(child_right);
		LBB [0] = lChild->min; LBB [1] = lChild->max;
		RBB [0] = rChild->min; RBB [1] = rChild->max;
		#endif

		ComputeAccessProb (CurIdx, child_left, Min, Max, LBB [0], LBB [1]);
		ComputeAccessProb (CurIdx, child_right, Min, Max, RBB [0], RBB [1]);
	}
}

// count nodes from child nodes of StartIdx with the constraints of the paret cluster
void OoCTreeAccess::ComputeContainedNodes (CTreeCluster & TreeLayout, bool RootContained)
{
	
	int NumNodes = 0;
	bool bContained;

	CIntHashMap & ContainedNodes = TreeLayout.m_ContainedNodes;

	int StartIdx = TreeLayout.m_StartIdx;
	stack <int> Stack;
	Stack.push (StartIdx);
	
	if (RootContained == true) {
		bContained = TreeLayout.IsBelongToParentDecomposedCluster (StartIdx);
		if (bContained) {
			
			CIntHashMap::iterator Iter = ContainedNodes.find (StartIdx);
			if (Iter == ContainedNodes.end ()) {	
				CIntHashMap::value_type NewElement1 (StartIdx, 1);
				ContainedNodes.insert (NewElement1);
			}
			else {
				printf ("Data duplication in hash map for layout computation 1\n");
				exit (-1);
			}

			NumNodes++;

		}
	}

	
	while (! Stack.empty ()) {
		int CurIdx = Stack.top ();
		Stack.pop ();

		#if HIERARCHY_TYPE == TYPE_BVH
		BSPArrayTreeNodePtr pCurNode = GETNODE (CurIdx);
		#else
		BSPArrayTreeNodePtr pCurNode = GETNODE (CurIdx << 1);
		#endif
		if (ISNOLEAF(pCurNode)) {	
			unsigned int LeftIdx = GETLEFTCHILD(pCurNode);
			unsigned int RightIdx = GETRIGHTCHILD(pCurNode);

			#if HIERARCHY_TYPE == TYPE_BVH
			#else
			LeftIdx = LeftIdx >> 1;
			RightIdx = RightIdx >> 1;
			#endif

			bContained = TreeLayout.IsBelongToParentDecomposedCluster (RightIdx);
			if (bContained) {
				Stack.push (RightIdx);
				
				CIntHashMap::iterator Iter = ContainedNodes.find (RightIdx);
				if (Iter == ContainedNodes.end ()) {	
					CIntHashMap::value_type NewElement1 (RightIdx, 1);
					ContainedNodes.insert (NewElement1);
				}
				else {
					printf ("Data duplication in hash map for layout computation 2\n");
					exit (-1);
				}

				NumNodes++;
			}

			bContained = TreeLayout.IsBelongToParentDecomposedCluster (LeftIdx);
			if (bContained) {
				Stack.push (LeftIdx);

				CIntHashMap::iterator Iter = ContainedNodes.find (LeftIdx);
				if (Iter == ContainedNodes.end ()) {	
					CIntHashMap::value_type NewElement1 (LeftIdx, 1);
					ContainedNodes.insert (NewElement1);
				}
				else {
					printf ("Data duplication in hash map for layout computation 3\n");
					exit (-1);
				}

				NumNodes++;
			}
			

		}	// end "ISNO_LEAF"

	}	// end of while

	TreeLayout.m_NumNodes = NumNodes;

	assert (TreeLayout.m_ContainedNodes.size () == TreeLayout.m_NumNodes);

}


// for each node in the front, we store its two children.
// Node of StartIdx is already stored in the parent cluster.
// we layout nodes from the childs of the StartIdx.
bool OoCTreeAccess::ComputeLayout_VEB (
		unsigned int StartIdx, int MaxDepth, VArray <int> & NextFront)
{	
	if (MaxDepth == 1) {
		

		#ifdef FOUR_BYTE_FOR_KD_NODE
			BSPArrayTreeNodePtr pCurNode = GETNODE(StartIdx);

			if (ISNOLEAF(pCurNode)) {	
				unsigned int LeftIdx = GETLEFTCHILD(pCurNode);
				unsigned int RightIdx = GETRIGHTCHILD(pCurNode);

				m_pLayoutWrite->Add (LeftIdx);
				m_pLayoutWrite->Add (RightIdx);
				
				m_pProg->step ();
				m_pProg->step ();

				// compute next front nodes
				NextFront.Append (LeftIdx);
				NextFront.Append (RightIdx);
			}
		#else
			m_pLayoutWrite->Add (StartIdx);
			m_pProg->step ();

			#if HIERARCHY_TYPE == TYPE_BVH
			BSPArrayTreeNodePtr pCurNode = GETNODE((StartIdx));
			#else
			BSPArrayTreeNodePtr pCurNode = GETNODE((StartIdx << 1));
			#endif

			if (ISNOLEAF(pCurNode)) {	
				unsigned int LeftIdx = GETLEFTCHILD(pCurNode);
				unsigned int RightIdx = GETRIGHTCHILD(pCurNode);

				#if HIERARCHY_TYPE == TYPE_BVH
				#else
				LeftIdx = LeftIdx >> 1;
				RightIdx = RightIdx >> 1;
				#endif

				// compute next front nodes
				NextFront.Append (LeftIdx);
				NextFront.Append (RightIdx);
			}
		#endif

		return true;
	}

	// recursive divide the hierarchy
	int i, j, MidDepth = MaxDepth / 2;

	// hold fronts for next recursion
	VArray <int> NextFrontofTopCluster;	// maintain next front of top cluster

	// recursive divide the hierary and layout
	ComputeLayout_VEB (StartIdx, MidDepth, NextFrontofTopCluster);

	VArray <int> NextSubFront;
	for (i = 0;i < NextFrontofTopCluster.Size ();i++)
	{
		unsigned int StartIdx = NextFrontofTopCluster [i];
		NextSubFront.Clear (false);
		ComputeLayout_VEB (StartIdx, MaxDepth - MidDepth, NextSubFront);

		for (j = 0;j < NextSubFront.Size ();j++)
			NextFront.Append (NextSubFront [j]);
	}

	return true;
}

// for each node in the front, we store its two children.
// Node of StartIdx is already stored in the parent cluster.
// we layout nodes from the childs of the StartIdx.
bool OoCTreeAccess::ComputeLayout_VEB_CONTI (
		unsigned int StartIdx, int MaxDepth, VArray <int> & NextFront)
{	
	if (MaxDepth == 1) {
		

		#ifdef FOUR_BYTE_FOR_KD_NODE
			BSPArrayTreeNodePtr pCurNode = GETNODE(StartIdx);

			if (ISNOLEAF(pCurNode)) {	
				unsigned int LeftIdx = GETLEFTCHILD(pCurNode);
				unsigned int RightIdx = GETRIGHTCHILD(pCurNode);

				m_pLayoutWrite->Add (LeftIdx);
				m_pLayoutWrite->Add (RightIdx);
				
				m_pProg->step ();
				m_pProg->step ();

				// compute next front nodes
				NextFront.Append (LeftIdx);
				NextFront.Append (RightIdx);
			}
		#else
			
			m_pProg->step ();

			#if HIERARCHY_TYPE == TYPE_BVH
			BSPArrayTreeNodePtr pCurNode = GETNODE((StartIdx));
			#else
			BSPArrayTreeNodePtr pCurNode = GETNODE((StartIdx << 1));
			#endif

			if (ISNOLEAF(pCurNode)) {	
				unsigned int LeftIdx = GETLEFTCHILD(pCurNode);
				unsigned int RightIdx = GETRIGHTCHILD(pCurNode);

				#if HIERARCHY_TYPE == TYPE_BVH
				#else
				LeftIdx = LeftIdx >> 1;
				RightIdx = RightIdx >> 1;
				#endif

				m_pLayoutWrite->Add (LeftIdx);
				m_pLayoutWrite->Add (RightIdx);

				// compute next front nodes
				NextFront.Append (LeftIdx);
				NextFront.Append (RightIdx);
			}
		#endif

		return true;
	}

	// recursive divide the hierarchy
	int i, j, MidDepth = MaxDepth / 2;

	// hold fronts for next recursion
	VArray <int> NextFrontofTopCluster;	// maintain next front of top cluster

	// recursive divide the hierary and layout
	ComputeLayout_VEB_CONTI (StartIdx, MidDepth, NextFrontofTopCluster);

	VArray <int> NextSubFront;
	for (i = 0;i < NextFrontofTopCluster.Size ();i++)
	{
		unsigned int StartIdx = NextFrontofTopCluster [i];
		NextSubFront.Clear (false);
		ComputeLayout_VEB_CONTI (StartIdx, MaxDepth - MidDepth, NextSubFront);

		for (j = 0;j < NextSubFront.Size ();j++)
			NextFront.Append (NextSubFront [j]);
	}

	return true;
}

/*

// for each node in the front, we store its two children.
// Node of StartIdx is already stored in the parent cluster.
// we layout nodes from the childs of the StartIdx.
bool OoCTreeAccess::ComputeLayout_VEB_Contiguous (
		unsigned int StartIdx, int MaxDepth, VArray <int> & NextFront)
{	
	if (MaxDepth == 1) {
		

		#ifdef FOUR_BYTE_FOR_KD_NODE
			BSPArrayTreeNodePtr pCurNode = GETNODE(StartIdx);

			if (ISNOLEAF(pCurNode)) {	
				unsigned int LeftIdx = GETLEFTCHILD(pCurNode);
				unsigned int RightIdx = GETRIGHTCHILD(pCurNode);

				m_pLayoutWrite->Add (LeftIdx);
				m_pLayoutWrite->Add (RightIdx);
				
				m_pProg->step ();
				m_pProg->step ();

				// compute next front nodes
				NextFront.Append (LeftIdx);
				NextFront.Append (RightIdx);
			}
		#else
			m_pLayoutWrite->Add (StartIdx);
			m_pProg->step ();

			#if HIERARCHY_TYPE == TYPE_BVH
			BSPArrayTreeNodePtr pCurNode = GETNODE((StartIdx));
			#else
			BSPArrayTreeNodePtr pCurNode = GETNODE((StartIdx << 1));
			#endif

			if (ISNOLEAF(pCurNode)) {	
				unsigned int LeftIdx = GETLEFTCHILD(pCurNode);
				unsigned int RightIdx = GETRIGHTCHILD(pCurNode);

				#if HIERARCHY_TYPE == TYPE_BVH
				#else
				LeftIdx = LeftIdx >> 1;
				RightIdx = RightIdx >> 1;
				#endif
				// compute next front nodes
				NextFront.Append (LeftIdx);
				NextFront.Append (RightIdx);
			}
		#endif

		return true;
	}

	// recursive divide the hierarchy
	int i, j, MidDepth = MaxDepth / 2;

	// hold fronts for next recursion
	VArray <int> NextFrontofTopCluster;	// maintain next front of top cluster

	// recursive divide the hierary and layout
	ComputeLayout_VEB (StartIdx, MidDepth, NextFrontofTopCluster);

	VArray <int> NextSubFront;
	for (i = 0;i < NextFrontofTopCluster.Size ();i++)
	{
		unsigned int StartIdx = NextFrontofTopCluster [i];
		NextSubFront.Clear (false);
		ComputeLayout_VEB (StartIdx, MaxDepth - MidDepth, NextSubFront);

		for (j = 0;j < NextSubFront.Size ();j++)
			NextFront.Append (NextSubFront [j]);
	}

	return true;
}
*/

// this is incore-version
// Assume: root node, 0, is already stored.
bool OoCTreeAccess::ComputeLayout_BFS (void)
{
	queue <int> Front;
	Front.push (0);

	printf ("BFS computation in in-core manner\n");

	while (!Front.empty ()) {
		int StartIdx = Front.front ();
		Front.pop ();

		#if HIERARCHY_TYPE == TYPE_BVH
		BSPArrayTreeNodePtr pCurNode = GETNODE(StartIdx);
		#else
		BSPArrayTreeNodePtr pCurNode = GETNODE(StartIdx << 1);
		#endif

		if (ISNOLEAF(pCurNode)) {	
			unsigned int LeftIdx = GETLEFTCHILD(pCurNode);
			unsigned int RightIdx = GETRIGHTCHILD(pCurNode);

			#if HIERARCHY_TYPE == TYPE_BVH
			#else
			LeftIdx = LeftIdx >> 1;
			RightIdx = RightIdx >> 1;
			#endif

			m_pLayoutWrite->Add (LeftIdx);
			m_pLayoutWrite->Add (RightIdx);
			
			m_pProg->step ();
			m_pProg->step ();


			Front.push (LeftIdx);
			Front.push (RightIdx);
		}
	}

	return true;
}


bool OoCTreeAccess::ComputeLayout_DFS (void)
{
	#ifdef FOUR_BYTE_FOR_KD_NODE
		printf ("There is no point of using this function.\n");
		printf ("Therefore, this function is not fully implemeted for this case.\n");
		exit (-1);
	#endif

	stack <int> Front;
	Front.push (0);

	printf ("DFS computationr\n");

	while (!Front.empty ()) {
		int StartIdx = Front.top ();
		Front.pop ();

		m_pLayoutWrite->Add (StartIdx);
		m_pProg->step ();

		#if HIERARCHY_TYPE == TYPE_BVH
		BSPArrayTreeNodePtr pCurNode = GETNODE((StartIdx));
		#else
		BSPArrayTreeNodePtr pCurNode = GETNODE((StartIdx << 1));
		#endif

		if (ISNOLEAF(pCurNode)) {	
			unsigned int LeftIdx = GETLEFTCHILD(pCurNode);
			unsigned int RightIdx = GETRIGHTCHILD(pCurNode);

			#if HIERARCHY_TYPE == TYPE_BVH
			#else
			LeftIdx = LeftIdx >> 1;
			RightIdx = RightIdx >> 1;
			#endif

			Front.push (RightIdx);
			Front.push (LeftIdx);
		}
	}

	return true;
}


bool OoCTreeAccess::ComputeLayout_CA (CTreeCluster & TreeLayout)
{
	OOC_NodeLayout_FILECLASS<CNodeLayout> & NodeLayout = * m_NodeLayout;

	int BVInBlock = g_NumNodeInBlock;	
	int i, NumSavedNode = 0;

	// for each node in the front, we store its two children.
	// So, its probability is to have collision given its parent collision.
	VArray <CSortedNodeLayout> * pNextFront = new VArray <CSortedNodeLayout>;// it keeps nodes for next front tracking
	VArray <CSortedNodeLayout> * ptempFront = new VArray <CSortedNodeLayout>;		

	//	cache-aware
	bool start = true;

	m_pLayoutWrite->Add (NULL_IDX);		// to have even nodes in the top cluster. so, we have extra one bv
	
	CSortedNodeLayout Root (NodeLayout, NULL, 0,		// root node
	TreeLayout.m_BB[0], TreeLayout.m_BB[1], false);	// this node is not under the current root cluster

	pNextFront->Append (Root);
	
	while (pNextFront->Size () != 0) {
		printf ("Size of Front = %d\n", pNextFront->Size ());

		// Step2: store data into a new file
		// here we get a node having biggest prob. 
		// Staring from the node, we store nodes and add its child into the heap again.
		
		VArray <CSortedNodeLayout> & NextFront = * pNextFront;
		// Note: we do not consider any spatical locality between nodes
		for (i = 0;i < NextFront.Size ();i++)
		{
			
			CSortedNodeLayout Root = NextFront [i];
		
			// store nodes from the local maximum.
			priority_queue <CSortedNodeLayout> LocalHeap;
			LocalHeap.push (Root);

			int TempNumSavedNode;
			if (start) {
				TempNumSavedNode = 2;		// at first time, we store 2 nodes including real root
				start = false;
			}
			else
				TempNumSavedNode = 0;
			
			while (!LocalHeap.empty () && (TempNumSavedNode < BVInBlock)) {
				CSortedNodeLayout Node = LocalHeap.top ();
				LocalHeap.pop ();

				// we store two child together since we access them together
				// put two children				
				BSPArrayTreeNodePtr pCurNode = GETNODE(Node.m_Idx);		

				if (ISLEAF(pCurNode))
					continue;

				Vector3 Min = Node.m_BB [0];
				Vector3 Max = Node.m_BB [1];

				unsigned int LeftIdx = GETLEFTCHILD(pCurNode);
				unsigned int RightIdx = GETRIGHTCHILD(pCurNode);

				Vector3 LBB [2], RBB [2];
				#if HIERARCHY_TYPE == TYPE_KD_TREE
				LBB [0] = Min; LBB [1] = Max;
				RBB [0] = Min; RBB [1] = Max;

				int currentAxis = AXIS (pCurNode);

				LBB [1].e [currentAxis] = pCurNode->splitcoord;
				RBB [0].e [currentAxis] = pCurNode->splitcoord;
				#endif	
				#if HIERARCHY_TYPE == TYPE_BVH
				BSPArrayTreeNodePtr lChild = GETNODE(LeftIdx);
				BSPArrayTreeNodePtr rChild = GETNODE(RightIdx);
				LBB [0] = lChild->min; LBB [1] = lChild->max;
				RBB [0] = rChild->min; RBB [1] = rChild->max;
				#endif

				CSortedNodeLayout Left (NodeLayout, NULL, LeftIdx,
					LBB[0], LBB[1], false);
				LocalHeap.push (Left);

				CSortedNodeLayout Right (NodeLayout, NULL, RightIdx,
					RBB[0], RBB[1], false);
				LocalHeap.push (Right);

				m_pLayoutWrite->Add (LeftIdx);
				m_pLayoutWrite->Add (RightIdx);

				m_pProg->step ();
				m_pProg->step ();


				NumSavedNode += 2;
				TempNumSavedNode += 2;
			}

			// indicate one block cluster is processed.
			m_pLayoutWrite->Finalize (false);

			// make left nodde in the queue into Next front
			while (!LocalHeap.empty ()) {
				CSortedNodeLayout Node = LocalHeap.top ();
				LocalHeap.pop ();

				ptempFront->Append (Node);
			}

		}	// end of for --> front node

		if (ptempFront->Size () == 0) {	// end of process
			VArray <CSortedNodeLayout> * pSwapArr = pNextFront;
			pNextFront->Clear (false);
			pNextFront = ptempFront;
			ptempFront = pNextFront;

			continue;
		}

		pNextFront->Clear (false);

		// Compute Cache-Coherent Layout between front nodes ------
		// since we merge several clusters into one block, there should be good spatial locality
		// between successive clusters
		// Here we use cache-coherent method since we do not know any cache-aware method.

		// Compute weight between clusters as a probability to access another cluster.
		// Since we do not compute a probability between clusters if there is no overlap, we only
		// consider clusters whose bonding boxes have some overlap.

		// Note: but, now, we just use minimum distance between cluster as a weight for the simple 
		//			implementation.
		//		--> just use KNN.

		VArray <CSortedNodeLayout> & Nodes = * ptempFront;
		VArray <CSortedNodeLayout> & NewFront = * pNextFront;
		int NumNodes = Nodes.Size ();

		// To make it work in out-of-core manner, we simply process nodes in some decomposed
		// way
		int Unit_Processed = GetGoodProcessingUnitSize (NumNodes);
		int LeftNumNodes = NumNodes;
		int StartNodeIdx = 0;
		int Pass = 0;
		while (LeftNumNodes > 0) {
			Pass++;

			
			int NumNodesBatch = Unit_Processed;

			if (NumNodesBatch > LeftNumNodes)
				NumNodesBatch = LeftNumNodes;

			printf ("%d node processed at %d pass\n",NumNodesBatch, Pass);

			LeftNumNodes -= NumNodesBatch;
			
			YOON::CANN Ann;
			int * Orders = new int [NumNodesBatch];

			// add nodes into KNN structure
			Ann.SetNumVertices (NumNodesBatch);
			Vec3f Sum;
			for (i = 0;i < NumNodesBatch;i++)
			{
				CSortedNodeLayout Node = Nodes [StartNodeIdx + i];

				Vector3 _Center = (Nodes [StartNodeIdx + i].m_BB [0] + Nodes [StartNodeIdx + i].m_BB [1]) / 2.f;
				Vec3f Center;
				CONVERT_VECTOR3_VEC3F (_Center, Center)

				Ann.SetGeomVertex (i, Center);
			}

			// create a graph with a KNN
			OpenCCL::CLayoutGraph Graph (NumNodesBatch);

			int Neighbors [50], NumNeighbors;
			for (i = 0;i < NumNodesBatch;i++)
			{
				CSortedNodeLayout Node = Nodes [StartNodeIdx + i];

				Vector3 _Center = (Nodes [StartNodeIdx+i].m_BB [0] + Nodes [StartNodeIdx+i].m_BB [1]) / 2.f;
				Vec3f Center;
				CONVERT_VECTOR3_VEC3F (_Center, Center)

				NumNeighbors = Ann.GetApproximateNNs (Center, Neighbors);

				int k;
				for (k = 0;k < NumNeighbors;k++)
					Graph.AddEdge (i, Neighbors [k]);
			}

			Graph.ComputeOrdering (Orders);

			for (i = 0;i < NumNodesBatch;i++)
				NewFront.Append (Nodes [StartNodeIdx + Orders [i]]);
	     
			
			delete [] Orders;

			StartNodeIdx += NumNodesBatch;
		}
		Nodes.Clear (false);
	}

	// force write any data
	m_pLayoutWrite->Finalize (true);


	return true;
}

int GetGoodProcessingUnitSize (int NumNode)
{
	//const int GoodProcessingUnitSize = 1000*50;	// 50k
	const int GoodProcessingUnitSize = 10;

	int Division = 1;
	while (float (NumNode) / float (Division) > GoodProcessingUnitSize)
		Division++;

	return (float (NumNode) / float (Division) + 1); // ceil

}
