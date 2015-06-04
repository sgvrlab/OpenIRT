#ifndef TREE_LAYOUT_H_
#define TREE_LAYOUT_H_
// Layout computation --------------------------------

#include "VDTActiveList.h"
#include "math.h"
#include "vec3f.hpp"
#include <hash_map>
#include "VArray.h"
using namespace std;


#ifndef INT_HASH_MAP
#define INT_HASH_MAP
// hashmap to detect unique vertex
struct eqInt
{
	bool operator()(int V1, int V2) const
	{
		if (V1 ==  V2)
			return 1;
		return 0;
	}
};
typedef stdext::hash_map <int, int> CIntHashMap;
#endif




const unsigned int NULL_IDX = 0xffffffff;	// maximum number

class CNodeLayout
{
public:

	float m_AccProb;		// probability that the node is accessed during traversal
	unsigned int m_ParentIdx;		// parent index -- for layout computation


	CNodeLayout (void);
	

};

class CSortedNodeLayout : public CNodeLayout
{
public:

	unsigned int m_Idx;
	Vector3 m_BB [2];
	//Vec3f	m_Center;			// center of the kd-node
								// Based on this info, we will compute order between clusters

	CSortedNodeLayout * m_pPrev, * m_pNext;

	CSortedNodeLayout (void) {}
	CSortedNodeLayout (OOC_NodeLayout_FILECLASS<CNodeLayout> & NodeLayout, 
		CIntHashMap * pHashMap, unsigned int Idx, Vector3 & Min, Vector3 & Max,
		bool ContainedInCluster = true);

	void Set (OOC_NodeLayout_FILECLASS<CNodeLayout> & NodeLayout, 
		CIntHashMap * pHashMap, unsigned int Idx, Vector3 & Min, Vector3 & Max,
		bool ContainedInCluster = true);


	friend inline bool operator< (CSortedNodeLayout t1, CSortedNodeLayout t2);
};

inline bool operator < (CSortedNodeLayout t1, CSortedNodeLayout t2) 
{ 
	return t1.m_AccProb < t2.m_AccProb;
}



/*
class CTreeCluster 
{
public:
	//char m_Filename [500];
	
	//int m_NumNode;				// # of node that the cluster has
	//Vector3 m_Min, m_Max;		// Bounding box
	

	//CTreeCluster * m_pPrev, * m_pNext;
};
*/

// kind of cluster node
class CTreeCluster
{
public:
	int		m_NumNodes;
	CIntHashMap	m_ContainedNodes;	// nodes that the parent cluster has.
										// since it cannot have more than 1M node, 
										// just store it at RAM

	unsigned int		m_StartIdx;				// Start Idx of the cluster

	Vector3 m_BB [2];					// BB of the cluster
	Vec3f	m_Center;			// center of the kd-node
								// Based on this info, we will compute order between clusters


	CTreeCluster * m_pParentInDecomposition;			// parent cluster in the decomposition
										// recursion
	CTreeCluster * m_SubClusters;		// sub divided clusters from the bigger cluster
										
	int m_NumSubClusters;
	int * m_OrderSubClusters;				// order of child clusters
											// , which are inm_Clusters

 
//	VArray m_StartIdxForSubClusters;
	

	CTreeCluster (void);
	CTreeCluster (int MaxNumClusters);
	~CTreeCluster (void);
	void ReleaseMemory (void);
	void SetMaxNumClusters (int MaxNumClusters);
	bool IsBelongToParentDecomposedCluster (int NodeIdx);


};


int GetAvgNodesInCluster (int TotalNodes);




#endif

