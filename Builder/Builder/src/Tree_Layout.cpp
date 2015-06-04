#include "common.h"
//#include "kDTree.h"
//#include "OutOfCoreTree.h"
#include "OoCTreeAccess.h"

#include <math.h>
#include <direct.h>
#include "BufferedOutputs.h"
#include "CashedBoxFiles.h"
#include "Vertex.h"
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



#include "TreeLayout.h"

// mininum volume reduction between two nodes of LOD
const int LOD_DEPTH_GRANUALITY = 3; 
const float MIN_VOLUME_REDUCTION_BW_LODS = pow ((float)2, (int)LOD_DEPTH_GRANUALITY);
const int MIN_NUM_TRIANGLES_LOD = pow ((float)2, (int)LOD_DEPTH_GRANUALITY);
const unsigned int MAX_NUM_LODs = __int64 (1) << 31 - 1;
const unsigned int LOD_BIT = 1;
//#define HAS_LOD(idx) (idx & LOD_BIT)
#define HAS_LOD(idx) (idx != 0)
const unsigned int ERR_BITs = 5;
#define GET_REAL_IDX(idx) (idx >> ERR_BITs)
const unsigned int _QUAN_ERR_BIT_MASK = ((1 << ERR_BITs) - 1);
#define GET_ERR_QUANTIZATION_IDX(idx) (idx & _QUAN_ERR_BIT_MASK)


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


CNodeLayout::CNodeLayout (void)
{
	m_AccProb = 0;
	m_ParentIdx = NULL_IDX;
	//m_pPrev = m_pNext = NULL;
}


CSortedNodeLayout::CSortedNodeLayout (OOC_NodeLayout_FILECLASS<CNodeLayout> & NodeLayout, 
							  CIntHashMap * pHashMap, unsigned int Idx,
							  Vector3 & Min, Vector3 & Max, bool ContainedInCluster)
{
	m_pPrev = m_pNext = NULL;
	m_Idx = Idx;
	m_AccProb = NodeLayout [Idx].m_AccProb;

	m_BB [0] = Min;
	m_BB [1] = Max;

	if (pHashMap != NULL && ContainedInCluster == true) {
		CIntHashMap::iterator Iter = pHashMap->find (Idx);
		if (Iter == pHashMap->end ()) {	
			CIntHashMap::value_type NewElement1 (Idx, 1);
			pHashMap->insert (NewElement1);
		}
		else {
			printf ("Data duplication in hash map for layout computation 4\n");
			exit (-1);
		}
	}
}

void CSortedNodeLayout::Set (OOC_NodeLayout_FILECLASS<CNodeLayout> & NodeLayout, 
							  CIntHashMap * pHashMap, unsigned int Idx,
							  Vector3 & Min, Vector3 & Max, bool ContainedInCluster)
{
	m_pPrev = m_pNext = NULL;
	m_Idx = Idx;
	m_AccProb = NodeLayout [Idx].m_AccProb;

	m_BB [0] = Min;
	m_BB [1] = Max;

	if (pHashMap != NULL && ContainedInCluster == true) {
		CIntHashMap::iterator Iter = pHashMap->find (Idx);
		if (Iter == pHashMap->end ()) {	
			CIntHashMap::value_type NewElement1 (Idx, 1);
			pHashMap->insert (NewElement1);
		}
		else {
			printf ("Data duplication in hash map for layout computation 5\n");
			exit (-1);
		}
	}
}

CTreeCluster::CTreeCluster (void)
{
	m_SubClusters = NULL;
	m_OrderSubClusters = NULL;

}

void CTreeCluster::SetMaxNumClusters (int MaxNumClusters)
{
	m_SubClusters = new CTreeCluster [MaxNumClusters];
	m_OrderSubClusters = new int [MaxNumClusters];

	int i;
	for (i = 0;i < MaxNumClusters;i++)
		m_SubClusters [i].m_pParentInDecomposition = this;
}

void CTreeCluster::ReleaseMemory (void)
{
	if (m_SubClusters != NULL) {
		delete [] m_SubClusters;
		m_SubClusters = NULL;
	}

	if (m_OrderSubClusters != NULL) {
		delete [] m_OrderSubClusters;
		m_OrderSubClusters = NULL;
	}

		

}
CTreeCluster::~CTreeCluster (void)
{
	ReleaseMemory ();
}

bool CTreeCluster::IsBelongToParentDecomposedCluster (int NodeIdx)
{

	if (m_pParentInDecomposition == NULL)
		return true;

	if (m_pParentInDecomposition->m_ContainedNodes.size () == 0)
		return true;
	else {
		CIntHashMap & Nodes = m_pParentInDecomposition->m_ContainedNodes;

		CIntHashMap::iterator Iter = Nodes.find (NodeIdx);
		if (Iter == Nodes.end ())
			return false;

		return true;
	}

	return true;
}



int GetAvgNodesInCluster (int TotalNodes)
{
	// recursive divide the hierarchy
	// if a top cluster has B node, its child is B+1
	// Assume that each cluster has B nodes
	// So, B(B+2) >= NumNode
	// -> B^2 + 2B - NumNode >= 0
	// -> (B+ 1)^2 - 1 - NumNode >= 0
	// -> B >= (NumNode + 1)^(1/2) - 1

	int AvgNodes = int (ceil (sqrt (float (TotalNodes + 1) - 1) ));

	// if the tree is very irregular, the child cluster can be huge.
	// so, we have more node in the root cluster, which can be controled in main memory
	if (TotalNodes > 100)
		AvgNodes = float (AvgNodes) * 1.5;

	if (TotalNodes <= 3)
		return 1;

	return AvgNodes;
}


// add src node into one current possible position
void CDelayWrite::Add (int Src)
{
	m_Buffer.Append (Src);

	if (m_Buffer.Size () == m_NumNodeBlock)
		Flush (BUFFER);

}

// indicate one cluster is finished
// So, if there are left node in buffer, it need to go m_Delayed.
// Then, the m_Delayed is full, it need to be flushed.
void CDelayWrite::Finalize (bool force)
{
	int i, j;

	for (i = 0;i < m_Buffer.Size ();i++)
	{
		m_Delayed.Append (m_Buffer [i]);

		if (m_Delayed.Size () == m_NumNodeBlock)
			Flush (DELAYED);
	}

	m_Buffer.Clear (false);

	// this is called at the last step of storing nodes of BVH
	if (force) {
		Flush (DELAYED);
		m_Delayed.Clear (false);
	}
}

int CDelayWrite::GetNumSavedNode (void)
{
	return m_CurPos;
}

// save buffered BVs into destination of BVH
void CDelayWrite::Flush (int Type)
{
	int i, Size;

	if (Type == BUFFER)	// m_buffer
		Size = m_Buffer.Size ();
	else			// delayed buffer
		Size = m_Delayed.Size ();

	for (i = 0;i < Size;i++)
	{
		int SrcIdx;
		
		if (Type == BUFFER)
			SrcIdx = m_Buffer [i];
		else
			SrcIdx = m_Delayed [i];

		if (SrcIdx == NULL_IDX)	{
			// store dummy: it is used to make top cluster have even number of nodes
			//BSPArrayTreeNode & DestBV = m_DestNodes.GetRef (m_CurPos);
			//BSPArrayTreeNode Temp;
			//DestBV = Temp;
			
			m_CurPos++;
			continue;
		}

		
		#ifdef FOUR_BYTE_FOR_KD_NODE
			BSPArrayTreeNode SrcBV = m_SrcNodes [SrcIdx];
			BSPArrayTreeNodePtr pSrcBV = &SrcBV;

			BSPArrayTreeNode & DestBV = m_DestNodes.GetRef (m_CurPos);
			DestBV = SrcBV;

			if (ISNOLEAF(pSrcBV))
				DestBV.children = NULL_IDX;
		#else
			BSPArrayTreeNode SrcBV = m_SrcNodes [(SrcIdx)];
			BSPArrayTreeNodePtr pSrcBV = &SrcBV;

			static int count = 0;
			count++;
			BSPArrayTreeNode & DestBV = m_DestNodes.GetRef ((m_CurPos));
			DestBV = SrcBV;

			if (ISNOLEAF(pSrcBV)) {
				unsigned int lower_2bit = DestBV.children & 3;
				DestBV.children = NULL_IDX;

				// initialize 2 bit
				DestBV.children = (DestBV.children >> 2);
				DestBV.children = (DestBV.children << 2);
				
				// put axis and leaf info
				DestBV.children |= lower_2bit;

				#ifdef KDTREENODE_16BYTES
				#ifndef _USE_CONTI_NODE
					DestBV.children2 = NULL_IDX;
				#endif
				#endif

			}
		#endif
		

		#ifdef KDTREENODE_16BYTES
			#ifdef _USE_LOD
			if (ISNOLEAF(pSrcBV) && HAS_LOD (SrcBV.lodIndex)) {
				int lodIndex = GET_REAL_IDX (SrcBV.lodIndex);
				//const LODNode & LOD = GET_LOD (lodIndex);
				const LODNode SrcLOD = m_SrcLODs [lodIndex];
				LODNode & DestLOD = m_DestLODs.GetRef (m_LODCurPos);
				DestLOD = SrcLOD;
				int QuanIdx = GET_ERR_QUANTIZATION_IDX(SrcBV.lodIndex);
					
				DestBV.lodIndex = (m_LODCurPos << ERR_BITs);
				DestBV.lodIndex |= QuanIdx;

				m_LODCurPos++;
			}
			#endif
		#endif

		// When child's stored, we need to put its index on the nodes child
		// for that, make link

		if (ISNOLEAF(pSrcBV)) {
			unsigned int LeftIdx = GETLEFTCHILD(pSrcBV);
			unsigned int RightIdx = GETRIGHTCHILD(pSrcBV);

			CNodeLayout & Left = m_NodeLayout.GetRef (LeftIdx);
			assert (Left.m_ParentIdx == NULL_IDX);
			Left.m_ParentIdx = m_CurPos;
		
			CNodeLayout & Right = m_NodeLayout.GetRef (RightIdx);
			assert (Right.m_ParentIdx == NULL_IDX);
			Right.m_ParentIdx = m_CurPos;

		}
		
		#ifdef FOUR_BYTE_FOR_KD_NODE
			// If there is link, set parent's left and right child index
			CNodeLayout & CurLink = m_NodeLayout.GetRef (SrcIdx);
			if (CurLink.m_ParentIdx != NULL_IDX) {
				BSPArrayTreeNode & ParentBV = m_DestNodes.GetRef (CurLink.m_ParentIdx);

				// we make sure that we always meet left node first
				if (ParentBV.children == NULL_IDX) {
					ParentBV.children = m_CurPos;
				}
				else {
					// we're saving right node right next the left node.
					// So, here, we store right child.
					assert (ParentBV.children == m_CurPos - 1);
				}
			}
		#else
			// If there is link, set parent's left and right child index
			#if HIERARCHY_TYPE == TYPE_BVH
			CNodeLayout & CurLink = m_NodeLayout.GetRef ((SrcIdx));
			#else
			CNodeLayout & CurLink = m_NodeLayout.GetRef ((SrcIdx << 1));
			#endif
			if (CurLink.m_ParentIdx != NULL_IDX) {
				#if HIERARCHY_TYPE == TYPE_BVH
				BSPArrayTreeNode & ParentBV = m_DestNodes.GetRef ((CurLink.m_ParentIdx));
				#else
				BSPArrayTreeNode & ParentBV = m_DestNodes.GetRef ((CurLink.m_ParentIdx << 1));
				#endif
			
				unsigned int lower_2bit = ParentBV.children & 3;
				unsigned int Flag = ParentBV.children >> 2;

				// we make sure that we always meet left node first
				if (Flag == (NULL_IDX >> 2)) {
					ParentBV.children = lower_2bit;
					#if HIERARCHY_TYPE == TYPE_BVH
					ParentBV.children |= (m_CurPos << 2);
					#else
					ParentBV.children |= (m_CurPos << 3);
					#endif
				}
				else {
					// we're saving right node right next the left node.
					// So, here, we store right child.

					#ifdef KDTREENODE_16BYTES
						#if HIERARCHY_TYPE == TYPE_BVH
						#ifndef _USE_CONTI_NODE
						assert (ParentBV.children2 == NULL_IDX);
						ParentBV.children2 = (m_CurPos << 2);
						#endif
						#else
						ParentBV.children2 = (m_CurPos << 3);
						#endif
					#endif
				}
			}
		#endif
			
		m_CurPos++;
	}

	if (Type == BUFFER)	// m_buffer
		m_Buffer.Clear (false);
	else
		m_Delayed.Clear (false);
}

