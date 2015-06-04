#ifndef DelayWrite_H_
#define DelayWrite_H_
#include "cmdef.h"
#include "VArray.h"

class CDelayWrite {
public:
	enum {BUFFER, DELAYED};

	VArray <int> m_Delayed;
	VArray <int> m_Buffer;

	int m_NumNodeBlock;	// maximum node that can fit into a block
	int m_CurPos;		// where we store input node

	OOC_BSPNODE_FILECLASS<BSPArrayTreeNode> & m_SrcNodes, & m_DestNodes;
#ifdef _USE_LOD
	OOC_LOD_FILECLASS<LODNode> & m_SrcLODs, & m_DestLODs;
	int m_LODCurPos;	// postion for LOD layout
#endif
	OOC_NodeLayout_FILECLASS<CNodeLayout> & m_NodeLayout;

	
	CDelayWrite (
		OOC_BSPNODE_FILECLASS<BSPArrayTreeNode> & SrcNodes, 
		OOC_BSPNODE_FILECLASS<BSPArrayTreeNode> & DestNodes,
#ifdef _USE_LOD
		OOC_LOD_FILECLASS<LODNode> & SrcLODs,
		OOC_LOD_FILECLASS<LODNode> & DestLODs,
#endif
		OOC_NodeLayout_FILECLASS<CNodeLayout> & NodeLayout,
		int NumNodeBlock)
		:m_SrcNodes(SrcNodes), m_DestNodes(DestNodes), 
#ifdef _USE_LOD
		 m_SrcLODs(SrcLODs),
		 m_DestLODs(DestLODs), 
		 m_LODCurPos(0), 
#endif
		 m_NodeLayout(NodeLayout),
		 m_CurPos(0), m_NumNodeBlock(NumNodeBlock) 
	{ };
	
	void Add (int Src);
	void Finalize (bool force);	// finalize one block
	void Flush (int Type);
	int GetNumSavedNode (void);
};
#endif