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

#include "OOC_PCA.h"
#include <assert.h>
#include "limits.h"
#include "DelayWrite.h"

// mininum volume reduction between two nodes of LOD
const int LOD_DEPTH_GRANUALITY = 1; 
const float MIN_VOLUME_REDUCTION_BW_LODS = pow ((float)2, (int)LOD_DEPTH_GRANUALITY);
const int MIN_NUM_TRIANGLES_LOD = pow ((float)2, (int)LOD_DEPTH_GRANUALITY);
const unsigned int MAX_NUM_LODs = UINT_MAX;
const unsigned int LOD_BIT = 1;
//#define HAS_LOD(idx) (idx & LOD_BIT)
#define HAS_LOD(idx) (idx != 0)
const unsigned int ERR_BITs = 5;
#define GET_REAL_IDX(idx) (idx >> ERR_BITs)
//#define GET_REAL_IDX(idx) (idx >> LOD_BIT)

#ifdef _USE_OOC
#define GET_LOD(Idx) (*m_LodList)[Idx]
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
#define GETVERTEX(idx) (*m_VertexList)[idx]
#else
#define GETTRI(idx) m_TriangleList[idx]
#define GETVERTEX(idx) m_VertexList[idx]
#endif


#define CONVERT_VEC3F_VECTOR3(s,d) {d.e [0] = s.x;d.e [1] = s.y; d.e[2] = s.z;}
#define CONVERT_VECTOR3_VEC3F(s,d) {d.x = s.e [0];;d.y = s.e [1];d.z = s.e [2];}

Vector3 g_BBMin, g_BBMax;
int g_NumNodes = 0;

bool OoCTreeAccess::ComputeSimpRep (void)
{
	// traverse the kd-tree from top-down
	// if the BB of the current node is much smaller than a pivot node, 
	//		we compute simplification representation
	//		RQ1: what is optimal condition for that given runtime performance?
	// if we decided to have a simplification representation, we tag it.
	// we continue this process until we hit the leaf
	// Then, we collect all the vertex, and run PCA to compute a plane or two triangles
	//		RQ2: how do we compute good representative representation for irregular meshes?
	//		we use weight according to area of face and use weighted vertex for PCA
	//		one possible way of doing it is to use centroid of the triangle.
	//		RQ3: which representation is faster for intersection and ray-differentials?
	//		RQ4: we also want to extend this method for out-of-core to handle massive models.
	//		RQ5: what is runtime error metric?

	assert (sizeof (LODNode) == 32);

	OptionManager *opt = OptionManager::getSingletonPtr();

	char fileNameTri[MAX_PATH], fileNameVertex[MAX_PATH], fileNameMaterial[MAX_PATH];
	char filenameNodes[MAX_PATH], fileNameIndices[MAX_PATH], fileNameLODs[MAX_PATH];
	char output[1000];
	char header[100];
	char bspfilestring[50];
	size_t ret;

	strcpy (fileNameVertex, getVertexFileName ().c_str ());
	strcpy (fileNameTri, getTriangleFileName ().c_str ());

	m_TriangleList = new OOC_TRI_FILECLASS<Triangle>(fileNameTri, 
						   			  1024*1024*32,
									  1024*1024);
	m_VertexList = new OOC_VERTEX_FILECLASS<Vertex>(fileNameVertex, 
									1024*1024*32,
		    						1024*1024);
									//1024*opt->getOptionAsInt("ooc", "cacheEntrySizeVerticesKB", 1536*4*2));									
	


	sprintf(filenameNodes, "%s.node", getkDTreeFileName ().c_str ());
	sprintf(fileNameIndices, "%s.idx", getkDTreeFileName ().c_str ());
	sprintf(fileNameLODs, "%s.lod", getkDTreeFileName ().c_str ());

	m_Tree = new OOC_BSPNODE_FILECLASS<BSPArrayTreeNode>(filenameNodes, 										 
										 1024*1024*256,
										 1024*1024, "w");

	m_Indexlists = new OOC_BSPIDX_FILECLASS<unsigned int>(fileNameIndices, 
										   1024*1024*32,
										   1024*1024);
	
	m_pLODBuf =  new BufferedOutputs<LODNode> (fileNameLODs, 100000);
	m_pLODBuf->clear();

	LODNode Dummy;
	m_pLODBuf->appendElement (Dummy);		// index 0 is non LOD, so we make sure that we do not have it
	m_NumLODs = 1;
	//m_NumLODs = 0;


	COOCPCAwoExtent RootPCA;
	Vector3 BBMin, BBMax, Diff;
	

	g_BBMin = BBMin = grid.p_min;
	g_BBMax = BBMax = grid.p_max;

	Diff = BBMax - BBMin;

	float VolumeBB = Diff.e [0] * Diff.e [1] * Diff.e [2];

	if (VolumeBB < 1) {
		printf ("\n");
		printf ("Warning: current impl. has numerical issues. Please increase scale of the model.\n");
		printf ("\n");
	}

	// start from the root (0)
	//ComputeSimpRep (0, BBMin, BBMax, VolumeBB/ MIN_VOLUME_REDUCTION_BW_LODS, RootPCA);
	Vec3f RealBB [2];
	RealBB [0].Set ( 1e15, 1e15, 1e15);
	RealBB [1].Set (-1e15,-1e15,-1e15);

	m_pProg = new Progression ("Compute R-LODs", m_treeStats.numNodes, 100);



	sprintf (output, "Simplification start\n");
	printf ("%s", output); 
	m_pLog->logMessage (LOG_INFO, output);
	

	ComputeSimpRep (0, BBMin, BBMax, VolumeBB, RootPCA, RealBB);
	
	sprintf (output, "Simplification end\n");
	printf ("%s", output); 
	m_pLog->logMessage (LOG_INFO, output);


	delete m_pProg;

	printf ("Num LODs = %d\n", m_NumLODs);

	delete m_pLODBuf;
	delete m_Indexlists;
	delete m_Tree;

	delete m_TriangleList;
	delete m_VertexList;


	// flush material file
	m_outputs_mat->flush();
	delete m_outputs_mat;


	//delete m_pLog;

	return true;

}



// BBMin and BBMax is a volume of previous LOD in the search path in the tree.
// RealBB [2] are conservative BB that cover all the geometry that are contained in the voxel.
bool OoCTreeAccess::ComputeSimpRep (unsigned int NodeIdx, Vector3 & BBMin, Vector3 & BBMax,
									float MinVolume, COOCPCAwoExtent & PCA, Vec3f * RealBB)
{
	#ifdef KDTREENODE_16BYTES

	// detect whether we need a LOD representation.
	//		- if there is big volume change or depth change, we need to use a LOD at runtime.
	//		- we take care of depth change at runtime, so here we only consider volume change.

	/*
	if (NoLODs > MIN_NO_LODS) {
	*/
	int i;
	//BSPArrayTreeNodePtr _pCurNode = GETNODE (NodeIdx);	
	BSPArrayTreeNode curNode = *GETNODE(NodeIdx);	// local copy, point can be removed due to out-of-core access
	BSPArrayTreeNodePtr pCurNode = &curNode;

	m_pProg->step ();

	if (ISLEAF (pCurNode)) {
		// collect data

		BSPArrayTreeNode & CurNode = m_Tree->GetRef (NodeIdx);
		//CurNode.lodIndex = 0;

		int count = GETCHILDCOUNT(pCurNode);
		unsigned int idxList = GETIDXOFFSET(pCurNode);	

		if (count == 0)	{ // if empty cell, fall back to hierarchically compute BB
			CONVERT_VECTOR3_VEC3F(BBMin, RealBB [0]);
			CONVERT_VECTOR3_VEC3F(BBMax, RealBB [1]);
		}

		for (i=0; i<count; i++,idxList++) {	
			//assert(idxList < treeStats.sumTris);		
			unsigned int triID = *MAKEIDX_PTR(idxList);		
			const Triangle &tri = GETTRI(triID);

			const Vector3 V0 = GETVERTEX (tri.p[0]).v;
			const Vector3 V1 = GETVERTEX (tri.p[1]).v;
			const Vector3 V2 = GETVERTEX (tri.p[2]).v;

			Vec3f V [4];
			V [0].x = V0.e [0];V [0].y = V0.e [1];V [0].z = V0.e [2]; // position
			V [1].x = V1.e [0];V [1].y = V1.e [1];V [1].z = V1.e [2];
			V [2].x = V2.e [0];V [2].y = V2.e [1];V [2].z = V2.e [2];
			V [3].x = tri.n.e [0];	// normal
			V [3].y = tri.n.e [1];
			V [3].z = tri.n.e [2];

			for (int j = 0;j < 3;j++)
				V [j].UpdateMinMax (RealBB [0], RealBB [1]);
	
			// Note: we need to generalize this later
			//rgb color (0.7, 0.7, 0.7);
			rgb color = m_MaterialColor [tri.material];
			PCA.InsertTriangle (V, V[3], color);
		}

		return true;
	}

	Vector3 Diff = BBMax - BBMin;
	float VolumeBB = Diff.e [0] * Diff.e [1] * Diff.e [2];
	float TargetMinVolume;

	BSPArrayTreeNode & CurNode = m_Tree->GetRef (NodeIdx);
	if (VolumeBB <= MinVolume) {
		CurNode.lodIndex = LOD_BIT;
		TargetMinVolume = VolumeBB/MIN_VOLUME_REDUCTION_BW_LODS;
	}
	else {
		CurNode.lodIndex = 0;
		TargetMinVolume = MinVolume;
	}

	// compute BB of two child nodes
	Vector3 LBB [2], RBB [2];
	BSPArrayTreeNode lChild = *GETNODE(GETLEFTCHILD(pCurNode));
	BSPArrayTreeNode rChild = *GETNODE(GETRIGHTCHILD(pCurNode));
	LBB [0] = lChild.min; LBB [1] = lChild.max;
	RBB [0] = rChild.min; RBB [1] = rChild.max;

	int currentAxis = AXIS (pCurNode);

	Diff = BBMax - BBMin;
	
	// continue top-down process
	COOCPCAwoExtent LPCA, RPCA;
	Vec3f LRealBB [2], RRealBB [2];
	
	LRealBB [0].Set ( 1e15, 1e15, 1e15);
	LRealBB [1].Set (-1e15,-1e15,-1e15);
	RRealBB [0].Set ( 1e15, 1e15, 1e15);
	RRealBB [1].Set (-1e15,-1e15,-1e15);

	ComputeSimpRep (GETLEFTCHILD(pCurNode), LBB [0], LBB [1], TargetMinVolume, LPCA, LRealBB);
	ComputeSimpRep (GETRIGHTCHILD(pCurNode), RBB [0], RBB [1], TargetMinVolume, RPCA, RRealBB);

	PCA = LPCA + RPCA;		// compute PCA for parent one with constant complexity

	// compute new conservative BB
	for (i = 0;i < 2;i++)
	{
		LRealBB [i].UpdateMinMax (RealBB [0], RealBB [1]);
		RRealBB [i].UpdateMinMax (RealBB [0], RealBB [1]);
	}


	// continue bottom-up process
	// if we detect LOD tag, perform (linear time complexity) PCA in an out-of-core manner
	ComputeRLOD (NodeIdx, PCA, BBMin, BBMax, RealBB);


	#endif

	return true;

}


float OoCTreeAccess::ComputeRLOD (unsigned int NodeIdx, COOCPCAwoExtent & PCA, 
								Vector3 & BBMin, Vector3 & BBMax, Vec3f * RealBB)
{

	#ifdef KDTREENODE_16BYTES

	BSPArrayTreeNode & CurNode = m_Tree->GetRef (NodeIdx);

	if (HAS_LOD (CurNode.lodIndex)) {
		
		if (PCA.IsEmpty ()) {	// empty node
			CurNode.lodIndex = 0;
			return true;
		}

 		if (PCA.GetNumContainedTriangle () < MIN_NUM_TRIANGLES_LOD) {	// contained too smal triangles
			CurNode.lodIndex = 0;
			return true;
		}

		// compute LOD
		Vec3f Center,  Extents [3];
		//Vec3f ShadingNormal;

		CExtLOD LOD;
		PCA.ComputePC (Center, Extents);
		PCA.SetLOD (LOD, BBMin, BBMax, RealBB);
	

		m_StatErr.update (LOD.m_ErrBnd);

		#ifdef _USE_TRI_MATERIALS
		// Note: we need to generalize this later
		rgb color = PCA.GetMeanColor (); 

		//rgb color ((rand ()%3 + 2)/4., (rand ()%3 + 2)/4., (rand ()%3 + 2)/4.);
		//rgb color (0.7, 0.7, 0.7);
		#ifdef USE_QUANTIZED_COLOR
			color = PCA.GetQuantizedMeanColor ();
		#endif
		//rgb color = PCA.GetMeanColor ();

		unsigned int hash = (unsigned int)(color.r() + 256*color.g() + 256*256*color.b());

		ColorTableIterator colorIter;
		if ((colorIter = m_usedColors.find(hash)) != m_usedColors.end()) {
			LOD.m_material = colorIter->second;
		}
		else {
			MaterialDiffuse newMat(color);					
			m_outputs_mat->appendElement(newMat);				
			LOD.m_material = m_materialIndex;
			m_usedColors [hash] = LOD.m_material;
			m_MaterialColor.Append (color);

			m_materialIndex++;
		}
		#endif

		assert (LOD.m_i1 <= 2 && LOD.m_i2 <= 2);

		// store LOD into a file
		m_pLODBuf->appendElement (LOD);
		assert (m_NumLODs < MAX_NUM_LODs);

		CurNode.lodIndex = (m_NumLODs << ERR_BITs);
		//CurNode.lodIndex = (m_NumLODs << 1) | LOD_BIT;
		m_NumLODs++;
	
		return true;
	}

	#endif

	return true;
}


bool OoCTreeAccess::Check (bool LODCheck)
{
	OptionManager *opt = OptionManager::getSingletonPtr();
	LogManager *log = LogManager::getSingletonPtr();
 
  
	char fileNameTri[MAX_PATH], fileNameVertex[MAX_PATH], fileNameMaterial[MAX_PATH];
	char filenameNodes[MAX_PATH], fileNameIndices[MAX_PATH], fileNameLODs[MAX_PATH], fileNameTreeInfo[MAX_PATH];
	char output[1000];
	char header[100];
	char bspfilestring[50];
	size_t ret;
 
	g_NumNodes = 0;

	/*
	sprintf(fileNameTreeInfo, "%s", getkDTreeFileName ().c_str ());
	FILE *fp = fopen(fileNameTreeInfo, "rb");

	ret = fread(header, 1, BSP_FILEIDSTRINGLEN + 1, fp);
	if (ret != (BSP_FILEIDSTRINGLEN + 1)) {
		sprintf(output, "Could not read header from BSP tree file '%s', aborting. (empty file?)", fileNameTreeInfo);
		log->logMessage(LOG_ERROR, output);
		return false;
	}

	// test header format:
	strcpy(bspfilestring, BSP_FILEIDSTRING);
	for (unsigned int i = 0; i < BSP_FILEIDSTRINGLEN; i++) {
		if (header[i] != bspfilestring[i]) {
			printf(output, "Invalid BSP tree header, aborting. (expected:'%c', found:'%c')", bspfilestring[i], header[i]);
			log->logMessage(LOG_ERROR, output);
			return false;		
		}
	}

	// test file version:
	if (header[BSP_FILEIDSTRINGLEN] != BSP_FILEVERSION) {
		printf(output, "Wrong BSP tree file version (expected:%d, found:%d)", BSP_FILEVERSION, header[BSP_FILEIDSTRINGLEN]);
		log->logMessage(LOG_ERROR, output);
		return false;		
	}

	// format correct, read in full BSP tree info structure:

	// write count of nodes and tri indices:
	ret = fread(&m_treeStats, sizeof(BSPTreeInfo), 1, fp);
	if (ret != 1) {
		sprintf(output, "Could not read tree info header!");
		log->logMessage(LOG_ERROR, output);
		return false;
	}

	*/

	strcpy (fileNameVertex, getVertexFileName ().c_str ());
	strcpy (fileNameTri, getTriangleFileName ().c_str ());
 
	m_TriangleList = new OOC_TRI_FILECLASS<Triangle>(fileNameTri, 
						   			  1024*1024*32,									   
									  1024*1024);	

	m_VertexList = new OOC_VERTEX_FILECLASS<Vertex>(fileNameVertex, 
									1024*1024*32,
		    						1024*1024);									
	
	sprintf(filenameNodes, "%s.node", getkDTreeFileName ().c_str ());
	sprintf(fileNameIndices, "%s.idx", getkDTreeFileName ().c_str ());
	sprintf(fileNameLODs, "%s.lod", getkDTreeFileName ().c_str ());

	m_Tree = new OOC_BSPNODE_FILECLASS<BSPArrayTreeNode>(filenameNodes, 										 
										 1024*1024*256,
										 1024*1024, "w");

	m_Indexlists = new OOC_BSPIDX_FILECLASS<unsigned int>(fileNameIndices, 
										   1024*1024*32,
										   1024*1024);
	
	if (LODCheck) {
		m_LodList = new OOC_LOD_FILECLASS<LODNode>(fileNameLODs, 
										   1024*1024*32,
										   1024*1024, "w");
	}

	Vec3f RealBB [2];
	RealBB [0].Set ( 1e15, 1e15, 1e15);
	RealBB [1].Set (-1e15,-1e15,-1e15);
	

	m_pProg = new Progression ("Check the kd-tree", m_treeStats.numNodes, 100);

	Check (0, RealBB, LODCheck, -1, 1);
	
	delete m_pProg;

	assert (g_NumNodes = m_treeStats.numNodes);
	if (g_NumNodes != m_treeStats.numNodes) {
		printf ("Error: The number of counted nodes (%d) is more than stored ones (%d)", g_NumNodes,
			m_treeStats.numNodes);
	}	

	if (LODCheck) {
		delete m_LodList;
	}

	delete m_Indexlists;
	delete m_Tree;

	delete m_TriangleList;
	delete m_VertexList;

	if (LODCheck) {
		cout << "Error stats: -------------" << endl;
		cout << m_StatErr << endl;
		cout << "Error Ratio stats: -------" << endl;
		cout << m_StatErrRatio << endl;
	}

	cout << "Depth of leaves: --------------" << endl;
	cout << m_DepthStats << endl;
	return true; 

}



bool OoCTreeAccess::Check (int NodeIdx, Vec3f BBox [2], bool LODCheck, float PreError, 
						   int Depth)
{

	#ifdef KDTREENODE_16BYTES

	float CurError = PreError;

	g_NumNodes++;
//	assert (g_NumNodes <= m_treeStats.numNodes);
	if (g_NumNodes > m_treeStats.numNodes) {
		printf ("Error: we're visiting too many nodes: accessing %d nodes (max: %d)\n",
			g_NumNodes, m_treeStats.numNodes);
	}

	//assert (NodeIdx >> 1 < m_treeStats.numNodes);
	#ifdef FOUR_BYTE_FOR_KD_NODE
	if (NodeIdx >= m_treeStats.numNodes)
		printf ("Too big nodeIdx, %d (%d)\n", NodeIdx >> 1, m_treeStats.numNodes);
	#else
	if ((NodeIdx) >> 1 >= m_treeStats.numNodes)
		printf ("Too big nodeIdx, %d (%d)\n", NodeIdx >> 1, m_treeStats.numNodes);

	#endif


	m_pProg->step ();

	BSPArrayTreeNode curNode = *GETNODE(NodeIdx);
	BSPArrayTreeNodePtr pCurNode = &curNode;

#ifdef USE_LOD
	if (!LODCheck) {
		//assert (curNode.lodIndex == 0);
//		curNode.lodIndex = 0;
		BSPArrayTreeNode & CurNode = m_Tree->GetRef (NodeIdx);
		if(!ISLEAF(&CurNode))
			CurNode.lodIndex = 0;

	}
#endif
	
	if (!ISLEAF (&curNode) && LODCheck && HAS_LOD (curNode.lodIndex)) {
//		printf ("LOD idx = %d\n", curNode.lodIndex);

		int lodIndex = GET_REAL_IDX (curNode.lodIndex);

		assert (lodIndex < m_NumLODs);
		
		const LODNode & LOD = GET_LOD (lodIndex);
		int i;
		for (i = 0;i < 3;i++)
			assert (LOD.m_n.e [i] >= -1.f && LOD.m_n.e [i] <= 1.f);

		assert (LOD.m_i1 <= 2);
		assert (LOD.m_i2 <= 2);
		assert (LOD.m_ErrBnd > 0);		

		if (PreError != -1)
			m_StatErrRatio.update (PreError/LOD.m_ErrBnd);

		CurError = LOD.m_ErrBnd;

		//m_StatErr.update (LOD.m_ErrBnd);
	}
	
	if (ISLEAF (&curNode)) {
		m_DepthStats.update (Depth);

		// collect data
		Vector3 V[3];
		int i, _i;
		unsigned int triID;
		Triangle tri;
		//BSPArrayTreeNodePtr pCurNode = GETNODE (NodeIdx);

		int count = GETCHILDCOUNT(&curNode);
		unsigned int idxList = GETIDXOFFSET(&curNode);	

		for (i=0; i<count; i++,idxList++) {	
			//assert(idxList < treeStats.sumTris);		
			triID = *MAKEIDX_PTR(idxList);		

			if (triID >= m_treeStats.numTris) {
				cerr << "triID >= m_treeStats.numTris : " << triID << " >= " << m_treeStats.numTris << endl;
			}

			tri = GETTRI(triID);

			V[0] = GETVERTEX(tri.p[0]).v;
			V[1] = GETVERTEX(tri.p[1]).v;
			V[2] = GETVERTEX(tri.p[2]).v;
			
			for (_i = 0;_i < 3;_i++)
			{
				int j;
				for (j = 0;j < 3;j++)
				{
					assert (V [_i].e [j] >= grid.p_min.e[j]);
					assert (V [_i].e [j] <= grid.p_max.e[j]);
				}
			}


			/*
			for (_i = 0;_i < 3;_i++)
			{
				IFWARN (! (tri.n.e [_i] >= -1.f && tri.n.e [_i] <= 1.f), 
					0, 
					"Found degenerated triangle: " << triID)
					exit (-1);
			}
			*/

		}

		return true;
	}

	// compute BB of two child nodes
	Vector3 LBB [2], RBB [2];
	BSPArrayTreeNode lChild = *GETNODE(GETLEFTCHILD(pCurNode));
	BSPArrayTreeNode rChild = *GETNODE(GETRIGHTCHILD(pCurNode));
	LBB [0] = lChild.min; LBB [1] = lChild.max;
	RBB [0] = rChild.min; RBB [1] = rChild.max;
	//LBB [0] = BBox [0]; LBB [1] = BBBox [1];
	//RBB [0] = BBox [0]; RBB [1] = BBBox [1];

	CONVERT_VEC3F_VECTOR3 (BBox [0], LBB [0]);
	CONVERT_VEC3F_VECTOR3 (BBox [1], LBB [1]);
	CONVERT_VEC3F_VECTOR3 (BBox [0], RBB [0]);
	CONVERT_VEC3F_VECTOR3 (BBox [1], RBB [1]);

	int currentAxis = AXIS (&curNode);

	Vec3f LRealBB [2], RRealBB [2];
	
	LRealBB [0].Set ( 1e15, 1e15, 1e15);
	LRealBB [1].Set (-1e15,-1e15,-1e15);
	RRealBB [0].Set ( 1e15, 1e15, 1e15);
	RRealBB [1].Set (-1e15,-1e15,-1e15);


	/*
	if (curNode.children == 63417 && curNode.children2 == 63424)
		printf ("Hello\n");
	*/

	unsigned int child_left = GETLEFTCHILD(pCurNode);
	unsigned int child_right = GETRIGHTCHILD(pCurNode);
	Check(child_left, LRealBB, LODCheck, CurError, Depth + 1);
	Check(child_right, RRealBB, LODCheck, CurError, Depth + 1);


	#endif

	return true;
}

bool OoCTreeAccess::PrepareDataForRLODs (char * fileName, bool Material)
{
	OptionManager *opt = OptionManager::getSingletonPtr();
	char fileNameMaterial[MAX_PATH], tempfileNameMaterial[MAX_PATH], output[1000], command [255];
	FILE *materialFile = NULL;

	const char *baseDirName = opt->getOption("global", "scenePath", "");
	sprintf(outDirName, "%s%s.ooc", baseDirName, fileName);
	//mkdir(outDirName);

	unsigned int i;

	// read materials
	if (Material == true) {
		sprintf(fileNameMaterial, "%s", getMaterialListName ().c_str ());
		strcpy (tempfileNameMaterial, fileNameMaterial);
		strcat (tempfileNameMaterial, "_");
		// move it into temporary file for creating new for BufferedOutput
		sprintf (output, "move materials.ooc materials.ooc_");
	

		char oldDir[MAX_PATH+1]; // save working directory
		getcwd(oldDir, MAX_PATH);
		chdir(outDirName);		

		system (output);

		chdir(oldDir);		

		if ((materialFile = fopen(tempfileNameMaterial, "rb")) == NULL) {
			sprintf(output, "PrepareDataForRLODs(): could not find %s in directory!", fileNameMaterial);
			//log->logMessage(LOG_ERROR, output);
			//fclose(vertexFile);
			//fclose(triFile);
			//return ERROR_FILENOTFOUND;
			return false;
		}

		struct stat fileInfo;
		stat(tempfileNameMaterial, &fileInfo);
		unsigned int nMaterials = fileInfo.st_size / sizeof(MaterialDiffuse);


		// materials:
		MaterialDiffuse *tempMaterialList = new MaterialDiffuse[nMaterials];
		fread(tempMaterialList, sizeof(MaterialDiffuse), nMaterials, materialFile);
		fclose(materialFile);

		

		// init out file for all materials
		m_outputs_mat = new BufferedOutputs<MaterialDiffuse>(getMaterialListName().c_str(), 1000);
		m_outputs_mat->clear();
		m_materialIndex = 0;

		// handle materials if necessary:
		ColorTable usedColors;
		ColorTableIterator colorIter;

		#ifdef _USE_TRI_MATERIALS
		for (i = 0 ; i < nMaterials; i++)
		{
			rgb color = tempMaterialList [i].getColor ();
			unsigned int hash = (unsigned int)(color.r() + 256*color.g() + 256*256*color.b());


			if ((colorIter = m_usedColors.find(hash)) != m_usedColors.end()) {
				assert (0);
			}
			else {
				MaterialDiffuse newMat(color);					
				m_outputs_mat->appendElement(newMat);				
				//tri.material = m_materialIndex;
				m_usedColors[hash] = m_materialIndex;
				m_MaterialColor.Append (color);

				m_materialIndex++;
			}
		}

		#endif
	}	// end of noMaterial


	// read bounding box info
	// this code came from "
	//		bool SIMDBSPTree::loadFromFiles(const char* filename)"	

	LogManager *log = LogManager::getSingletonPtr();	
	//OptionManager *opt = OptionManager::getSingletonPtr();
	char bspFileName[MAX_PATH];
	char header[100];
	char bspfilestring[50];
	size_t ret;

	sprintf(bspFileName, "%s", getkDTreeFileName ().c_str ());

	FILE *fp = fopen(bspFileName, "rb");
	if (fp == NULL) {
		sprintf(output, "Could not open BSP tree file '%s'!", bspFileName);
		log->logMessage(LOG_WARNING, output);
		return false;
	}

	
	sprintf(output, "Loading BSP tree from files ('%s')...", bspFileName);
	log->logMessage(LOG_INFO, output);

	ret = fread(header, 1, BSP_FILEIDSTRINGLEN + 1, fp);
	if (ret != (BSP_FILEIDSTRINGLEN + 1)) {
		sprintf(output, "Could not read header from BSP tree file '%s', aborting. (empty file?)", bspFileName);
		log->logMessage(LOG_ERROR, output);
		return false;
	}

	// test header format:
	strcpy(bspfilestring, BSP_FILEIDSTRING);
	for (i = 0; i < BSP_FILEIDSTRINGLEN; i++) {
		if (header[i] != bspfilestring[i]) {
			printf(output, "Invalid BSP tree header, aborting. (expected:'%c', found:'%c')", bspfilestring[i], header[i]);
			log->logMessage(LOG_ERROR, output);
			return false;		
		}
	}

	// test file version:
	if (header[BSP_FILEIDSTRINGLEN] != BSP_FILEVERSION) {
		printf(output, "Wrong BSP tree file version (expected:%d, found:%d)", BSP_FILEVERSION, header[BSP_FILEIDSTRINGLEN]);
		log->logMessage(LOG_ERROR, output);
		return false;		
	}

	// format correct, read in full BSP tree info structure:

	// write count of nodes and tri indices:
	ret = fread(&m_treeStats, sizeof(BSPTreeInfo), 1, fp);
	if (ret != 1) {
		sprintf(output, "Could not read tree info header!");
		log->logMessage(LOG_ERROR, output);
		return false;
	}

	/* 
#ifdef _USE_OOC

#else
	sprintf(output, "Allocating memory...");
	log->logMessage(LOG_INFO, output);

	tree = new BSPArrayTreeNode[treeStats.numNodes];	
	indexlists = new unsigned int[treeStats.sumTris];

	// read tree node array:
	sprintf(output, "  ... reading %d tree nodes ...", treeStats.numNodes);
	log->logMessage(LOG_INFO, output);
	ret = fread(tree, sizeof(BSPArrayTreeNode), treeStats.numNodes, fpNodes);

	if (ret != treeStats.numNodes) {
		sprintf(output, "Could only read %u nodes, expecting %u!", ret, treeStats.numNodes);
		log->logMessage(LOG_ERROR, output);
		return false;
	}

	// read tri index array
	sprintf(output, "  ... reading %d tri indices ...", treeStats.sumTris);
	log->logMessage(LOG_INFO, output);
	ret = fread(indexlists, sizeof(int), treeStats.sumTris, fpIndices);

	if (ret != treeStats.sumTris) {
		sprintf(output, "Could only read %u indices, expecting %u!", ret, treeStats.sumTris);
		log->logMessage(LOG_ERROR, output);
		return false;
	}

	sprintf(output, "  done!");
	log->logMessage(LOG_INFO, output);
#endif
	*/

	fclose(fp);
	

	PrintTree ();

	//this->min = treeStats.min;
	//this->max = treeStats.max;
	this->grid.p_min = m_treeStats.min;
	this->grid.p_max = m_treeStats.max;
	

	m_pLog = new LoggerImplementationFileout (getLogFileName ().c_str ());
	

	return true;

}

void OoCTreeAccess::PrintTree(const char *LoggerName) {
	LogManager *log = LogManager::getSingletonPtr();
	char outputBuffer[2000];
	log->logMessage("-------------------------------------------", LoggerName);
	log->logMessage("BSP Tree Statistics", LoggerName);
	log->logMessage("-------------------------------------------", LoggerName);
	sprintf(outputBuffer, "Time to build:\t%d seconds, %d milliseconds", (int)m_treeStats.timeBuild, (int)((m_treeStats.timeBuild - floor(m_treeStats.timeBuild)) * 1000));
	log->logMessage(outputBuffer, LoggerName);
	sprintf(outputBuffer, "Triangles:\t%d", m_treeStats.numTris);
	log->logMessage(outputBuffer, LoggerName);
	sprintf(outputBuffer, "Nodes:\t\t%d", m_treeStats.numNodes);
	log->logMessage(outputBuffer, LoggerName);
	sprintf(outputBuffer, "Leafs:\t\t%d", m_treeStats.numLeafs);
	log->logMessage(outputBuffer, LoggerName);
	sprintf(outputBuffer, "Max. leaf depth:\t%d (of %d)", m_treeStats.maxLeafDepth, m_treeStats.maxDepth);
	log->logMessage(outputBuffer, LoggerName);
	sprintf(outputBuffer, "Max. tri count/leaf:\t%d", m_treeStats.maxTriCountPerLeaf);
	log->logMessage(outputBuffer, LoggerName);
	if (m_treeStats.numLeafs > 0) {
		sprintf(outputBuffer, "Avg. leaf depth:\t%.2f", (float)m_treeStats.sumDepth / (float)m_treeStats.numLeafs);
		log->logMessage(outputBuffer, LoggerName);

		sprintf(outputBuffer, "Avg. tris/leaf:\t%.2f", (float)m_treeStats.sumTris / (float)m_treeStats.numLeafs);
		log->logMessage(outputBuffer, LoggerName);

		sprintf(outputBuffer, "Tri refs total:\t\t%d", m_treeStats.sumTris);
		log->logMessage(outputBuffer, LoggerName);

	}
	sprintf(outputBuffer, "Used memory:\t%d KB", (m_treeStats.numNodes*sizeof(BSPArrayTreeNode) + (m_treeStats.sumTris * sizeof(int))) / 1024);
	log->logMessage(outputBuffer, LoggerName);

}

bool OoCTreeAccess::PrintError (void)
{
	OptionManager *opt = OptionManager::getSingletonPtr();
	LogManager *log = LogManager::getSingletonPtr();


	char fileNameTri[MAX_PATH], fileNameVertex[MAX_PATH], fileNameMaterial[MAX_PATH];
	char filenameNodes[MAX_PATH], fileNameIndices[MAX_PATH], fileNameLODs[MAX_PATH], fileNameTreeInfo[MAX_PATH];
	char output[1000];
	char header[100];
	char bspfilestring[50];
	size_t ret;

	sprintf(filenameNodes, "%s.node", getkDTreeFileName ().c_str ());
	sprintf(fileNameIndices, "%s.idx", getkDTreeFileName ().c_str ());
	sprintf(fileNameLODs, "%s.lod", getkDTreeFileName ().c_str ());


	m_Tree = new OOC_BSPNODE_FILECLASS<BSPArrayTreeNode>(filenameNodes, 										 
										 1024*1024*256,
										 1024*1024, "w");

	m_Indexlists = new OOC_BSPIDX_FILECLASS<unsigned int>(fileNameIndices, 
										   1024*1024*32,
										   1024*1024);
	
	m_LodList = new OOC_LOD_FILECLASS<LODNode>(fileNameLODs, 
										   1024*1024*32,
										   1024*1024, "w");

	Vec3f RealBB [2];
	RealBB [0].Set ( 1e15, 1e15, 1e15);
	RealBB [1].Set (-1e15,-1e15,-1e15);
	

//	m_pProg = new Progression ("Check the kd-tree", m_treeStats.numNodes, 100);

	srand (time (NULL));
	PrintError (0);
	
//	delete m_pProg;
	delete m_LodList;
	delete m_Indexlists;
	delete m_Tree;

	return true;

}

bool OoCTreeAccess::PrintError (int NodeIdx)
{

	#ifdef KDTREENODE_16BYTES

	BSPArrayTreeNode curNode = *GETNODE(NodeIdx);
	BSPArrayTreeNodePtr pCurNode = &curNode;


	if (HAS_LOD (curNode.lodIndex)) {


		int lodIndex = GET_REAL_IDX (curNode.lodIndex);

		assert (lodIndex < m_NumLODs);
		
		const LODNode & LOD = GET_LOD (lodIndex);

		printf ("LOD error = %f\n", LOD.m_ErrBnd);
		
	}
	
	if (ISLEAF (&curNode))
		return true;

	unsigned int child_left = GETLEFTCHILD(pCurNode);
	unsigned int child_right = GETRIGHTCHILD(pCurNode);

	int where = rand () % 2;

	if (where == 0) 
		PrintError (child_left);
	else
		PrintError (child_right);

	#endif


	return true;
}

bool OoCTreeAccess::QuantizeErr (void)
{
	OptionManager *opt = OptionManager::getSingletonPtr();

	char fileNameTri[MAX_PATH], fileNameVertex[MAX_PATH], fileNameMaterial[MAX_PATH];
	char filenameNodes[MAX_PATH], fileNameIndices[MAX_PATH], fileNameLODs[MAX_PATH];
	char output[1000];

	sprintf(filenameNodes, "%s.node", getkDTreeFileName ().c_str ());
	sprintf(fileNameIndices, "%s.idx", getkDTreeFileName ().c_str ());
	sprintf(fileNameLODs, "%s.lod", getkDTreeFileName ().c_str ());

	m_Tree = new OOC_BSPNODE_FILECLASS<BSPArrayTreeNode>(filenameNodes, 										 
										 1024*1024*32,
										 1024*1024, "w");

	m_LodList = new OOC_LOD_FILECLASS<LODNode>(fileNameLODs, 
										   1024*1024*32,
										   1024*1024, "w"); 
	

	m_pProg = new Progression ("Quantize Erros of R-LODs", m_treeStats.numNodes, 100);

	//m_pLog = new LoggerImplementationFileout (getLogFileName ().c_str ());
	m_pLog->logMessage (LOG_INFO, "Quantization start.");


	// Assumption: we assume that simplification ratio bewteen erros of two LODs of child
	//				parent nodes is close to LOD_DEPTH_GRANUALITY.
	float AvgDepth = (float)m_treeStats.sumDepth / (float)m_treeStats.numLeafs;
	float NumLODsinPath = AvgDepth * 4 / float (LOD_DEPTH_GRANUALITY);
	int AllowedBit = ceil (log (NumLODsinPath)/log(2.));
	int ForcedBit = 5;

	
	sprintf (output, "AllowdBit = %d, But we use %d, \n", AllowedBit, ForcedBit);
	printf ("%s", output); 
	m_pLog->logMessage (LOG_INFO, output);
	
	//m_QuanErr.Set (m_StatErr.getMin (), m_StatErr.getMax (), m_StatErrRatio.getAvg (), AllowedBit);
	m_QuanErr.Set (m_StatErr.getMin (), m_StatErr.getMax (), m_StatErrRatio.getAvg (), ForcedBit);

	char fileNameQErr[MAX_PATH];
	sprintf(fileNameQErr, "%s.QErr", getkDTreeFileName ().c_str ());
	FILE *fpQErr = fopen(fileNameQErr, "wb");
	fwrite(&m_QuanErr, sizeof(CErrorQuan), 1, fpQErr);
	fclose(fpQErr);

	// update BVH header


	m_treeStats.m_MinErr = m_QuanErr.m_MinErr;
	m_treeStats.m_ErrRatio = m_QuanErr.m_ErrRatio;
	m_treeStats.m_NumAllowedBit = m_QuanErr.m_NumUpBit;

	printf ("Min %f, ErrRatio %f, # of Bits = %d\n", m_treeStats.m_MinErr, m_treeStats.m_ErrRatio, 
		m_treeStats.m_NumAllowedBit);
	SaveStatBVH ();


	QuantizeErr (0);

	// logging
	sprintf (output, "Quantized error bit: %d\n", m_QuanErr.m_NumUpBit);
	printf ("%s", output); 
	m_pLog->logMessage (LOG_INFO, output);

	m_NumLODBit = ceil (log (float (m_NumLODs))/ log (2.));
	int TotalUsedBit = m_NumLODBit + m_QuanErr.m_NumUpBit + m_QuanErr.m_NumLowBit;
	sprintf (output, "LOD bit: %d, Total used bit: %d\n", m_NumLODBit, TotalUsedBit);
	printf ("%s", output); 
	m_pLog->logMessage (LOG_INFO, output);


	if (TotalUsedBit > 32) {
		sprintf (output, "Total bit is overflow\n");
		printf ("%s", output); 
		m_pLog->logMessage (LOG_INFO, output);
	}

	sprintf (output, "Avg. Quantization Error = %f\n", m_QuanErr.m_SumQuanErr / m_QuanErr.m_NumFittedData);
	printf ("%s", output); 
	m_pLog->logMessage (LOG_INFO, output);



	m_pLog->logMessage (LOG_INFO, "Quantization end.");
	delete m_pProg;

	delete m_LodList;
	delete m_Tree;

	
	//delete m_pLog;

	return true;
}

bool OoCTreeAccess::QuantizeErr (int NodeIdx)
{
	#ifdef KDTREENODE_16BYTES

	BSPArrayTreeNode curNode = *GETNODE(NodeIdx);
	BSPArrayTreeNodePtr pCurNode = &curNode;

	m_pProg->step ();

	if (ISLEAF (&curNode))
		return true;

	if (HAS_LOD (curNode.lodIndex)) {

		int lodIndex = GET_REAL_IDX (curNode.lodIndex);		
		const LODNode & LOD = GET_LOD (lodIndex);

		unsigned int QuanErr;
		m_QuanErr.Fit (LOD.m_ErrBnd, QuanErr);

		BSPArrayTreeNode & CurNode = m_Tree->GetRef (NodeIdx);

		assert (QuanErr < pow ((float) 2, (float) 5));
		assert (CurNode.lodIndex >= m_QuanErr.m_MaxNumUp);
		CurNode.lodIndex |= QuanErr;

	}

	unsigned int child_left = GETLEFTCHILD(pCurNode);
	unsigned int child_right = GETRIGHTCHILD(pCurNode);

	QuantizeErr (child_left);
	QuantizeErr (child_right);

	#endif

	return true;
}

bool OoCTreeAccess::SaveStatBVH (void)
{
	LogManager *log = LogManager::getSingletonPtr();	
	char output[255], filename [255];
	size_t ret;


	sprintf(filename, "%s", getkDTreeFileName ().c_str ());

	FILE *fp = fopen(filename, "wb");

	if (fp == NULL) {
		sprintf(output, "Could not open BSP tree file '%s'!", filename);
		log->logMessage(LOG_ERROR, output);
		return false;
	}

	sprintf(output, "Saving BSP tree to file '%s'...", filename);
	log->logMessage(LOG_INFO, output);

	// write header and version:

	fwrite(BSP_FILEIDSTRING, 1, BSP_FILEIDSTRINGLEN, fp);
	fputc(BSP_FILEVERSION, fp);

	// write stats:
	fwrite(&m_treeStats, sizeof(BSPTreeInfo), 1, fp);

	fclose(fp);

	return true;


}

