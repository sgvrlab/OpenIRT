#ifndef COMP_REP_H
#define COMP_REP_H
#include "BVHNodeDefine.h"
#include "Vertex.h"
#include "Triangle.h"
#include "Photon.h"
#include "OOCFile6464.h"
#include "OptionManager.h"
#include "DictionaryCompression.h"
#include "Progression.h"
#include <vector>
#include <queue>
#include <hash_set>

#define MAX_SIZE_TEMPLATE 15
#define DEPTH_TEMPLATE 4
#define MAX_NUM_TEMPLATES 26

#define BIT_MASK_16 0xFFFF
#define BIT_MASK_14 0x3FFF
#define BIT_MASK_9 0x1FF
#define BIT_MASK_5 0x1F
#define BIT_MASK_3 0x7
#define BIT_MASK_2 0x3

//#define USE_COMPLETE_TREE
//#define USE_LOD
//#define CONVERT_PHOTON
//#define USE_VERTEX_QUANTIZE

//#define USE_GZIP

/*////
#ifdef GENERATE_OUT_OF_CORE_REP
//#define USE_RANGE_ENCODER
#endif
*/////
#ifdef CONVERT_PHOTON
#undef USE_LOD
#undef USE_VERTEX_QUANTIZE
#endif

#ifdef USE_LOD
#include "LOD_header.h"
#include "OOC_PCA.h"
#include "Statistics.h"
#include "err_quantization.h"
#endif

#include "positionquantizer_new.h"
#ifdef CONVERT_PHOTON
#include "StaticDictionary.hpp"
#endif

/*////
#ifdef GENERATE_OUT_OF_CORE_REP
#ifdef USE_RANGE_ENCODER
#include "smreader_smc_v.h"
#include "smwriter_smc_v.h"
#else
#include "smreader_smc_v_d.h"
#include "smwriter_smc_v_d.h"
#endif
#endif
*/////

#ifdef CONVERT_PHOTON
#define TREE_CLASS	Photon
#define CTREE_CLASS	CompTreePhoton
#else
#define TREE_CLASS	BSPArrayTreeNode
#define CTREE_CLASS	CompTreeNode
#endif

#ifdef USE_GZIP
#include "zlib.h"
#include "zconf.h"
#pragma comment(lib, "zdll.lib")
#endif

typedef struct CompTreeNode_t
{
	unsigned int data;
} CompTreeNode, *CompTreeNodePtr;

typedef struct CompTreePhoton_t
{
	Vector3 pos;		// position
	unsigned char phi, theta;	// incident angle (spherical coords)
	short plane;		// plane flag for tree
	unsigned short power[4];	// Light power(2 byte for each R,G,B,A)
} CompTreePhoton, *CompTreePhotonPtr;

typedef struct CompTreeSupp_t
{
	unsigned short leftIndex;
	unsigned short rightIndex;
	unsigned int data;
} CompTreeSupp, *CompTreeSuppPtr;

#pragma pack(push, 1)
typedef struct CompTreeVert_t
{
#ifdef USE_VERTEX_QUANTIZE
	unsigned short qV[3];
	unsigned int data;
#else
	_Vector4 vert;
#endif
} CompTreeVert, *CompTreeVertPtr;
#pragma pack(pop)

#ifdef USE_LOD
typedef struct CompLOD_t
{
	LODNode lod;
} CompLOD, *CompLODPtr;
#endif

typedef struct CompClusterHeader_t
{
	unsigned int fileSize;
	unsigned int numNode;
	unsigned int numSupp;
	unsigned int numVert;
	unsigned int rootType;
	Vector3 BBMin;
	Vector3 BBMax;
} CompClusterHeader, *CompClusterHeaderPtr;

typedef struct TravStat_t
{
	int cluster;
	int index;
	unsigned int rootTemplate;
	int type;
	int isLeft;
	TREE_CLASS node;
	int axis;
} TravStat, *TravStatPtr;

typedef struct TempTri_t
{
	unsigned short p[3];
	unsigned char i1, i2;
} TempTri, *TempTriPtr;

typedef struct TriList_t
{
	unsigned short offsetOrBackup;
	unsigned char numTris;
	vector<TempTri> tris;
} TriList, *TriListPtr;

typedef struct Vert16_t
{
	Vector3 v;
	unsigned int m;
} Vert16;

class HCCMesh
{
protected :
	OOCFile6464<TREE_CLASS> *tree;
	OOCFile6464<Triangle> *tris;
	OOCFile6464<unsigned int> *indices;
	OOCFile6464<Vertex> *verts;

	int numClusters;
	int numLowLevelNodes;
	int maxNodesPerCluster;
	int maxVertsPerCluster;

	int nQuantize;

	typedef struct TemplateTable_t {
		int numNodes;
		int numLeafs;
		TREE_CLASS tree[MAX_SIZE_TEMPLATE];
		unsigned int listLeaf[(MAX_SIZE_TEMPLATE+1)/2];
	} TemplateTable, *TemplateTablePtr;

	int numTemplates;

	class TreeStat {
	public :
		int numTris;		// number of triangles
		int numNodes;		// number of nodes in tree
		int numLeafs;		// number of leaf nodes

		int minDepth;		// minimum depth
		int maxDepth;		// maximum depth

		Vector3 min, max;	// bounding box

		TreeStat()
		{
			numTris = 0;
			numNodes = 0;
			numLeafs = 0;
			minDepth = INT_MAX;
			maxDepth = 0;
		}
	};

	BSPTreeInfo m_treeStats;

	typedef stdext::hash_map<unsigned int, unsigned int> VertexHash;
	typedef VertexHash::iterator VertexHashIterator;

	typedef stdext::hash_map<unsigned int, unsigned int> LinkHash;
	typedef LinkHash::iterator LinkHashIterator;

	typedef stdext::hash_set<unsigned int> VertexSet;
	typedef VertexSet::iterator VertexSetIterator;

	vector<TREE_CLASS> treeHighNode;
	typedef stdext::hash_map<unsigned int, TREE_CLASS> TreeHighNodeHash;
	typedef TreeHighNodeHash::iterator TreeHighNodeHashIterator;
	TreeHighNodeHash treeHighNodeHash;

	typedef stdext::hash_set<unsigned int> ProcessedClusters;
	ProcessedClusters processedClusters;

#ifdef USE_LOD
	OOCFile64<LODNode> *LODs;
	vector<LODNode> highLOD;
	vector<unsigned int> highLODIndexList;
	typedef stdext::hash_map<unsigned int, unsigned int> HighLODIndexHash;
	typedef HighLODIndexHash::iterator HighLODIndexHashIterator;
	HighLODIndexHash highLODIndexHash;

	vector<CompLOD> compLOD;

	FILE *fpLOD;
	FILE *fpLODCluster;
	FILE *fpLODClusterOut;

	int m_NumLODs;

	int m_curLODIndexTrav;
	vector<unsigned int> m_LODIndexMap;

	CErrorQuan m_QuanErr;

	/*
	Stats<float> m_StatErr, m_StatErrRatio;	// m_Ratio is ratio of error between parent and its children		
	Stats<double> m_DepthStats;		// for depth of leaf
	*/

	bool ComputeSimpRep(CompClusterHeader &header);
	bool ComputeSimpRep(unsigned int NodeIdx, Vector3 & BBMin, Vector3 & BBMax,
						float MinVolume, COOCPCAwoExtent & PCA, Vec3f * RealBB, TravStat &ts);
	float ComputeRLOD(unsigned int NodeIdx, unsigned int NodeIdxTrav, COOCPCAwoExtent & PCA, 
					  Vector3 & BBMin, Vector3 & BBMax, Vec3f * RealBB);

	bool QuantizeErr(CompClusterHeader &header);
	bool QuantizeErr(int NodeIdx, TravStat &ts);

	int compressClusterHCCLOD();
#endif

	unsigned int curCluster;
	unsigned int curCNodeIndex;
	unsigned int curCSuppIndex;
	unsigned int curCVertIndex;
	unsigned int curCTriIndex;
	VertexHash *vClusterHash;
	LinkHash *lClusterHash;

	vector<CTREE_CLASS> compTreeNode;
	vector<CompTreeSupp> compTreeSupp;
	vector<CompTreeVert> clusterVert;
	vector<TempTri> tempTri;

	int curTempTri;
	vector<TriList> triList2;

	unsigned int curVertOffset;
	vector<unsigned int> vertOffsetTable;
/*////
#ifdef GENERATE_OUT_OF_CORE_REP
	SMwriter* smwriter;
	SMreader* smreader;

	BitCompression *compTreeNodeOut;
	BitCompression *compTreeSuppOut;
	BitCompression *compVertColorOut;
	BitCompression *compNumTrisOut;
	FILE *fpCompOut;
	FILE *fpClusterOut;
	FILE *fpClusterOutGeom;

	__int64 offset, bOffset;
	__int64 offsetGeom, bOffsetGeom;
	__int64 fileSizeHeaderOut;
	__int64 fileSizeHighNodeOut;
	__int64 fileSizeNodeOut;
	__int64 fileSizeSuppOut;
	__int64 fileSizeMeshOut;
	__int64 fileSizeAddVertOut;
	__int64 fileSizePhotonOut;

	typedef stdext::hash_map<unsigned int, unsigned int> VertColorHash;
	typedef VertColorHash::iterator VertColorHashIterator;
	VertColorHash vertColorHash;

	Vector3 globalBBMin, globalBBMax;

	vector<unsigned int> clusterFileSizeOut;
	vector<unsigned int> clusterGeomFileSizeOut;
	vector<unsigned int> clusterFileSizeIn;

	inline int getBits(unsigned int x)
	{
		if(x < 0)
			cout << "out of range : (getBits)" << endl;
		int bits = 1;
		while((x >>= 1) != 0) bits++;
		return bits;
	}
#endif
*/////
	__int64 fileSizeHeader;
	__int64 fileSize;
	__int64 fileSizeNode;
	__int64 fileSizeSupp;
	__int64 fileSizeVert;
	__int64 fileSizeTris;

	char filePath[255];
	FILE *fpComp;
	FILE *fpCluster;
	int rootType;
	//FILE *fpTri;

	FILE *fpTreeH2;
	FILE *fpTrisH2;
	FILE *fpVertsH2;

	int numAxis[3];

#ifdef USE_VERTEX_QUANTIZE
	PositionQuantizerNew pqVert;
#endif

	int convertHCCMesh(unsigned int nodeIndex, int &numNodes, VertexHash &v, int depth, unsigned int parentIndex, int type);
	unsigned int makeCluster(unsigned int nodeIndex, int numNodes, unsigned int parentIndex, int type);

	int convertHCCMesh2(unsigned int nodeIndex, VertexSet &vs);
	int makeCluster2(unsigned int nodeIndex);
	int makeCluster2(unsigned int nodeIndex, VertexHash &vh);

	int compressClusterHCCMesh(unsigned int clusterID, unsigned int localRootIdx, CompClusterHeader &header);
#ifdef CONVERT_PHOTON
	int compressClusterHCCPhoton(unsigned int clusterID, unsigned int localRootIdx, CompClusterHeader &header);
	int compressPhoton(
		BitCompression *encoder, StaticDictionary *dic[], PositionQuantizerNew &pq, 
		unsigned int nodeIndex, unsigned int parentIndex, 
		I32 qBBMin[], I32 qBBMax[], TravStat &ts);
#endif

	int computeBB(unsigned int nodeIndex, unsigned int vList[], TravStat ts, int encode = true);

	int getTreeStat(TreeStat &treeStat, unsigned int startIndex, int depth);

#ifdef USE_COMPLETE_TREE
	int StoreCompleteTreeNodes(vector<unsigned int> &listNodeIndex);
#endif
	int StoreGeneralNodes(vector<unsigned int> &listNodeIndex, TreeStat &treeStat);
	int StoreGeneralNodes(unsigned int startIndex, int maxDepth, vector<unsigned int> &nextFront);

	int findCorrIndex(unsigned int startIndexS, unsigned int startIndexT, TREE_CLASS t[], unsigned int map[]);

	unsigned int CGETLEFTCHILD(CTREE_CLASS* node, TravStat &ts, unsigned int &minBB);
	unsigned int CGETRIGHTCHILD(CTREE_CLASS* node, TravStat &ts, unsigned int &maxBB);
	TREE_CLASS* CGETNODE(unsigned int index, TravStat &ts, unsigned int minBB, unsigned int maxBB);

	int reassignHighNodeStruct(unsigned int nodeSrcIndex, int isLeft, unsigned int parentIndex);

	int testTraverse(unsigned int startIndex, TreeStat &treeStat, TravStat ts);

	int argmin(float x[], int size);
	int argmax(float x[], int size);

	Vector3 quantizedNormals[104*104*6];
	int quantizeVector(Vector3 &v);
	void calculateQNormals();

	Progression prog;

#ifdef USE_GZIP
	char gzipDir[255];
#endif

public :
	HCCMesh();
	~HCCMesh();
	int convertHCCMesh(const char* filepath, unsigned int maxNodesPerCluster = 2048, unsigned int maxVertsPerCluster = 512);
	int convertSimple(const char* filepath);
	int convertHCCMesh2(const char* fileName);

	TemplateTable templates[MAX_NUM_TEMPLATES];

	int generateTemplates();
	int generateTemplates(const char* srcFileName);
	int isSameTree(unsigned int startIndexA, unsigned int startIndexB, int depth, TREE_CLASS t[] = NULL);

	int test();

/*////
#ifdef GENERATE_OUT_OF_CORE_REP
	int OutRepToInRep(const char *outFileName, const char *inFileName);
#endif
*/////

	FORCEINLINE void updateBB(Vector3 &min, Vector3 &max, Vector3 &vec)
	{
		min.e[0] = ( min.e[0] < vec.e[0] ) ? min.e[0] : vec.e[0];
		min.e[1] = ( min.e[1] < vec.e[1] ) ? min.e[1] : vec.e[1];
		min.e[2] = ( min.e[2] < vec.e[2] ) ? min.e[2] : vec.e[2];

		max.e[0] = ( max.e[0] > vec.e[0] ) ? max.e[0] : vec.e[0];
		max.e[1] = ( max.e[1] > vec.e[1] ) ? max.e[1] : vec.e[1];
		max.e[2] = ( max.e[2] > vec.e[2] ) ? max.e[2] : vec.e[2];
	}
};
#endif