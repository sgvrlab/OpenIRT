#include <windows.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "OptionManager.h"
#include "BVHNodeDefine.h"
#include "positionquantizer_new.h"
#include "integercompressorRACBVH.h"
#include <io.h>
#include "stopwatch.hpp"

#include "BVHCompression.h"

#define GETROOT() (1)
#define FLOOR 0
#define CEIL 1

#define _USE_DUMMY_NODE
#ifdef _USE_DUMMY_NODE
#define DUMMY_NODE 0
#endif

//#define DEBUG_CODEC     // enable extra codec info to verify correctness

#define SIZE_BASE_PAGE 4096
#define SIZE_BASE_PAGE_POWER 12

//#define STATISTICS

//#define USE_RACM

#define COMPRESS_TRI

#ifdef COMPRESS_TRI
//#define USE_TRI_3_TYPE_ENCODING
#define USE_TRI_DELTA_ENCODING
#endif

#ifdef STATISTICS
//#define STAT_ERRORS
#endif

#ifdef USE_RACM
#include "compressed_mesh.h"
CMeshAbstract *g_pMesh;
#endif

#include "dynamicvector.h"

#define AXIS(node) ((node)->children & 3)
#define ISLEAF(node) (((node)->children & 3) == 3)
#define GETNODEOFFSET(idx) ((idx >> 2) << 3)

#ifdef _DEBUG
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#define new new(_NORMAL_BLOCK, __FILE__, __LINE__) 
#endif

BVHCompression::BVHCompression(int isComp)
{
	this->isComp = isComp;
}

BVHCompression::~BVHCompression(void)
{
}

struct FrontNode;
typedef struct FrontNode
{
	FrontNode* buffer_next;
	int IsInDV;
	int index;
	int count;
} FrontNode;

struct FrontTri;
typedef struct FrontTri
{
	FrontTri* buffer_next;
	int IsInDV;
	unsigned int index;
} FrontTri;

struct BaseTri;
typedef struct BaseTri
{
	BaseTri* buffer_next;
	int IsInDV;
	unsigned int index;
} BaseTri;

struct ParentCluster;
typedef struct ParentCluster
{
	ParentCluster* buffer_next;
	int IsInDV;
	unsigned int parentCN;
} ParentCluster;

struct ParentRootIndex;
typedef struct ParentRootIndex
{
	unsigned int parentIndex;
	int count;
} ParentRootIndex;

void BVHCompression::storeParentIndex(FILE *fpo, FILE *fpi, unsigned int index)
{
	BSPArrayTreeNode node;
	_fseeki64(fpo, (__int64)index*sizeof(BSPArrayTreeNode), SEEK_SET);
	fread(&node, sizeof(BSPArrayTreeNode), 1, fpo);
	if(ISLEAF(&node)) return;
	unsigned int left = node.children >> 4;
	unsigned int right = node.children2 >> 4;
	_fseeki64(fpi, (__int64)left*sizeof(unsigned int), SEEK_SET);
	fwrite(&index, sizeof(unsigned int), 1, fpi);
	_fseeki64(fpi, (__int64)right*sizeof(unsigned int), SEEK_SET);
	fwrite(&index, sizeof(unsigned int), 1, fpi);
	storeParentIndex(fpo, fpi, left);
	storeParentIndex(fpo, fpi, right);
}

void BVHCompression::storeParentIndex(FILE *fpo, FILE *fpi)
{
	__int64 posFpo = _ftelli64(fpo);
	__int64 posFpi = _ftelli64(fpi);
	storeParentIndex(fpo, fpi, 0);
	_fseeki64(fpo, posFpo, SEEK_SET);
	_fseeki64(fpi, posFpi, SEEK_SET);
}

int BVHCompression::compressQuantize(const char* filepath)
{
	char filename[MAX_PATH];
	sprintf(filename, "%s/BVH", filepath);
	char nodeName[MAX_PATH], compressedName[MAX_PATH];
	FILE *fpo;
	FILE *fpc;
	BSPArrayTreeNode currentNode;

	sprintf(nodeName, "%s.node", filename);
	sprintf(compressedName, "%s.qtz", filename);

	cout << 0 << endl;
	cout << filename << endl;

	fpo = fopen(nodeName, "rb");
	fpc = fopen(compressedName, "wb");

	// first node is the root node
	BSPArrayTreeNode root;
	fread(&root, sizeof(BSPArrayTreeNode), 1, fpo);
	_fseeki64(fpo, 0, SEEK_SET);

	float *bb_min_f = root.min.e;
	float *bb_max_f = root.max.e;

	int nbits = 16;
	PositionQuantizerNew pq;
	pq.SetMinMax(bb_min_f, bb_max_f);
	pq.SetPrecision(nbits);
	pq.SetupQuantizer();

	unsigned int numNodes = _filelengthi64(fileno(fpo))/sizeof(BSPArrayTreeNode);
	int beforeStep = 0;
	for(int i=0;i<numNodes;i++)
	{
		BSPArrayTreeNode node;
		fread(&node, sizeof(BSPArrayTreeNode), 1, fpo);
		BSPArrayTreeNode qNode;
		qNode.children = node.children;
		qNode.children2 = node.children2;
		I32 minQ[3];
		I32 maxQ[3];
		pq.EnQuantize(node.min.e, minQ, FLOOR);
		pq.EnQuantize(node.max.e, maxQ, CEIL);
		unsigned int code;
		for(int k=0;k<3;k++)
		{
			code = (unsigned int)minQ[k];
			code = (code << nbits) + (unsigned int)maxQ[k];
			memcpy(&node.min.e[k], &code, sizeof(unsigned int));
		}
		fwrite(&node, sizeof(BSPArrayTreeNode)-sizeof(Vector3), 1, fpc);
		if(i*100/numNodes != beforeStep)
		{
			beforeStep = i*100/numNodes;
			cout << beforeStep << "% ";
		}
	}
	fclose(fpo);
	fclose(fpc);
	return 1;
}

#include "zlib.h"
int BVHCompression::compressgzip(const char* filepath)
{
	OptionManager *opt = OptionManager::getSingletonPtr();
	unsigned int numNodes = 0;
	unsigned int numClusters = 0;
	unsigned int nodesPerCluster = opt->getOptionAsInt("raytracing", "nodesPerCluster", 16384);
	unsigned int maxDepth = opt->getOptionAsInt("raytracing", "maxBSPTreeDepth", 100);

	char filename[MAX_PATH];
	sprintf(filename, "%s/BVH", filepath);
	char nodeName[MAX_PATH], compressedName[MAX_PATH], headerName[MAX_PATH];
	FILE *fpo;
	gzFile fpc, fph;
	BSPArrayTreeNode currentNode;

	sprintf(nodeName, "%s.node", filename);
	sprintf(compressedName, "%s.cgz", filename);
	sprintf(headerName, "%s.gzh", filename);

	cout << 0 << endl;
	cout << filename << endl;

	fpo = fopen(nodeName, "rb");
	fpc = gzopen(compressedName, "wb");
	fph = gzopen(headerName, "wb");

	numNodes = _filelengthi64(fileno(fpo))/sizeof(BSPArrayTreeNode);
	numClusters = ceil(((float)numNodes)/nodesPerCluster);

	cout << 1 << endl;

	// first node is the root node
	BSPArrayTreeNode root;
	fread(&root, sizeof(BSPArrayTreeNode), 1, fpo);
	_fseeki64(fpo, 0, SEEK_SET);

	float *bb_min_f = root.min.e;
	float *bb_max_f = root.max.e;

	z_off_t *offsets = new z_off_t[numClusters+1];

	unsigned int sizeBasePage = SIZE_BASE_PAGE;
	unsigned int sizeBasePagePower = SIZE_BASE_PAGE_POWER;
	gzwrite(fpc, &nodesPerCluster, sizeof(unsigned int));
	gzwrite(fpc, &sizeBasePage, sizeof(unsigned int));
	gzwrite(fpc, &sizeBasePagePower, sizeof(unsigned int));
	gzwrite(fpc, &numNodes, sizeof(unsigned int));
	gzwrite(fpc, &numClusters, sizeof(unsigned int));
	gzwrite(fpc, bb_min_f, sizeof(float)*3);
	gzwrite(fpc, bb_max_f, sizeof(float)*3);
	offsets[0] = gztell(fpc);
	gzclose(fpc);

	cout << 2 << endl;

	char* clusterCache = new char[nodesPerCluster*sizeof(BSPArrayTreeNode)];
	int beforeStep = 0;
	for(int i=0;i<numClusters;i++)
	{
		unsigned int clusterSize = fread(clusterCache, sizeof(BSPArrayTreeNode), nodesPerCluster, fpo);
		char nCompressedName[MAX_PATH];
		sprintf(nCompressedName, "%s%d", compressedName, i);
		fpc = gzopen(nCompressedName, "wb");
		gzwrite(fpc, clusterCache, sizeof(BSPArrayTreeNode)*clusterSize);
		offsets[i+1] = gztell(fpc);
		gzclose(fpc);
		if(i*100/numClusters != beforeStep)
		{
			beforeStep = i*100/numClusters;
			cout << beforeStep << "% ";
		}
	}
	delete[] clusterCache;

	gzwrite(fph, offsets, sizeof(z_off_t)*(numClusters+1));

	delete[] offsets;

	fclose(fpo);
	gzclose(fph);
	return 1;
}

int BVHCompression::compress(const char* filepath)
{
	OptionManager *opt = OptionManager::getSingletonPtr();
	unsigned int numNodes = 0;
	unsigned int numClusters = 0;
	unsigned int nodesPerCluster = opt->getOptionAsInt("raytracing", "nodesPerCluster", 65536);
	unsigned int maxDepth = opt->getOptionAsInt("raytracing", "maxBSPTreeDepth", 100);
	unsigned int numBoundary = opt->getOptionAsInt("raytracing", "numBoundary", 1);
	const char *listBoundaryConstTemp = opt->getOption("raytracing", "listBoundary", "10");
	char listBoundaryTemp[1024];
	strcpy(listBoundaryTemp, listBoundaryConstTemp);

	unsigned int *listBoundary = new unsigned int[numBoundary];

	for(int i=0;i<numBoundary;i++)
	{
		char *token;
		if(i==0)
		{
			token = strtok(listBoundaryTemp, " ");
		}
		else
		{
			token = strtok(NULL, " ");
		}
		listBoundary[i] = atoi(token);
	}

	char filename[MAX_PATH];
	sprintf(filename, "%s/BVH", filepath);
	char nodeName[MAX_PATH], compressedName[MAX_PATH], parentIndexName[MAX_PATH];
	char childCountName[MAX_PATH];
	FILE *fpo, *fpc, *fpi;
	BSPArrayTreeNode currentNode;

	sprintf(nodeName, "%s.node", filename);
	sprintf(compressedName, "%s.cmp", filename);
	sprintf(parentIndexName, "%s.pi", filename);

	#ifdef USE_RACM
	sprintf(filename, "%s/RACM", filepath);
	g_pMesh = new CCompressedMesh;
	g_pMesh->PrepareData(filename ,true);
	#endif


	fpo = fopen(nodeName, "rb");
	fpc = fopen(compressedName, "wb");
	fpi = fopen(parentIndexName, "rb");

	if(fpi == NULL)
	{
		fpi = fopen(parentIndexName, "wb");

		// first, store parent index of each nodes.
		storeParentIndex(fpo, fpi);

		fclose(fpi);
		fpi = fopen(parentIndexName, "rb");
	}

	numNodes = _filelengthi64(fileno(fpo))/sizeof(BSPArrayTreeNode);
	numClusters = ceil(((float)numNodes)/nodesPerCluster);



	/*
	childCounts = new unsigned int[numNodes];

	FILE *fpChild;
	sprintf(childCountName, "%s.child", filename);

	fpChild = fopen(childCountName, "rb");
	if(fpChild == NULL)
	{
		for(int i=0;i<numNodes;i++)
		{
			childCounts[i] = (unsigned int)-1;
		}

		fpChild = fopen(childCountName, "wb");

		getAllChildCount(fpo);

		fwrite(childCounts, sizeof(unsigned int), numNodes, fpChild);
	}
	else
	{
		fread(childCounts, sizeof(unsigned int), numNodes, fpChild);
	}
	fclose(fpChild);

	cout << "childCounts[0] = " << childCounts[0] << endl;
	*/




	// first node is the root node
	BSPArrayTreeNode root;
	fread(&root, sizeof(BSPArrayTreeNode), 1, fpo);
	_fseeki64(fpo, 0, SEEK_SET);

	float *bb_min_f = root.min.e;
	float *bb_max_f = root.max.e;

	unsigned int sizeBasePage = SIZE_BASE_PAGE;
	unsigned int sizeBasePagePower = SIZE_BASE_PAGE_POWER;
	fwrite(&nodesPerCluster, sizeof(unsigned int), 1, fpc);
	fwrite(&sizeBasePage, sizeof(unsigned int), 1, fpc);
	fwrite(&sizeBasePagePower, sizeof(unsigned int), 1, fpc);
	fwrite(&numNodes, sizeof(unsigned int), 1, fpc);
	fwrite(&numClusters, sizeof(unsigned int), 1, fpc);
	fwrite(bb_min_f, sizeof(float), 3, fpc);
	fwrite(bb_max_f, sizeof(float), 3, fpc);
	fwrite(&numBoundary, sizeof(unsigned int), 1, fpc);
	fwrite(listBoundary, sizeof(unsigned int), numBoundary, fpc);

	unsigned int nodesPerClusterPower = log((double)nodesPerCluster)/log(2.0);

	// reserve for offset list
	long posOffset = ftell(fpc);
	fseek(fpc, sizeof(long)*numClusters, SEEK_CUR);

	long *offsets = new long[numClusters];

	#ifdef STATISTICS
	float g_geom = 0;
	float g_geom_raw = 0;
	float g_geom_error = 0;
	float g_index = 0;
	float g_parent_cluster = 0;
	float g_parent_index_offset = 0;
	float g_index_axis = 0;
	float g_index_isChildInThis = 0;
	float g_index_outCluster = 0;
	float g_index_offset = 0;
	float g_avg_front_size_per_cluster = 0;
	#ifdef USE_TRI_3_TYPE_ENCODING
	float g_index_tri_type = 0;
	float g_index_tri_cache = 0;
	float g_index_tri_front = 0;
	float g_index_tri_base = 0;
	float g_index_tri_offset = 0;
	#endif
	#ifdef USE_TRI_DELTA_ENCODING
	float g_index_tri = 0;
	#endif
	float g_etc = 0;
	float g_debug = 0;
	#ifdef USE_TRI_3_TYPE_ENCODING
	unsigned int g_tri_index_type1 = 0;
	unsigned int g_tri_index_type2 = 0;
	unsigned int g_tri_index_type3 = 0;
	#endif
	unsigned int g_total_node = 0;
	unsigned int g_root_nodes_of_cluster = 0;
	unsigned int g_out_cluster_index = 0;
	unsigned int g_max_parent_root_hash = 0;
	unsigned int g_max_inter_cluster_nodes = 0;
	unsigned int g_total_inter_cluster_nodes = 0;
	unsigned int g_front_size = 0;
	unsigned int g_front_size_count = 0;
	unsigned int g_num_parent_cluster = 0;
	#ifdef USE_TRI_DELTA_ENCODING
	unsigned int g_num_out_boundary_child_index = 0;
	unsigned int g_num_out_boundary_tri_index = 0;
	unsigned int g_num_in_boundary_child_index = 0;
	unsigned int g_num_in_boundary_tri_index = 0;
	#endif
	#endif

	#ifdef STATISTICS
	FILE *fpTestAllErrors = fopen("all_errors", "w");
	FILE *fpTestMinErrors = fopen("min_errors", "w");
	FILE *fpTestMaxErrors = fopen("max_errors", "w");
	FILE *fpTestBiggestAxisErrors = fopen("biggest_axis_errors", "w");
	FILE *fpTestOtherAxisErrors = fopen("other_axis_errors", "w");
	#endif

	Stopwatch tCompression("Compression time");
	BSPArrayTreeNode mCurrentNode;
	for(int i=0;i<numClusters;i++)
	{
		if(i==0) tCompression.Start();
		if(i==5)
		{
			tCompression.Stop();
			cout << tCompression << endl;
		}
		offsets[i] = ftell(fpc);

		RangeEncoder *re_geom;
		RangeEncoder *re_geom_raw;
		RangeEncoder *re_geom_error;
		RangeEncoder *re_index;
		RangeEncoder *re_parent_cluster;
		RangeEncoder *re_parent_index_offset;
		RangeEncoder *re_index_axis;
		RangeEncoder *re_index_isChildInThis;
		RangeEncoder *re_index_outCluster;
		RangeEncoder *re_index_offset;
		#ifdef USE_TRI_3_TYPE_ENCODING
		RangeEncoder *re_index_tri_type;
		RangeEncoder *re_index_tri_cache;
		RangeEncoder *re_index_tri_front;
		RangeEncoder *re_index_tri_base;
		RangeEncoder *re_index_tri_offset;
		#endif
		#ifdef USE_TRI_DELTA_ENCODING
		RangeEncoder *re_index_tri;
		#endif
		RangeEncoder *re_etc;
		RangeEncoder *re_debug;
		#ifndef STATISTICS
		re_geom = new RangeEncoder(fpc);
		re_geom_raw = re_geom;
		re_geom_error = re_geom;
		re_index = re_geom;
		re_parent_cluster = re_geom;
		re_parent_index_offset = re_geom;
		re_index_axis = re_geom;
		re_index_isChildInThis = re_geom;
		re_index_outCluster = re_geom;
		re_index_offset = re_geom;
		#ifdef USE_TRI_3_TYPE_ENCODING
		re_index_tri_type = re_geom;
		re_index_tri_cache = re_geom;
		re_index_tri_front = re_geom;
		re_index_tri_base = re_geom;
		re_index_tri_offset = re_geom;
		#endif
		#ifdef USE_TRI_DELTA_ENCODING
		re_index_tri = re_geom;
		#endif
		re_etc = re_geom;
		re_debug = re_geom;
		#endif
		#ifdef STATISTICS
		re_geom = new RangeEncoder(0, false);
		re_geom_raw = new RangeEncoder(0, false);
		re_geom_error = new RangeEncoder(0, false);
		re_index = new RangeEncoder(0, false);
		re_parent_cluster = new RangeEncoder(0, false);
		re_parent_index_offset = new RangeEncoder(0, false);
		re_index_axis = new RangeEncoder(0, false);
		re_index_isChildInThis = new RangeEncoder(0, false);
		re_index_outCluster = new RangeEncoder(0, false);
		re_index_offset = new RangeEncoder(0, false);
		#ifdef USE_TRI_3_TYPE_ENCODING
		re_index_tri_type = new RangeEncoder(0, false);
		re_index_tri_cache = new RangeEncoder(0, false);
		re_index_tri_front = new RangeEncoder(0, false);
		re_index_tri_base = new RangeEncoder(0, false);
		re_index_tri_offset = new RangeEncoder(0, false);
		#endif
		#ifdef USE_TRI_DELTA_ENCODING
		re_index_tri = new RangeEncoder(0, false);
		#endif
		re_etc = new RangeEncoder(0, false);
		re_debug = new RangeEncoder(0, false);
		#endif

		int nbits = 16;

		PositionQuantizerNew* pq;
		IntegerCompressorRACBVH* ic[2];
		RangeModel **rmDeltaChildIndex = new RangeModel*[numBoundary];
		RangeModel **rmDeltaTriIndex = new RangeModel*[numBoundary];
		RangeModel *rmIsChildInThis;
		#if defined(USE_RACM) && defined(COMPRESS_TRI)
		//RangeModel *rmTriIndexType;
		#endif

		pq = new PositionQuantizerNew();
		pq->SetMinMax(bb_min_f, bb_max_f);
		pq->SetPrecision(nbits);
		pq->SetupQuantizer();

		ic[0] = new IntegerCompressorRACBVH();
		ic[1] = new IntegerCompressorRACBVH();

		I32 maxRange = pq->m_aiQuantRange[0] > pq->m_aiQuantRange[1] ? pq->m_aiQuantRange[0] : pq->m_aiQuantRange[1];
		maxRange = maxRange > pq->m_aiQuantRange[2] ? maxRange : pq->m_aiQuantRange[2];
		ic[0]->SetRange(maxRange);
		ic[1]->SetRange(maxRange);

		ic[0]->SetPrecision(nbits);
		ic[1]->SetPrecision(nbits);

		ic[0]->SetupCompressor(re_geom_error);
		ic[1]->SetupCompressor(re_geom_error);

		typedef stdext::hash_map<unsigned int, unsigned int> NodeHashTable;
		typedef NodeHashTable::iterator NodeHashTableIterator;
		NodeHashTable *nodeHashTable = new NodeHashTable;
		char* clusterCache = new char[nodesPerCluster*sizeof(BSPArrayTreeNode)];
		#ifdef _USE_DUMMY_NODE
		unsigned int clusterSize = fread(clusterCache+sizeof(BSPArrayTreeNode)*(i==DUMMY_NODE), sizeof(BSPArrayTreeNode), nodesPerCluster-(i==DUMMY_NODE), fpo);
		for(int j=0;j<clusterSize;j++)
		{
			BSPArrayTreeNodePtr node = (BSPArrayTreeNodePtr)(clusterCache + j*sizeof(BSPArrayTreeNode));
			if(i==0 && j==0) 
			{
				continue;
			}
			if(ISLEAF(node)) continue;
			node->children += 16;
			node->children2 += 16;
		}
		#else
		unsigned int clusterSize = fread(clusterCache, sizeof(BSPArrayTreeNode), nodesPerCluster, fpo);
		#endif
		#ifdef STATISTICS
		g_total_node += clusterSize;
		#endif
		re_etc->encodeInt(clusterSize);

		for(int j=0;j<numBoundary;j++)
		{
			rmDeltaChildIndex[j] = new RangeModel(listBoundary[j]+1, 0, TRUE);
			rmDeltaTriIndex[j] = new RangeModel(listBoundary[j]+1, 0, TRUE);
		}
		#if defined(USE_RACM) && defined(COMPRESS_TRI)
//		rmTriIndexType = new RangeModel(3, 0, TRUE);
		DynamicVector *dvTriIndexFront = new DynamicVector();
		DynamicVector *dvTriIndexBase = new DynamicVector();
		typedef stdext::hash_map<unsigned int, FrontTri*> FrontHashTableTri;
		typedef stdext::hash_map<unsigned int, BaseTri*> BaseHashTableTri;
		typedef FrontHashTableTri::iterator FrontHashTableTriIterator;
		typedef BaseHashTableTri::iterator BaseHashTableTriIterator;
		FrontHashTableTri *frontHashTableTri = new FrontHashTableTri;
		BaseHashTableTri *baseHashTableTri = new BaseHashTableTri;
		bool isFirstTri = true;
		#else
		DynamicVector *dvTriIndexBase = new DynamicVector();
		typedef stdext::hash_map<unsigned int, BaseTri*> BaseHashTableTri;
		typedef BaseHashTableTri::iterator BaseHashTableTriIterator;
		BaseHashTableTri *baseHashTableTri = new BaseHashTableTri;
		#endif
		rmIsChildInThis = new RangeModel(4, 0, TRUE);

		DynamicVector *dvNodeIndex = new DynamicVector();
		typedef stdext::hash_map<int, FrontNode*> FrontHashTableNode;
		typedef stdext::hash_map<unsigned int, ParentCluster*> ParentClusterHashTable;
		typedef FrontHashTableNode::iterator FrontHashTableNodeIterator;
		typedef ParentClusterHashTable::iterator ParentClusterHashTableIterator;
		FrontHashTableNode *frontHashTableNode = new FrontHashTableNode;
		ParentClusterHashTable *parentClusterHashTable = new ParentClusterHashTable;

		int maxFrontSize = 0;
		int totalNodes = 0;
		int contNodes = 0;
		int numInterClusterNodes = 0;
		int numParentCluster = 0;
		int numLocalRoot = 0;

		// find inter-cluster nodes in advance.
		DynamicVector *dvParentCluster = new DynamicVector();
		for(int pass=0;pass<3;pass++)
		{
			if(pass == 1)
			{
				numParentCluster = dvParentCluster->size();
				re_parent_cluster->encode(numClusters, numParentCluster);
				unsigned int parentCN = 0;
				for(int k=0;k<dvParentCluster->size();k++)
				{
					parentCN = ((ParentCluster*)dvParentCluster->getElementWithRelativeIndex(k))->parentCN;
					re_parent_cluster->encode(numClusters, parentCN);
				}
				re_parent_cluster->encode(nodesPerCluster, numLocalRoot);
				continue;
			}
			for(int j=0;j<clusterSize;j++)
			{
				BSPArrayTreeNodePtr node = (BSPArrayTreeNodePtr)(clusterCache + j*sizeof(BSPArrayTreeNode));
				unsigned int curNodeIdx = i*nodesPerCluster+j;
				if(curNodeIdx == DUMMY_NODE) continue;

				NodeHashTableIterator it = nodeHashTable->find(curNodeIdx);
				if(curNodeIdx != GETROOT() && it == nodeHashTable->end())
				{
					unsigned int parentIndex;

					#ifdef _USE_DUMMY_NODE
					_fseeki64(fpi, sizeof(unsigned int)*(__int64)(curNodeIdx-1), SEEK_SET);
					fread(&parentIndex, sizeof(unsigned int), 1, fpi);
					parentIndex++;
					#else
					_fseeki64(fpi, sizeof(unsigned int)*(__int64)curNodeIdx, SEEK_SET);
					fread(&parentIndex, sizeof(unsigned int), 1, fpi);
					#endif
					unsigned int parentCN = parentIndex / nodesPerCluster;
					ParentCluster* parentCluster = NULL;
					ParentClusterHashTableIterator itParent = parentClusterHashTable->find(parentCN);
					if(pass == 0)
					{
						numLocalRoot++;
						if(parentCN != i && itParent == parentClusterHashTable->end())
						{
							parentCluster = new ParentCluster;
							parentCluster->buffer_next = 0;
							parentCluster->IsInDV = 0;
							parentCluster->parentCN = parentCN;

							parentClusterHashTable->insert(std::pair<unsigned int, ParentCluster*>(parentCN, parentCluster));
							dvParentCluster->addElement(parentCluster);
						}
					}
					if(pass == 2)
					{
						parentCluster = (ParentCluster*)itParent->second;
						re_parent_cluster->encode(numParentCluster, dvParentCluster->getRelativeIndex(parentCluster));
						re_parent_index_offset->encode(nodesPerCluster, parentIndex % nodesPerCluster);
					}
				}
				if(!ISLEAF(node))
				{
					nodeHashTable->insert(std::pair<unsigned int, unsigned int>(node->children >> 4, curNodeIdx));
					nodeHashTable->insert(std::pair<unsigned int, unsigned int>(node->children2 >> 4, curNodeIdx));
				}
			}
			nodeHashTable->clear();
		}
		while(dvParentCluster->size() > 0)
		{
			delete dvParentCluster->getAndRemoveFirstElement();
		}
		delete dvParentCluster;

		#ifdef STATISTICS
		numInterClusterNodes = parentClusterHashTable->size();
//		printf("[%d] numInterClusterNodes = %d\n", i, numInterClusterNodes);
		g_total_inter_cluster_nodes += numInterClusterNodes;
		if(g_max_inter_cluster_nodes < numInterClusterNodes)
			g_max_inter_cluster_nodes = numInterClusterNodes;
		g_num_parent_cluster += numParentCluster;
		#endif
		nodeHashTable->clear();
		delete parentClusterHashTable;

		unsigned int beforeOutClusterChildIndex = 0;
		unsigned int beforeTriID = 0;

		#if defined(USE_RACM) && defined(COMPRESS_TRI)
		int triIDCache[3];
		#endif

		for(int j=0;j<clusterSize;j++)
		{
			BSPArrayTreeNodePtr node = (BSPArrayTreeNodePtr)(clusterCache + j*sizeof(BSPArrayTreeNode));
			unsigned int curNodeIdx = i*nodesPerCluster+j;
			if(curNodeIdx == DUMMY_NODE) continue;

			//if(childCounts[curNodeIdx] < 14) continue;

			unsigned int parentIndex;
			BSPArrayTreeNode parentNodeInstance;
			BSPArrayTreeNodePtr parentNode = &parentNodeInstance;
			
			if(curNodeIdx == GETROOT())
			{
				// global root
				re_index->encode(dvNodeIndex->size()+1, 0);
			}
			else
			{
				bool isLocalRoot = false;
				NodeHashTableIterator it = nodeHashTable->find(curNodeIdx);
				if(it == nodeHashTable->end())
				{
					// current node is a root node
					
					re_index->encode(dvNodeIndex->size()+1, 0);
					#ifdef STATISTICS
					g_root_nodes_of_cluster++;
					#endif
					// local root
					#ifdef _USE_DUMMY_NODE
					_fseeki64(fpi, sizeof(unsigned int)*(__int64)(curNodeIdx-1), SEEK_SET);
					int r = fread(&parentIndex, sizeof(unsigned int), 1, fpi);
					parentIndex++;
					#else
					_fseeki64(fpi, sizeof(unsigned int)*(__int64)curNodeIdx, SEEK_SET);
					int r = fread(&parentIndex, sizeof(unsigned int), 1, fpi);
					#endif

					__int64 pos = _ftelli64(fpo);

					#ifdef _USE_DUMMY_NODE
					_fseeki64(fpo, sizeof(BSPArrayTreeNode)*(__int64)(parentIndex-1), SEEK_SET);
					#else
					_fseeki64(fpo, sizeof(BSPArrayTreeNode)*(__int64)parentIndex, SEEK_SET);
					#endif
					fread(parentNode, sizeof(BSPArrayTreeNode), 1, fpo);
					_fseeki64(fpo, pos, SEEK_SET);

					#ifdef _USE_DUMMY_NODE
					parentNode->children += 16;
					parentNode->children2 += 16;
					#endif
					if(!(parentNode->children >> 4 == curNodeIdx || parentNode->children2 >> 4 == curNodeIdx))
					{
						printf("wrong mapping! parent = %d, left = %d, right = %d, curNode = %d\n", parentIndex, parentNode->children>>4, parentNode->children2>>4, curNodeIdx);
					}
					isLocalRoot = true;
				}
				else
				{
					// current node is a non-root node
					parentIndex = it->second;
					parentNode = (BSPArrayTreeNodePtr)(clusterCache + (parentIndex%nodesPerCluster)*sizeof(BSPArrayTreeNode));

					FrontHashTableNodeIterator frontIt = frontHashTableNode->find(parentIndex);
					assert(frontIt != frontHashTableNode->end());
					FrontNode* front = frontIt->second;
					int frontIndex = dvNodeIndex->getRelativeIndex(front);
					re_index->encode(dvNodeIndex->size()+1, frontIndex+1);

					if(front->count == 0)
					{
						front->count = front->count+1;
					}
					else
					{
						dvNodeIndex->removeElement(front);
						delete front;
					}
				}

				I32 parentMinQ[3];
				I32 parentMaxQ[3];
				I32 predictedQ;
				pq->EnQuantize(parentNode->min.e, parentMinQ, FLOOR);
				pq->EnQuantize(parentNode->max.e, parentMaxQ, CEIL);
		
				int biggestaxis = getBiggestAxis(parentMinQ, parentMaxQ);
				int axis1;
				int axis2;
				switch(biggestaxis)
				{
				case 0 : axis1 = 1; axis2 = 2; break;
				case 1 : axis1 = 2; axis2 = 0; break;
				case 2 : axis1 = 0; axis2 = 1; break;
				}

				predictedQ = (parentMinQ[biggestaxis] + parentMaxQ[biggestaxis]) >> 1;

				Vector3 cMin, cMax;
				cMin = node->min;
				cMax = node->max;

				I32 qMin[3];
				I32 qMax[3];
				pq->EnQuantize(node->min.e, qMin, FLOOR);
				pq->EnQuantize(node->max.e, qMax, CEIL);

				static I32 adjustMin[3] = {0, 0, 0};
				static I32 adjustMax[3] = {0, 0, 0};
				if(parentNode->children >> 4 == curNodeIdx)
				{
					// this is a left child
					ic[0]->Compress(qMin[biggestaxis], parentMinQ[biggestaxis], 1);
					ic[0]->Compress(qMax[biggestaxis], predictedQ, 0);
					ic[1]->Compress(qMin[axis1], parentMinQ[axis1], 1);
					ic[1]->Compress(qMax[axis1], parentMaxQ[axis1], 0);
					ic[1]->Compress(qMin[axis2], parentMinQ[axis2], 1);
					ic[1]->Compress(qMax[axis2], parentMaxQ[axis2], 0);
				}
				else
				{
					// this is a right child
					ic[0]->Compress(qMin[biggestaxis], predictedQ, 1);
					ic[0]->Compress(qMax[biggestaxis], parentMaxQ[biggestaxis], 0);
					ic[1]->Compress(qMin[axis1], parentMinQ[axis1], 1);
					ic[1]->Compress(qMax[axis1], parentMaxQ[axis1], 0);
					ic[1]->Compress(qMin[axis2], parentMinQ[axis2], 1);
					ic[1]->Compress(qMax[axis2], parentMaxQ[axis2], 0);
				}

				if(parentNode->children >> 4 == curNodeIdx)
				{
					if((parentNode->children >> 4) + 1 == parentNode->children2 >> 4) contNodes++;
				}
				totalNodes++;
			}

			if(!ISLEAF(node))
			{
				nodeHashTable->insert(std::pair<unsigned int, unsigned int>(node->children >> 4, curNodeIdx));
				nodeHashTable->insert(std::pair<unsigned int, unsigned int>(node->children2 >> 4, curNodeIdx));
			}

			re_index_axis->encode(2, (node->children & 3) == 3);

			if(ISLEAF(node))
			{
				unsigned int TriID = node->indexOffset;

				#ifdef USE_TRI_3_TYPE_ENCODING
				// Compress triangle index
				#if defined(USE_RACM) && defined(COMPRESS_TRI)
				unsigned int NewTris [10];
				static int beforeTriID = TriID;

				bool encoded = false;
				unsigned int triCompType = 0;
				unsigned int Corner;
				unsigned int NextCorner;
				unsigned int PrevCorner;
				unsigned int Vertex;
				unsigned int NextVertex;
				unsigned int PrevVertex;
				int NewTriNum;
				unsigned int v1, v2;
				int type = 0;

				if(!isFirstTri)
				{
					// Try type 1 : cache
					int whichTri = -1;
					if(triIDCache[0] == TriID) whichTri = 0;
					if(triIDCache[1] == TriID) whichTri = 1;
					if(triIDCache[2] == TriID) whichTri = 2;
					if(whichTri >= 0)
					{
						re_index_tri_type->encode(3, 0);
						re_index_tri_cache->encode(3, whichTri);
						encoded = true;
						type = 1;

						#ifdef STATISTICS
						g_tri_index_type1++;
						#endif
					}

					// Try type 2 : front
					if(!encoded && 0)
					{
						FrontHashTableTriIterator it = frontHashTableTri->find(TriID);
						if(it != frontHashTableTri->end())
						{
 							//re_index_tri_type->encode(rmTriIndexType, 1);
							re_index_tri_type->encode(3, 1);

							int triIndex = dvTriIndexFront->getRelativeIndex(it->second);
							re_index_tri_front->encode(dvTriIndexFront->size()+1, triIndex);//icTriIndexFront->CompressNone(triIndex);// re_index_tri_front->encode(rmTriIndexFront, triIndex);
							encoded = true;
							type = 2;

							#ifdef STATISTICS
							g_tri_index_type2++;
							#endif
						}
					}
				}

				// type 3 : base + offset
				if(!encoded)
				{
					unsigned int base = TriID >> SIZE_BASE_PAGE_POWER;
					unsigned int offset = TriID & (SIZE_BASE_PAGE-1);
					BaseHashTableTriIterator it = baseHashTableTri->find(base);
					BaseTri *baseTri;
					re_index_tri_type->encode(3, 2);
					//re_index_tri_type->encode(rmTriIndexType, 2);
					type = 3;

					unsigned int baseIndex = 0;
					if(it == baseHashTableTri->end())
					{
						re_index_tri_base->encode(dvTriIndexBase->size()+1, baseIndex);//icTriIndexBase->CompressNone(baseIndex);
						//re_index_tri_base->encode(rmTriIndexBase, baseIndex);
						re_index_tri_base->encode(numClusters, base);

						baseTri = new BaseTri;
						baseTri->buffer_next = 0;
						baseTri->IsInDV = 0;
						baseTri->index = base;
						dvTriIndexBase->addElement(baseTri);
						baseHashTableTri->insert(std::pair<unsigned int, BaseTri*>(base, baseTri));
					}
					else
					{
						baseTri = it->second;
						baseIndex = dvTriIndexBase->getRelativeIndex(baseTri)+1;
						re_index_tri_base->encode(dvTriIndexBase->size()+1, baseIndex);//icTriIndexBase->CompressNone(baseIndex);
						//re_index_tri_base->encode(rmTriIndexBase, baseIndex);
					}
					re_index_tri_offset->encode(sizeBasePage, offset);
					encoded = true;

					#ifdef STATISTICS
					g_tri_index_type3++;
					#endif
				}

				// delete from front list
				FrontHashTableTriIterator it = frontHashTableTri->find(TriID);
				if(it != frontHashTableTri->end())
				{
					//unsigned int a = it->second;
					FrontTri *frontTri = (FrontTri*)it->second;
					dvTriIndexFront->removeElement(frontTri);
					frontHashTableTri->erase(it);
					delete frontTri;
				}
				Corner = g_pMesh->GetCornerFromTriID (TriID);
				NextCorner = g_pMesh->GetNextCornerGivenSameTri (Corner);
				PrevCorner = g_pMesh->GetPrevCornerGivenSameTri (Corner);

				Vertex = g_pMesh->GetIncidentVertexFromCorner (Corner);
				NextVertex = g_pMesh->GetIncidentVertexFromCorner (NextCorner);
				PrevVertex = g_pMesh->GetIncidentVertexFromCorner (PrevCorner);

				NewTriNum = 0;

				// insert to front list
				for(int c=0;c<3;c++)
				{
					triIDCache[c] = -1;
					switch(c)
					{
					case 0 : v1 = Vertex; v2 = NextVertex; break;
					case 1 : v1 = NextVertex; v2 = PrevVertex; break;
					case 2 : v1 = PrevVertex; v2 = Vertex; break;
					}
					if ((NewTriNum = g_pMesh->GetTrianglesSharingTwoVertices (v1, v2, NewTris, TriID, true)) == 1)
					{
						triIDCache[c] = NewTris[0];
						FrontHashTableTriIterator it = frontHashTableTri->find(NewTris[0]);
						if(it == frontHashTableTri->end())
						{
							FrontTri *frontTri = new FrontTri;
							frontTri->buffer_next = 0;
							frontTri->IsInDV = 0;
							frontTri->index = NewTris[0];
							dvTriIndexFront->addElement(frontTri);
							frontHashTableTri->insert(std::pair<unsigned int, FrontTri*>(NewTris[0], frontTri));
						}
					}
				}
				beforeTriID = TriID;
				isFirstTri = false;
				#else
				unsigned int base = TriID >> SIZE_BASE_PAGE_POWER;
				unsigned int offset = TriID & (SIZE_BASE_PAGE-1);
				BaseHashTableTriIterator it = baseHashTableTri->find(base);
				BaseTri *baseTri;

				unsigned int baseIndex = 0;
				if(it == baseHashTableTri->end())
				{
					re_index_tri_base->encode(dvTriIndexBase->size()+1, baseIndex);
					re_index_tri_base->encode(numClusters, base);

					baseTri = new BaseTri;
					baseTri->buffer_next = 0;
					baseTri->IsInDV = 0;
					baseTri->index = base;
					dvTriIndexBase->addElement(baseTri);
					baseHashTableTri->insert(std::pair<unsigned int, BaseTri*>(base, baseTri));
				}
				else
				{
					baseTri = it->second;
					baseIndex = dvTriIndexBase->getRelativeIndex(baseTri)+1;
					re_index_tri_base->encode(dvTriIndexBase->size()+1, baseIndex);
				}
				re_index_tri_offset->encode(sizeBasePage, offset);
//				re_index_offset->encodeInt(TriID);
				#endif
				#endif				

				#ifdef USE_TRI_DELTA_ENCODING
				int delta = TriID - beforeTriID;
				re_index_tri->encode(2, delta > 0);
				if(!encodeDeltaForChildIndexInOutCluster(delta > 0 ? delta : -delta, numBoundary, listBoundary, rmDeltaTriIndex, re_index_tri))
				{
					re_index_tri->encode((numNodes>>1)+1, TriID);
					#ifdef STATISTICS
					g_num_out_boundary_tri_index++;
					#endif
				}
				else
				{
					#ifdef STATISTICS
					g_num_in_boundary_tri_index++;
					#endif
				}
				beforeTriID = TriID;
				#endif
			}
			//if(childCounts[curNodeIdx] <= 14);
			else
			{
				int isChildInThis = 0;
				isChildInThis |= (curNodeIdx / nodesPerCluster == (node->children >> 4) / nodesPerCluster) << 1;
				isChildInThis |= (curNodeIdx / nodesPerCluster == (node->children2 >> 4) / nodesPerCluster);
				
				re_index_isChildInThis->encode(rmIsChildInThis, isChildInThis);
	
				ParentRootIndex *parentRootIndex;
				unsigned int delta = 0;
				switch(isChildInThis)
				{
				case 0 : 
					delta = (node->children >> 4) - beforeOutClusterChildIndex;
					if(!encodeDeltaForChildIndexInOutCluster(delta, numBoundary, listBoundary, rmDeltaChildIndex, re_index_outCluster))
					{
						re_index_outCluster->encode(numNodes, node->children >> 4);
						#ifdef STATISTICS
						g_num_out_boundary_child_index++;
						#endif
					}
					else
					{
						#ifdef STATISTICS
						g_num_in_boundary_child_index++;
						#endif
					}
					delta = (node->children2 >> 4) - (node->children >> 4);
					if(!encodeDeltaForChildIndexInOutCluster(delta, numBoundary, listBoundary, rmDeltaChildIndex, re_index_outCluster))
					{
						re_index_outCluster->encode(numNodes, node->children2 >> 4);
						#ifdef STATISTICS
						g_num_out_boundary_child_index++;
						#endif
					}
					else
					{
						#ifdef STATISTICS
						g_num_in_boundary_child_index++;
						#endif
					}
					beforeOutClusterChildIndex = node->children2 >> 4;
					#ifdef STATISTICS
					g_out_cluster_index++;
					g_out_cluster_index++;
					#endif
					break;
				case 1 :
					delta = (node->children >> 4) - beforeOutClusterChildIndex;
					if(!encodeDeltaForChildIndexInOutCluster(delta, numBoundary, listBoundary, rmDeltaChildIndex, re_index_outCluster))
					{
						re_index_outCluster->encode(numNodes, node->children >> 4);
						#ifdef STATISTICS
						g_num_out_boundary_child_index++;
						#endif
					}
					else
					{
						#ifdef STATISTICS
						g_num_in_boundary_child_index++;
						#endif
					}
					beforeOutClusterChildIndex = node->children >> 4;
					#ifdef STATISTICS
					g_out_cluster_index++;
					#endif
					break;
				case 2 :
					delta = (node->children2 >> 4) - beforeOutClusterChildIndex;
					if(!encodeDeltaForChildIndexInOutCluster(delta, numBoundary, listBoundary, rmDeltaChildIndex, re_index_outCluster))
					{
						re_index_outCluster->encode(numNodes, node->children2 >> 4);
						#ifdef STATISTICS
						g_num_out_boundary_child_index++;
						#endif
					}
					else
					{
						#ifdef STATISTICS
						g_num_in_boundary_child_index++;
						#endif
					}
					beforeOutClusterChildIndex = node->children2 >> 4;
					#ifdef STATISTICS
					g_out_cluster_index++;
					#endif
					break;
				}

				if(isChildInThis > 0)
				{
					FrontNode* frontNode = new FrontNode;
					frontNode->buffer_next = 0;
					frontNode->IsInDV = 0;
					frontNode->index = curNodeIdx;
					frontNode->count = isChildInThis == 3 ? 0 : 1;

					dvNodeIndex->addElement(frontNode);

					frontHashTableNode->insert(std::pair<int, FrontNode*>(curNodeIdx, frontNode));
				}
			}

			#ifdef STATISTICS
			g_front_size += dvNodeIndex->size();
			unsigned int ss = dvNodeIndex->size();
			g_front_size_count++;
			#endif
		}

		#ifdef STATISTICS
		g_avg_front_size_per_cluster += g_front_size / (float)clusterSize;
		g_front_size_count = 0;
		g_front_size = 0;
		#endif

		if(i % 1000 == 0 && i > 0)
		{
			printf("[%d] Clusters done\n", i);
		}

		delete frontHashTableNode;
		while(dvNodeIndex->size() > 0)
		{
			delete dvNodeIndex->getAndRemoveFirstElement();
		}
		delete dvNodeIndex;

		#if defined(USE_RACM) && defined(COMPRESS_TRI)
		//delete rmTriIndexType;
		while(dvTriIndexFront->size() > 0)
		{
			delete dvTriIndexFront->getAndRemoveFirstElement();
		}
		while(dvTriIndexBase->size() > 0)
		{
			delete dvTriIndexBase->getAndRemoveFirstElement();
		}
		delete dvTriIndexFront;
		delete dvTriIndexBase;
		delete frontHashTableTri;
		delete baseHashTableTri;
		#else
		while(dvTriIndexBase->size() > 0)
		{
			delete dvTriIndexBase->getAndRemoveFirstElement();
		}
		delete dvTriIndexBase;
		delete baseHashTableTri;
		#endif
		
		delete rmIsChildInThis;
		for(int k=0;k<numBoundary;k++)
		{
			delete rmDeltaChildIndex[k];
			delete rmDeltaTriIndex[k];
		}
		delete rmDeltaChildIndex;
		delete rmDeltaTriIndex;
		delete clusterCache;
		delete nodeHashTable;

		#ifndef STATISTICS
		re_geom->done();
		delete re_geom;
		#endif
		#ifdef STATISTICS
		re_geom->done();
		re_geom_raw->done();
		re_geom_error->done();
		re_index->done();
		re_parent_cluster->done();
		re_parent_index_offset->done();
		re_index_axis->done();
		re_index_isChildInThis->done();
		re_index_outCluster->done();
		re_index_offset->done();
		#ifdef USE_TRI_3_TYPE_ENCODING
		re_index_tri_type->done();
		re_index_tri_cache->done();
		re_index_tri_front->done();
		re_index_tri_base->done();
		re_index_tri_offset->done();
		#endif
		#ifdef USE_TRI_DELTA_ENCODING
		re_index_tri->done();
		#endif
		re_etc->done();
		re_debug->done();
		g_geom += re_geom->getNumberBits();
		g_geom_raw += re_geom_raw->getNumberBits();
		g_geom_error += re_geom_error->getNumberBits();
		g_index += re_index->getNumberBits();
		g_parent_cluster += re_parent_cluster->getNumberBits();
		g_parent_index_offset += re_parent_index_offset->getNumberBits();
		g_index_axis += re_index_axis->getNumberBits();
		g_index_isChildInThis += re_index_isChildInThis->getNumberBits();
		g_index_outCluster += re_index_outCluster->getNumberBits();
		g_index_offset += re_index_offset->getNumberBits();
		#ifdef USE_TRI_3_TYPE_ENCODING
		g_index_tri_type += re_index_tri_type->getNumberBits();
		g_index_tri_cache += re_index_tri_cache->getNumberBits();
		g_index_tri_front += re_index_tri_front->getNumberBits();
		g_index_tri_base += re_index_tri_base->getNumberBits();
		g_index_tri_offset += re_index_tri_offset->getNumberBits();
		#endif
		#ifdef USE_TRI_DELTA_ENCODING
		g_index_tri += re_index_tri->getNumberBits();
		#endif
		g_etc += re_etc->getNumberBits();
		g_debug += re_debug->getNumberBits();
		delete re_geom;
		delete re_geom_raw;
		delete re_geom_error;
		delete re_index;
		delete re_parent_cluster;
		delete re_parent_index_offset;
		delete re_index_axis;
		delete re_index_isChildInThis;
		delete re_index_outCluster;
		delete re_index_offset;
		#ifdef USE_TRI_3_TYPE_ENCODING
		delete re_index_tri_type;
		delete re_index_tri_cache;
		delete re_index_tri_front;
		delete re_index_tri_base;
		delete re_index_tri_offset;
		#endif
		#ifdef USE_TRI_DELTA_ENCODING
		delete re_index_tri;
		#endif
		delete re_etc;
		delete re_debug;
		#endif

		ic[0]->FinishCompressor();
		ic[1]->FinishCompressor();

		delete ic[0];
		delete ic[1];
		delete pq;
	}

	fseek(fpc, posOffset, SEEK_SET);
	fwrite(offsets, sizeof(long), numClusters, fpc);
	fseek(fpc, 0, SEEK_END);

	delete offsets;
	delete listBoundary;

	fclose(fpo);
	fclose(fpc);
	fclose(fpi);

	#ifdef USE_RACM
	delete g_pMesh;
	#endif

	#ifdef STATISTICS
	fclose(fpTestAllErrors);
	fclose(fpTestMinErrors);
	fclose(fpTestMaxErrors);
	fclose(fpTestBiggestAxisErrors);
	fclose(fpTestOtherAxisErrors);
	#endif

	#ifdef STATISTICS
	float g_total_geom =
//		g_geom_raw + 
		g_geom_error
		;
	float g_total_index = 
		g_index + 
		g_parent_cluster +
		g_parent_index_offset +
		g_index_axis +
		g_index_isChildInThis +
		g_index_outCluster +
		g_index_offset +
		#ifdef USE_TRI_3_TYPE_ENCODING
		g_index_tri_type +
		g_index_tri_cache +
		g_index_tri_front +
		g_index_tri_base +
		g_index_tri_offset
		#endif
		#ifdef USE_TRI_DELTA_ENCODING
		g_index_tri
		#endif
		;
	float g_parent_index = 
		g_parent_cluster +
		g_parent_index_offset
		;

	printf("---------------------------------\n");
	printf("Number of nodes : %d\n", g_total_node);
	printf("Total : %f bpn\n", (g_total_geom + g_total_index + g_etc + g_debug)/g_total_node);
	printf("Geom : %f bpn\n", g_total_geom/g_total_node);
	printf("Index : %f bpn\n", g_total_index/g_total_node);
	printf("---------------------------------\n");
	printf("Geom\n");
//	printf(" Raw data : %f bpn\n", g_geom_raw/g_total_node);
	printf(" Error : %f bpn\n", g_geom_error/g_total_node);
	printf("---------------------------------\n");
	printf("Index\n");
	printf(" Front index : %f bpn\n", g_index/g_total_node);
	printf(" LeafOrNot : %f bpn\n", g_index_axis/g_total_node);
	printf(" Is child in this cluster : %f bpn\n", g_index_isChildInThis/g_total_node);
	printf(" Out cluster index : %f bpn\n", g_index_outCluster/g_total_node);
	printf(" Parent Index : %f bpn\n", g_parent_index/g_total_node);
	printf("  Parent Cluster Number : %f bpn\n", g_parent_cluster/g_total_node);
	printf("  Parent Index Offset : %f bpn\n", g_parent_index_offset/g_total_node);
	#ifdef USE_TRI_3_TYPE_ENCODING
	printf(" Triangle index : %f bpn\n", (g_index_tri_type + g_index_tri_cache + g_index_tri_front + g_index_tri_base + g_index_tri_offset)/g_total_node);
	printf(" Triangle index type : %f bpn\n", g_index_tri_type/g_total_node);
	printf(" Triangle index cache : %f bpn\n", g_index_tri_cache/g_total_node);
	printf(" Triangle index front : %f bpn\n", g_index_tri_front/g_total_node);
	printf(" Triangle index base : %f bpn\n", g_index_tri_base/g_total_node);
	printf(" Triangle index offset : %f bpn\n", g_index_tri_offset/g_total_node);
	printf(" Number of triangle index type1(cache) : %d\n", g_tri_index_type1);
	printf(" Number of triangle index type2(front) : %d\n", g_tri_index_type2);
	printf(" Number of triangle index type3(base+offset) : %d\n", g_tri_index_type3);
	#endif
	#ifdef USE_TRI_DELTA_ENCODING
	printf(" Triangle index : %f bpn\n", g_index_tri/g_total_node);
	#endif
//	printf("Number of root nodes of clusters : %d\n", g_root_nodes_of_cluster);
//	printf("Number of out cluster indices : %d\n", g_out_cluster_index);
//	printf("Max number of inter cluster nodes : %d\n", g_max_inter_cluster_nodes);
	printf("---------------------------------\n");
//	printf("Maximum parent root hash size : %d\n", g_max_parent_root_hash);
//	printf("average number of inter cluster nodes per cluster : %f\n", ((float)g_total_inter_cluster_nodes)/numClusters);
//	printf("Debug : %f bpn\n", g_debug/g_total_node);
	printf("Average size of front : %f\n", g_avg_front_size_per_cluster/numClusters);
	printf("Average number of parent clusters per cluster : %f\n", float(g_num_parent_cluster)/numClusters);
	printf("Percentage of in-boundary delta child index : %.1f%%\n", float(g_num_in_boundary_child_index)/(float(g_num_in_boundary_child_index) + float(g_num_out_boundary_child_index))*100.0f);
	printf("Percentage of in-boundary delta tri index : %.1f%%\n", float(g_num_in_boundary_tri_index)/(float(g_num_in_boundary_tri_index) + float(g_num_out_boundary_tri_index))*100.0f);
	printf("Etc : %f bpn\n", g_etc/g_total_node);
	#endif
	_CrtDumpMemoryLeaks();
	return 0;
}

int BVHCompression::encodeDeltaForChildIndexInOutCluster(unsigned int delta, unsigned int numBoundary, unsigned int *listBoundary, RangeModel **rmDeltaChildIndex, RangeEncoder *re)
{
	for(unsigned int pass=0;pass<numBoundary;pass++)
	{
		if(delta < listBoundary[pass])
		{
			re->encode(rmDeltaChildIndex[pass], delta);
			return TRUE;
		}
		re->encode(rmDeltaChildIndex[pass], listBoundary[pass]);
	}
	return FALSE;
}

void BVHCompression::getAllChildCount(FILE *fp)
{
	__int64 posOrigin = _ftelli64(fp);

	getAllChildCountRec(fp, 0);

	_fseeki64(fp, posOrigin, SEEK_SET);
}

void BVHCompression::getAllChildCountRec(FILE *fp, unsigned int nodeIndex)
{
	BSPArrayTreeNode node;
	_fseeki64(fp, (__int64)nodeIndex*sizeof(BSPArrayTreeNode), SEEK_SET);
	fread(&node, sizeof(BSPArrayTreeNode), 1, fp);

	if(!ISLEAF(&node))
	{
		unsigned int left = node.children >> 4;
		unsigned int right = node.children2 >> 4;
		if(childCounts[left] == (unsigned int)-1)
		{
			getAllChildCountRec(fp, left);
		}
		if(childCounts[right] == (unsigned int)-1)
		{
			getAllChildCountRec(fp, right);
		}
		childCounts[nodeIndex] = childCounts[left] + childCounts[right] + 2;
	}
	else
	{
		childCounts[nodeIndex] = 0;
	}

}