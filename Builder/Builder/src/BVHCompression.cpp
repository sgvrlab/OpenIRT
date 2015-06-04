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

#include "Files.h"
#include "BVHCompression.h"

#ifdef USE_RACM
#include "compressed_mesh.h"
CMeshAbstract *g_pMesh;
#endif

#include "dynamicvector.h"

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

void BVHCompression::storeParentIndex(FILE *fpo, FILE *fpi, unsigned int index, unsigned int numNodes)
{
	BSPArrayTreeNode node;
	_fseeki64(fpo, (__int64)index*sizeof(BSPArrayTreeNode), SEEK_SET);
	fread(&node, sizeof(BSPArrayTreeNode), 1, fpo);
	if(ISLEAF(&node)) return;
	unsigned int left = node.children >> 2;
	#ifndef _USE_CONTI_NODE
	unsigned int right = node.children2 >> 2;
	#else
	unsigned int right = (node.children >> 2) + 1;
	#endif
	_fseeki64(fpi, (__int64)left*sizeof(unsigned int), SEEK_SET);
	fwrite(&index, sizeof(unsigned int), 1, fpi);
	_fseeki64(fpi, (__int64)right*sizeof(unsigned int), SEEK_SET);
	fwrite(&index, sizeof(unsigned int), 1, fpi);
	static unsigned int	processedNodes = 0;
	static unsigned int	beforeStep = 0;

	processedNodes += 2;
	if( processedNodes*100/numNodes != beforeStep)
	{
		beforeStep = processedNodes*100/numNodes;
		cout << beforeStep << "% ";
	}

	storeParentIndex(fpo, fpi, left, numNodes);
	storeParentIndex(fpo, fpi, right, numNodes);
}

void BVHCompression::storeParentIndex(FILE *fpo, FILE *fpi, unsigned int numNodes)
{
	__int64 posFpo = _ftelli64(fpo);
	__int64 posFpi = _ftelli64(fpi);
	storeParentIndex(fpo, fpi, 0, numNodes);
	_fseeki64(fpo, posFpo, SEEK_SET);
	_fseeki64(fpi, posFpi, SEEK_SET);
	cout << "Storing parent indices completed. " << endl;
}

int BVHCompression::compressQBVH(const char* filepath)
{
	char filename[MAX_PATH];
	sprintf(filename, "%s/BVH", filepath);
	char nodeName[MAX_PATH], compressedName[MAX_PATH], QBVHDir[MAX_PATH];
	FILE *fpo;
	FILE *fpc;
	BSPArrayTreeNode currentNode;

	sprintf(nodeName, "%s.node", filename);
	sprintf(QBVHDir, "%s/QBVH", filepath);
	if(!dirExists(QBVHDir))
	{
		mkdir(QBVHDir);
	}
	sprintf(compressedName, "%s/BVH.qtz", QBVHDir);

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
		#ifndef _USE_CONTI_NODE
		qNode.children2 = node.children2;
		#else
		qNode.lodindex = node.lodindex;
		#endif
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
int BVHCompression::compressgzipBVH(const char* filepath)
{
	OptionManager *opt = OptionManager::getSingletonPtr();
	unsigned int numNodes = 0;
	unsigned int numClusters = 0;
	unsigned int nodesPerCluster = opt->getOptionAsInt("raytracing", "nodesPerCluster", 16384);
	unsigned int maxDepth = opt->getOptionAsInt("raytracing", "maxBSPTreeDepth", 100);
	int maxNumTrisPerLeaf = opt->getOptionAsInt("raytracing", "MaxNumTrisPerLeaf", 1);

	char filename[MAX_PATH];
	sprintf(filename, "%s/BVH", filepath);
	char nodeName[MAX_PATH], compressedName[MAX_PATH], gzipBVHDir[MAX_PATH], headerName[MAX_PATH];
	FILE *fpo;
	gzFile fpc, fph;
	BSPArrayTreeNode currentNode;

	sprintf(nodeName, "%s.node", filename);
	sprintf(gzipBVHDir, "%s/gzipBVH", filepath);

	if(!dirExists(gzipBVHDir))
	{
		mkdir(gzipBVHDir);
	}
	sprintf(compressedName, "%s/BVH.cgz", gzipBVHDir);
	sprintf(headerName, "%s/BVH.gzh", gzipBVHDir);

	cout << 0 << endl;
	cout << filename << endl;

	fpo = fopen(nodeName, "rb");
	fpc = gzopen(compressedName, "wb");
	fph = gzopen(headerName, "wb");

	numNodes = _filelengthi64(fileno(fpo))/sizeof(BSPArrayTreeNode);
	numClusters = ceil(((float)numNodes)/nodesPerCluster);
	unsigned int maxNumTris = numNodes*maxNumTrisPerLeaf;

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
	gzwrite(fpc, &maxNumTris, sizeof(unsigned int));
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
	#ifdef _USE_BIT_ENCODER
	unsigned int numBoundary = opt->getOptionAsInt("raytracing", "numBoundaryBits", 1);
	const char *listBoundaryConstTemp = opt->getOption("raytracing", "listBoundaryBits", "10");
	#else
	unsigned int numBoundary = opt->getOptionAsInt("raytracing", "numBoundary", 1);
	const char *listBoundaryConstTemp = opt->getOption("raytracing", "listBoundary", "10");
	#endif
	unsigned int maxNumTrisPerLeaf = opt->getOptionAsInt("raytracing", "MaxNumTrisPerLeaf", 1);
	char listBoundaryTemp[1024];
	strcpy(listBoundaryTemp, listBoundaryConstTemp);

	unsigned int *listBoundary = new unsigned int[numBoundary];
	unsigned int *listBoundaryBits = new unsigned int[numBoundary];

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
		listBoundaryBits[i] = getBits(listBoundary[i]);
	}

	char filename[MAX_PATH];
	sprintf(filename, "%s/BVH", filepath);
	char nodeName[MAX_PATH], compressedName[MAX_PATH], parentIndexName[MAX_PATH];
	char childCountName[MAX_PATH];
	FILE *fpo, *fpc, *fpi;
	BSPArrayTreeNode currentNode;

	sprintf(nodeName, "%s.node", filename);
	#ifdef STATISTICS
	sprintf(compressedName, "%s.stat", filename);
	#else
	sprintf(compressedName, "%s.cmp", filename);
	#endif
	sprintf(parentIndexName, "%s.pi", filename);

	#ifdef USE_RACM
	sprintf(filename, "%s/RACM", filepath);
	g_pMesh = new CCompressedMesh;
	g_pMesh->PrepareData(filename ,true);
	#endif


	fpo = fopen(nodeName, "rb");
	fpc = fopen(compressedName, "wb");
	fpi = fopen(parentIndexName, "rb");

	numNodes = _filelengthi64(fileno(fpo))/sizeof(BSPArrayTreeNode);
	numClusters = ceil(((float)numNodes)/nodesPerCluster);
	unsigned int maxNumTris = numNodes*maxNumTrisPerLeaf;

	if(fpi == NULL)
	{
		cout << "Store parent indices." << endl;
		fpi = fopen(parentIndexName, "wb");

		// first, store parent index of each nodes.
		storeParentIndex(fpo, fpi, numNodes);

		fclose(fpi);
		fpi = fopen(parentIndexName, "rb");
	}
	else
	{
		cout << "Parent indices already exist. Skip" << endl;
	}





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
	fwrite(&maxNumTris, sizeof(unsigned int), 1, fpc);
	fwrite(&numClusters, sizeof(unsigned int), 1, fpc);
	fwrite(bb_min_f, sizeof(float), 3, fpc);
	fwrite(bb_max_f, sizeof(float), 3, fpc);
	fwrite(&numBoundary, sizeof(unsigned int), 1, fpc);
	fwrite(listBoundary, sizeof(unsigned int), numBoundary, fpc);

	unsigned int nodesPerClusterPower = log((double)nodesPerCluster)/log(2.0);
	unsigned int bitsNumNodes = getBits(numNodes-1);
	unsigned int bitsNodesPerCluster = getBits(nodesPerCluster-1);
	unsigned int bitsNumClusters = getBits(numClusters-1);

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
	#if COMPRESS_TYPE == COMPRESS_TYPE_STATIC_DICTIONARY
	float g_avg_dic_table_size = 0;
	#endif
	_int64 g_zero_error = 0;
	#endif

	#ifdef STATISTICS
	FILE *fpTestAllErrors = fopen("all_errors", "w");
	FILE *fpTestMinErrors = fopen("min_errors", "w");
	FILE *fpTestMaxErrors = fopen("max_errors", "w");
	FILE *fpTestBiggestAxisErrors = fopen("biggest_axis_errors", "w");
	FILE *fpTestOtherAxisErrors = fopen("other_axis_errors", "w");
	#endif

	BSPArrayTreeNode mCurrentNode;
	int numProccessedClusters = 0;
	// 1000 for statistics
	Stopwatch tLoadingTest("1000 clusters loading");
	tLoadingTest.Start();

	for(int i=0;i<numClusters;i++)
	{
		if(numProccessedClusters == 1000)
		{
			tLoadingTest.Stop();
			cout << tLoadingTest << endl;
		}		offsets[i] = ftell(fpc);

		Encoder *e_geom;
		Encoder *e_geom_raw;
		Encoder *e_geom_error;
		Encoder *e_index;
		Encoder *e_parent_cluster;
		Encoder *e_parent_index_offset;
		Encoder *e_index_axis;
		Encoder *e_index_isChildInThis;
		Encoder *e_index_outCluster;
		Encoder *e_index_offset;
		#ifdef USE_TRI_3_TYPE_ENCODING
		Encoder *e_index_tri_type;
		Encoder *e_index_tri_cache;
		Encoder *e_index_tri_front;
		Encoder *e_index_tri_base;
		Encoder *e_index_tri_offset;
		#endif
		#ifdef USE_TRI_DELTA_ENCODING
		Encoder *e_index_tri;
		#endif
		Encoder *e_etc;
		Encoder *e_debug;
		#ifndef STATISTICS
		e_geom = new Encoder(fpc, 0);
		e_geom_raw = e_geom;
		e_geom_error = e_geom;
		e_index = e_geom;
		e_parent_cluster = e_geom;
		e_parent_index_offset = e_geom;
		e_index_axis = e_geom;
		e_index_isChildInThis = e_geom;
		e_index_outCluster = e_geom;
		e_index_offset = e_geom;
		#ifdef USE_TRI_3_TYPE_ENCODING
		e_index_tri_type = e_geom;
		e_index_tri_cache = e_geom;
		e_index_tri_front = e_geom;
		e_index_tri_base = e_geom;
		e_index_tri_offset = e_geom;
		#endif
		#ifdef USE_TRI_DELTA_ENCODING
		e_index_tri = e_geom;
		#endif
		e_etc = e_geom;
		e_debug = e_geom;
		#endif
		#ifdef STATISTICS
		e_geom = new Encoder((FILE*)0, false);
		e_geom_raw = new Encoder((FILE*)0, false);
		e_geom_error = new Encoder((FILE*)0, false);
		e_index = new Encoder((FILE*)0, false);
		e_parent_cluster = new Encoder((FILE*)0, false);
		e_parent_index_offset = new Encoder((FILE*)0, false);
		e_index_axis = new Encoder((FILE*)0, false);
		e_index_isChildInThis = new Encoder((FILE*)0, false);
		e_index_outCluster = new Encoder((FILE*)0, false);
		e_index_offset = new Encoder((FILE*)0, false);
		#ifdef USE_TRI_3_TYPE_ENCODING
		e_index_tri_type = new Encoder(0, false);
		e_index_tri_cache = new Encoder(0, false);
		e_index_tri_front = new Encoder(0, false);
		e_index_tri_base = new Encoder(0, false);
		e_index_tri_offset = new Encoder(0, false);
		#endif
		#ifdef USE_TRI_DELTA_ENCODING
		e_index_tri = new Encoder((FILE*)0, false);
		#endif
		e_etc = new Encoder((FILE*)0, false);
		e_debug = new Encoder((FILE*)0, false);
		#endif

		int nbits = 16;

		PositionQuantizerNew* pq;
		#ifndef _USE_BIT_ENCODER
		RangeModel **rmDeltaChildIndex = new RangeModel*[numBoundary];
		RangeModel **rmDeltaTriIndex = new RangeModel*[numBoundary];
		RangeModel *rmIsChildInThis;
		#endif
		pq = new PositionQuantizerNew();
		pq->SetMinMax(bb_min_f, bb_max_f);
		pq->SetPrecision(nbits);
		pq->SetupQuantizer();
		#if defined(USE_RACM) && defined(COMPRESS_TRI)
		//RangeModel *rmTriIndexType;
		#endif

		#if COMPRESS_TYPE == COMPRESS_TYPE_ARITHMETIC
		IntegerCompressorRACBVH* ic[2];

		ic[0] = new IntegerCompressorRACBVH();
		ic[1] = new IntegerCompressorRACBVH();

		I32 maxRange = pq->m_aiQuantRange[0] > pq->m_aiQuantRange[1] ? pq->m_aiQuantRange[0] : pq->m_aiQuantRange[1];
		maxRange = maxRange > pq->m_aiQuantRange[2] ? maxRange : pq->m_aiQuantRange[2];
		ic[0]->SetRange(maxRange);
		ic[1]->SetRange(maxRange);

		ic[0]->SetPrecision(nbits);
		ic[1]->SetPrecision(nbits);

		ic[0]->SetupCompressor(e_geom_error);
		ic[1]->SetupCompressor(e_geom_error);
		#endif

		typedef stdext::hash_map<unsigned int, unsigned int> NodeHashTable;
		typedef NodeHashTable::iterator NodeHashTableIterator;
		NodeHashTable *nodeHashTable = new NodeHashTable;
		char* clusterCache = new char[nodesPerCluster*sizeof(BSPArrayTreeNode)];
		#ifdef _USE_DUMMY_NODE
		unsigned int clusterSize = fread(clusterCache+sizeof(BSPArrayTreeNode)*(i==DUMMY_NODE), sizeof(BSPArrayTreeNode), nodesPerCluster-(i==DUMMY_NODE), fpo);
		if(i==0) clusterSize++;
		for(int j=0;j<clusterSize;j++)
		{
			BSPArrayTreeNodePtr node = (BSPArrayTreeNodePtr)(clusterCache + j*sizeof(BSPArrayTreeNode));
			if(i==0 && j==0) 
			{
				continue;
			}
			if(ISLEAF(node)) continue;
			node->children += 4;
			#ifndef _USE_CONTI_NODE
			node->children2 += 4;
			#endif
		}
		#else
		unsigned int clusterSize = fread(clusterCache, sizeof(BSPArrayTreeNode), nodesPerCluster, fpo);
		#endif
		#ifdef STATISTICS
		g_total_node += clusterSize;
		#endif
		e_etc->encodeInt(clusterSize);

		#ifndef _USE_BIT_ENCODER
		for(int j=0;j<numBoundary;j++)
		{
			rmDeltaChildIndex[j] = new RangeModel(listBoundary[j]+1, 0, TRUE);
			rmDeltaTriIndex[j] = new RangeModel(listBoundary[j]+1, 0, TRUE);
		}
		#endif
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
		#ifndef _USE_BIT_ENCODER
		rmIsChildInThis = new RangeModel(4, 0, TRUE);
		#endif

		DynamicVector *dvNodeIndex = new DynamicVector();
		unsigned int bitsSizeDvNodeIndex = 0;
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
		unsigned int bitsNumParentCluster = 0;

		// find inter-cluster nodes in advance.
		DynamicVector *dvParentCluster = new DynamicVector();
		for(int pass=0;pass<3;pass++)
		{
			if(pass == 1)
			{
				numParentCluster = dvParentCluster->size();
				#ifdef _USE_BIT_ENCODER
				e_parent_cluster->encode(bitsNumClusters, numParentCluster);
				bitsNumParentCluster = numParentCluster == 0 ? 0 : getBits(numParentCluster-1);
				#else
				e_parent_cluster->encode(numClusters, numParentCluster);
				#endif
				unsigned int parentCN = 0;
				for(int k=0;k<dvParentCluster->size();k++)
				{
					parentCN = ((ParentCluster*)dvParentCluster->getElementWithRelativeIndex(k))->parentCN;
					#ifdef _USE_BIT_ENCODER
					e_parent_cluster->encode(bitsNumClusters, parentCN);
					#else
					e_parent_cluster->encode(numClusters, parentCN);
					#endif
				}
				#ifdef _USE_BIT_ENCODER
				e_parent_cluster->encode(bitsNodesPerCluster, numLocalRoot);
				#else
				e_parent_cluster->encode(nodesPerCluster, numLocalRoot);
				#endif
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
						#ifdef _USE_BIT_ENCODER
						e_parent_cluster->encode(bitsNumParentCluster, dvParentCluster->getRelativeIndex(parentCluster));
						e_parent_index_offset->encode(bitsNodesPerCluster, parentIndex % nodesPerCluster);
						#else
						e_parent_cluster->encode(numParentCluster, dvParentCluster->getRelativeIndex(parentCluster));
						e_parent_index_offset->encode(nodesPerCluster, parentIndex % nodesPerCluster);
						#endif
					}
				}
				if(!ISLEAF(node))
				{
					nodeHashTable->insert(std::pair<unsigned int, unsigned int>(node->children >> 2, curNodeIdx));
					#ifndef _USE_CONTI_NODE
					nodeHashTable->insert(std::pair<unsigned int, unsigned int>(node->children2 >> 2, curNodeIdx));
					#else
					nodeHashTable->insert(std::pair<unsigned int, unsigned int>((node->children >> 2) + 1, curNodeIdx));
					#endif
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

		#if COMPRESS_TYPE == COMPRESS_TYPE_STATIC_DICTIONARY
		typedef stdext::hash_map<unsigned int, unsigned int> DicHashTable;
		typedef DicHashTable::iterator DicHashTableIterator;
		DicHashTable *dicHashTable = new DicHashTable;
		unsigned int bitsSizeDicHashTable = 0;
		#endif

		#if COMPRESS_TYPE == COMPRESS_TYPE_STATIC_DICTIONARY_NEW
		StaticDictionary dicError(16), dicDeltaOutCluster(28);//, dicDeltaTri(27);
		dicError.setEncoder(e_geom_error);
		//dicDeltaTri.setEncoder(e_index_tri);
		dicDeltaOutCluster.setEncoder(e_index_outCluster);
		#endif

		#if COMPRESS_TYPE == COMPRESS_TYPE_QIC_LZW
		// Build dictionary table
		SimpleDictionary dicError, dicIdxTri, dicIdxOutCluster;
		dicError.setEncoder(e_geom_error);
		dicIdxTri.setEncoder(e_index_tri);
		dicIdxOutCluster.setEncoder(e_index_outCluster);

		for(int j=0;j<clusterSize;j++)
		{
			BSPArrayTreeNodePtr node = (BSPArrayTreeNodePtr)(clusterCache + j*sizeof(BSPArrayTreeNode));
			unsigned int curNodeIdx = i*nodesPerCluster+j;
			if(curNodeIdx == DUMMY_NODE) continue;

			unsigned int parentIndex;
			BSPArrayTreeNode parentNodeInstance;
			BSPArrayTreeNodePtr parentNode = &parentNodeInstance;

			if(curNodeIdx == GETROOT())
			{
				// global root
			}
			else
			{
				bool isLocalRoot = false;
				NodeHashTableIterator it = nodeHashTable->find(curNodeIdx);
				if(it == nodeHashTable->end())
				{
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
					parentNode->children += 4;
					parentNode->children2 += 4;
					#endif
					if(!(parentNode->children >> 2 == curNodeIdx || parentNode->children2 >> 2 == curNodeIdx))
					{
						printf("wrong mapping! parent = %d, left = %d, right = %d, curNode = %d\n", parentIndex, parentNode->children>>2, parentNode->children2>>2, curNodeIdx);
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

					if(front->count == 0)
					{
						front->count = front->count+1;
					}
					else
					{
						frontHashTableNode->erase(frontIt);
						maxFrontSize = max(maxFrontSize, frontHashTableNode->size());
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

				if(parentNode->children >> 2 == curNodeIdx)
				{
					// this is a left child
					dicError.push(qMin[biggestaxis] - parentMinQ[biggestaxis]);
					dicError.push(qMin[axis1] - parentMinQ[axis1]);
					dicError.push(qMin[axis2] - parentMinQ[axis2]);
					dicError.push(predictedQ - qMax[biggestaxis]);
					dicError.push(parentMaxQ[axis1] - qMax[axis1]);
					dicError.push(parentMaxQ[axis2] - qMax[axis2]);
				}
				else
				{
					// this is a right child
					dicError.push(qMin[biggestaxis] - predictedQ);
					dicError.push(parentMaxQ[biggestaxis] - qMax[biggestaxis]);
					dicError.push(qMin[axis1] - parentMinQ[axis1]);
					dicError.push(qMin[axis2] - parentMinQ[axis2]);
					dicError.push(parentMaxQ[axis1] - qMax[axis1]);
					dicError.push(parentMaxQ[axis2] - qMax[axis2]);
				}
			}

			if(!ISLEAF(node))
			{
				nodeHashTable->insert(std::pair<unsigned int, unsigned int>(node->children >> 2, curNodeIdx));
				nodeHashTable->insert(std::pair<unsigned int, unsigned int>(node->children2 >> 2, curNodeIdx));
			}

			if(ISLEAF(node))
			{
				unsigned int TriID = node->indexOffset;

				#ifdef USE_TRI_DELTA_ENCODING
				int delta = TriID - beforeTriID;

				dicIdxTri.push(delta);

				beforeTriID = TriID;
				#endif
			}
			else
			{
				int isChildInThis = 0;
				isChildInThis |= (curNodeIdx / nodesPerCluster == (node->children >> 2) / nodesPerCluster) << 1;
				isChildInThis |= (curNodeIdx / nodesPerCluster == (node->children2 >> 2) / nodesPerCluster);

				ParentRootIndex *parentRootIndex;
				unsigned int delta = 0;
				if(isChildInThis == 0)
				{
					delta = (node->children >> 2) - beforeOutClusterChildIndex;

					dicIdxOutCluster.push(delta);

					delta = (node->children2 >> 2) - (node->children >> 2);

					dicIdxOutCluster.push(delta);

					beforeOutClusterChildIndex = node->children2 >> 2;
				}
				if(isChildInThis > 0)
				{
					FrontNode* frontNode = new FrontNode;
					frontNode->buffer_next = 0;
					frontNode->IsInDV = 0;
					frontNode->index = curNodeIdx;
					frontNode->count = isChildInThis == 3 ? 0 : 1;

					frontHashTableNode->insert(std::pair<int, FrontNode*>(curNodeIdx, frontNode));
					maxFrontSize = max(maxFrontSize, frontHashTableNode->size());
				}
			}
		}

		dicError.buildDictionary();
		dicError.sortByFrequency();
		dicError.setSizeDicTableByFreq(0.99f);
		//dicError.setSizeDicTable(64);
		//dicError.setBitsOfMaxValue(16);

		dicIdxTri.buildDictionary();
		dicIdxTri.sortByFrequency();
		dicIdxTri.setSizeDicTableByFreq(0.99f);
		//dicIdxTri.setSizeDicTable(64);
		//dicIdxTri.setBitsOfMaxValue(16);

		dicIdxOutCluster.buildDictionary();
		dicIdxOutCluster.sortByFrequency();
		dicIdxOutCluster.setSizeDicTableByFreq(0.99f);
		//dicIdxOutCluster.setSizeDicTable(64);
		//dicIdxOutCluster.setBitsOfMaxValue(16);

		dicError.encodeDicTable();
		dicIdxTri.encodeDicTable();		
		dicIdxOutCluster.encodeDicTable();

		//LZW dicAxis(2), dicIdxIsChildInThis(2), dicFrontIndex(maxFrontSize+1);
		//dicAxis.setEncoder(e_index_axis);
		//dicIdxIsChildInThis.setEncoder(e_index_isChildInThis);
		//dicFrontIndex.setEncoder(e_index);
		
		nodeHashTable->clear();
		for(FrontHashTableNodeIterator it = frontHashTableNode->begin(); it != frontHashTableNode->end(); ++it)
		{
			FrontNode *frontNode = it->second;
			delete frontNode;
		}
		frontHashTableNode->clear();
		beforeTriID = 0;
		beforeOutClusterChildIndex = 0;
		#endif

		/*
		LZW dicAxis(2), dicIdxIsChildInThis(2);//, dicFrontIndex(maxFrontSize+1);
		dicAxis.setEncoder(e_index_axis);
		dicIdxIsChildInThis.setEncoder(e_index_isChildInThis);
		*/

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
				#ifdef _USE_BIT_ENCODER
				#if COMPRESS_TYPE == COMPRESS_TYPE_QIC_LZW && 0
				dicFrontIndex.encode(0);
				#else
				e_index->encode(bitsSizeDvNodeIndex, 0);
				#endif
				#else
				e_index->encode(dvNodeIndex->size()+1, 0);
				#endif
			}
			else
			{
				bool isLocalRoot = false;
				NodeHashTableIterator it = nodeHashTable->find(curNodeIdx);
				if(it == nodeHashTable->end())
				{
					// current node is a root node
					
					#ifdef _USE_BIT_ENCODER
					#if COMPRESS_TYPE == COMPRESS_TYPE_QIC_LZW && 0
					dicFrontIndex.encode(0);
					#else
					e_index->encode(bitsSizeDvNodeIndex, 0);
					#endif
					#else
					e_index->encode(dvNodeIndex->size()+1, 0);
					#endif
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
					parentNode->children += 4;
					#ifndef _USE_CONTI_NODE
					parentNode->children2 += 4;
					#endif
					#endif
					#ifndef _USE_CONTI_NODE
					if(!(parentNode->children >> 2 == curNodeIdx || parentNode->children2 >> 2 == curNodeIdx))
					{
						printf("wrong mapping! parent = %d, left = %d, right = %d, curNode = %d\n", parentIndex, parentNode->children>>2, parentNode->children2>>2, curNodeIdx);
					}
					#else
					if(!(parentNode->children >> 2 == curNodeIdx || (parentNode->children >> 2) + 1 == curNodeIdx))
					{
						printf("wrong mapping! parent = %d, left = %d, right = %d, curNode = %d\n", parentIndex, parentNode->children>>2, (parentNode->children>>2)+1, curNodeIdx);
					}
					#endif

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
					#ifdef _USE_BIT_ENCODER
					#if COMPRESS_TYPE == COMPRESS_TYPE_QIC_LZW && 0
					dicFrontIndex.encode(frontIndex+1);
					#else
					e_index->encode(bitsSizeDvNodeIndex, frontIndex+1);
					#endif
					#else
					e_index->encode(dvNodeIndex->size()+1, frontIndex+1);
					#endif

					if(front->count == 0)
					{
						front->count = front->count+1;
					}
					else
					{
						dvNodeIndex->removeElement(front);
						#ifdef _USE_BIT_ENCODER
						bitsSizeDvNodeIndex = getBits(dvNodeIndex->size());
						#endif
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

				#if COMPRESS_TYPE == COMPRESS_TYPE_ARITHMETIC
				if(parentNode->children >> 2 == curNodeIdx)
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
				#endif

				#if COMPRESS_TYPE == COMPRESS_TYPE_INCREMENTAL
				I32 bitQ;
				I32 rangeQ;
				I32 diffQ;
				I32 sumQ;
				I32 nextQ;
				if(parentNode->children >> 2 == curNodeIdx)
				{
					// this is a left child
					int sign = predictedQ < qMax[biggestaxis];
					#ifdef _USE_BIT_ENCODER
					e_geom_error->encode(1, sign);
					#else
					e_geom_error->encode(2, sign);
					#endif
					for(int k=0;k<3;k++)
					{
						bitQ = 1;
						rangeQ = 2;
						sumQ = 0;
						nextQ = 1;
						diffQ = qMin[k] - parentMinQ[k];
						while(sumQ + nextQ <= diffQ)
						{
							#ifdef _USE_BIT_ENCODER
							e_geom_error->encode(bitQ, nextQ);
							#else
							e_geom_error->encode(rangeQ, nextQ);
							#endif
							bitQ++;
							rangeQ <<= 1;
							sumQ += nextQ;
							nextQ = rangeQ - 1;
						}
						#ifdef _USE_BIT_ENCODER
						e_geom_error->encode(bitQ, diffQ - sumQ);
						#else
						e_geom_error->encode(rangeQ, diffQ - sumQ);
						#endif
					}
					for(int k=0;k<3;k++)
					{
						bitQ = 1;
						rangeQ = 2;
						sumQ = 0;
						nextQ = 1;
						diffQ = k == biggestaxis ? (sign ? qMax[k] - predictedQ : predictedQ - qMax[k]) : parentMaxQ[k] - qMax[k];
						while(sumQ + nextQ <= diffQ)
						{
							#ifdef _USE_BIT_ENCODER
							e_geom_error->encode(bitQ, nextQ);
							#else
							e_geom_error->encode(rangeQ, nextQ);
							#endif
							bitQ++;
							rangeQ <<= 1;
							sumQ += nextQ;
							nextQ = rangeQ - 1;
						}
						#ifdef _USE_BIT_ENCODER
						e_geom_error->encode(bitQ, diffQ - sumQ);
						#else
						e_geom_error->encode(rangeQ, diffQ - sumQ);
						#endif
					}
				}
				else
				{
					// this is a right child
					int sign = predictedQ > qMin[biggestaxis];
					#ifdef _USE_BIT_ENCODER
					e_geom_error->encode(1, sign);
					#else
					e_geom_error->encode(2, sign);
					#endif
					for(int k=0;k<3;k++)
					{
						bitQ = 1;
						rangeQ = 2;
						sumQ = 0;
						nextQ = 1;
						diffQ = k == biggestaxis ? (sign ? predictedQ - qMin[k] : qMin[k] - predictedQ) : qMin[k] - parentMinQ[k];
						while(sumQ + nextQ <= diffQ)
						{
							#ifdef _USE_BIT_ENCODER
							e_geom_error->encode(bitQ, nextQ);
							#else
							e_geom_error->encode(rangeQ, nextQ);
							#endif
							bitQ++;
							rangeQ <<= 1;
							sumQ += nextQ;
							nextQ = rangeQ - 1;
						}
						#ifdef _USE_BIT_ENCODER
						e_geom_error->encode(bitQ, diffQ - sumQ);
						#else
						e_geom_error->encode(rangeQ, diffQ - sumQ);
						#endif
					}
					for(int k=0;k<3;k++)
					{
						bitQ = 1;
						rangeQ = 2;
						sumQ = 0;
						nextQ = 1;
						diffQ = parentMaxQ[k] - qMax[k];
						while(sumQ + nextQ <= diffQ)
						{
							#ifdef _USE_BIT_ENCODER
							e_geom_error->encode(bitQ, nextQ);
							#else
							e_geom_error->encode(rangeQ, nextQ);
							#endif
							bitQ++;
							rangeQ <<= 1;
							sumQ += nextQ;
							nextQ = rangeQ - 1;
						}
						#ifdef _USE_BIT_ENCODER
						e_geom_error->encode(bitQ, diffQ - sumQ);
						#else
						e_geom_error->encode(rangeQ, diffQ - sumQ);
						#endif
					}
				}
				#endif

				#if COMPRESS_TYPE == COMPRESS_TYPE_ZERO_OR_ALL
				I32 rangeQ;
				I32 diffQ;
				for(int k=0;k<3;k++)
				{
					diffQ = qMin[k] - parentMinQ[k];
					if(diffQ == 0)
					{
						#ifdef STATISTICS
						g_zero_error++;
						#endif
						e_geom_error->encode(2, 0);
					}
					else
					{
						e_geom_error->encode(2, 1);
						rangeQ = parentMaxQ[k] - parentMinQ[k];
						e_geom_error->encode(16, getBits(rangeQ));
						e_geom_error->encode(rangeQ, diffQ);
					}
				}
				for(int k=0;k<3;k++)
				{
					diffQ = parentMaxQ[k] - qMax[k];
					if(diffQ == 0)
					{
						#ifdef STATISTICS
						g_zero_error++;
						#endif
						e_geom_error->encode(2, 0);
					}
					else
					{
						e_geom_error->encode(2, 1);
						rangeQ = parentMaxQ[k] - parentMinQ[k];
						e_geom_error->encode(16, getBits(rangeQ));
						e_geom_error->encode(rangeQ, diffQ);
					}
				}
				#endif

				#if COMPRESS_TYPE == COMPRESS_TYPE_STATIC_DICTIONARY
				I32 diffQ;
				for(int k=0;k<3;k++)
				{
					diffQ = qMin[k] - parentMinQ[k];
					if(diffQ == 0)
					{
						#ifdef STATISTICS
						g_zero_error++;
						#endif
						#ifdef _USE_BIT_ENCODER
						e_geom_error->encode(1, 0);
						#else
						e_geom_error->encode(2, 0);
						#endif
					}
					else
					{
						#ifdef _USE_BIT_ENCODER
						e_geom_error->encode(1, 1);
						#else
						e_geom_error->encode(2, 1);
						#endif
						DicHashTableIterator dicIt = dicHashTable->find(diffQ);
						if(dicIt == dicHashTable->end())
						{
							#ifdef _USE_BIT_ENCODER
							e_geom_error->encode(bitsSizeDicHashTable, 0);
							e_geom_error->encode(nbits, diffQ);
							dicHashTable->insert(std::pair<unsigned int, unsigned int>(diffQ, dicHashTable->size()));
							bitsSizeDicHashTable = getBits(dicHashTable->size());
							#else
							e_geom_error->encode(dicHashTable->size()+1, 0);
							e_geom_error->encode(65536, diffQ);
							dicHashTable->insert(std::pair<unsigned int, unsigned int>(diffQ, dicHashTable->size()));
							#endif
						}
						else
						{
							//e_geom_error->encode(2, 0);
							#ifdef _USE_BIT_ENCODER
							e_geom_error->encode(bitsSizeDicHashTable, dicIt->second+1);
							#else
							e_geom_error->encode(dicHashTable->size()+1, dicIt->second+1);
							#endif
						}
					}
				}
				for(int k=0;k<3;k++)
				{
					diffQ = parentMaxQ[k] - qMax[k];
					if(diffQ == 0)
					{
						#ifdef STATISTICS
						g_zero_error++;
						#endif
						#ifdef _USE_BIT_ENCODER
						e_geom_error->encode(1, 0);
						#else
						e_geom_error->encode(2, 0);
						#endif
					}
					else
					{
						#ifdef _USE_BIT_ENCODER
						e_geom_error->encode(1, 1);
						#else
						e_geom_error->encode(2, 1);
						#endif
						DicHashTableIterator dicIt = dicHashTable->find(diffQ);
						if(dicIt == dicHashTable->end())
						{
							#ifdef _USE_BIT_ENCODER
							e_geom_error->encode(bitsSizeDicHashTable, 0);
							e_geom_error->encode(nbits, diffQ);
							dicHashTable->insert(std::pair<unsigned int, unsigned int>(diffQ, dicHashTable->size()));
							bitsSizeDicHashTable = getBits(dicHashTable->size());
							#else
							e_geom_error->encode(dicHashTable->size()+1, 0);
							e_geom_error->encode(65536, diffQ);
							dicHashTable->insert(std::pair<unsigned int, unsigned int>(diffQ, dicHashTable->size()));
							#endif
						}
						else
						{
							//e_geom_error->encode(2, 0);
							#ifdef _USE_BIT_ENCODER
							e_geom_error->encode(bitsSizeDicHashTable, dicIt->second+1);
							#else
							e_geom_error->encode(dicHashTable->size()+1, dicIt->second+1);
							#endif
						}
					}
				}
				#endif
				#if COMPRESS_TYPE == COMPRESS_TYPE_QIC_LZW
				if(parentNode->children >> 2 == curNodeIdx)
				{
					// this is a left child
					dicError.encode(qMin[biggestaxis] - parentMinQ[biggestaxis]);
					dicError.encode(qMin[axis1] - parentMinQ[axis1]);
					dicError.encode(qMin[axis2] - parentMinQ[axis2]);
					dicError.encode(predictedQ - qMax[biggestaxis]);
					dicError.encode(parentMaxQ[axis1] - qMax[axis1]);
					dicError.encode(parentMaxQ[axis2] - qMax[axis2]);
				}
				else
				{
					// this is a right child
					dicError.encode(qMin[biggestaxis] - predictedQ);
					dicError.encode(qMin[axis1] - parentMinQ[axis1]);
					dicError.encode(qMin[axis2] - parentMinQ[axis2]);
					dicError.encode(parentMaxQ[biggestaxis] - qMax[biggestaxis]);
					dicError.encode(parentMaxQ[axis1] - qMax[axis1]);
					dicError.encode(parentMaxQ[axis2] - qMax[axis2]);
				}
				#endif
				#if COMPRESS_TYPE == COMPRESS_TYPE_STATIC_DICTIONARY_NEW
				for(int k=0;k<3;k++)
				{
					dicError.encodeNoneSpecial(qMin[k] - parentMinQ[k]);
				}
				for(int k=0;k<3;k++)
				{
					dicError.encodeNoneSpecial(parentMaxQ[k] - qMax[k]);
				}
				#endif
				if(parentNode->children >> 2 == curNodeIdx)
				{
					#ifndef _USE_CONTI_NODE
					if((parentNode->children >> 2) + 1 == parentNode->children2 >> 2) contNodes++;
					#else
					if((parentNode->children >> 2) + 1 == (parentNode->children >> 2) + 1) contNodes++;
					#endif
				}
				totalNodes++;
			}

			if(!ISLEAF(node))
			{
				nodeHashTable->insert(std::pair<unsigned int, unsigned int>(node->children >> 2, curNodeIdx));
				#ifndef _USE_CONTI_NODE
				nodeHashTable->insert(std::pair<unsigned int, unsigned int>(node->children2 >> 2, curNodeIdx));
				#else
				nodeHashTable->insert(std::pair<unsigned int, unsigned int>((node->children >> 2) + 1, curNodeIdx));
				#endif
			}

			#if COMPRESS_TYPE == COMPRESS_TYPE_QIC_LZW && 0
			dicAxis.encode((node->children & 3) == 3);
			#else
			#ifdef _USE_BIT_ENCODER
			e_index_axis->encode(1, (node->children & 3) == 3);
			//dicAxis.encode((node->children & 3) == 3);
			#else
			e_index_axis->encode(2, (node->children & 3) == 3);
			#endif
			#endif


			if(ISLEAF(node))
			{
				if(maxNumTrisPerLeaf != 1)
				{
					#ifdef _USE_BIT_ENCODER
					static unsigned int bitsMaxNumTrisPerLeaf = getBits(maxNumTrisPerLeaf-1);
					e_index_axis->encode(bitsMaxNumTrisPerLeaf, GETCHILDCOUNT(node)-1);
					#else
					e_index_axis->encode(maxNumTrisPerLeaf, GETCHILDCOUNT(node)-1);
					#endif
				}
				#ifndef _USE_ONE_TRI_PER_LEAF
				unsigned int TriID = node->indexOffset;
				#else
				unsigned int TriID = node->triIndex >> 2;
				#endif

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
						e_index_tri_type->encode(3, 0);
						e_index_tri_cache->encode(3, whichTri);
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
 							//e_index_tri_type->encode(rmTriIndexType, 1);
							e_index_tri_type->encode(3, 1);

							int triIndex = dvTriIndexFront->getRelativeIndex(it->second);
							e_index_tri_front->encode(dvTriIndexFront->size()+1, triIndex);//icTriIndexFront->CompressNone(triIndex);// e_index_tri_front->encode(rmTriIndexFront, triIndex);
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
					e_index_tri_type->encode(3, 2);
					//e_index_tri_type->encode(rmTriIndexType, 2);
					type = 3;

					unsigned int baseIndex = 0;
					if(it == baseHashTableTri->end())
					{
						e_index_tri_base->encode(dvTriIndexBase->size()+1, baseIndex);//icTriIndexBase->CompressNone(baseIndex);
						//e_index_tri_base->encode(rmTriIndexBase, baseIndex);
						e_index_tri_base->encode(numClusters, base);

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
						e_index_tri_base->encode(dvTriIndexBase->size()+1, baseIndex);//icTriIndexBase->CompressNone(baseIndex);
						//e_index_tri_base->encode(rmTriIndexBase, baseIndex);
					}
					e_index_tri_offset->encode(sizeBasePage, offset);
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
					e_index_tri_base->encode(dvTriIndexBase->size()+1, baseIndex);
					e_index_tri_base->encode(numClusters, base);

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
					e_index_tri_base->encode(dvTriIndexBase->size()+1, baseIndex);
				}
				e_index_tri_offset->encode(sizeBasePage, offset);
//				e_index_offset->encodeInt(TriID);
				#endif
				#endif				

				#ifdef USE_TRI_DELTA_ENCODING
				int delta = TriID - beforeTriID;

				#if COMPRESS_TYPE == COMPRESS_TYPE_QIC_LZW
				dicIdxTri.encode(delta);
				#else

				#if COMPRESS_TYPE == COMPRESS_TYPE_STATIC_DICTIONARY_NEW && 0
				dicDeltaTri.encodeNone(delta);
				#else

				#ifdef _USE_BIT_ENCODER

				e_index_tri->encode(1, delta > 0);
				static unsigned int bitsNumHalfNodes = getBits(maxNumTris>>1);
				if(!encodeDeltaForChildIndexInOutCluster(delta > 0 ? delta : -delta, numBoundary, listBoundary, listBoundaryBits, e_index_tri))
				{
					e_index_tri->encode(bitsNumHalfNodes, TriID);
					#ifdef STATISTICS
					g_num_out_boundary_tri_index++;
					#endif
				}
				#else

				e_index_tri->encode(2, delta > 0);
				if(!encodeDeltaForChildIndexInOutCluster(delta > 0 ? delta : -delta, numBoundary, listBoundary, rmDeltaTriIndex, e_index_tri))
				{
					e_index_tri->encode((maxNumTris>>1)+1, TriID);
					#ifdef STATISTICS
					g_num_out_boundary_tri_index++;
					#endif
				}
				#endif
				else
				{
					#ifdef STATISTICS
					g_num_in_boundary_tri_index++;
					#endif
				}	
				#endif
				#endif
				beforeTriID = TriID;
				#endif
			}
			//if(childCounts[curNodeIdx] <= 14);
			else
			{
				int isChildInThis = 0;
				isChildInThis |= (curNodeIdx / nodesPerCluster == (node->children >> 2) / nodesPerCluster) << 1;
				#ifndef _USE_CONTI_NODE
				isChildInThis |= (curNodeIdx / nodesPerCluster == (node->children2 >> 2) / nodesPerCluster);
				#else
				isChildInThis |= (curNodeIdx / nodesPerCluster == ((node->children >> 2) + 1) / nodesPerCluster);
				#endif
				
				#ifdef _USE_BIT_ENCODER
				#ifdef _USE_DUMMY_NODE
				#if COMPRESS_TYPE == COMPRESS_TYPE_QIC_LZW && 0
				dicIdxIsChildInThis.encode(isChildInThis == 3);
				#else
				e_index_isChildInThis->encode(1, isChildInThis == 3);
				//dicIdxIsChildInThis.encode(isChildInThis == 3);
				#endif
				#else
				cout << "error!!!! isChildInThis!!!" << endl;
				#endif
				#else
				e_index_isChildInThis->encode(rmIsChildInThis, isChildInThis);
				#endif
	
				ParentRootIndex *parentRootIndex;
				unsigned int delta = 0;
				#ifdef _USE_BIT_ENCODER
				if(isChildInThis == 0)
				{
					#if COMPRESS_TYPE == COMPRESS_TYPE_QIC_LZW
					delta = (node->children >> 2) - beforeOutClusterChildIndex;
					dicIdxOutCluster.encode(delta);
					delta = (node->children2 >> 2) - (node->children >> 2);
					dicIdxOutCluster.encode(delta);
					#else
					#if COMPRESS_TYPE == COMPRESS_TYPE_STATIC_DICTIONARY_NEW && 0
					delta = (node->children >> 2) - beforeOutClusterChildIndex;
					dicDeltaOutCluster.encodeNone(delta);
					delta = (node->children2 >> 2) - (node->children >> 2);
					dicDeltaOutCluster.encodeNone(delta);
					#else
					delta = (node->children >> 2) - beforeOutClusterChildIndex;
					if(!encodeDeltaForChildIndexInOutCluster(delta, numBoundary, listBoundary, listBoundaryBits, e_index_outCluster))
					{
						e_index_outCluster->encode(bitsNumNodes, node->children >> 2);
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
					#ifndef _USE_CONTI_NODE
					delta = (node->children2 >> 2) - (node->children >> 2);
					#else
					delta = 1;
					#endif
					if(!encodeDeltaForChildIndexInOutCluster(delta, numBoundary, listBoundary, listBoundaryBits, e_index_outCluster))
					{
						#ifndef _USE_CONTI_NODE
						e_index_outCluster->encode(bitsNumNodes, node->children2 >> 2);
						#else
						e_index_outCluster->encode(bitsNumNodes, (node->children >> 2) + 1);
						#endif
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
					#endif
					#endif
					#ifndef _USE_CONTI_NODE
					beforeOutClusterChildIndex = node->children2 >> 2;
					#else
					beforeOutClusterChildIndex = (node->children >> 2) + 1;
					#endif
					#ifdef STATISTICS
					g_out_cluster_index++;
					g_out_cluster_index++;
					#endif
				}
				#else
				switch(isChildInThis)
				{
				case 0 : 
					delta = (node->children >> 2) - beforeOutClusterChildIndex;
					if(!encodeDeltaForChildIndexInOutCluster(delta, numBoundary, listBoundary, rmDeltaChildIndex, e_index_outCluster))
					{
						e_index_outCluster->encode(numNodes, node->children >> 2);
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
					delta = (node->children2 >> 2) - (node->children >> 2);
					if(!encodeDeltaForChildIndexInOutCluster(delta, numBoundary, listBoundary, rmDeltaChildIndex, e_index_outCluster))
					{
						e_index_outCluster->encode(numNodes, node->children2 >> 2);
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
					beforeOutClusterChildIndex = node->children2 >> 2;
					#ifdef STATISTICS
					g_out_cluster_index++;
					g_out_cluster_index++;
					#endif
					break;
				case 1 :
					delta = (node->children >> 2) - beforeOutClusterChildIndex;
					if(!encodeDeltaForChildIndexInOutCluster(delta, numBoundary, listBoundary, rmDeltaChildIndex, e_index_outCluster))
					{
						e_index_outCluster->encode(numNodes, node->children >> 2);
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
					beforeOutClusterChildIndex = node->children >> 2;
					#ifdef STATISTICS
					g_out_cluster_index++;
					#endif
					break;
				case 2 :
					delta = (node->children2 >> 2) - beforeOutClusterChildIndex;
					if(!encodeDeltaForChildIndexInOutCluster(delta, numBoundary, listBoundary, rmDeltaChildIndex, e_index_outCluster))
					{
						e_index_outCluster->encode(numNodes, node->children2 >> 2);
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
					beforeOutClusterChildIndex = node->children2 >> 2;
					#ifdef STATISTICS
					g_out_cluster_index++;
					#endif
					break;
				}
				#endif

				if(isChildInThis > 0)
				{
					FrontNode* frontNode = new FrontNode;
					frontNode->buffer_next = 0;
					frontNode->IsInDV = 0;
					frontNode->index = curNodeIdx;
					frontNode->count = isChildInThis == 3 ? 0 : 1;

					dvNodeIndex->addElement(frontNode);
					#ifdef _USE_BIT_ENCODER
					bitsSizeDvNodeIndex = getBits(dvNodeIndex->size());
					#endif

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

		//dicAxis.done();
		//dicIdxIsChildInThis.done();
		//dicFrontIndex.done();

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
		
		#ifndef _USE_BIT_ENCODER
		delete rmIsChildInThis;
		for(int k=0;k<numBoundary;k++)
		{
			delete rmDeltaChildIndex[k];
			delete rmDeltaTriIndex[k];
		}
		delete rmDeltaChildIndex;
		delete rmDeltaTriIndex;
		#endif
		delete clusterCache;
		delete nodeHashTable;

		#if COMPRESS_TYPE == COMPRESS_TYPE_STATIC_DICTIONARY
		#ifdef STATISTICS
		g_avg_dic_table_size += dicHashTable->size();
		#endif
		dicHashTable->clear();
		delete dicHashTable;
		#endif

		#if COMPRESS_TYPE == COMPRESS_TYPE_STATIC_DICTIONARY_NEW
		dicError.done();
		//dicDeltaTri.done();
		//dicDeltaOutCluster.done();
		#endif

		#ifndef STATISTICS
		e_geom->done();
		delete e_geom;
		#endif
		#ifdef STATISTICS
		e_geom->done();
		e_geom_raw->done();
		e_geom_error->done();
		e_index->done();
		e_parent_cluster->done();
		e_parent_index_offset->done();
		e_index_axis->done();
		e_index_isChildInThis->done();
		e_index_outCluster->done();
		e_index_offset->done();
		#ifdef USE_TRI_3_TYPE_ENCODING
		e_index_tri_type->done();
		e_index_tri_cache->done();
		e_index_tri_front->done();
		e_index_tri_base->done();
		e_index_tri_offset->done();
		#endif
		#ifdef USE_TRI_DELTA_ENCODING
		e_index_tri->done();
		#endif
		e_etc->done();
		e_debug->done();
		g_geom += e_geom->getNumberBits();
		g_geom_raw += e_geom_raw->getNumberBits();
		g_geom_error += e_geom_error->getNumberBits();
		g_index += e_index->getNumberBits();
		g_parent_cluster += e_parent_cluster->getNumberBits();
		g_parent_index_offset += e_parent_index_offset->getNumberBits();
		g_index_axis += e_index_axis->getNumberBits();
		g_index_isChildInThis += e_index_isChildInThis->getNumberBits();
		g_index_outCluster += e_index_outCluster->getNumberBits();
		g_index_offset += e_index_offset->getNumberBits();
		#ifdef USE_TRI_3_TYPE_ENCODING
		g_index_tri_type += e_index_tri_type->getNumberBits();
		g_index_tri_cache += e_index_tri_cache->getNumberBits();
		g_index_tri_front += e_index_tri_front->getNumberBits();
		g_index_tri_base += e_index_tri_base->getNumberBits();
		g_index_tri_offset += e_index_tri_offset->getNumberBits();
		#endif
		#ifdef USE_TRI_DELTA_ENCODING
		g_index_tri += e_index_tri->getNumberBits();
		#endif
		g_etc += e_etc->getNumberBits();
		g_debug += e_debug->getNumberBits();
		delete e_geom;
		delete e_geom_raw;
		delete e_geom_error;
		delete e_index;
		delete e_parent_cluster;
		delete e_parent_index_offset;
		delete e_index_axis;
		delete e_index_isChildInThis;
		delete e_index_outCluster;
		delete e_index_offset;
		#ifdef USE_TRI_3_TYPE_ENCODING
		delete e_index_tri_type;
		delete e_index_tri_cache;
		delete e_index_tri_front;
		delete e_index_tri_base;
		delete e_index_tri_offset;
		#endif
		#ifdef USE_TRI_DELTA_ENCODING
		delete e_index_tri;
		#endif
		delete e_etc;
		delete e_debug;
		#endif

		#if COMPRESS_TYPE == COMPRESS_TYPE_ARITHMETIC
		ic[0]->FinishCompressor();
		ic[1]->FinishCompressor();

		delete ic[0];
		delete ic[1];
		delete pq;
		#endif
		numProccessedClusters++;
	}

	fseek(fpc, posOffset, SEEK_SET);
	fwrite(offsets, sizeof(long), numClusters, fpc);
	fseek(fpc, 0, SEEK_END);

	delete offsets;
	delete[] listBoundary;
	delete[] listBoundaryBits;

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
//	printf("average number of inter cluster nodes per cluster : %f\n", ((float)g_total_inter_cluster_nodes)/numProccessedClusters);
//	printf("Debug : %f bpn\n", g_debug/g_total_node);
	printf("Average size of front : %f\n", g_avg_front_size_per_cluster/numProccessedClusters);
	printf("Average number of parent clusters per cluster : %f\n", float(g_num_parent_cluster)/numProccessedClusters);
	printf("Percentage of in-boundary delta child index : %.1f%%\n", float(g_num_in_boundary_child_index)/(float(g_num_in_boundary_child_index) + float(g_num_out_boundary_child_index))*100.0f);
	printf("Percentage of in-boundary delta tri index : %.1f%%\n", float(g_num_in_boundary_tri_index)/(float(g_num_in_boundary_tri_index) + float(g_num_out_boundary_tri_index))*100.0f);
	printf("Etc : %f bpn\n", g_etc/g_total_node);
	#if COMPRESS_TYPE == COMPRESS_TYPE_STATIC_DICTIONARY
	printf("Average size of dictionary hash table : %f\n", g_avg_dic_table_size/numProccessedClusters);
	#endif
	printf("Percentage of zero error : %.1f%%\n", float(g_zero_error)/float(__int64(g_total_node)*6)*100.0f);
	#endif
	_CrtDumpMemoryLeaks();
	return 0;
}

int BVHCompression::encodeDeltaForChildIndexInOutCluster(unsigned int delta, unsigned int numBoundary, unsigned int *listBoundary, unsigned int *listBoundaryBits, BitCompression *e)
{
	for(unsigned int pass=0;pass<numBoundary;pass++)
	{
		if(delta < listBoundary[pass])
		{
			e->encode(listBoundaryBits[pass], delta);
			return TRUE;
		}
		e->encode(listBoundaryBits[pass], listBoundary[pass]);
	}
	return FALSE;
}

int BVHCompression::encodeDeltaForChildIndexInOutCluster(unsigned int delta, unsigned int numBoundary, unsigned int *listBoundary, RangeModel **rmDeltaChildIndex, RangeEncoder *e)
{
	for(unsigned int pass=0;pass<numBoundary;pass++)
	{
		if(delta < listBoundary[pass])
		{
			e->encode(rmDeltaChildIndex[pass], delta);
			return TRUE;
		}
		e->encode(rmDeltaChildIndex[pass], listBoundary[pass]);
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
		unsigned int left = node.children >> 2;
		#ifndef _USE_CONTI_NODE
		unsigned int right = node.children2 >> 2;
		#else
		unsigned int right = (node.children >> 2) + 1;
		#endif
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