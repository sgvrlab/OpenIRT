#ifndef RACBVH_H
#define RACBVH_H

#include "common.h"
#include <queue>
#include "RangeDecoder_File.h"
#include "RangeDecoder_Mem.h"
#include "positionquantizer_new.h"
#include "integercompressorRACBVH.h"
#include "stopwatch.hpp"

//#define DEBUG_CODEC     // enable extra codec info to verify correctness

#define USE_DYNAMIC_VECTOR
//#define USE_LIST

#define USE_MEM_MANAGER

#define USE_MM
//#define USE_DM

#define NBITS 16
#define BIT_MASK 0xFFFF

#define FLOOR 0
#define CEIL 1

#ifdef _USE_RACBVH
#ifdef GETIDXOFFSET
#undef GETIDXOFFSET
#define GETIDXOFFSET(node) ((node)->indexOffset >> 2)
#endif
#endif

#ifdef USE_DYNAMIC_VECTOR
#include "dynamicvector.h"
#endif

#ifdef USE_MEM_MANAGER
#include "mem_managerRACBVH.h"
#endif

#ifdef _USE_RACM
#include "compressed_mesh.h"
#endif

template <class T>
class RACBVH
{
public :
	RACBVH(const char * pFileName, int maxAllowedMem, int blockSize);
//	FORCEINLINE const T &operator[](unsigned int i); 
	FORCEINLINE const T &operator[](unsigned int i); 
	~RACBVH();

#ifdef USE_MM
	int readInt();
	unsigned int readUInt();
	long readLong();
	float readFloat();
#endif

#ifdef USE_DM
	int readInt(long &pos);
	unsigned int readUInt(long &pos);
	long readLong(long &pos);
	float readFloat(long &pos);
#endif

	int maxAllowedMem;

	int loadClusterTable();
	bool loadCluster(unsigned int CN, T* posCluster, long diskClusterOffset, int threadNum);

	//BSPArrayTreeNodePtr getNode(unsigned int index, unsigned int depth = 0);
	void completeBB(BSPArrayTreeNodePtr parentNode, BSPArrayTreeNodePtr node, int leftOrRight, unsigned int CN);
	unsigned int decodeDeltaForChildIndexInOutCluster(unsigned int numBoundary, unsigned int *listBoundary, RangeModel **rmDeltaChildIndex, RangeDecoder *rd);
	unsigned int decodeDeltaForTriIndex(unsigned int numBoundary, unsigned int *listBoundary, RangeModel **rmDeltaTriIndex, RangeDecoder *rd);
	int getBiggestAxis(I32 *minQ, I32 *maxQ)
	{
		I32 diffQ[3] = {maxQ[0]-minQ[0], maxQ[1]-minQ[1], maxQ[2]-minQ[2]};
		return (diffQ[0]> diffQ[1] && diffQ[0]> diffQ[2]) ? 0 : (diffQ[1] > diffQ[2] ? 1 : 2);
	}

	PositionQuantizerNew* pq;

	struct FrontNode;
	typedef struct FrontNode
	{
		FrontNode* buffer_next;
		int IsInDV;
		int index;
		int count;
		int isChildInThis;
		int axis;
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

	/////////////////////////////
	// Variables
	/////////////////////////////

	#if defined(_USE_RACM) && defined(_USE_TRI_3_TYPE_ENCODING)
	CMeshAbstract * m_pMesh;
	#endif
	CMemManagerRACBVH<BSPArrayTreeNode> m_physicalMemory;
	unsigned int m_nodesPerCluster;
	unsigned int m_sizeBasePage;
	unsigned int m_sizeBasePagePower;
	unsigned int m_numNodes;
	unsigned int m_numClusters;
	unsigned int m_maxNumPCN;
	unsigned int m_nodesPerClusterPower;
	float m_bb_min_f[3];
	float m_bb_max_f[3];
	unsigned int m_numBoundary;
	unsigned int *m_listBoundary;
	I32 m_maxRange;

	#if !defined(USE_MM) && !defined(USE_DM)
	FILE *fp[NUM_THREADS];
	#endif

	#ifdef USE_MM
	// Compressed data
	CMemoryMappedFile <unsigned char> m_CompressedFile[NUM_THREADS];
	#endif

	#ifdef USE_DM
	unsigned char *m_CompressedFile;
	#endif
};

#include "io.h"
#include <math.h>
#include "stopwatch.hpp"

template <class T>
RACBVH<T>::RACBVH(const char * pFileName, int maxAllowedMem, int blockSize)
{
	m_listBoundary = NULL;
	pq = NULL;
	this->maxAllowedMem = maxAllowedMem;

	#if !defined(USE_MM) && !defined(USE_DM)
	int i;
	for(i=0;i<NUM_THREADS;i++)
		fp[i] = fopen(pFileName, "rb");
	#endif
	#ifdef USE_MM
	FILE *fpTemp = fopen(pFileName, "rb");
	I64 fileSize = _filelengthi64(fileno(fpTemp));
	fclose(fpTemp);

	int i;
	for(i=0;i<NUM_THREADS;i++)
		m_CompressedFile[i].Init(pFileName, "r", 1024*1024*32/(64*1024), fileSize);
	#endif

	#ifdef USE_DM
	FILE *fpTemp = fopen(pFileName, "rb");
	I64 fileSize = _filelengthi64(fileno(fpTemp));

	m_CompressedFile = new unsigned char[fileSize];
	fread(m_CompressedFile, sizeof(unsigned char), fileSize, fpTemp);
	fclose(fpTemp);
	#endif

	loadClusterTable();
}

template <class T>
RACBVH<T>::~RACBVH()
{
	if (m_listBoundary)
		delete m_listBoundary;
	if (pq)
		delete pq;

	#if !defined(USE_MM) && !defined(USE_DM)
	int i;
	for(i=0;i<NUM_THREADS;i++)
		if(fp[i])
			fclose(fp[i]);
	#endif

	#ifdef USE_DM
	int i;
	for(i=0;i<NUM_THREADS;i++)
		delete m_CompressedFile;
	#endif
}

/*
FORCEINLINE const BSPArrayTreeNode& RACBVH<BSPArrayTreeNode>::operator[](unsigned int i)
{
	return *getNode(i);
}

const BSPArrayTreeNode& RACBVH<BSPArrayTreeNode>::operator[](unsigned int index)
{
//	printf("getNode(%d)\n", index);
	static unsigned int SHIFT_NPC_BSPN = m_nodesPerClusterPower + BSPTREENODESIZEPOWER;
	static unsigned int CN_SHIFT = SHIFT_NPC_BSPN - 3;
	unsigned int CN = index >> CN_SHIFT;
	int TPCN = clusterTable[CN].PCN;
	if(TPCN == -1)
	{
		static int pp = 0;

		// Page Fault
		long DCO = clusterTable[CN].DCO;

		int PCN = -1;
		if(emptyList.empty())
		{
			#ifdef _VICTIM_POLICY_RANDOM
			PCN = rand()%m_maxNumPCN;
			#endif
			#ifdef _VICTIM_POLICY_SECOND_CHANCE
			unsigned int beforeVictim = currentVictim;
			ClusterTableEntry *curEntry = &clusterTable[currentVictim];
			while(curEntry->chance > 0)
			{
				curEntry->chance--;
				if(curEntry->chance < 0) curEntry->chance = 0;

				currentVictim = (currentVictim+1)%m_maxNumPCN;
				if(currentVictim == beforeVictim)
				{
					printf("There is no victim!!!\n");
				}
				curEntry = &clusterTable[currentVictim];
			}
			PCN = currentVictim;
			currentVictim = (currentVictim+1)%m_maxNumPCN;
			#endif
			clusterTable[PMM[PCN]].PCN = -1;
		}
		else
		{
			PCN = emptyList.front();
			emptyList.pop();
		}

		clusterTable[CN].PCN = TPCN = PCN;
		PMM[PCN] = CN;

//		loadCluster(CN, PCN, DCO);

	}

	#ifdef _VICTIM_POLICY_SECOND_CHANCE
	clusterTable[CN].chance = 1;
	#endif

	long offset = (index << 3) & offsetMask;
	BSPArrayTreeNodePtr node = (BSPArrayTreeNodePtr)(m_physicalMemory + (TPCN << SHIFT_NPC_BSPN) + offset);
	if((node->children2 & 1) == 1)
	{
		// Bounding box isn't completed

		char* PosCluster = m_physicalMemory + (TPCN << SHIFT_NPC_BSPN);
	
		BSPArrayTreeNodePtr localRoot = node;
		// find local root
		unsigned int parentIndex;
		while(true)
		{
			memcpy(&parentIndex, &localRoot->min.e[0], sizeof(unsigned int));
			if(parentIndex / m_nodesPerCluster != CN) break;
			offset = (parentIndex << 5) & offsetMask;
			localRoot = (BSPArrayTreeNodePtr)(PosCluster + offset);
		}
		BSPArrayTreeNodePtr parentNode = ((BSPArrayTreeNodePtr)&((*this)[parentIndex << 2]));
		int leftOrRight = GETLEFTCHILD(parentNode) == index;
		completeBB(parentNode, localRoot, leftOrRight, PosCluster, CN);
	}
	return *node;
}
*/

const BSPArrayTreeNode& RACBVH<BSPArrayTreeNode>::operator[](unsigned int index)
{
	BSPArrayTreeNodePtr node = &m_physicalMemory[index >> 2];
	if((node->children2 & 1) == 1)
	{
		extern Stopwatch **TComplete;
		#ifdef _USE_OPENMP
		int threadNum = omp_get_thread_num();
		#else
		int threadNum = 0;
		#endif

		TComplete[threadNum]->Start();
		// Bounding box isn't completed
		BSPArrayTreeNodePtr parentNode = node;
		BSPArrayTreeNodePtr localRoot = node;
		// find local root and get parent node of the local root.
		unsigned int parentIndex;
		unsigned int CN = (index >> 2) >> m_nodesPerClusterPower;
		while(true)
		{
			memcpy(&parentIndex, &localRoot->min.e[0], sizeof(unsigned int));
			// busy wait when another processor is completing
			if(((parentNode->children2) >> 1) & 1)
			{
				while(((parentNode->children2) >> 1) & 1)
				{
					#ifdef _USE_OPENMP
					#pragma omp critical
					cout << omp_get_thread_num() << " is wating for " << CN << " " << parentIndex / m_nodesPerCluster << endl;
					#endif
				}
				assert((node->children2 & 3) == 0);
				TComplete[threadNum]->Stop();
				return *node;
			}
			localRoot = parentNode;
			parentNode = &m_physicalMemory[parentIndex];
			// parent node is in another cluster or has completed BB
			if(parentIndex / m_nodesPerCluster != CN) break;
		}
		int leftOrRight = GETLEFTCHILD(parentNode) == index;
		completeBB(parentNode, localRoot, leftOrRight, CN);
		TComplete[threadNum]->Stop();
	}
	return *node;
}

// link module between CMemManager and RACBVH
template <class T>
bool loadCluster(RACBVH<T> *pBVH, unsigned int CN, T* posCluster, long diskClusterOffset, int threadNum = 0)
{
	extern Stopwatch **TCluster;
	TCluster[threadNum]->Start();
	bool returnValue = pBVH->loadCluster(CN, posCluster, diskClusterOffset, threadNum);
	TCluster[threadNum]->Stop();
	return returnValue;
}

template <class T>
bool RACBVH<T>::loadCluster(unsigned int CN, T* posCluster, long diskClusterOffset, int threadNum)
{
	#if !defined(USE_MM) && !defined(USE_DM)
	fseek(fp[threadNum], diskClusterOffset, SEEK_SET);
	//RangeDecoder *rd_geom = new RangeDecoderFile(fp);
	unsigned char m_CompressedFile[4096*sizeof(BSPArrayTreeNode)];
	fread(m_CompressedFile, 4096*sizeof(BSPArrayTreeNode), 1, fp[threadNum]);

	RangeDecoder *rd_geom = new RangeDecoderFile(m_CompressedFile, 4096*sizeof(BSPArrayTreeNode));
	#endif

	#ifdef USE_MM
	m_CompressedFile[threadNum].SetCurrentPointer(diskClusterOffset);
	RangeDecoder *rd_geom = new RangeDecoderMemFile(m_CompressedFile[threadNum]);
	#endif

	#ifdef USE_DM
	RangeDecoder *rd_geom = new RangeDecoderFile(m_CompressedFile+diskClusterOffset, 4096*sizeof(BSPArrayTreeNode));
	#endif

	IntegerCompressorRACBVH* ic[2];
	RangeModel **rmDeltaChildIndex = new RangeModel*[m_numBoundary];
	#ifdef _USE_TRI_DELTA_ENCODING
	RangeModel **rmDeltaTriIndex = new RangeModel*[m_numBoundary];
	#endif
	RangeModel *rmIsChildInThis;
	#if defined(_USE_RACM) && defined(_USE_TRI_3_TYPE_ENCODING)
	//RangeModel *rmTriIndexType;
	#endif

	ic[0] = new IntegerCompressorRACBVH();
	ic[1] = new IntegerCompressorRACBVH();

	m_maxRange = pq->m_aiQuantRange[0] > pq->m_aiQuantRange[1] ? pq->m_aiQuantRange[0] : pq->m_aiQuantRange[1];
	m_maxRange = m_maxRange > pq->m_aiQuantRange[2] ? m_maxRange : pq->m_aiQuantRange[2];

	ic[0]->SetRange(m_maxRange);
	ic[1]->SetRange(m_maxRange);

	ic[0]->SetPrecision(NBITS);
	ic[1]->SetPrecision(NBITS);

	ic[0]->SetupDecompressor(rd_geom);
	ic[1]->SetupDecompressor(rd_geom);

	unsigned int clusterSize = rd_geom->decodeInt();

	for(int i=0;i<m_numBoundary;i++)
	{
		rmDeltaChildIndex[i] = new RangeModel(m_listBoundary[i]+1, 0, FALSE);
		#ifdef _USE_TRI_DELTA_ENCODING
		rmDeltaTriIndex[i] = new RangeModel(m_listBoundary[i]+1, 0, FALSE);
		#endif
	}

	#if defined(_USE_RACM) && defined(_USE_TRI_3_TYPE_ENCODING)
	//rmTriIndexType = new RangeModel(3, 0, FALSE);
	DynamicVector *dvTriIndexFront = new DynamicVector();
	DynamicVector *dvTriIndexBase = new DynamicVector();

	typedef stdext::hash_map<unsigned int, FrontTri*> FrontHashTableTri;
	typedef FrontHashTableTri::iterator FrontHashTableTriIterator;
	FrontHashTableTri *frontHashTableTri = new FrontHashTableTri;

	#else
	DynamicVector *dvTriIndexBase = new DynamicVector();
	#endif

	rmIsChildInThis = new RangeModel(4, 0, FALSE);

	DynamicVector *dvNodeIndex = new DynamicVector();
	int beforeTriID = 0;

	unsigned int numParentCluster = 0;
	unsigned int numLocalRoot = 0;

	numParentCluster = rd_geom->decode(m_numClusters);
	unsigned int *listParentCN = new unsigned int[numParentCluster];
	for(int i=0;i<numParentCluster;i++)
	{
		listParentCN[i] = rd_geom->decode(m_numClusters);
	}

	numLocalRoot = rd_geom->decode(m_nodesPerCluster);
	unsigned int *listParentIndex = new unsigned int[numLocalRoot];
	for(int i=0;i<numLocalRoot;i++)
	{
//			unsigned int parentCN = listParentCN[rd_geom->decode(rmParentClusterOffset)];
		unsigned int parentCN = listParentCN[rd_geom->decode(numParentCluster)];
		listParentIndex[i] = (parentCN << m_nodesPerClusterPower) + rd_geom->decode(m_nodesPerCluster);
	}

	delete listParentCN;


	unsigned int curLocalRoot = 0;
	unsigned int beforeOutClusterChildIndex = 0;

	int triIDCache[3];

	for(int i=0;i<clusterSize;i++)
	{
		BSPArrayTreeNodePtr node = &posCluster[i];//(BSPArrayTreeNodePtr)(m_physicalMemory + ((PCN << SHIFT_NPC_BSPN) + (i << BSPTREENODESIZEPOWER)));
		unsigned int curNodeIdx = CN*m_nodesPerCluster+i;

		unsigned int parentIndex;
		int isChildInThis = 0;
		int parentAxis = 0;
		BOOL leftOrRight = 0;
		BSPArrayTreeNodePtr parentNode = NULL;
		int indexCode = rd_geom->decode(dvNodeIndex->size()+1);
		unsigned int hasInCompleteBB = 0;
		int biggestaxis = 0;

		if(indexCode == 0)
		{
			if(curNodeIdx == 0)
			{
				// global root
				node->min.e[0] = m_bb_min_f[0];
				node->min.e[1] = m_bb_min_f[1];
				node->min.e[2] = m_bb_min_f[2];
				node->max.e[0] = m_bb_max_f[0];
				node->max.e[1] = m_bb_max_f[1];
				node->max.e[2] = m_bb_max_f[2];
//					Vector3 diff = node->max - node->min;
//					node->children |= diff.indexOfMaxComponent();
			}
			else
			{
				// local root
				parentIndex = listParentIndex[curLocalRoot++];//rd_geom->decodeInt();
				memcpy(&node->min.e[0], &parentIndex, sizeof(int));
				unsigned int code = 0;

				// change Last(0) to None -------------------------
				code = (unsigned int)ic[0]->Decompress(0, 1);
				code = (code << NBITS) + (unsigned int)ic[0]->Decompress(0, 1);
				memcpy(&node->max.e[0], &code, sizeof(unsigned int));
				code = (unsigned int)ic[1]->Decompress(0, 1);
				code = (code << NBITS) + (unsigned int)ic[1]->Decompress(0, 1);
				memcpy(&node->max.e[1], &code, sizeof(unsigned int));
				code = (unsigned int)ic[1]->Decompress(0, 1);
				code = (code << NBITS) + (unsigned int)ic[1]->Decompress(0, 1);
				memcpy(&node->max.e[2], &code, sizeof(unsigned int));
				hasInCompleteBB = 1;
			}
		}
		else
		{
			parentIndex = indexCode-1;

			FrontNode* frontNode = (FrontNode *)dvNodeIndex->getElementWithRelativeIndex(parentIndex);
			parentIndex = frontNode->index;
			isChildInThis = frontNode->isChildInThis;
			parentAxis = frontNode->axis;

			if(frontNode->count == 0)
			{
				frontNode->count = frontNode->count+1;
				leftOrRight = 1;
			}
			else
			{
				leftOrRight = (isChildInThis & 1) == 1 ? 0 : 1;
				dvNodeIndex->removeElement(frontNode);
				delete frontNode;
			}

			unsigned int pMod = parentIndex - ((parentIndex >> m_nodesPerClusterPower) << m_nodesPerClusterPower);
			parentNode = &posCluster[pMod];//(BSPArrayTreeNodePtr)(m_physicalMemory + ((PCN << SHIFT_NPC_BSPN) + (pMod << BSPTREENODESIZEPOWER)));

			if(leftOrRight)
			{
				parentNode->children = (curNodeIdx << 4);
			}
			else
			{
				parentNode->children2 = (curNodeIdx << 4) | (parentNode->children2 & 1);
			}

			if((parentNode->children2 & 1) == 1)
			{
				// parent node's BB isn't completed
				hasInCompleteBB = 1;
				memcpy(&node->min.e[0], &parentIndex, sizeof(int));
				unsigned int code = 0;
				if(leftOrRight)
				{
					// this is a left child

					// change Last to none
					code = (unsigned int)ic[0]->Decompress(0, 1);
					code = (code << NBITS) + (unsigned int)ic[0]->Decompress(0, 1);
					memcpy(&node->max.e[0], &code, sizeof(unsigned int));
					code = (unsigned int)ic[1]->Decompress(0, 1);
					code = (code << NBITS) + (unsigned int)ic[1]->Decompress(0, 1);
					memcpy(&node->max.e[1], &code, sizeof(unsigned int));
					code = (unsigned int)ic[1]->Decompress(0, 1);
					code = (code << NBITS) + (unsigned int)ic[1]->Decompress(0, 1);
					memcpy(&node->max.e[2], &code, sizeof(unsigned int));
				}
				else
				{
					// this is a right child
					code = (unsigned int)ic[0]->Decompress(0, 1);
					code = (code << NBITS) + (unsigned int)ic[0]->Decompress(0, 1);
					memcpy(&node->max.e[0], &code, sizeof(unsigned int));
					code = (unsigned int)ic[1]->Decompress(0, 1);
					code = (code << NBITS) + (unsigned int)ic[1]->Decompress(0, 1);
					memcpy(&node->max.e[1], &code, sizeof(unsigned int));
					code = (unsigned int)ic[1]->Decompress(0, 1);
					code = (code << NBITS) + (unsigned int)ic[1]->Decompress(0, 1);
					memcpy(&node->max.e[2], &code, sizeof(unsigned int));
				}
			}
			else
			{
				// parent node's BB is completed.
				hasInCompleteBB = 0;
				I32 parentMinQ[3];
				I32 parentMaxQ[3];
				I32 predictedQ;

				pq->EnQuantize(parentNode->min.e, parentMinQ);
				pq->EnQuantize(parentNode->max.e, parentMaxQ);

				biggestaxis = getBiggestAxis(parentMinQ, parentMaxQ);
				int axis1;
				int axis2;
				switch(biggestaxis)
				{
				case 0 : axis1 = 1; axis2 = 2; break;
				case 1 : axis1 = 2; axis2 = 0; break;
				case 2 : axis1 = 0; axis2 = 1; break;
				}

				predictedQ = (parentMinQ[biggestaxis] + parentMaxQ[biggestaxis]) >> 1;

				I32 qMin[3];
				I32 qMax[3];

				if(leftOrRight)
				{
					// this is a left child
					/*
					unsigned int sym = 63;
					if((sym & 1) == 1) qMin[biggestaxis] = ic[0]->DecompressLast(parentMinQ[biggestaxis], 1);
					else qMin[biggestaxis] = parentMinQ[biggestaxis];
					if((sym & 2) == 2) qMax[biggestaxis] = ic[0]->DecompressLast(predictedQ, 0);
					else qMax[biggestaxis] = parentMaxQ[biggestaxis];
					if((sym & 4) == 4) qMin[axis1] = ic[1]->DecompressLast(parentMinQ[axis1], 1);
					else qMin[axis1] = parentMinQ[axis1];
					if((sym & 8) == 8) qMax[axis1] = ic[1]->DecompressLast(parentMaxQ[axis1], 0);
					else qMax[axis1] = parentMaxQ[axis1];
					if((sym & 16) == 16) qMin[axis2] = ic[2]->DecompressLast(parentMinQ[axis2], 1);
					else qMin[axis2] = parentMinQ[axis2];
					if((sym & 32) == 32) qMax[axis2] = ic[2]->DecompressLast(parentMaxQ[axis2], 0);
					else qMax[axis2] = parentMaxQ[axis2];
					*/
					
					qMin[biggestaxis] = ic[0]->Decompress(parentMinQ[biggestaxis], 1);
					qMax[biggestaxis] = ic[0]->Decompress(predictedQ, 0);
					qMin[axis1] = ic[1]->Decompress(parentMinQ[axis1], 1);
					qMax[axis1] = ic[1]->Decompress(parentMaxQ[axis1], 0);
					qMin[axis2] = ic[1]->Decompress(parentMinQ[axis2], 1);
					qMax[axis2] = ic[1]->Decompress(parentMaxQ[axis2], 0);
					
					parentNode->children |= biggestaxis;
				}
				else
				{
					// this is a right child
					/*
					unsigned int sym = 63;
					if((isChildInThis & 2) == 2)
					{
						// left child is in this cluster
						sym = 0;
						BSPArrayTreeNodePtr lChild = (BSPArrayTreeNodePtr)(m_physicalMemory + (PCN*m_nodesPerCluster*sizeof(BSPArrayTreeNode) + ((parentNode->children >> 4)%m_nodesPerCluster)*sizeof(BSPArrayTreeNode)));
						BSPArrayTreeNodePtr rChild = node;
						I32 sMinQ[3];
						I32 sMaxQ[3];
						pq->EnQuantize(lChild->min.e, sMinQ);//, FLOOR);
						pq->EnQuantize(lChild->max.e, sMaxQ);//, CEIL);
						if(sMinQ[biggestaxis] == parentMinQ[biggestaxis]) sym |= 1;
						if(sMaxQ[biggestaxis] == parentMaxQ[biggestaxis]) sym |= 2;
						if(sMinQ[axis1] == parentMinQ[axis1]) sym |= 4;
						if(sMaxQ[axis1] == parentMaxQ[axis1]) sym |= 8;
						if(sMinQ[axis2] == parentMinQ[axis2]) sym |= 16;
						if(sMaxQ[axis2] == parentMaxQ[axis2]) sym |= 32;
					}
					sym = 63;
					*/
					/*
					if((sym & 1) == 1) qMin[biggestaxis] = ic[0]->DecompressLast(predictedQ, 1);
					else qMin[biggestaxis] = parentMinQ[biggestaxis];
					if((sym & 2) == 2) qMax[biggestaxis] = ic[0]->DecompressLast(parentMaxQ[biggestaxis], 0);
					else qMax[biggestaxis] = parentMaxQ[biggestaxis];
					if((sym & 4) == 4) qMin[axis1] = ic[1]->DecompressLast(parentMinQ[axis1], 1);
					else qMin[axis1] = parentMinQ[axis1];
					if((sym & 8) == 8) qMax[axis1] = ic[1]->DecompressLast(parentMaxQ[axis1], 0);
					else qMax[axis1] = parentMaxQ[axis1];
					if((sym & 16) == 16) qMin[axis2] = ic[2]->DecompressLast(parentMinQ[axis2], 1);
					else qMin[axis2] = parentMinQ[axis2];
					if((sym & 32) == 32) qMax[axis2] = ic[2]->DecompressLast(parentMaxQ[axis2], 0);
					else qMax[axis2] = parentMaxQ[axis2];
					*/
					qMin[biggestaxis] = ic[0]->Decompress(predictedQ, 1);
					qMax[biggestaxis] = ic[0]->Decompress(parentMaxQ[biggestaxis], 0);
					qMin[axis1] = ic[1]->Decompress(parentMinQ[axis1], 1);
					qMax[axis1] = ic[1]->Decompress(parentMaxQ[axis1], 0);
					qMin[axis2] = ic[1]->Decompress(parentMinQ[axis2], 1);
					qMax[axis2] = ic[1]->Decompress(parentMaxQ[axis2], 0);
				}
				pq->DeQuantize(qMin, node->min.e);
				pq->DeQuantize(qMax, node->max.e);
			}
		}

		int nodeAxis = rd_geom->decode(2);

		if(nodeAxis)
		{
			node->indexCount = 7;
			unsigned int TriID = 0;

			#ifdef _USE_TRI_3_TYPE_ENCODING
			// Compress triangle index
			#if defined(_USE_RACM) && defined(_USE_TRI_3_TYPE_ENCODING)
			unsigned int NewTris [10];

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

			//int type = rd_geom->decode(rmTriIndexType)+1;
			int type = rd_geom->decode(3)+1;
			// type 1 : cache
			if(type == 1 && i != 0)
			{
				TriID = triIDCache[rd_geom->decode(3)];
			}

			// type 2 : front
			if(type == 2 && i != 0)
			{
				int triIndex = rd_geom->decode(dvTriIndexFront->size()+1);//icTriIndexFront->DecompressNone();// rd_geom->decode(rmTriIndexFront);
				FrontTri *frontTri = (FrontTri*)dvTriIndexFront->getElementWithRelativeIndex(triIndex);
				TriID = frontTri->index;
			}

			// type 3 : base + offset
			if(type == 3)
			{
				unsigned int base;
				unsigned int offset;
				unsigned int baseIndex;
//					baseIndex = rd_geom->decode(rmTriIndexBase);
				baseIndex = rd_geom->decode(dvTriIndexBase->size()+1);//icTriIndexBase->DecompressNone();
				BaseTri *baseTri;
				if(baseIndex == 0)
				{
					base = rd_geom->decode(m_numClusters);
					baseTri = new BaseTri;
					baseTri->buffer_next = 0;
					baseTri->IsInDV = 0;
					baseTri->index = base;
					dvTriIndexBase->addElement(baseTri);
				}
				else
				{
					baseTri = (BaseTri*)dvTriIndexBase->getElementWithRelativeIndex(baseIndex-1);
					base = baseTri->index;
				}
				offset = rd_geom->decode(m_sizeBasePage);
				TriID = (base << m_sizeBasePagePower) + offset;
			}

			/*
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
			*/

			Corner = m_pMesh->GetCornerFromTriID (TriID);
			NextCorner = m_pMesh->GetNextCornerGivenSameTri (Corner);
			PrevCorner = m_pMesh->GetPrevCornerGivenSameTri (Corner);

			Vertex = m_pMesh->GetIncidentVertexFromCorner (Corner);
			NextVertex = m_pMesh->GetIncidentVertexFromCorner (NextCorner);
			PrevVertex = m_pMesh->GetIncidentVertexFromCorner (PrevCorner);

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
				if ((NewTriNum = m_pMesh->GetTrianglesSharingTwoVertices (v1, v2, NewTris, TriID, true)) == 1)
				{
					triIDCache[c] = NewTris[0];
					/*
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
					*/
				}
			}
			beforeTriID = TriID;
			node->indexOffset = TriID << 2;

			#else
			unsigned int base;
			unsigned int offset;
			unsigned int baseIndex;
			baseIndex = rd_geom->decode(dvTriIndexBase->size()+1);
			BaseTri *baseTri;
			if(baseIndex == 0)
			{
				base = rd_geom->decode(m_numClusters);
				baseTri = new BaseTri;
				baseTri->buffer_next = 0;
				baseTri->IsInDV = 0;
				baseTri->index = base;
				dvTriIndexBase->addElement(baseTri);
			}
			else
			{
				baseTri = (BaseTri*)dvTriIndexBase->getElementWithRelativeIndex(baseIndex-1);
				base = baseTri->index;
			}
			offset = rd_geom->decode(m_sizeBasePage);
			node->indexOffset = ((base << m_sizeBasePagePower) + offset) << 2;
			//node->indexOffset = rd_geom->decodeInt() << 2;
			#endif
			#endif
			
			#ifdef _USE_TRI_DELTA_ENCODING
			int sign = rd_geom->decode(2);
			int delta = decodeDeltaForTriIndex(m_numBoundary, m_listBoundary, rmDeltaTriIndex, rd_geom);
			if(delta == INT_MAX)
				TriID = rd_geom->decode((m_numNodes>>1)+1);
			else
				TriID = (beforeTriID + (sign ? delta : -delta));
			beforeTriID = TriID;
			node->indexOffset = TriID << 2;
			#endif
		}
		else
		{
			int isChildInThis = rd_geom->decode(rmIsChildInThis);
			unsigned int delta = 0;
			switch(isChildInThis)
			{
			case 0 : 
				delta = decodeDeltaForChildIndexInOutCluster(m_numBoundary, m_listBoundary, rmDeltaChildIndex, rd_geom);
				if(delta == UINT_MAX)
					node->children = rd_geom->decode(m_numNodes) << 4;
				else
					node->children = (beforeOutClusterChildIndex + delta) << 4;
				beforeOutClusterChildIndex = node->children >> 4;
				delta = decodeDeltaForChildIndexInOutCluster(m_numBoundary, m_listBoundary, rmDeltaChildIndex, rd_geom);
				if(delta == UINT_MAX)
					node->children2 = rd_geom->decode(m_numNodes) << 4;
				else
					node->children2 = (beforeOutClusterChildIndex + delta) << 4;
				beforeOutClusterChildIndex = node->children2 >> 4;
				break;
			case 1 :
				delta = decodeDeltaForChildIndexInOutCluster(m_numBoundary, m_listBoundary, rmDeltaChildIndex, rd_geom);
				if(delta == UINT_MAX)
					node->children = rd_geom->decode(m_numNodes) << 4;
				else
					node->children = (beforeOutClusterChildIndex + delta) << 4;
				beforeOutClusterChildIndex = node->children >> 4;
				break;
			case 2 :
				//beforeOutClusterChildIndex = node->children >> 4;
				delta = decodeDeltaForChildIndexInOutCluster(m_numBoundary, m_listBoundary, rmDeltaChildIndex, rd_geom);
				if(delta == UINT_MAX)
					node->children2 = rd_geom->decode(m_numNodes) << 4;
				else
					node->children2 = (beforeOutClusterChildIndex + delta) << 4;
				beforeOutClusterChildIndex = node->children2 >> 4;
				break;
			}

			if(isChildInThis > 0)
			{
				FrontNode* frontNode = new FrontNode;
				frontNode->buffer_next = 0;
				frontNode->IsInDV = 0;
				frontNode->index = curNodeIdx;
				frontNode->count = isChildInThis == 3 ? 0 : 1;
				frontNode->isChildInThis = isChildInThis;
				frontNode->axis = biggestaxis;

				dvNodeIndex->addElement(frontNode);
			}
		}
		// if parent node's BB isn't complete, this node, neither.
		node->children2 &= ~1u;
		node->children2 |= hasInCompleteBB;
	}

	while(dvNodeIndex->size() > 0)
	{
		delete dvNodeIndex->getAndRemoveFirstElement();
	}
	delete dvNodeIndex;

	#if defined(_USE_RACM) && defined(_USE_TRI_3_TYPE_ENCODING)
	delete frontHashTableTri;
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
	//delete rmTriIndexType;
	#else
	while(dvTriIndexBase->size() > 0)
	{
		delete dvTriIndexBase->getAndRemoveFirstElement();
	}
	delete dvTriIndexBase;
	#endif
	delete listParentIndex;
	delete rmIsChildInThis;
//		delete rmParentClusterOffset;
	for(int j=0;j<m_numBoundary;j++)
	{
		delete rmDeltaChildIndex[j];
		#ifdef _USE_TRI_DELTA_ENCODING
		delete rmDeltaTriIndex[j];
		#endif
	}
	delete rmDeltaChildIndex;
	#ifdef _USE_TRI_DELTA_ENCODING
	delete rmDeltaTriIndex;
	#endif

	rd_geom->done();
	delete rd_geom;

	ic[0]->FinishDecompressor();
	ic[1]->FinishDecompressor();

	delete ic[0];
	delete ic[1];
	return true;
}


#ifdef USE_MM
template <class T>
int RACBVH<T>::readInt()
{
	unsigned char c1 = m_CompressedFile[0].GetNextElement();
	unsigned char c2 = m_CompressedFile[0].GetNextElement();
	unsigned char c3 = m_CompressedFile[0].GetNextElement();
	unsigned char c4 = m_CompressedFile[0].GetNextElement();
	return (c4 << 24) | (c3 << 16) | (c2 << 8) | (c1 << 0);
}
template <class T>
unsigned int RACBVH<T>::readUInt()
{
	unsigned char c1 = m_CompressedFile[0].GetNextElement();
	unsigned char c2 = m_CompressedFile[0].GetNextElement();
	unsigned char c3 = m_CompressedFile[0].GetNextElement();
	unsigned char c4 = m_CompressedFile[0].GetNextElement();
	return (c4 << 24) | (c3 << 16) | (c2 << 8) | (c1 << 0);
}

template <class T>
long RACBVH<T>::readLong()
{
	unsigned char c1 = m_CompressedFile[0].GetNextElement();
	unsigned char c2 = m_CompressedFile[0].GetNextElement();
	unsigned char c3 = m_CompressedFile[0].GetNextElement();
	unsigned char c4 = m_CompressedFile[0].GetNextElement();
	return (c4 << 24) | (c3 << 16) | (c2 << 8) | (c1 << 0);
}

template <class T>
float RACBVH<T>::readFloat()
{
	unsigned char c1 = m_CompressedFile[0].GetNextElement();
	unsigned char c2 = m_CompressedFile[0].GetNextElement();
	unsigned char c3 = m_CompressedFile[0].GetNextElement();
	unsigned char c4 = m_CompressedFile[0].GetNextElement();
	float returnValue = 0;
	unsigned int temp = (c4 << 24) | (c3 << 16) | (c2 << 8) | (c1 << 0);
	memcpy(&returnValue, &temp, 4);
	return returnValue;
}
#endif

#ifdef USE_DM
template <class T>
int RACBVH<T>::readInt(long &pos)
{
	unsigned char c1 = m_CompressedFile[pos++];
	unsigned char c2 = m_CompressedFile[pos++];
	unsigned char c3 = m_CompressedFile[pos++];
	unsigned char c4 = m_CompressedFile[pos++];
	return (c4 << 24) | (c3 << 16) | (c2 << 8) | (c1 << 0);
}
template <class T>
unsigned int RACBVH<T>::readUInt(long &pos)
{
	unsigned char c1 = m_CompressedFile[pos++];
	unsigned char c2 = m_CompressedFile[pos++];
	unsigned char c3 = m_CompressedFile[pos++];
	unsigned char c4 = m_CompressedFile[pos++];
	return (c4 << 24) | (c3 << 16) | (c2 << 8) | (c1 << 0);
}

template <class T>
long RACBVH<T>::readLong(long &pos)
{
	unsigned char c1 = m_CompressedFile[pos++];
	unsigned char c2 = m_CompressedFile[pos++];
	unsigned char c3 = m_CompressedFile[pos++];
	unsigned char c4 = m_CompressedFile[pos++];
	return (c4 << 24) | (c3 << 16) | (c2 << 8) | (c1 << 0);
}

template <class T>
float RACBVH<T>::readFloat(long &pos)
{
	unsigned char c1 = m_CompressedFile[pos++];
	unsigned char c2 = m_CompressedFile[pos++];
	unsigned char c3 = m_CompressedFile[pos++];
	unsigned char c4 = m_CompressedFile[pos++];
	float returnValue = 0;
	unsigned int temp = (c4 << 24) | (c3 << 16) | (c2 << 8) | (c1 << 0);
	memcpy(&returnValue, &temp, 4);
	return returnValue;
}
#endif

template <class T>
int RACBVH<T>::loadClusterTable()
{
	#if !defined(USE_MM) && !defined(USE_DM)
	fread(&m_nodesPerCluster, sizeof(unsigned int), 1, fp[0]);
	fread(&m_sizeBasePage, sizeof(unsigned int), 1, fp[0]);
	fread(&m_sizeBasePagePower, sizeof(unsigned int), 1, fp[0]);
	fread(&m_numNodes, sizeof(unsigned int), 1, fp[0]);
	fread(&m_numClusters, sizeof(unsigned int), 1, fp[0]);
	fread(m_bb_min_f, sizeof(float), 3, fp[0]);
	fread(m_bb_max_f, sizeof(float), 3, fp[0]);

	fread(&m_numBoundary, sizeof(unsigned int), 1, fp[0]);
	m_listBoundary = new unsigned int[m_numBoundary];
	fread(m_listBoundary, sizeof(unsigned int), m_numBoundary, fp[0]);

	pq = new PositionQuantizerNew();
	
	pq->SetMinMax(m_bb_min_f, m_bb_max_f);
	pq->SetPrecision(NBITS);
	pq->SetupQuantizer();

	#endif

	#ifdef USE_MM
	m_nodesPerCluster = readUInt();
	m_sizeBasePage = readUInt();
	m_sizeBasePagePower = readUInt();
	m_numNodes = readUInt();
	m_numClusters = readUInt();
	m_bb_min_f[0] = readFloat();
	m_bb_min_f[1] = readFloat();
	m_bb_min_f[2] = readFloat();
	m_bb_max_f[0] = readFloat();
	m_bb_max_f[1] = readFloat();
	m_bb_max_f[2] = readFloat();
	m_numBoundary = readUInt();
	m_listBoundary = new unsigned int[m_numBoundary];
	for(int i=0;i<m_numBoundary;i++)
	{
		m_listBoundary[i] = readUInt();
	}

	pq = new PositionQuantizerNew();

	pq->SetMinMax(m_bb_min_f, m_bb_max_f);
	pq->SetPrecision(NBITS);
	pq->SetupQuantizer();

	#endif

	#ifdef USE_DM
	long pos = 0;
	m_nodesPerCluster = readUInt(pos);
	m_sizeBasePage = readUInt(pos);
	m_sizeBasePagePower = readUInt(pos);
	m_numNodes = readUInt(pos);
	m_numClusters = readUInt(pos);
	m_bb_min_f[0] = readFloat(pos);
	m_bb_min_f[1] = readFloat(pos);
	m_bb_min_f[2] = readFloat(pos);
	m_bb_max_f[0] = readFloat(pos);
	m_bb_max_f[1] = readFloat(pos);
	m_bb_max_f[2] = readFloat(pos);
	m_numBoundary = readUInt(pos);
	m_listBoundary = new unsigned int[m_numBoundary];
	for(int i=0;i<m_numBoundary;i++)
	{
		m_listBoundary[i] = readUInt(pos);
	}

	pq = new PositionQuantizerNew();

	pq->SetMinMax(m_bb_min_f, m_bb_max_f);
	pq->SetPrecision(NBITS);
	pq->SetupQuantizer();

	#endif

	m_nodesPerClusterPower = log((double)m_nodesPerCluster)/log(2.0);
	int offsetPower = log((double)(m_nodesPerCluster*sizeof(BSPArrayTreeNode)))/log(2.0);
	assert(pow(2.0, (int)m_nodesPerClusterPower) == m_nodesPerCluster);
	assert(pow(2.0, offsetPower) == m_nodesPerCluster*sizeof(BSPArrayTreeNode));

	OptionManager *opt = OptionManager::getSingletonPtr();
	m_maxNumPCN = min(m_numClusters, maxAllowedMem/(m_nodesPerCluster * sizeof(BSPArrayTreeNode)));

	#ifdef USE_DM
	#else
	#endif
	m_physicalMemory.Init("RACBVH", m_numNodes, maxAllowedMem/(m_nodesPerCluster * sizeof(BSPArrayTreeNode)), m_nodesPerCluster);
	m_physicalMemory.m_pRACBVH = this;

	#if !defined(USE_MM) && !defined(USE_DM)
	for(int i=0;i<m_numClusters;i++)
	{
		long offset;
		fread(&offset, sizeof(long), 1, fp[0]);
		m_physicalMemory.m_DiskClusterOffset[i] = offset;
	}
	#endif

	#ifdef USE_MM
	for(int i=0;i<m_numClusters;i++)
	{
		m_physicalMemory.m_DiskClusterOffset[i] = readLong();
	}
	#endif

	#ifdef USE_DM
	for(int i=0;i<m_numClusters;i++)
	{
		m_physicalMemory.m_DiskClusterOffset[i] = readLong(pos);
	}
	#endif

	printf("Load cluster table complete.\n");
	return 1;
}


template <class T>
void RACBVH<T>::completeBB(BSPArrayTreeNodePtr parentNode, BSPArrayTreeNodePtr node, int leftOrRight, unsigned int CN)
{
	node->children2 |= 3u;

	I32 parentMinQ[3];
	I32 parentMaxQ[3];
	I32 predictedQ;

	pq->EnQuantize(parentNode->min.e, parentMinQ);
	pq->EnQuantize(parentNode->max.e, parentMaxQ);

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

	I32 qMin[3];
	I32 qMax[3];

	I32 error[6];
	unsigned int code;
	memcpy(&code, &node->max[0], sizeof(unsigned int));
	error[0] = code >> NBITS;
	error[1] = code & BIT_MASK;
	memcpy(&code, &node->max[1], sizeof(unsigned int));
	error[2] = code >> NBITS;
	error[3] = code & BIT_MASK;
	memcpy(&code, &node->max[2], sizeof(unsigned int));
	error[4] = code >> NBITS;
	error[5] = code & BIT_MASK;

	if(leftOrRight)
	{
		// this is a left child
		/*
		unsigned int sym = 63;
		if((sym & 1) == 1) qMin[biggestaxis] = parentMinQ[biggestaxis] + error[0];
		else qMin[biggestaxis] = parentMinQ[biggestaxis];
		if((sym & 2) == 2) qMax[biggestaxis] = predictedQ - error[1];
		else qMax[biggestaxis] = parentMaxQ[biggestaxis];
		if((sym & 4) == 4) qMin[axis1] = parentMinQ[axis1] + error[2];
		else qMin[axis1] = parentMinQ[axis1];
		if((sym & 8) == 8) qMax[axis1] = parentMaxQ[axis1] - error[3];
		else qMax[axis1] = parentMaxQ[axis1];
		if((sym & 16) == 16) qMin[axis2] = parentMinQ[axis2] + error[4];
		else qMin[axis2] = parentMinQ[axis2];
		if((sym & 32) == 32) qMax[axis2] = parentMaxQ[axis2] - error[5];
		else qMax[axis2] = parentMaxQ[axis2];
		*/

		parentNode->children |= biggestaxis;

		qMin[biggestaxis] = parentMinQ[biggestaxis] + error[0];
		qMax[biggestaxis] = predictedQ - error[1];
		qMin[axis1] = parentMinQ[axis1] + error[2];
		qMax[axis1] = parentMaxQ[axis1] - error[3];
		qMin[axis2] = parentMinQ[axis2] + error[4];
		qMax[axis2] = parentMaxQ[axis2] - error[5];
	}
	else
	{
		// this is a right child
		/*
		unsigned int sym = 63;
		if(CN == (GETLEFTCHILD(node) >> 2) / m_nodesPerCluster)
		{
			// left child is in this cluster
			sym = 0;
			long offset = (GETLEFTCHILD(parentNode) << 3) & offsetMask;
			BSPArrayTreeNodePtr lChild = (BSPArrayTreeNodePtr)(PosCluster + offset);
			BSPArrayTreeNodePtr rChild = node;
			I32 sMinQ[3];
			I32 sMaxQ[3];
			pq->EnQuantize(lChild->min.e, sMinQ);//, FLOOR);
			pq->EnQuantize(lChild->max.e, sMaxQ);//, CEIL);
			if(sMinQ[biggestaxis] == parentMinQ[biggestaxis]) sym |= 1;
			if(sMaxQ[biggestaxis] == parentMaxQ[biggestaxis]) sym |= 2;
			if(sMinQ[axis1] == parentMinQ[axis1]) sym |= 4;
			if(sMaxQ[axis1] == parentMaxQ[axis1]) sym |= 8;
			if(sMinQ[axis2] == parentMinQ[axis2]) sym |= 16;
			if(sMaxQ[axis2] == parentMaxQ[axis2]) sym |= 32;
		}
		sym = 63;
		if((sym & 1) == 1) qMin[biggestaxis] = predictedQ + error[0];
		else qMin[biggestaxis] = parentMinQ[biggestaxis];
		if((sym & 2) == 2) qMax[biggestaxis] = parentMaxQ[biggestaxis] - error[1];
		else qMax[biggestaxis] = parentMaxQ[biggestaxis];
		if((sym & 4) == 4) qMin[axis1] = parentMinQ[axis1] + error[2];
		else qMin[axis1] = parentMinQ[axis1];
		if((sym & 8) == 8) qMax[axis1] = parentMaxQ[axis1] - error[3];
		else qMax[axis1] = parentMaxQ[axis1];
		if((sym & 16) == 16) qMin[axis2] = parentMinQ[axis2] + error[4];
		else qMin[axis2] = parentMinQ[axis2];
		if((sym & 32) == 32) qMax[axis2] = parentMaxQ[axis2] - error[5];
		else qMax[axis2] = parentMaxQ[axis2];
		*/
		qMin[biggestaxis] = predictedQ + error[0];
		qMax[biggestaxis] = parentMaxQ[biggestaxis] - error[1];
		qMin[axis1] = parentMinQ[axis1] + error[2];
		qMax[axis1] = parentMaxQ[axis1] - error[3];
		qMin[axis2] = parentMinQ[axis2] + error[4];
		qMax[axis2] = parentMaxQ[axis2] - error[5];
	}
	for(int i=0;i<3;i++)
	{
		if(qMin[i] >= m_maxRange) qMin[i] -= m_maxRange;
		if(qMin[i] < 0) qMin[i] += m_maxRange;
		if(qMax[i] >= m_maxRange) qMax[i] -= m_maxRange;
		if(qMax[i] < 0) qMax[i] += m_maxRange;
	}

	pq->DeQuantize(qMin, node->min.e);
	pq->DeQuantize(qMax, node->max.e);

	node->children2 &= ~1u;

	if(!ISLEAF(node))
	{

		if(CN == (GETLEFTCHILD(node) >> 2) / m_nodesPerCluster)
		{
			BSPArrayTreeNodePtr left = &m_physicalMemory[node->children >> 4];
			completeBB(node, left, 1, CN);
		}
		if(CN == (GETRIGHTCHILD(node) >> 2) / m_nodesPerCluster)
		{
			BSPArrayTreeNodePtr right = &m_physicalMemory[node->children2 >> 4];
			completeBB(node, right, 0, CN);
		}
	}

	node->children2 &= ~3u;
}

template <class T>
unsigned int RACBVH<T>::decodeDeltaForChildIndexInOutCluster(unsigned int numBoundary, unsigned int *listBoundary, RangeModel **rmDeltaChildIndex, RangeDecoder *rd)
{
	for(unsigned int pass=0;pass<numBoundary;pass++)
	{
		unsigned int delta = rd->decode(rmDeltaChildIndex[pass]);
		if(delta < listBoundary[pass])
		{
			return delta;
		}
	}
	return UINT_MAX;
}

template <class T>
unsigned int RACBVH<T>::decodeDeltaForTriIndex(unsigned int numBoundary, unsigned int *listBoundary, RangeModel **rmDeltaTriIndex, RangeDecoder *rd)
{
	for(unsigned int pass=0;pass<numBoundary;pass++)
	{
		unsigned int delta = rd->decode(rmDeltaTriIndex[pass]);
		if(delta < listBoundary[pass])
		{
			return delta;
		}
	}
	return INT_MAX;
}

#endif