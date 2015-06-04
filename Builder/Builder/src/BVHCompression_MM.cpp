#include <windows.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "OptionManager.h"
#include "BVHNodeDefine.h"
#include "rangeencoder.h"
#include "positionquantizer_new.h"
#include "integercompressor_new.h"

#include "BVHCompression.h"

BVHCompression::BVHCompression(void)
{
}

BVHCompression::~BVHCompression(void)
{
}

static PositionQuantizerNew* pq;
static IntegerCompressorNew* ic[3];

static void compressVertexPosition(float* n)
{
#ifdef PRINT_CONTROL_OUTPUT
  prediction_none++;
#endif

  if (pq)
  {
    pq->EnQuantize(n, (int*)n);
    for (int i = 0; i < 3; i++)
    {
      ic[i]->CompressNone(((int*)n)[i]);
    }
  }
  /*
  else
  {
    for (int i = 0; i < 3; i++)
    {
      n[i] = fc[i]->CompressNone(n[i]);
    }
  }
  */
}

static void compressVertexPosition(const float* l, float* n)
{
#ifdef PRINT_CONTROL_OUTPUT
  prediction_last++;
#endif

  if (pq)
  {
    pq->EnQuantize(l, (int*)l);
    pq->EnQuantize(n, (int*)n);
    for (int i = 0; i < 3; i++)
    {
      ic[i]->CompressLast(((const int*)l)[i],((int*)n)[i]);
    }
  }
  /*
  else
  {
    for (int i = 0; i < 3; i++)
    {
      n[i] = fc[i]->CompressLast(l[i],n[i]);
    }
  }
  */
}


#define ISLEAF(node) (((node)->children & 3) == 3)
#define GETNODEOFFSET(idx) ((idx >> 2) << 3)
int BVHCompression::compress(const char* filename)
{
	OptionManager *opt = OptionManager::getSingletonPtr();
	unsigned int numNodes = 0;
	unsigned int numClusters = 0;
	unsigned int nodesPerCluster = opt->getOptionAsInt("raytracing", "nodesPerCluster", 65536);
	unsigned int maxDepth = opt->getOptionAsInt("raytracing", "maxBSPTreeDepth", 100);
	char nodeName[MAX_PATH], mNodeName[MAX_PATH], compressedName[MAX_PATH];
	FILE *fpo, *fpm, *fpc;
	BSPArrayTreeNode currentNode;

	sprintf(nodeName, "%s.node", filename);
	sprintf(mNodeName, "%s.tmp", filename);
	sprintf(compressedName, "%s.cmp", filename);

	fpo = fopen(nodeName, "rb");
	fpm = fopen(mNodeName, "wb");

	typedef struct {
		unsigned int index;
		Vector3 min;
		Vector3 max;
	} StackElem;

	StackElem *stack = new StackElem[maxDepth];
	unsigned int curIndex;
	int stackPtr = 0;

	float bb_min_f[3] = {0, };
	float bb_max_f[3] = {0, };
	bool isSetBB = false;

	// root
	BSPArrayTreeNode parrentNode;
	curIndex = 0;
	stack[0].index = curIndex;
	stack[0].min = parrentNode.min;
	stack[0].max = parrentNode.max;
	stackPtr++;

	while(stackPtr > 0)
	{
		BSPArrayTreeNode mCurrentNode;
		fseek(fpo, GETNODEOFFSET(curIndex), SEEK_SET);
		fseek(fpm, GETNODEOFFSET(curIndex), SEEK_SET);

		fread(&currentNode, sizeof(BSPArrayTreeNode), 1, fpo);
		mCurrentNode = currentNode;
		mCurrentNode.min -= parrentNode.min;
		mCurrentNode.max -= parrentNode.max;
		if(!isSetBB)
		{
			bb_min_f[0] = currentNode.min.e[0];
			bb_min_f[1] = currentNode.min.e[1];
			bb_min_f[2] = currentNode.min.e[2];
			bb_max_f[0] = currentNode.max.e[0];
			bb_max_f[1] = currentNode.max.e[1];
			bb_max_f[2] = currentNode.max.e[2];
			isSetBB = true;
		}
		fwrite(&mCurrentNode, sizeof(BSPArrayTreeNode), 1, fpm);
		numNodes++;

		if(!ISLEAF(&currentNode))
		{
			Vector3 diff = currentNode.max - currentNode.min;
			int biggestaxis = diff.indexOfMaxComponent();
			curIndex = currentNode.children;
			parrentNode = currentNode;
			parrentNode.max.e[biggestaxis] = (float)(0.5 * diff.e[biggestaxis] + currentNode.min.e[biggestaxis]);
			stack[stackPtr].index = currentNode.children2;
			stack[stackPtr].min = currentNode.min;
			stack[stackPtr].max = currentNode.max;
			stack[stackPtr].min.e[biggestaxis] = (float)(0.5 * diff.e[biggestaxis] + currentNode.min.e[biggestaxis]);

			stackPtr++;
			continue;
		}

		stackPtr--;
		curIndex = stack[stackPtr].index;
		parrentNode.min = stack[stackPtr].min;
		parrentNode.max = stack[stackPtr].max;
	}
	numClusters = ceil(((float)numNodes)/nodesPerCluster);

	delete stack;

	fclose(fpm);
	fclose(fpo);

	fpo = fopen(nodeName, "rb");
	fpm = fopen(mNodeName, "rb");
	fpc = fopen(compressedName, "wb");

	fwrite(&nodesPerCluster, sizeof(unsigned int), 1, fpc);
	fwrite(&numNodes, sizeof(unsigned int), 1, fpc);
	fwrite(&numClusters, sizeof(unsigned int), 1, fpc);
	fwrite(bb_min_f, sizeof(float), 3, fpc);
	fwrite(bb_max_f, sizeof(float), 3, fpc);

	// reserve for offset list
	long posOffset = ftell(fpc);
	fseek(fpc, sizeof(long)*numClusters, SEEK_CUR);

	long *offsets = new long[numClusters];


	/*
	int nbits = 16;

	pq = new PositionQuantizerNew();
    pq->SetMinMax(bb_min_f, bb_max_f);
    pq->SetPrecision(nbits);
    pq->SetupQuantizer();

    ic[0] = new IntegerCompressorNew();
    ic[1] = new IntegerCompressorNew();
    ic[2] = new IntegerCompressorNew();

	ic[0]->SetRange(pq->m_aiQuantRange[0]);
    ic[1]->SetRange(pq->m_aiQuantRange[1]);
    ic[2]->SetRange(pq->m_aiQuantRange[2]);

	ic[0]->SetPrecision(nbits);
    ic[1]->SetPrecision(nbits);
    ic[2]->SetPrecision(nbits);

	ic[0]->SetupCompressor(re_geom);
    ic[1]->SetupCompressor(re_geom);
    ic[2]->SetupCompressor(re_geom);
	*/
	BSPArrayTreeNode mCurrentNode;
	for(int i=0;i<numClusters;i++)
	{

		offsets[i] = ftell(fpc);

	RangeEncoder *re_geom = new RangeEncoder(fpc);
	
		for(int j=0;j<nodesPerCluster && !feof(fpm);j++)
		{
			/*
			fread(&currentNode, sizeof(BSPArrayTreeNode), 1, fpo);
			fread(&mCurrentNode, sizeof(BSPArrayTreeNode), 1, fpm);
			re_geom->encodeInt(currentNode.children);
			re_geom->encodeInt(currentNode.children2);
			compressVertexPosition(mCurrentNode.min.e, currentNode.min.e);
			compressVertexPosition(mCurrentNode.max.e, currentNode.max.e);
			*/
			fread(&mCurrentNode, sizeof(BSPArrayTreeNode), 1, fpm);
			re_geom->encodeInt(mCurrentNode.children);
			re_geom->encodeInt(mCurrentNode.children2);
			re_geom->encodeFloat(mCurrentNode.min.e[0]);
			re_geom->encodeFloat(mCurrentNode.min.e[1]);
			re_geom->encodeFloat(mCurrentNode.min.e[2]);
			re_geom->encodeFloat(mCurrentNode.max.e[0]);
			re_geom->encodeFloat(mCurrentNode.max.e[1]);
			re_geom->encodeFloat(mCurrentNode.max.e[2]);
		}
	re_geom->done();
	delete re_geom;
	}

	/*
    ic[0]->FinishCompressor();
    ic[1]->FinishCompressor();
    ic[2]->FinishCompressor();

	delete pq;
	delete ic[0];
	delete ic[1];
	delete ic[2];
	*/

	fseek(fpc, posOffset, SEEK_SET);
	fwrite(offsets, sizeof(long), numClusters, fpc);
	fseek(fpc, 0, SEEK_END);

	delete offsets;

	fclose(fpo);
	fclose(fpc);
	fclose(fpm);
	return 0;
}
