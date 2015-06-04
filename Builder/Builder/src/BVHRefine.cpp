#include <windows.h>
#include <string.h>
#include <stdio.h>
#include <io.h>
#include "OptionManager.h"
#include "BVHRefine.h"

BVHRefine::BVHRefine(void)
{
}

BVHRefine::~BVHRefine(void)
{
}

#define ISLEAF(node) (((node)->children & 3) == 3)
#define GETNODEOFFSET(idx) ((idx >> 2) << 3)
bool BVHRefine::refine(const char* filename)
{
	OptionManager *opt = OptionManager::getSingletonPtr();
	unsigned int maxDepth = opt->getOptionAsInt("raytracing", "maxBSPTreeDepth", 100);
	char nodeNameO[MAX_PATH];
	char nodeNameD[MAX_PATH];
	sprintf(nodeNameO, "%s.node", filename);
	sprintf(nodeNameD, "%s.rfd", filename);
	fpo = fopen(nodeNameO, "rb");
	fpd = fopen(nodeNameD, "wb");

	BSPArrayTreeNode root;
	fseek(fpo, 0, SEEK_SET);
	fread(&root, sizeof(BSPArrayTreeNode), 1, fpo);

	float *bb_min_f = root.min.e;
	float *bb_max_f = root.max.e;

	int nbits = 16;

	pq = new PositionQuantizerNew();
	pq->SetMinMax(bb_min_f, bb_max_f);
	pq->SetPrecision(nbits);
	pq->SetupQuantizer();

	Vector3 min, max;
	refineRecursive(0, min, max);

	delete pq;

	fclose(fpo);
	fclose(fpd);
	return true;
}

bool BVHRefine::refineRecursive(unsigned int idx, Vector3 &min, Vector3 &max)
{
	BSPArrayTreeNode node;
	fseek(fpo, GETNODEOFFSET(idx), SEEK_SET);
	fread(&node, sizeof(BSPArrayTreeNode), 1, fpo);
	I32 minQ[3], maxQ[3];
	pq->EnQuantize(node.min.e, minQ);
	pq->EnQuantize(node.max.e, maxQ);
	pq->DeQuantize(minQ, node.min.e);
	pq->DeQuantize(maxQ, node.max.e);
	min = node.min;
	max = node.max;

	if(ISLEAF(&node)) return true;

	Vector3 lChildMin, lChildMax;
	Vector3 rChildMin, rChildMax;
	refineRecursive(node.children, lChildMin, lChildMax);
	refineRecursive(node.children2, rChildMin, rChildMax);
	updateBB(node.min, node.max, lChildMin);
	updateBB(node.min, node.max, lChildMax);
	updateBB(node.min, node.max, rChildMin);
	updateBB(node.min, node.max, rChildMax);

	fseek(fpd, GETNODEOFFSET(idx), SEEK_SET);
	fwrite(&node, sizeof(BSPArrayTreeNode), 1, fpd);
	min = node.min;
	max = node.max;

	return true;
}