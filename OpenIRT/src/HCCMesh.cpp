#include "CommonOptions.h"
#include "defines.h"
#include "CommonHeaders.h"
#include "handler.h"

#include "HCCMesh.h"
#include "Matrix.h"
#include <io.h>

#include "OpenIRT.h"

#define EN_QUANTIZE(pos, qPos) \
	(qPos)[0] = (unsigned short)(m_qEnMult * ((double)(pos)[0] - (double)m_BB.min.e[0])); \
	(qPos)[1] = (unsigned short)(m_qEnMult * ((double)(pos)[1] - (double)m_BB.min.e[1])); \
	(qPos)[2] = (unsigned short)(m_qEnMult * ((double)(pos)[2] - (double)m_BB.min.e[2]));

#define DE_QUANTIZE(qPos, pos) \
	(pos)[0] = (float)(m_qDeMult*(qPos)[0] + (double)m_BB.min.e[0]); \
	(pos)[1] = (float)(m_qDeMult*(qPos)[1] + (double)m_BB.min.e[1]); \
	(pos)[2] = (float)(m_qDeMult*(qPos)[2] + (double)m_BB.min.e[2]);

#define DE_QUANTIZE_1(qPos, pos, idx) \
	*(pos) = (float)(m_qDeMult*(qPos)[idx] + (double)m_BB.min.e[idx]); \


#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif

using namespace irt;

HCCMesh::HCCMesh(void)
{
	m_numClusters = 0;
	m_compFile = NULL;
	m_compHighTree = NULL;
	m_compCluster = NULL;

	stacks = new StackElem *[MAX_NUM_THREADS*MAX_NUM_INTERSECTION_STREAM];

	for(int i=0;i<MAX_NUM_THREADS*MAX_NUM_INTERSECTION_STREAM;i++)
		stacks[i] = (StackElem *)_aligned_malloc(150 * sizeof(StackElem), 16);
}

HCCMesh::~HCCMesh(void)
{
	if(m_compFile) delete[] m_compFile;
	if(m_compHighTree) delete[] m_compHighTree;

	if(m_compCluster)
	{
		for(int i=1;i<(int)m_numClusters+1;i++)
		{
			delete[] m_compCluster[i].node;
			delete[] m_compCluster[i].supp;
			delete[] m_compCluster[i].vert;
		}
		delete[] m_compCluster;
	}

	if(stacks)
	{
		for(int i=0;i<MAX_NUM_THREADS;i++)
			_aligned_free(stacks[i]);
		delete[] stacks;
	}
}

void HCCMesh::calculateQuantizedNormals()
{
	// calculate quantized normals;
	float delta = 2.0f/(NUM_QUANTIZE_NORMALS-1);
	float u, v;
	int i0, i1, i2;
	float sign = -1.0f;

	for(int plane=0;plane<6;plane++)
	{
		i0 = plane/2;
		i1 = (i0 + 1) % 3;
		i2 = (i0 + 2) % 3;
		v = -1.0f;
		for(int i=0;i<NUM_QUANTIZE_NORMALS;i++)
		{
			u = -1.0f;
			for(int j=0;j<NUM_QUANTIZE_NORMALS;j++)
			{
				Vector3 &n = m_quantizedNormals[i*NUM_QUANTIZE_NORMALS+j + NUM_QUANTIZE_NORMALS*NUM_QUANTIZE_NORMALS*plane];
				n.e[i0] = sign;
				n.e[i1] = u;
				n.e[i2] = v;
				n.makeUnitVector();
				u += delta;
			}
			v += delta;
		}
		sign *= -1.0f;
	}
}

void HCCMesh::readHeader(FILE *fp)
{
	// load templates
	fread(&m_numTemplates, sizeof(int), 1, fp);
	fread(m_templates, sizeof(TemplateTable), m_numTemplates, fp);

	// load number of clusters
	fread(&m_numClusters, sizeof(unsigned int), 1, fp);
	// ignore first cluster (cluster 0)
	m_numClusters--;

	CompClusterHeader header;

	// load high level tree (cluster 0)
	fread(&header, sizeof(CompClusterHeader), 1, fp);
	m_compHighTree = new BVHNode[header.numNode];
	m_BB.min = header.BBMin;
	m_BB.max = header.BBMax;
	

	/*
#ifdef USE_VERTEX_QUANTIZE
	pqVert.SetMinMax(header.BBMin.e, header.BBMax.e);
	pqVert.SetPrecision(16);
	pqVert.SetupQuantizer();
#endif
	*/
	Vector3 diff = header.BBMax - header.BBMin;
	double maxQRange = max(max(diff.e[0], diff.e[1]), diff.e[2]);
	m_qEnMult = (double)m_qStep / maxQRange;
	m_qDeMult = maxQRange / (double)m_qStep;

	fread(m_compHighTree, sizeof(BVHNode), header.numNode, fp);
}

bool HCCMesh::load(const char *fileName)
{
	strcpy_s(m_fileName, 256, fileName);
	char compFileName[MAX_PATH];
	char matFileName[MAX_PATH];
	
	sprintf_s(compFileName, MAX_PATH, "%s\\data.hccmesh", fileName);
	sprintf_s(matFileName, MAX_PATH, "%s\\material.mtl", fileName);

	FILE *fp;
	errno_t err;

	if(err = fopen_s(&fp, compFileName, "rb"))
	{
		printf("File open error [%d] : %s", err, compFileName);
		return false;
	}

	// read header and high level BVH
	readHeader(fp);

	Progress &prog = OpenIRT::getSingletonPtr()->getProgress();
	prog.reset(m_numClusters+1);
	prog.setText("Header");

	prog.step();

	char progText[256];

#	ifdef USE_HCCMESH_QUANTIZATION
	m_compCluster = new QCompCluster[m_numClusters+1];
#	else
	m_compCluster = new CompCluster[m_numClusters+1];
#	endif

	CompClusterHeader header;

#	ifndef TEST_VOXEL
	for(int i=1;i<(int)m_numClusters+1;i++)
	{
		if(i % 1000 == 0)
		{
			sprintf_s(progText, 256, "Cluster %d/%d", i, m_numClusters+1);
			prog.setText(progText);
		}

		fread(&header, sizeof(CompClusterHeader), 1, fp);
		m_compCluster[i].header = header;
		m_compCluster[i].node = new CompTreeNode[header.numNode];
		m_compCluster[i].supp = new CompTreeSupp[header.numSupp];
#		ifdef USE_HCCMESH_QUANTIZATION
		m_compCluster[i].vert = new QCompTreeVert[header.numVert];
#		else
		m_compCluster[i].vert = new CompTreeVert[header.numVert];
#		endif
#ifdef USE_HCCMESH_MT
		m_compCluster[i].tris = new unsigned char[header.sizeTris];
#endif
		if(m_compCluster[i].node == NULL || m_compCluster[i].supp == NULL || m_compCluster[i].vert == NULL)
		{
			printf("Our of memory!\n");
			exit(-1);
		}
		fread(m_compCluster[i].node, sizeof(CompTreeNode), header.numNode, fp);
		fread(m_compCluster[i].supp, sizeof(CompTreeSupp), header.numSupp, fp);

#		ifdef USE_HCCMESH_QUANTIZATION
		CompTreeVert *temp = new CompTreeVert[header.numVert];
		fread(temp, sizeof(CompTreeVert), header.numVert, fp);
		for(unsigned int j=0;j<header.numVert;j++)
		{
			EN_QUANTIZE(temp[j].vert.e, m_compCluster[i].vert[j].qV);
			m_compCluster[i].vert[j].data = *((unsigned int*)&(temp[j].vert.m_alpha));
		}
		delete[] temp;
#		else
		fread(m_compCluster[i].vert, sizeof(CompTreeVert), header.numVert, fp);
#		endif
#ifdef USE_HCCMESH_MT
		fread(m_compCluster[i].tris, header.sizeTris, 1, fp);
#endif
		prog.step();
	}
#	endif

	fclose(fp);

	calculateQuantizedNormals();

	if(!loadMaterialFromMTL(matFileName, m_matList))
	{
		printf("Load material file error : %s\n", matFileName);

		char matOOCFileName[MAX_PATH];

		sprintf_s(matOOCFileName, MAX_PATH, "%s\\materials.ooc", fileName);
		printf("Generate MTL from file : %s\n", matOOCFileName);

		if(generateMTLFromOOCMaterial(matOOCFileName, matFileName))
		{
			loadMaterialFromMTL(matFileName, m_matList);
		}
		else
		{
			printf("Failed! Use default material\n");
			Material mat;
			m_matList.push_back(mat);
		}
	}

	return true;
}

BVHNode *HCCMesh::getBV(unsigned int index, TravStat &ts, unsigned int minBB, unsigned int maxBB)
{
	if(ts.cluster == 0)
	{
		// in high level tree
		ts.axis = m_compHighTree[ts.index].left & 0x3;
		return &m_compHighTree[ts.index];
	}

#ifdef USE_O_HCCMESH
	/*if(m_useOHCCMesh)*/ access(ts.cluster, threadID);
#endif

	ts.node.left = m_compCluster[ts.cluster].node[ts.index].data;
	ts.axis = ts.node.left & 0x3;

	if(minBB == 0 || maxBB == 0) return &ts.node;

#	ifdef USE_HCCMESH_QUANTIZATION
	QCompTreeVert *vert = m_compCluster[ts.cluster].vert;
#	else
	CompTreeVert *vert = m_compCluster[ts.cluster].vert;
#	endif

#ifndef USE_HCCMESH_MT
	if(CISLEAF(&ts.node))
	{
		unsigned int vi[3] = {
			(ts.node.left >> 23) & BIT_MASK_9,
			(ts.node.left >> 14) & BIT_MASK_9,
			(ts.node.left >>  5) & BIT_MASK_9};

		/*
#ifdef USE_VERTEX_QUANTIZE
#if 0
		Vector3 v[3];
		pqVert.DeQuantize(vert[vi[0]].qV, v[0].e);
		pqVert.DeQuantize(vert[vi[1]].qV, v[1].e);
		pqVert.DeQuantize(vert[vi[2]].qV, v[2].e);
		float xv[3] = {v[0].e[0], v[1].e[0], v[2].e[0]};
		float yv[3] = {v[0].e[1], v[1].e[1], v[2].e[1]};
		float zv[3] = {v[0].e[2], v[1].e[2], v[2].e[2]};

		ts.node.min.e[0] = (xv[0] < xv[1] && xv[0] < xv[2]) ? xv[0] : (xv[1] < xv[2] ? xv[1] : xv[2]);
		ts.node.min.e[1] = (yv[0] < yv[1] && yv[0] < yv[2]) ? yv[0] : (yv[1] < yv[2] ? yv[1] : yv[2]);
		ts.node.min.e[2] = (zv[0] < zv[1] && zv[0] < zv[2]) ? zv[0] : (zv[1] < zv[2] ? zv[1] : zv[2]);
		ts.node.max.e[0] = (xv[0] > xv[1] && xv[0] > xv[2]) ? xv[0] : (xv[1] > xv[2] ? xv[1] : xv[2]);
		ts.node.max.e[1] = (yv[0] > yv[1] && yv[0] > yv[2]) ? yv[0] : (yv[1] > yv[2] ? yv[1] : yv[2]);
		ts.node.max.e[2] = (zv[0] > zv[1] && zv[0] > zv[2]) ? zv[0] : (zv[1] > zv[2] ? zv[1] : zv[2]);
#else
		__m128 v[3];
		pqVert.DeQuantize(vert[vi[0]].qV, &v[0]);
		pqVert.DeQuantize(vert[vi[1]].qV, &v[1]);
		pqVert.DeQuantize(vert[vi[2]].qV, &v[2]);

		__m128 min, max;
		float minF[4], maxF[4];
		min = _mm_min_ps(_mm_min_ps(v[0], v[1]), v[2]);
		max = _mm_max_ps(_mm_max_ps(v[0], v[1]), v[2]);
		_mm_storeu_ps(minF, min);
		_mm_storeu_ps(maxF, max);
		ts.node.min.set(minF);
		ts.node.max.set(maxF);
#endif
		*/
#		ifdef USE_HCCMESH_QUANTIZATION
		Vector3 v[3];
		DE_QUANTIZE(vert[vi[0]].qV, v[0].e);
		DE_QUANTIZE(vert[vi[1]].qV, v[1].e);
		DE_QUANTIZE(vert[vi[2]].qV, v[2].e);
		float xv[3] = {v[0].e[0], v[1].e[0], v[2].e[0]};
		float yv[3] = {v[0].e[1], v[1].e[1], v[2].e[1]};
		float zv[3] = {v[0].e[2], v[1].e[2], v[2].e[2]};

		ts.node.min.e[0] = (xv[0] < xv[1] && xv[0] < xv[2]) ? xv[0] : (xv[1] < xv[2] ? xv[1] : xv[2]);
		ts.node.min.e[1] = (yv[0] < yv[1] && yv[0] < yv[2]) ? yv[0] : (yv[1] < yv[2] ? yv[1] : yv[2]);
		ts.node.min.e[2] = (zv[0] < zv[1] && zv[0] < zv[2]) ? zv[0] : (zv[1] < zv[2] ? zv[1] : zv[2]);
		ts.node.max.e[0] = (xv[0] > xv[1] && xv[0] > xv[2]) ? xv[0] : (xv[1] > xv[2] ? xv[1] : xv[2]);
		ts.node.max.e[1] = (yv[0] > yv[1] && yv[0] > yv[2]) ? yv[0] : (yv[1] > yv[2] ? yv[1] : yv[2]);
		ts.node.max.e[2] = (zv[0] > zv[1] && zv[0] > zv[2]) ? zv[0] : (zv[1] > zv[2] ? zv[1] : zv[2]);
#else
		_Vector4 *v[3] = {&vert[vi[0]].vert, &vert[vi[1]].vert, &vert[vi[2]].vert};
		float xv[3] = {v[0]->e[0], v[1]->e[0], v[2]->e[0]};
		float yv[3] = {v[0]->e[1], v[1]->e[1], v[2]->e[1]};
		float zv[3] = {v[0]->e[2], v[1]->e[2], v[2]->e[2]};

		ts.node.min.e[0] = (xv[0] < xv[1] && xv[0] < xv[2]) ? xv[0] : (xv[1] < xv[2] ? xv[1] : xv[2]);
		ts.node.min.e[1] = (yv[0] < yv[1] && yv[0] < yv[2]) ? yv[0] : (yv[1] < yv[2] ? yv[1] : yv[2]);
		ts.node.min.e[2] = (zv[0] < zv[1] && zv[0] < zv[2]) ? zv[0] : (zv[1] < zv[2] ? zv[1] : zv[2]);
		ts.node.max.e[0] = (xv[0] > xv[1] && xv[0] > xv[2]) ? xv[0] : (xv[1] > xv[2] ? xv[1] : xv[2]);
		ts.node.max.e[1] = (yv[0] > yv[1] && yv[0] > yv[2]) ? yv[0] : (yv[1] > yv[2] ? yv[1] : yv[2]);
		ts.node.max.e[2] = (zv[0] > zv[1] && zv[0] > zv[2]) ? zv[0] : (zv[1] > zv[2] ? zv[1] : zv[2]);
#endif

		float diff[3] = {
			ts.node.max.e[0] - ts.node.min.e[0],
			ts.node.max.e[1] - ts.node.min.e[1],
			ts.node.max.e[2] - ts.node.min.e[2]};
		ts.axis = (diff[0] > diff[1] && diff[0] > diff[2]) ? 0 : (diff[1] > diff[2] ? 1 : 2);
		
		return &ts.node;
	}
#endif

	unsigned int vi[6] = {
		(minBB >> 23) & BIT_MASK_9,
		(minBB >> 14) & BIT_MASK_9,
		(minBB >>  5) & BIT_MASK_9,
		(maxBB >> 23) & BIT_MASK_9,
		(maxBB >> 14) & BIT_MASK_9,
		(maxBB >>  5) & BIT_MASK_9};
	unsigned int flag = (minBB >> 2) & BIT_MASK_3;
	flag = (flag << 3) | ((maxBB >> 2) & BIT_MASK_3);
	if(ts.isLeft == 0) flag = ~flag;

	unsigned int curPos = 0x20;

	float v[6];
	/*
#ifdef USE_VERTEX_QUANTIZE
	pqVert.DeQuantize(vert[vi[0]].qV, &v[0], 0);
	pqVert.DeQuantize(vert[vi[1]].qV, &v[1], 1);
	pqVert.DeQuantize(vert[vi[2]].qV, &v[2], 2);
	pqVert.DeQuantize(vert[vi[3]].qV, &v[3], 0);
	pqVert.DeQuantize(vert[vi[4]].qV, &v[4], 1);
	pqVert.DeQuantize(vert[vi[5]].qV, &v[5], 2);
	*/
#	ifdef USE_HCCMESH_QUANTIZATION
	DE_QUANTIZE_1(vert[vi[0]].qV, &v[0], 0);
	DE_QUANTIZE_1(vert[vi[1]].qV, &v[1], 1);
	DE_QUANTIZE_1(vert[vi[2]].qV, &v[2], 2);
	DE_QUANTIZE_1(vert[vi[3]].qV, &v[3], 0);
	DE_QUANTIZE_1(vert[vi[4]].qV, &v[4], 1);
	DE_QUANTIZE_1(vert[vi[5]].qV, &v[5], 2);
#	else
	v[0] = vert[vi[0]].vert.e[0];
	v[1] = vert[vi[1]].vert.e[1];
	v[2] = vert[vi[2]].vert.e[2];
	v[3] = vert[vi[3]].vert.e[0];
	v[4] = vert[vi[4]].vert.e[1];
	v[5] = vert[vi[5]].vert.e[2];
#	endif
	ts.node.min.e[0] = (flag & curPos) ? v[0] : ts.node.min.e[0];
	curPos >>= 1;
	ts.node.min.e[1] = (flag & curPos) ? v[1] : ts.node.min.e[1];
	curPos >>= 1;
	ts.node.min.e[2] = (flag & curPos) ? v[2] : ts.node.min.e[2];
	curPos >>= 1;
	ts.node.max.e[0] = (flag & curPos) ? v[3] : ts.node.max.e[0];
	curPos >>= 1;
	ts.node.max.e[1] = (flag & curPos) ? v[4] : ts.node.max.e[1];
	curPos >>= 1;
	ts.node.max.e[2] = (flag & curPos) ? v[5] : ts.node.max.e[2];

	float diff[3] = {
		ts.node.max.e[0] - ts.node.min.e[0],
		ts.node.max.e[1] - ts.node.min.e[1],
		ts.node.max.e[2] - ts.node.min.e[2]};
	ts.axis = (diff[0] > diff[1] && diff[0] > diff[2]) ? 0 : (diff[1] > diff[2] ? 1 : 2);

	return &ts.node;
}

Index_t HCCMesh::getRootIdx(TravStat &ts)
{
	ts.cluster = 0;
	ts.index = Model::getRootIdx();
	ts.rootTemplate = -1;
	ts.type = -1;
	return ts.index;
}

Index_t HCCMesh::getLeftChildIdx(BVHNode *node, TravStat &ts, unsigned int &minBB)
{
	Index_t leftChild;
	if(ts.cluster == 0)
	{
		// in high level tree
		if((node->right & 0x2) != 0x2)
		{
			ts.index = leftChild = Model::getLeftChildIdx(node);
			return leftChild;
		}
		ts.index = leftChild = 0;
		ts.cluster = Model::getLeftChildIdx(node);

#ifdef USE_O_HCCMESH
		/*if(m_useOHCCMesh)*/ access(ts.cluster, threadID);
#endif

		ts.type = m_compCluster[ts.cluster].header.rootType;
		ts.rootTemplate = 0;
		ts.node.min = m_compCluster[ts.cluster].header.BBMin;
		ts.node.max = m_compCluster[ts.cluster].header.BBMax;
		minBB = m_compCluster[ts.cluster].node[leftChild].data;
		return leftChild;
	}

	// in low level tree

#ifdef USE_O_HCCMESH
	/*if(m_useOHCCMesh)*/ access(ts.cluster, threadID);
#endif

#	ifdef USE_HCCMESH_QUANTIZATION
	QCompCluster &cluster = m_compCluster[ts.cluster];
#	else
	CompCluster &cluster = m_compCluster[ts.cluster];
#	endif

	if(CISLEAFOFPATCH(node))
	{
		// leaf of a patch or complete tree
		unsigned int suppIndex = (node->left >> 5) & BIT_MASK_9;
		CompTreeSuppPtr supp = &(cluster.supp[suppIndex]);
		leftChild = supp->leftIndex >> 5;
		ts.type = supp->leftIndex & BIT_MASK_5;
		ts.rootTemplate = leftChild;
	}
#ifdef USE_COMPLETE_TREE
	else if(CISINCOMPLETETREE(node))
	{
		leftChild = ts.index * 2 + 1;
	}
#endif
	else
	{
		unsigned int offset = ts.index - ts.rootTemplate;
		BVHNode * tNode = &m_templates[ts.type].tree[offset];
		leftChild = Model::getLeftChildIdx(tNode) + ts.rootTemplate;
	}
	CompTreeNodePtr lChild = &cluster.node[leftChild];
	minBB = lChild->data;
	if(CISLEAFOFPATCH(lChild))
	{
		CompTreeSuppPtr supp = &(cluster.supp[(lChild->data >> 5) & BIT_MASK_9]);
		minBB &= ~0x3FE0;
		minBB |= ((supp->data >> 5) & BIT_MASK_9) << 5;
	}
#ifdef USE_HCCMESH_MT
	if(CISLEAF(lChild))
	{
		unsigned int offset = (lChild->data & (0xFFFF << 5)) >> 5;
		unsigned int backup = *((unsigned short*)(cluster.tris + offset));
		minBB &= (~(0xFFFF << 5));
		minBB |= (backup << 5);
	}
#endif

	ts.index = leftChild;
	ts.isLeft = 1;
	return leftChild;
}

Index_t HCCMesh::getRightChildIdx(BVHNode * node, TravStat &ts, unsigned int &maxBB)
{
	Index_t rightChild;
	if(ts.cluster == 0)
	{
		// in high level tree
		if((node->right & 0x1) != 0x1)
		{
			ts.index = rightChild = (node->right >> 2);
			return rightChild;
		}
		ts.index = rightChild = 0;
		ts.cluster = (node->right >> 2);

#ifdef USE_O_HCCMESH
		/*if(m_useOHCCMesh)*/ access(ts.cluster, threadID);
#endif

		ts.type = ts.type = m_compCluster[ts.cluster].header.rootType;
		ts.rootTemplate = 0;

		ts.node.min = m_compCluster[ts.cluster].header.BBMin;
		ts.node.max = m_compCluster[ts.cluster].header.BBMax;
		maxBB = m_compCluster[ts.cluster].node[rightChild].data;
		return rightChild;
	}
	// in low level tree


#ifdef USE_O_HCCMESH
	/*if(m_useOHCCMesh)*/ access(ts.cluster, threadID);
#endif

#	ifdef USE_HCCMESH_QUANTIZATION
	QCompCluster &cluster = m_compCluster[ts.cluster];
#	else
	CompCluster &cluster = m_compCluster[ts.cluster];
#	endif

	if(CISLEAFOFPATCH(node))
	{
		// leaf of a patch or complete tree
		unsigned int suppIndex = (node->left >> 5) & BIT_MASK_9;
		CompTreeSuppPtr supp = &(cluster.supp[suppIndex]);
		rightChild = supp->rightIndex >> 5;
		ts.type = supp->rightIndex & BIT_MASK_5;
		ts.rootTemplate = rightChild;
	}
#ifdef USE_COMPLETE_TREE
	else if(CISINCOMPLETETREE(node))
	{
		rightChild = ts.index * 2 + 2;
	}
#endif
	else 
	{
		unsigned int offset = ts.index - ts.rootTemplate;
		BVHNode * tNode = &m_templates[ts.type].tree[offset];
		rightChild = (tNode->right >> 2) + ts.rootTemplate;
	}
	CompTreeNodePtr rChild = &cluster.node[rightChild];
	maxBB = rChild->data;
	if(CISLEAFOFPATCH(rChild))
	{
		CompTreeSuppPtr supp = &(cluster.supp[(rChild->data >> 5) & BIT_MASK_9]);
		maxBB &= ~0x3FE0;
		maxBB |= ((supp->data >> 5) & BIT_MASK_9) << 5;
	}
#ifdef USE_HCCMESH_MT
	if(CISLEAF(rChild))
	{
		unsigned int offset = (rChild->data & (0xFFFF << 5)) >> 5;
		unsigned int backup = *((unsigned short*)(cluster.tris + offset));
		maxBB &= (~(0xFFFF << 5));
		maxBB |= (backup << 5);
	}
#endif
	ts.index = rightChild;
	ts.isLeft = 0;
	return rightChild;
}

#ifdef USE_HCCMESH_QUANTIZATION
_Vector4 HCCMesh::getVertexC(unsigned int idx, TravStat &ts)
#else
_Vector4 &HCCMesh::getVertexC(unsigned int idx, TravStat &ts)
#endif
{
	/*
#ifdef USE_VERTEX_QUANTIZE
	_Vector4 vert;

#ifdef USE_O_HCCMESH
	//if(m_useOHCCMesh)
		access(ts.cluster, threadID);
#endif

#if 1
	pqVert.DeQuantize(m_compCluster[threadID][ts.cluster].vert[idx].qV, vert.e);
#else
	__m128 temp;
	pqVert.DeQuantize(m_compCluster[threadID][ts.cluster].vert[idx].qV, &temp);
	_mm_storeu_ps(vert.e, temp);
#endif
	vert.m_alpha = *((float *)&m_compCluster[threadID][ts.cluster].vert[idx].data);
	return vert;
	*/
#	ifdef USE_HCCMESH_QUANTIZATION
	_Vector4 vert;

	DE_QUANTIZE(m_compCluster[ts.cluster].vert[idx].qV, vert.e);
	vert.m_alpha = *((float *)&m_compCluster[ts.cluster].vert[idx].data);
	return vert;
#else
	return m_compCluster[ts.cluster].vert[idx].vert;
#endif
}

Vertex HCCMesh::getVertex(unsigned int idx, TravStat &ts)
{
	_Vector4 &vC = getVertexC(idx, ts);
	unsigned int data = *((unsigned int*)&vC.m_alpha);
	Vertex v;
	v.v = vC;
	#ifdef USE_VERTEX_NORMALS
	v.n = m_quantizedNormals[data >> 16];
	#endif
	#ifdef USE_VERTEX_COLORS
	v.c = Vector3((data >> 11) & 0x1F, (data >> 5) & 0x3F, (data) & 0x1F);
	#else
	v.c[0] = vC.m_alpha;
	#endif
	return v;
}

bool HCCMesh::getIntersection(const Ray &oriRay, HitPointInfo &hitPointInfo, float tLimit, int stream)
{
	int threadID = omp_get_thread_num();
	StackElem *stack = stacks[threadID+MAX_NUM_THREADS*stream];

	int stackPtr;
	TravStat currentTS, tempTS;
	BVHNode * currentNode, parentNode;
	int CurrentDepth = 0;
	float min, max;	
	bool hasHit = false;
#ifdef USE_LOD
	int QuanIdx;
	float ErrBnd;
	float traveledDist = hit->m_TraveledDist;
	int lodIndex = 0;
#endif

	Ray ray = oriRay;
	ray.transform(m_invTransfMatrix);

	//hitPointInfo.t = FLT_MAX;
	//hitPointInfo.modelPtr = NULL;

	Index_t rootIndex = getRootIdx(currentTS);
	currentNode = getBV(rootIndex, currentTS, 0, 0);

	stack[0].index = Model::getRootIdx();
	stack[0].ts = currentTS;
	stackPtr = 1;

	// Note!!!
	// We should calculate error bound depend on model size. Because it prevent secondary ray's reflection to only next tri.
	// We will anneal this later.

	float error_bound = 0.000005f;
	unsigned int lChild, rChild;
	TravStat leftTS, rightTS;
	int axis;
	bool hitTest;
	unsigned int minBB, maxBB;

	// traverse BVH tree:
	while (true) {
		// is current node intersected and also closer than previous hit?
		hitTest = getIntersection(ray, &currentNode->min, min, max);

#ifdef _DEBUG_OUTPUT
		g_NumTraversed++;
#endif

		if ( hitTest && min < hitPointInfo.t && max > error_bound) {
			

			// is inner node?
			if (!isLeaf(currentNode)) {
				#ifdef USE_LOD

				lodIndex = CGET_LOD_INDEX(object, currentNode, currentTS);
				if ( lodIndex > 0) { // has LOD

					QuanIdx = GET_ERR_QUANTIZATION_IDX(lodIndex);

					assert (QuanIdx >= 0 && QuanIdx < (1<< ERR_BITs));

					 ErrBnd = GETLODERROR(object, QuanIdx);

					if (ErrBnd < g_MaxAllowModifier * (min + traveledDist)) 
					{
						if (RayCompLODIntersect(ray, object, lodIndex, hit, max, min, currentTS)) {
							hasHit = true;
							goto LOD_RETURN; // was hit
						}

						goto LOD_END; // traverse other node that are not descendent on the current node					
					}
					

				}
				#endif

				//axis = AXIS(currentNode);
				axis = currentTS.axis;
				
				if(ray.posneg[axis])
				{
					stack[stackPtr].ts = currentTS;
					lChild = getLeftChildIdx(currentNode, currentTS, minBB);
					rChild = getRightChildIdx(currentNode, stack[stackPtr].ts, maxBB);

					currentNode = getBV(lChild, currentTS, minBB, maxBB);

					stack[stackPtr].index = rChild;
					stack[stackPtr].minBB = minBB;
					stack[stackPtr].maxBB = maxBB;
				}
				else
				{
					stack[stackPtr].ts = currentTS;
					lChild = getLeftChildIdx(currentNode, stack[stackPtr].ts, minBB);
					rChild = getRightChildIdx(currentNode, currentTS, maxBB);

					currentNode = getBV(rChild, currentTS, minBB, maxBB);

					stack[stackPtr].index = lChild;
					stack[stackPtr].minBB = minBB;
					stack[stackPtr].maxBB = maxBB;
				}
				#ifdef USE_LOD
				stack[stackPtr].m_MinDim [0] = axis;
				stack[stackPtr].m_MaxDim [0] = g_MinMaxDim [1];
				#endif
				++stackPtr;
				continue;
			}
			else {				
				// is leaf node:
				// intersect with current node's members
#ifdef USE_HCCMESH_MT
				hasHit = RayMultiTriIntersect(ray, object, currentNode, hit, min(max, hit->t), currentTS, threadID) || hasHit;
#else
				hasHit = getIntersection(ray, currentNode, hitPointInfo, fminf(max, hitPointInfo.t), currentTS) || hasHit;
#endif
			}
		}
				
		#ifdef USE_LOD
		LOD_END:	
		#endif
		if (--stackPtr == 0) break;

		// fetch next node from stack
		currentTS = stack[stackPtr].ts;
		currentNode = getBV(stack[stackPtr].index, currentTS, stack[stackPtr].minBB, stack[stackPtr].maxBB);
		#ifdef USE_LOD
		g_MinMaxDim [0] = stack[stackPtr].m_MinDim [0];
		g_MinMaxDim [1] = stack[stackPtr].m_MaxDim [0];
		#endif
	}

	#ifdef USE_LOD
	LOD_RETURN:	
	#endif

	if(hasHit)
	{
		Vector3 hitX = ray.origin() + ray.direction() * hitPointInfo.t;
		hitX = transformLoc(m_transfMatrix, hitX);
		int idx = oriRay.direction().indexOfMaxComponent();
		hitPointInfo.t = (hitX.e[idx] - oriRay.origin().e[idx]) / oriRay.direction().e[idx];
		hitPointInfo.n = transformVec(m_transfMatrix, hitPointInfo.n);
		hitPointInfo.n.makeUnitVector();
	}

	return hasHit;
}

bool HCCMesh::getIntersection(const Ray &ray, BVHNode * node, HitPointInfo &hitPointInfo, float tmax, TravStat &ts)
{
	static int Mod3[] = {0, 1, 2, 0, 1};
	float point[2];
	float vdot;
	float alpha, beta;
	float t, u0, v0, u1, v1, u2, v2;

	unsigned int p[3];
	p[0] = (node->left >> 23) & BIT_MASK_9;
	p[1] = (node->left >> 14) & BIT_MASK_9;
	p[2] = (node->left >>  5) & BIT_MASK_9;

	_Vector4 &tri_p0 = getVertexC(p[0], ts);
	_Vector4 &tri_p1 = getVertexC(p[1], ts);
	_Vector4 &tri_p2 = getVertexC(p[2], ts);

	Vector3 triN = cross(tri_p1-tri_p0, tri_p2-tri_p0);
	unsigned int matID = (*((unsigned int*)&tri_p0.m_alpha)) & 0xFFFF;
	if(getMaterial(matID).getMat_d() < 0.1f) return false;

	if(triN.squaredLength() == 0.0f) return false;
	/*
	unsigned int INF = 4290772992u;
	if(*((unsigned int*)&triN.e[0]) == INF) return false;
	*/

	vdot = dot(ray.direction(), triN);
	if(vdot == 0.0f) return false;

	t = dot(tri_p0-ray.origin(), triN)/vdot;

	// if either too near or further away than a previous hit, we stop
	//if (t < INTERSECT_EPSILON || t > hitPoint->t)
	if (t < INTERSECT_EPSILON || t > (tmax + INTERSECT_EPSILON))
		return false;

	unsigned char i1, i2;
	// find best projection plane (YZ, XZ, XY)
	if (fabs(triN.e[0]) > fabs(triN.e[1]) && fabs(triN.e[0]) > fabs(triN.e[2])) {								
		i1 = 1;
		i2 = 2;
	}
	else if (fabs(triN.e[1]) > fabs(triN.e[2])) {								
		i1 = 0;
		i2 = 2;
	}
	else {								
		i1 = 0;
		i2 = 1;
	}

	int firstIdx;
	float u1list[3];
	u1list[0] = fabs(tri_p1[i1] - tri_p0[i1]);
	u1list[1] = fabs(tri_p2[i1] - tri_p1[i1]);
	u1list[2] = fabs(tri_p0[i1] - tri_p2[i1]);

	if (u1list[0] >= u1list[1] && u1list[0] >= u1list[2])
		firstIdx = 0;
	else if (u1list[1] >= u1list[2])
		firstIdx = 1;
	else
		firstIdx = 2;

	int secondIdx = Mod3[firstIdx + 1];
	int thirdIdx = Mod3[firstIdx + 2];

	// apply coordinate order to tri structure:
	_Vector4 *temp[3], *pTri[3];
	temp[0] = &tri_p0;
	temp[1] = &tri_p1;
	temp[2] = &tri_p2;

	pTri[0] = temp[firstIdx];
	pTri[1] = temp[secondIdx];
	pTri[2] = temp[thirdIdx];

	// intersection point with plane
	point[0] = ray.data[0].e[i1] + ray.data[1].e[i1] * t;
	point[1] = ray.data[0].e[i2] + ray.data[1].e[i2] * t;

	float p0_1 = pTri[0]->e[i1], p0_2 = pTri[0]->e[i2]; 
	u0 = point[0] - p0_1; 
	v0 = point[1] - p0_2; 
	u1 = pTri[1]->e[i1] - p0_1; 
	v1 = pTri[1]->e[i2] - p0_2; 
	u2 = pTri[2]->e[i1] - p0_1; 
	v2 = pTri[2]->e[i2] - p0_2;

	beta = (v0 * u1 - u0 * v1) / (v2 * u1 - u2 * v1);
	if (beta < 0 || beta > 1)
	//if (beta < -TRI_INTERSECT_EPSILON || beta > 1 + TRI_INTERSECT_EPSILON)
		return false;
	alpha = (u0 - beta * u2) / u1;	

	// not in triangle ?	
	if (alpha < 0 || (alpha + beta) > 1)
	//if (alpha < -TRI_INTERSECT_EPSILON || (alpha + beta) > 1.0f + TRI_INTERSECT_EPSILON)
		return false;

	// we have a hit:	
	hitPointInfo.alpha = alpha;  // .. and barycentric coords
	hitPointInfo.beta  = beta;

	// catch degenerate cases:
	if (tmax < 0.0f)
		return false;

	// Fill hitpoint structure:
	//

	unsigned int data0 = *((unsigned int*)&pTri[0]->m_alpha);
	unsigned int data1 = *((unsigned int*)&pTri[1]->m_alpha);
	unsigned int data2 = *((unsigned int*)&pTri[2]->m_alpha);

	/*
	float r, g, b;
	r = (data0 >> 11) & 0x1F;
	g = (data0 >> 5) & 0x3F;
	b = (data0) & 0x1F;

	Vector3 c0((data0 >> 11) & 0x1F, (data0 >> 5) & 0x3F, (data0) & 0x1F);
	Vector3 c1((data1 >> 11) & 0x1F, (data1 >> 5) & 0x3F, (data1) & 0x1F);
	Vector3 c2((data2 >> 11) & 0x1F, (data2 >> 5) & 0x3F, (data2) & 0x1F);

	Vector3 c = c0 + hitPoint->alpha * (c1 - c0) + hitPoint->beta * (c2 - c0);
	
	hitPoint->diffuseColor = rgb(c.e[0]/31.0f, c.e[1]/63.0f, c.e[2]/31.0f);
	*/

	hitPointInfo.m = data0 & 0xFFFF;

	Vector3 *qN = m_quantizedNormals;
	unsigned int v0idx, v1idx, v2idx;
	v0idx = data0 >> 16;
	v1idx = data1 >> 16;
	v2idx = data2 >> 16;

#	ifdef USE_VERTEX_NORMALS
	hitPointInfo.n = qN[v0idx] + hitPointInfo.alpha * (qN[v1idx]-qN[v0idx]) + hitPointInfo.beta * (qN[v2idx]-qN[v0idx]);
#	else
	hitPointInfo.n = triN;
#	endif

	hitPointInfo.t = t;

	hitPointInfo.modelPtr = this;

	if (vdot > 0.0f)
		hitPointInfo.n *= -1.0f;

	hitPointInfo.n.makeUnitVector();

	hitPointInfo.x = ray.origin() + ray.direction()*t;
	return true;
}

bool HCCMesh::getIntersection(const Ray &ray, Vector3 *box, float &interval_min, float &interval_max)  
{
	interval_min = -FLT_MAX;
	interval_max = FLT_MAX;

	#ifdef USE_LOD
	float t0 = (box[r.posneg[3]].e[0] - r.data[0].e[0]) * r.data[2].e[0];
	float t1 = (box[r.posneg[0]].e[0] - r.data[0].e[0]) * r.data[2].e[0];

	if (t0 > interval_min) {interval_min = t0; g_MinMaxDim [0] = 0;}
	if (t1 < interval_max) {interval_max = t1; g_MinMaxDim [1] = 0;}

	if (interval_min > interval_max) return false;

	t0 = (box[r.posneg[4]].e[1] - r.data[0].e[1]) * r.data[2].e[1];
	t1 = (box[r.posneg[1]].e[1] - r.data[0].e[1]) * r.data[2].e[1];

	if (t0 > interval_min) {interval_min = t0; g_MinMaxDim [0] = 1;}
	if (t1 < interval_max) {interval_max = t1; g_MinMaxDim [1] = 1;}

	if (interval_min > interval_max) return false;

	t0 = (box[r.posneg[5]].e[2] - r.data[0].e[2]) * r.data[2].e[2];
	t1 = (box[r.posneg[2]].e[2] - r.data[0].e[2]) * r.data[2].e[2];

	if (t0 > interval_min) {interval_min = t0; g_MinMaxDim [0] = 2;}
	if (t1 < interval_max) {interval_max = t1; g_MinMaxDim [1] = 2;}
	#else
	float t0 = (box[ray.posneg[3]].e[0] - ray.data[0].e[0]) * ray.data[2].e[0];
	float t1 = (box[ray.posneg[0]].e[0] - ray.data[0].e[0]) * ray.data[2].e[0];

	interval_min = max(t0, interval_min);
	interval_max = min(t1, interval_max);

	if (interval_min > interval_max) return false;

	t0 = (box[ray.posneg[4]].e[1] - ray.data[0].e[1]) * ray.data[2].e[1];
	t1 = (box[ray.posneg[1]].e[1] - ray.data[0].e[1]) * ray.data[2].e[1];

	interval_min = max(t0, interval_min);
	interval_max = min(t1, interval_max);

	if (interval_min > interval_max) return false;

	t0 = (box[ray.posneg[5]].e[2] - ray.data[0].e[2]) * ray.data[2].e[2];
	t1 = (box[ray.posneg[2]].e[2] - ray.data[0].e[2]) * ray.data[2].e[2];

	interval_min = max(t0, interval_min);
	interval_max = min(t1, interval_max);
	#endif
	return (interval_min <= interval_max);
}

void HCCMesh::updateTransformedBB(AABB &bb, const Matrix &mat)
{
	bb.min = Vector3(FLT_MAX, FLT_MAX, FLT_MAX);
	bb.max = Vector3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

	for(int i=1;i<(int)m_numClusters+1;i++)
	{
#		ifdef USE_HCCMESH_QUANTIZATION
		const QCompCluster &cluster = m_compCluster[i];
#		else
		const CompCluster &cluster = m_compCluster[i];
#		endif
		int numVert = cluster.header.numVert;
		for(int j=0;j<numVert;j++)
		{
#			ifdef USE_HCCMESH_QUANTIZATION
			Vector3 vert;
			DE_QUANTIZE(cluster.vert[j].qV, vert.e);
			vert = mat * vert;
#			else
			const _Vector4 &vert = mat * cluster.vert[j].vert;
#			endif
			bb.min.setX(min(bb.min.x(), vert.x()));
			bb.min.setY(min(bb.min.y(), vert.y()));
			bb.min.setZ(min(bb.min.z(), vert.z()));
			bb.max.setX(max(bb.max.x(), vert.x()));
			bb.max.setY(max(bb.max.y(), vert.y()));
			bb.max.setZ(max(bb.max.z(), vert.z()));
		}
	}
}