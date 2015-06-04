#include "BVHBuilder.h"

using namespace irt;

BVHBuilder::BVHBuilder(void)
{
}

BVHBuilder::~BVHBuilder(void)
{
}

void BVHBuilder::init(Model *mesh)
{
	clear();

	m_mesh = mesh;
}

void BVHBuilder::clear(void)
{
}

bool BVHBuilder::build(void)
{
	if(m_mesh->m_nodeList) delete[] m_mesh->m_nodeList;
	if(m_mesh->m_numTris == 0) 
	{
		m_mesh->m_nodeList = NULL;
		return true;
	}

	m_mesh->m_nodeList = new BVHNode[m_mesh->m_numTris*2 - 1];	// binary tree

	BVHNode *root = &m_mesh->m_nodeList[0];
	root->min.set(FLT_MAX);
	root->max.set(-FLT_MAX);
	m_mesh->m_numNodes = 1;

	unsigned int *triIDs = new unsigned int[m_mesh->m_numTris];
	for(int i=0;i<m_mesh->m_numTris;i++)
	{
		triIDs[i] = i;

		for(int j=0;j<3;j++)
		{
			const Vertex &v = m_mesh->m_vertList[m_mesh->m_triList[i].p[j]];
			updateBB(root->min, root->max, v.v);
		}
	}

	subDivide(triIDs, 0, m_mesh->m_numTris-1);

	delete[] triIDs;
	return true;
}

bool BVHBuilder::build(Model *mesh)
{
	BVHBuilder *builder = new BVHBuilder;
	builder->init(mesh);
	bool ret = builder->build();
	delete builder;
	return ret;
}

void BVHBuilder::updateBB(Vector3 &min, Vector3 &max, const Vector3 &vec)
{
	min.e[0] = ( min.e[0] < vec.e[0] ) ? min.e[0] : vec.e[0];
	min.e[1] = ( min.e[1] < vec.e[1] ) ? min.e[1] : vec.e[1];
	min.e[2] = ( min.e[2] < vec.e[2] ) ? min.e[2] : vec.e[2];

	max.e[0] = ( max.e[0] > vec.e[0] ) ? max.e[0] : vec.e[0];
	max.e[1] = ( max.e[1] > vec.e[1] ) ? max.e[1] : vec.e[1];
	max.e[2] = ( max.e[2] > vec.e[2] ) ? max.e[2] : vec.e[2];
}

bool BVHBuilder::subDivide(unsigned int *triIDs, unsigned int left, unsigned int right, unsigned int myIndex, unsigned int nextIndex, int depth)
{
#	define BVHNODE_BYTES 32

	if((int)myIndex > m_mesh->m_numTris*2 - 1 || (int)nextIndex + 1 > m_mesh->m_numTris*2 - 1)
	{
		printf("Error: Out of index range, %d/%d\n", myIndex, nextIndex + 1);
		return false;
	}

	BVHNode *node = &m_mesh->m_nodeList[myIndex];

	// find biggest axis:
	Vector3 diff = node->max - node->min;
	int biggestaxis = diff.indexOfMaxComponent();
	float split_pt = 0.5f * diff[biggestaxis] + node->min[biggestaxis];

	// compute average primitive location:
	unsigned int tsz = right - left + 1;
	float avgloc;
	unsigned int curLeft = left, curRight = right;
	for(unsigned int i=0;i<tsz;i++) 
	{
		const Triangle &tri = m_mesh->m_triList[triIDs[curLeft]];

		avgloc = 0.0f;

		avgloc = m_mesh->m_vertList[tri.p[0]].v.e[biggestaxis];
		avgloc += m_mesh->m_vertList[tri.p[1]].v.e[biggestaxis];
		avgloc += m_mesh->m_vertList[tri.p[2]].v.e[biggestaxis];
		avgloc /= 3.0f;

		if (avgloc < split_pt) 
		{
			curLeft++;
		} else 
		{ // swap with last unprocessed element
			unsigned int temp = triIDs[curLeft];
			triIDs[curLeft] = triIDs[curRight];
			triIDs[curRight] = temp;			

			curRight--;
		}
	}	

	unsigned int numLeft = curLeft - left;
	
	// special case: subdivision did not work out, just go half/half
	if (numLeft == 0 || numLeft == tsz) 
	{
		numLeft = tsz/2;		
	}
	
	node->left = ((nextIndex) << 2) | biggestaxis;
	node->right = ((nextIndex+1) << 2);
	
	m_mesh->m_numNodes += 2;

	BVHNode *lChild = &m_mesh->m_nodeList[nextIndex];
	BVHNode *rChild = &m_mesh->m_nodeList[nextIndex+1];

	if(numLeft == 1) 
	{
		lChild->left = (numLeft << 2) | 3;
		lChild->right = triIDs[left];

		lChild->min.set(FLT_MAX);
		lChild->max.set(-FLT_MAX);

		const Triangle &tri = m_mesh->m_triList[lChild->right];

		updateBB(lChild->min, lChild->max, m_mesh->m_vertList[tri.p[0]].v);
		updateBB(lChild->min, lChild->max, m_mesh->m_vertList[tri.p[1]].v);
		updateBB(lChild->min, lChild->max, m_mesh->m_vertList[tri.p[2]].v);
	}	
	else 
	{ 
		lChild->min.set(FLT_MAX);
		lChild->max.set(-FLT_MAX);
		for(unsigned int i=left;i<=(left+numLeft-1);i++) 
		{
			const Triangle &tri = m_mesh->m_triList[triIDs[i]];

			updateBB(lChild->min, lChild->max, m_mesh->m_vertList[tri.p[0]].v);
			updateBB(lChild->min, lChild->max, m_mesh->m_vertList[tri.p[1]].v);
			updateBB(lChild->min, lChild->max, m_mesh->m_vertList[tri.p[2]].v);
		}

		subDivide(triIDs, left, left+numLeft-1, nextIndex, nextIndex + 2, depth + 1);
	}

	if(tsz - numLeft == 1) 
	{
		rChild->left = ((tsz - numLeft) << 2) | 3;
		rChild->right = triIDs[left+numLeft];

		const Triangle &tri = m_mesh->m_triList[rChild->right];

		rChild->min.set(FLT_MAX);
		rChild->max.set(-FLT_MAX);
		updateBB(rChild->min, rChild->max, m_mesh->m_vertList[tri.p[0]].v);
		updateBB(rChild->min, rChild->max, m_mesh->m_vertList[tri.p[1]].v);
		updateBB(rChild->min, rChild->max, m_mesh->m_vertList[tri.p[2]].v);
	}	
	else 
	{ 
		rChild->min.set(FLT_MAX);
		rChild->max.set(-FLT_MAX);
		for(unsigned int i=left+numLeft;i<=right;i++) 
		{
			const Triangle &tri = m_mesh->m_triList[triIDs[i]];

			updateBB(rChild->min, rChild->max, m_mesh->m_vertList[tri.p[0]].v);
			updateBB(rChild->min, rChild->max, m_mesh->m_vertList[tri.p[1]].v);
			updateBB(rChild->min, rChild->max, m_mesh->m_vertList[tri.p[2]].v);
		}

		subDivide(triIDs, left+numLeft, right, nextIndex + 1, m_mesh->m_numNodes, depth + 1);
	}

	return true;
}
