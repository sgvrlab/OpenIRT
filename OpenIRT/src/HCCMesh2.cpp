#include "defines.h"
#include "CommonHeaders.h"
#include "handler.h"

#include "HCCMesh2.h"
#include "Matrix.h"
#include <io.h>

#include "OpenIRT.h"

#define INTERSECT_EPSILON 0.01f
#define BIT_MASK_10 0x3FF


#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif

using namespace irt;

HCCMesh2::HCCMesh2(void)
{
	stacks = new StackElem *[MAX_NUM_THREADS*MAX_NUM_INTERSECTION_STREAM];

	for(int i=0;i<MAX_NUM_THREADS*MAX_NUM_INTERSECTION_STREAM;i++)
		stacks[i] = (StackElem *)_aligned_malloc(150 * sizeof(StackElem), 16);

	m_clusterVertOffset = 0;
}

HCCMesh2::~HCCMesh2(void)
{
	if(stacks)
	{
		for(int i=0;i<MAX_NUM_THREADS;i++)
			_aligned_free(stacks[i]);
		delete[] stacks;
	}

	if(m_clusterVertOffset) delete[] m_clusterVertOffset;
}

bool HCCMesh2::load(const char *fileName)
{
	strcpy_s(m_fileName, 256, fileName);
	char headerFileName[MAX_PATH];
	char vertFileName[MAX_PATH];
	char triFileName[MAX_PATH];
	char nodeFileName[MAX_PATH];
	char matFileName[MAX_PATH];

	sprintf_s(headerFileName, MAX_PATH, "%s\\header", fileName);
	sprintf_s(vertFileName, MAX_PATH, "%s\\vertex.hccmesh2", fileName);
	sprintf_s(triFileName, MAX_PATH, "%s\\tris.hccmesh2", fileName);
	sprintf_s(nodeFileName, MAX_PATH, "%s\\BVH.hccmesh2", fileName);
	sprintf_s(matFileName, MAX_PATH, "%s\\material.mtl", fileName);

	FILE *fpHeader, *fpVert, *fpTri, *fpNode;
	errno_t err;

	// open files
	if(err = fopen_s(&fpHeader, headerFileName, "rb"))
	{
		printf("File open error [%d] : %s\n", err, headerFileName);
		return false;
	}

	if(err = fopen_s(&fpVert, vertFileName, "rb"))
	{
		printf("File open error [%d] : %s\n", err, vertFileName);
		return false;
	}

	if(err = fopen_s(&fpTri, triFileName, "rb"))
	{
		printf("File open error [%d] : %s\n", err, triFileName);
		return false;
	}

	if(err = fopen_s(&fpNode, nodeFileName, "rb"))
	{
		printf("File open error [%d] : %s\n", err, nodeFileName);
		return false;
	}

	// get file sizes
	__int64 sizeHeader, sizeVert, sizeTri, sizeNode;
	sizeHeader = _filelengthi64(_fileno(fpHeader));
	sizeVert = _filelengthi64(_fileno(fpVert));
	sizeTri = _filelengthi64(_fileno(fpTri));
	sizeNode = _filelengthi64(_fileno(fpNode));
	m_numVerts = (int)(sizeVert / sizeof(Vert16));
	m_numTris = (int)(sizeTri / sizeof(unsigned int));
	m_numNodes = (int)(sizeNode / sizeof(BVHNode));

	m_numClusters = (int)(sizeHeader/ sizeof(unsigned int));

	Progress &prog = OpenIRT::getSingletonPtr()->getProgress();
	prog.reset(5);
	prog.setText("Allocating memory...");

	// allocate memory space
	if(!(m_clusterVertOffset = new unsigned int[m_numClusters]))
	{
		printf("Memory allocation error : %s\n", headerFileName);
		return false;
	}

	if(!(m_vertList = new Vert16[m_numVerts]))
	{
		printf("Memory allocation error : %s\n", vertFileName);
		return false;
	}

	if(!(m_triList = new unsigned int[m_numTris]))
	{
		printf("Memory allocation error : %s\n", triFileName);
		return false;
	}

	if(!(m_nodeList = new BVHNode[m_numNodes]))
	{
		printf("Memory allocation error : %s\n", nodeFileName);
		return false;
	}

	prog.step();
	prog.setText(headerFileName);

	// load files
	if(!fread(m_clusterVertOffset, (size_t)sizeHeader, 1, fpHeader))
	{
		printf("Read file error : %s\n", headerFileName);
		return false;
	}

	prog.step();
	prog.setText(vertFileName);

	if(!fread(m_vertList, (size_t)sizeVert, 1, fpVert))
	{
		printf("Read file error : %s\n", vertFileName);
		return false;
	}

	prog.step();
	prog.setText(triFileName);

	if(!fread(m_triList, (size_t)sizeTri, 1, fpTri))
	{
		printf("Read file error : %s\n", triFileName);
		return false;
	}

	prog.step();
	prog.setText(nodeFileName);

	if(!fread(m_nodeList, (size_t)sizeNode, 1, fpNode))
	{
		printf("Read file error : %s\n", nodeFileName);
		return false;
	}

	prog.step();

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

	// close files
	fclose(fpVert);
	fclose(fpTri);
	fclose(fpNode);

	m_BB.min = getBV(getRootIdx())->min;
	m_BB.max = getBV(getRootIdx())->max;
	
	return true;
}

HCCMesh2::Vert16 *HCCMesh2::getVertex(const int clusterID, const Index_t n)
{
	return &m_vertList[m_clusterVertOffset[clusterID] + n];
}

unsigned int HCCMesh2::getTriangle(const Index_t n)
{
	return m_triList[n];
}

int HCCMesh2::getNumTriangles(const Index_t n)
{
	return (m_nodeList[n].left & 0x3FF) >> 2;
}
int HCCMesh2::getNumTriangles(const BVHNode *n)
{
	return (n->left & 0x3FF) >> 2;
}
int HCCMesh2::getClusterID(const Index_t n)
{
	return m_nodeList[n].left >> 10;
}
int HCCMesh2::getClusterID(const BVHNode *n)
{
	return n->left >> 10;
}

bool HCCMesh2::getIntersection(const Ray &ray, Vector3 *box, float &interval_min, float &interval_max)  
{
	interval_min = -FLT_MAX;
	interval_max = FLT_MAX;

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

	return (interval_min <= interval_max);
}

bool HCCMesh2::getIntersection(const Ray &oriRay, HitPointInfo &hitPointInfo, float tLimit, int stream)
{
	extern __int64 g_trav;
	int threadID = omp_get_thread_num();
	StackElem *stack = stacks[threadID+MAX_NUM_THREADS*stream];

	int stackPtr;
	BVHNode *currentNode;
	bool hasHit = false;
	float tmin, tmax;

	Ray ray = oriRay;
	ray.transform(m_invTransfMatrix);

	stack[0].index = getRootIdx();
	stackPtr = 1;

	currentNode = getBV(stack[0].index);

	float error_bound = 0.000005f;

	unsigned int lChild;//, rChild;
	int axis;
	bool hitTest;

	// traverse BVH tree:
	while (true) {
		// is current node intersected and also closer than previous hit?
		hitTest = HCCMesh2::getIntersection(ray, &currentNode->min, tmin, tmax);
		//g_trav++;

		if ( hitTest && tmin < hitPointInfo.t && tmax > error_bound) {


			// is inner node?
			if (!isLeaf(currentNode)) {
				// Store ordered children
				lChild = getLeftChildIdx(currentNode);

				axis = getAxis(currentNode);

				stack[stackPtr].index = (ray.posneg[axis]) + lChild;
				/*
				if(ray.posneg[axis])
				{
					stack[stackPtr].index = ray.data[axis] > getBV(lChild)->max.e[axis] ? lChild : lChild + 1;
				}
				else
				{
					stack[stackPtr].index = ray.data[axis] < getBV(lChild+1)->min.e[axis] ? lChild + 1 : lChild;
				}
				*/
				currentNode =  getBV(lChild + (lChild == stack[stackPtr].index));

				++stackPtr;
				continue;
			}
			else {				
				// is leaf node:
				// intersect with current node's members
				hasHit = getIntersection(ray, currentNode, hitPointInfo, min(tmax, hitPointInfo.t)) || hasHit;
				if(tLimit > 0.0f && hasHit)
				{
					if(hitPointInfo.t < tLimit) return true;
				}
			}
		}
		if (--stackPtr == 0) break;

		// fetch next node from stack
		currentNode = getBV(stack[stackPtr].index);
	}

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

bool HCCMesh2::getIntersection(const Ray &ray, BVHNode *node, HitPointInfo &hitPointInfo, float tmax)
{
	float point[2];
	float vdot, vdot2, fvdot;
	float alpha, beta;
	float t, u0, v0, u1, v1, u2, v2;
	int foundTri = -1;


	int p[3];
	int i1, i2;

	int count = getNumTriangles(node);
	unsigned int idxList = getTriangleIdx(node);
	int clusterID = getClusterID(node);

	Vector3 triN, curTriN;
	int triM;

	for(int i=0;i<count;i++, idxList++)
	{
		unsigned int triID = idxList;

		unsigned int tri = getTriangle(triID);
		p[0] = (tri >>  2) & BIT_MASK_10;
		p[1] = (tri >> 12) & BIT_MASK_10;
		p[2] = (tri >> 22) & BIT_MASK_10;
		i1 = (tri & 0x2) >> 1;
		i2 = (tri & 0x1) + 1;
		
		/*
		if(clusterID < 54)
		{
			if(m_clusterVertOffset[clusterID+1] - m_clusterVertOffset[clusterID] <= p[0]) 
				printf("[%d] p[0] = %d / %d\n", clusterID, p[0], m_clusterVertOffset[clusterID+1] - m_clusterVertOffset[clusterID+1]);
			if(m_clusterVertOffset[clusterID+1] - m_clusterVertOffset[clusterID] <= p[1]) 
				printf("[%d] p[1] = %d / %d\n", clusterID, p[1], m_clusterVertOffset[clusterID+1] - m_clusterVertOffset[clusterID+1]);
			if(m_clusterVertOffset[clusterID+1] - m_clusterVertOffset[clusterID] <= p[2]) 
				printf("[%d] p[2] = %d / %d\n", clusterID, p[2], m_clusterVertOffset[clusterID+1] - m_clusterVertOffset[clusterID+1]);
		}
		*/

		// begin barycentric intersection algorithm 
		const Vector3 &tri_p0 = (*getVertex(clusterID, p[0])).v; 
		const Vector3 &tri_p1 = (*getVertex(clusterID, p[1])).v; 
		const Vector3 &tri_p2 = (*getVertex(clusterID, p[2])).v;

		curTriN = cross(tri_p1-tri_p0, tri_p2-tri_p0);
		if(curTriN.e[0] + curTriN.e[1] + curTriN.e[2] > 0.0f) curTriN.makeUnitVector();

		// is ray parallel to plane or a back face ?
		vdot = dot(ray.direction(), curTriN);

		if(vdot == 0.0f) continue;

		// find parameter t of ray -> intersection point
		vdot2 = dot(ray.origin(),curTriN);
		t = dot(tri_p0-ray.origin(), curTriN)/vdot;

		// if either too near or further away than a previous hit, we stop
		if (t < INTERSECT_EPSILON || t > tmax + INTERSECT_EPSILON)
			continue;

		// intersection point with plane
		point[0] = ray.data[0].e[i1] + ray.data[1].e[i1] * t;
		point[1] = ray.data[0].e[i2] + ray.data[1].e[i2] * t;

		float p0_1 = tri_p0.e[i1], p0_2 = tri_p0.e[i2]; 
		u0 = point[0] - p0_1; 
		v0 = point[1] - p0_2; 
		u1 = tri_p1[i1] - p0_1; 
		v1 = tri_p1[i2] - p0_2; 
		u2 = tri_p2[i1] - p0_1; 
		v2 = tri_p2[i2] - p0_2;

		beta = (v0 * u1 - u0 * v1) / (v2 * u1 - u2 * v1);
		//if (beta < 0 || beta > 1)
		if (beta < 0.0f || beta > 1.0f)
			continue;
		alpha = (u0 - beta * u2) / u1;	

		// not in triangle ?	
		if (alpha < 0.0f || (alpha + beta) > 1.0f)
			continue;

		// we have a hit:	
		hitPointInfo.alpha = alpha;  // .. and barycentric coords
		hitPointInfo.beta  = beta;
		hitPointInfo.t = t;
		fvdot = vdot;
		foundTri = triID;
		tmax = t;
		triM = getVertex(clusterID, p[0])->m;
		triN = curTriN;
	}

	if(foundTri >= 0)
	{
		// catch degenerate cases:
		if (tmax < 0.0f)
			return false;

		// Fill hitpoint structure:
		//
		hitPointInfo.n = fvdot > 0 ? -triN : triN;
		hitPointInfo.m = triM;
		hitPointInfo.modelPtr = this;

		return true;
	}
	return false;	

}

void HCCMesh2::updateTransformedBB(AABB &bb, const Matrix &mat)
{
	bb.min = Vector3(FLT_MAX, FLT_MAX, FLT_MAX);
	bb.max = Vector3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
	for(int i=0;i<m_numVerts;i++)
	{
		const Vector3 &vert = mat * m_vertList[i].v;
		bb.min.setX(min(bb.min.x(), vert.x()));
		bb.min.setY(min(bb.min.y(), vert.y()));
		bb.min.setZ(min(bb.min.z(), vert.z()));
		bb.max.setX(max(bb.max.x(), vert.x()));
		bb.max.setY(max(bb.max.y(), vert.y()));
		bb.max.setZ(max(bb.max.z(), vert.z()));
	}
}