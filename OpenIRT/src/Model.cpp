#include "CommonOptions.h"
#include "defines.h"
#include "CommonHeaders.h"
#include "handler.h"

#include "Model.h"
#include "Matrix.h"
#include <io.h>

#include "OpenIRT.h"
#include "GeometryConverter.h"
#include "FileMapper.h"

#define INTERSECT_EPSILON 0.01f

#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif

using namespace irt;

Model::Model(void) : 
m_vertList(0),
m_triList(0),
m_nodeList(0),
m_numVerts(0),
m_numTris(0),
m_useMTL(0),
m_visible(true),
m_enabled(true)
{
	stacks = new StackElem *[MAX_NUM_THREADS*MAX_NUM_INTERSECTION_STREAM];

	for(int i=0;i<MAX_NUM_THREADS*MAX_NUM_INTERSECTION_STREAM;i++)
		stacks[i] = (StackElem *)_aligned_malloc(150 * sizeof(StackElem), 16);

	m_fileName[0] = 0;
	m_name[0] = 0;
}

Model::~Model(void)
{
	unload();

	if(stacks)
	{
		for(int i=0;i<MAX_NUM_THREADS*MAX_NUM_INTERSECTION_STREAM;i++)
			_aligned_free(stacks[i]);
		delete[] stacks;
	}

}

bool Model::load(const char *fileName)
{
	strcpy_s(m_fileName, 256, fileName);
	char vertFileName[MAX_PATH];
	char triFileName[MAX_PATH];
	char nodeFileName[MAX_PATH];
	char matFileName[MAX_PATH];

	sprintf_s(vertFileName, MAX_PATH, "%s\\vertex.ooc", fileName);
	sprintf_s(triFileName, MAX_PATH, "%s\\tris.ooc", fileName);
	sprintf_s(nodeFileName, MAX_PATH, "%s\\BVH.node", fileName);
	sprintf_s(matFileName, MAX_PATH, "%s\\material.mtl", fileName);

	//GeometryConverter::convertVert(vertFileName);
	//GeometryConverter::convertTri(triFileName);
	//makeCornellBoxModel("E:\\Projects\\Resources\\cornellbox.ooc");

	errno_t err;
	FILE *fpVert, *fpTri, *fpNode;

	// open files
	if(err = fopen_s(&fpVert, vertFileName, "rb"))
	{
		printf("File open error [%d] : %s", err, vertFileName);
		return false;
	}

	if(err = fopen_s(&fpTri, triFileName, "rb"))
	{
		printf("File open error [%d] : %s", err, triFileName);
		return false;
	}

	if(err = fopen_s(&fpNode, nodeFileName, "rb"))
	{
		printf("File open error [%d] : %s", err, nodeFileName);
		return false;
	}

	// get file sizes
	__int64 sizeVert, sizeTri, sizeNode;
	sizeVert = _filelengthi64(_fileno(fpVert));
	sizeTri = _filelengthi64(_fileno(fpTri));
	sizeNode = _filelengthi64(_fileno(fpNode));
	m_numVerts = (int)(sizeVert / sizeof(Vertex));
	m_numTris = (int)(sizeTri / sizeof(Triangle));
	m_numNodes = (int)(sizeNode / sizeof(BVHNode));

#	ifdef USE_MM
	fclose(fpVert);
	fclose(fpTri);
	fclose(fpNode);
	m_vertList = (Vertex*)FileMapper::map(vertFileName);
	m_triList = (Triangle*)FileMapper::map(triFileName);
	m_nodeList = (BVHNode*)FileMapper::map(nodeFileName);
#	else

	// allocate memory space
	if(!(m_vertList = new Vertex[m_numVerts]))
	{
		printf("Memory allocation error : %s\n", vertFileName);
		return false;
	}

	if(!(m_triList = new Triangle[m_numTris]))
	{
		printf("Memory allocation error : %s\n", triFileName);
		return false;
	}

	if(!(m_nodeList = new BVHNode[m_numNodes]))
	{
		printf("Memory allocation error : %s\n", nodeFileName);
		return false;
	}

	Progress &prog = OpenIRT::getSingletonPtr()->getProgress();
	prog.reset(3);
	prog.setText(vertFileName);

	// load files
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
#	endif

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
#	ifndef USE_MM
	fclose(fpVert);
	fclose(fpTri);
	fclose(fpNode);
#	endif

	m_BB.min = getBV(getRootIdx())->min;
	m_BB.max = getBV(getRootIdx())->max;
	
	return true;
}

bool Model::load(Vertex *vertList, int numVerts, Face *faceList, int numFaces, const Material &mat)
{
	m_numVerts = numVerts;
	
	m_matList.clear();
	m_matList.push_back(mat);

	if(m_vertList) delete[] m_vertList;
	if(m_triList) delete[] m_triList;

	m_vertList = new Vertex[numVerts];

	m_BB.min.set(FLT_MAX);
	m_BB.max.set(-FLT_MAX);

	bool computeNormals = vertList[0].v.e[0] == 0.0f && vertList[0].v.e[1] == 0.0f && vertList[0].v.e[2] == 0.0f;

	for(int i=0;i<numVerts;i++)
	{
		Vertex &vert = m_vertList[i];
		vert = vertList[i];

		m_BB.update(vert.v);
	}

	std::vector<Triangle> *pTriList = new std::vector<Triangle>;
	std::vector<Triangle> &triList = *pTriList;

	Index_t indexList[3];

	for(int i=0;i<numFaces;i++)
	{
		const Face &face = faceList[i];

		// simple triangulation
		for(int j=0;j<face.n-2;j++)
		{
			Triangle tri;

			//tri.p[0] = face.verts[0];
			//tri.p[1] = face.verts[j+1];
			//tri.p[2] = face.verts[j+2];

			//tri.n = cross(m_vertList[tri.p[1]].v - m_vertList[tri.p[0]].v, m_vertList[tri.p[2]].v - m_vertList[tri.p[0]].v);
			//tri.n.makeUnitVector();
			//tri.d = dot(tri.n, m_vertList[tri.p[0]].v);

			//tri.material = 0;

			indexList[0] = face.verts[0];
			indexList[1] = face.verts[j+1];
			indexList[2] = face.verts[j+2];
			
			tri = makeTriangle(m_vertList[indexList[0]].v, m_vertList[indexList[1]].v, m_vertList[indexList[2]].v, indexList, 0);

			if(computeNormals)
			{
				m_vertList[tri.p[0]].n += tri.n;
				m_vertList[tri.p[1]].n += tri.n;
				m_vertList[tri.p[2]].n += tri.n;
			}
			triList.push_back(tri);
		}
	}

	if(computeNormals)
	{
		for(int i=0;i<numVerts;i++)
			m_vertList[i].n.makeUnitVector();
	}

	m_numTris = (int)triList.size();
	m_triList = new Triangle[m_numTris];
	memcpy_s(m_triList, sizeof(Triangle)*m_numTris, &triList[0], sizeof(Triangle)*triList.size());

	delete pTriList;

	return true;
}

void Model::unload()
{
#	ifdef USE_MM
	if(m_vertList) FileMapper::unmap(m_vertList);
	if(m_triList) FileMapper::unmap(m_triList);
	if(m_nodeList) FileMapper::unmap(m_nodeList);
#	else
	if(m_vertList) delete[] m_vertList;
	if(m_triList) delete[] m_triList;
	if(m_nodeList) delete[] m_nodeList;
#	endif

	m_vertList = NULL;
	m_triList = NULL;
	m_nodeList = NULL;
	m_numVerts = m_numTris = m_numNodes = 0;
}

Vertex *Model::getVertex(const Index_t n)
{
	return &m_vertList[n];
}
Triangle *Model::getTriangle(const Index_t n)
{
	return &m_triList[n];
}
Index_t Model::getRootIdx()
{
	return 0;
}
BVHNode *Model::getBV(const Index_t n)
{
	return &m_nodeList[n];
}
bool Model::isLeaf(const Index_t n)
{
	return (m_nodeList[n].left & 3) == 3;
}
Index_t Model::getLeftChildIdx(const Index_t n)
{
	return m_nodeList[n].left >> 2;
}
Index_t Model::getRightChildIdx(const Index_t n)
{
	return m_nodeList[n].right >> 2;
}
Index_t Model::getTriangleIdx(const Index_t n)
{
	return m_nodeList[n].right;
}
int Model::getNumTriangles(const Index_t n)
{
	return m_nodeList[n].left >> 2;
}
int Model::getAxis(const Index_t n)
{
	return m_nodeList[n].left & 3;
}

bool Model::isLeaf(const BVHNode *n)
{
	return (n->left & 3) == 3;
}
Index_t Model::getLeftChildIdx(const BVHNode *n)
{
	return n->left >> 2;
}
Index_t Model::getRightChildIdx(const BVHNode *n)
{
	return n->right >> 2;
}
Index_t Model::getTriangleIdx(const BVHNode *n)
{
	return n->right;
}
int Model::getNumTriangles(const BVHNode *n)
{
	return n->left >> 2;
}
int Model::getAxis(const BVHNode *n)
{
	return n->left & 3;
}
bool Model::getIntersection(const Ray &ray, Vector3 *box, float &interval_min, float &interval_max)  
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

bool Model::getIntersection(const Ray &oriRay, HitPointInfo &hitPointInfo, float tLimit, int stream)
{
	if(!m_nodeList) return false;

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

	Index_t lChild;//, rChild;
	int axis;
	bool hitTest;

	// traverse BVH tree:
	while (true) {
		// is current node intersected and also closer than previous hit?
		hitTest = getIntersection(ray, &currentNode->min, tmin, tmax);

		if ( hitTest && tmin < hitPointInfo.t && tmax > error_bound) {


			// is inner node?
			if (!isLeaf(currentNode)) {
				// Store ordered children
				lChild = getLeftChildIdx(currentNode);

				axis = getAxis(currentNode);

				stack[stackPtr].index = (ray.posneg[axis]) + lChild;
				currentNode =  getBV(((ray.posneg[axis]^1)) + lChild);

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
		hitPointInfo.x = hitX;
	}

	return hasHit;
}

bool Model::getIntersection(const Ray &ray, BVHNode *node, HitPointInfo &hitPointInfo, float tmax)
{
	float point[2];
	float vdot, vdot2, fvdot;
	float alpha, beta;
	float t, u0, v0, u1, v1, u2, v2;
	int foundTri = -1;

	int count = getNumTriangles(node);
	Index_t idxList = getTriangleIdx(node);

	Vector3 triN;

	for(int i=0;i<count;i++, idxList++)
	{
		Index_t triID = idxList;

		const Triangle &tri = *getTriangle(triID);

		if(tri.p[0] == tri.p[1] || tri.p[1] == tri.p[2] || tri.p[2] == tri.p[0]) continue;

		assert(tri.i1 <= 2);
		assert(tri.i2 <= 2);

		// is ray parallel to plane or a back face ?
		vdot = dot(ray.direction(), tri.n);

		if(vdot == 0.0f) continue;

		// find parameter t of ray -> intersection point
		vdot2 = dot(ray.origin(),tri.n);
		t = (tri.d - vdot2) / vdot;

		// if either too near or further away than a previous hit, we stop
		if (t < INTERSECT_EPSILON || t > tmax + INTERSECT_EPSILON)
			continue;

		// intersection point with plane
		point[0] = ray.data[0].e[tri.i1] + ray.data[1].e[tri.i1] * t;
		point[1] = ray.data[0].e[tri.i2] + ray.data[1].e[tri.i2] * t;

		// begin barycentric intersection algorithm 
		const Vector3 &tri_p0 = (*getVertex(tri.p[0])).v; 
		const Vector3 &tri_p1 = (*getVertex(tri.p[1])).v; 
		const Vector3 &tri_p2 = (*getVertex(tri.p[2])).v;

		float p0_1 = tri_p0.e[tri.i1], p0_2 = tri_p0.e[tri.i2]; 
		u0 = point[0] - p0_1; 
		v0 = point[1] - p0_2; 
		u1 = tri_p1[tri.i1] - p0_1; 
		v1 = tri_p1[tri.i2] - p0_2; 
		u2 = tri_p2[tri.i1] - p0_1; 
		v2 = tri_p2[tri.i2] - p0_2;

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
	}

	if(foundTri >= 0)
	{
		// catch degenerate cases:
		if (tmax < 0.0f)
			return false;

		// Fill hitpoint structure:
		//
		const Triangle &tri = *getTriangle(foundTri);

		const Vertex &v0 = *getVertex(tri.p[0]); 
		const Vertex &v1 = *getVertex(tri.p[1]); 
		const Vertex &v2 = *getVertex(tri.p[2]);

		hitPointInfo.m = tri.material;

		hitPointInfo.modelPtr = this;

		//extern bool g_useVertexNormals;
		//if(g_useVertexNormals)
		//{
			//hitPointInfo.n = v0.n + hitPointInfo.alpha * (v1.n-v0.n) + hitPointInfo.beta * (v2.n-v0.n);
		//}
		//else hitPointInfo.n = *((Vector3*)&tri.n);
#		ifdef USE_VERTEX_NORMALS
		hitPointInfo.n = v0.n + hitPointInfo.alpha * (v1.n-v0.n) + hitPointInfo.beta * (v2.n-v0.n);
#		else
		hitPointInfo.n = *((Vector3*)&tri.n);
#		endif

		if (fvdot > 0.0f)
			hitPointInfo.n *= -1.0f;

		// interpolate tex coords..
		hitPointInfo.uv = v0.uv + hitPointInfo.alpha * (v1.uv-v0.uv) + hitPointInfo.beta * (v2.uv-v0.uv);

		hitPointInfo.n.makeUnitVector();

		hitPointInfo.tri = foundTri;

		return true;
	}
	return false;	

}

void Model::updateTransformedBB(AABB &bb, const Matrix &mat)
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

void Model::saveMaterials(const char *fileName)
{
	char matFileName[MAX_PATH];
	sprintf_s(matFileName, MAX_PATH, "%s\\material.mtl", fileName);
#	if 0
	Material mat1 = m_matList[2708], mat2 = m_matList[2708];
	mat2.setMatKs(RGBf(0.5f, 0.5f, 0.5f));
	mat2.setMat_Ns(10.0f);
	modifyMaterial(m_matList, mat1, mat2);
#	endif
	saveMaterialToMTL(matFileName, m_matList);
}

Triangle Model::makeTriangle(const Vector3 &v0, const Vector3 &v1, const Vector3 &v2, Index_t indexList[], unsigned short material)
{
	Triangle tri;
	tri.n = cross(v1 - v0, v2 - v0);
	tri.n.makeUnitVector();
	tri.d = dot(v0, tri.n);

	// find best projection plane (YZ, XZ, XY)
	if (fabs(tri.n[0]) > fabs(tri.n[1]) && fabs(tri.n[0]) > fabs(tri.n[2])) {								
		tri.i1 = 1;
		tri.i2 = 2;
	}
	else if (fabs(tri.n[1]) > fabs(tri.n[2])) {								
		tri.i1 = 0;
		tri.i2 = 2;
	}
	else {								
		tri.i1 = 0;
		tri.i2 = 1;
	}

	Index_t firstIdx;
	float u1list[3];
	u1list[0] = fabs(v1.e[tri.i1] - v0.e[tri.i1]);
	u1list[1] = fabs(v2.e[tri.i1] - v1.e[tri.i1]);
	u1list[2] = fabs(v0.e[tri.i1] - v2.e[tri.i1]);

	if (u1list[0] >= u1list[1] && u1list[0] >= u1list[2])
		firstIdx = 0;
	else if (u1list[1] >= u1list[2])
		firstIdx = 1;
	else
		firstIdx = 2;

	Index_t secondIdx = (firstIdx + 1) % 3;
	Index_t thirdIdx = (firstIdx + 2) % 3;

	// apply coordinate order to tri structure:
	tri.p[0] = indexList[firstIdx];
	tri.p[1] = indexList[secondIdx];
	tri.p[2] = indexList[thirdIdx];

	tri.material = material;

	return tri;
}

void Model::makeCornellBoxModel(const char *filePath)
{
	char vertFileName[MAX_PATH];
	char triFileName[MAX_PATH];
	char nodeFileName[MAX_PATH];
	char matFileName[MAX_PATH];

	sprintf_s(vertFileName, MAX_PATH, "%s\\vertex.ooc", filePath);
	sprintf_s(triFileName, MAX_PATH, "%s\\tris.ooc", filePath);
	sprintf_s(nodeFileName, MAX_PATH, "%s\\BVH.node", filePath);
	sprintf_s(matFileName, MAX_PATH, "%s\\material.mtl", filePath);

	FILE *fpVert, *fpTri, *fpNode, *fpMat;
	fopen_s(&fpVert, vertFileName, "wb");
	fopen_s(&fpTri, triFileName, "wb");
	fopen_s(&fpNode, nodeFileName, "wb");
	fopen_s(&fpMat, matFileName, "w");

	float maxX = 556.0f, maxY = 548.88f, maxZ = 559.2f;

	Vertex vert[4];
	float vList[20][3] = 
	{
		{0.0f, 0.0f, 0.0f}, 
		{0.0f, 0.0f, maxZ}, 
		{maxX, 0.0f, maxZ}, 
		{maxX, 0.0f, 0.0f}, 

		{0.0f, maxY, 0.0f}, 
		{maxX, maxY, 0.0f}, 
		{maxX, maxY, maxZ}, 
		{0.0f, maxY, maxZ}, 

		{0.0f, 0.0f, maxZ}, 
		{0.0f, maxY, maxZ}, 
		{maxX, maxY, maxZ}, 
		{maxX, 0.0f, maxZ}, 

		{0.0f, 0.0f, 0.0f}, 
		{0.0f, maxY, 0.0f}, 
		{0.0f, maxY, maxZ}, 
		{0.0f, 0.0f, maxZ}, 

		{maxX, 0.0f, 0.0f}, 
		{maxX, 0.0f, maxZ}, 
		{maxX, maxY, maxZ}, 
		{maxX, maxY, 0.0f}
	};
	float nList[5][3] = 
	{
		{0.0f, 1.0f, 0.0f}, 
		{0.0f, -1.0f, 0.0f}, 
		{0.0f, 0.0f, -1.0f}, 
		{1.0f, 0.0f, 0.0f}, 
		{-1.0f, 0.0f, 0.0f}
	};
	float cList[5][3] = 
	{
		{1.0f, 1.0f, 1.0f}, 
		{1.0f, 1.0f, 1.0f}, 
		{1.0f, 1.0f, 1.0f}, 
		{0.0f, 1.0f, 0.0f}, 
		{1.0f, 0.0f, 0.0f}
	};
	unsigned short matList[5] = {0, 0, 0, 1, 2};
	Triangle tri;
	BVHNode node;
	Index_t indexList[3] = {0, };

	for(int i=0;i<5;i++)
	{
		for(int j=0;j<4;j++)
		{
			for(int k=0;k<3;k++)
			{
				vert[j].v.e[k] = vList[i*4+j][k];
				vert[j].n.e[k] = nList[i][k];
				vert[j].c.e[k] = cList[i][k];
			}
		}
		fwrite(&vert[0], sizeof(Vertex), 4, fpVert);

		indexList[0] = i*4+0; indexList[1] = i*4+1; indexList[2] = i*4+2;
		tri = makeTriangle(vert[0].v, vert[1].v, vert[2].v, indexList, matList[i]);
		fwrite(&tri, sizeof(Triangle), 1, fpTri);

		indexList[0] = i*4+0; indexList[1] = i*4+2; indexList[2] = i*4+3;
		tri = makeTriangle(vert[0].v, vert[2].v, vert[3].v, indexList, matList[i]);
		fwrite(&tri, sizeof(Triangle), 1, fpTri);
	}

	AABB boxBB;
	boxBB.min = Vector3(0.0f, 0.0f, 0.0f);
	boxBB.max = Vector3(maxX, maxY, maxZ);

	Index_t leftList[19] = {1, 5, 3, 9, 7, 11, 13, 15, 17, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
	Index_t rightList[19] = {2, 6, 4, 10, 8, 12, 14, 16, 18, 2, 3, 0, 1, 6, 7, 8, 9, 4, 5};
	Index_t axis[19] = {2, 2, 2, 2, 2, 2, 2, 2, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
	float minList[19][3] =
	{
		{0.0f, 0.0f, 0.0f}, 
		{0.0f, 0.0f, 0.0f}, 
		{0.0f, 0.0f, 0.0f}, 
		{0.0f, 0.0f, 0.0f}, 
		{0.0f, 0.0f, 0.0f}, 
		{0.0f, 0.0f, 0.0f}, 
		{0.0f, 0.0f, 0.0f}, 
		{0.0f, 0.0f, 0.0f}, 
		{0.0f, 0.0f, maxZ}, 

		{0.0f, maxY, 0.0f}, 
		{0.0f, maxY, 0.0f}, 
		{0.0f, 0.0f, 0.0f}, 
		{0.0f, 0.0f, 0.0f}, 
		{0.0f, 0.0f, 0.0f}, 
		{0.0f, 0.0f, 0.0f}, 
		{maxX, 0.0f, 0.0f}, 
		{maxX, 0.0f, 0.0f}, 
		{0.0f, 0.0f, maxZ}, 
		{0.0f, 0.0f, maxZ}
	};

	float maxList[19][3] =
	{
		{maxX, maxY, maxZ}, 
		{maxX, maxY, maxZ}, 
		{maxX, maxY, maxZ}, 
		{maxX, maxY, maxZ}, 
		{maxX, maxY, maxZ}, 
		{maxX, maxY, maxZ}, 
		{maxX, maxY, maxZ}, 
		{maxX, maxY, maxZ}, 
		{maxX, maxY, maxZ}, 

		{maxX, maxY, maxZ}, 
		{maxX, maxY, maxZ}, 
		{maxX, 0.0f, maxZ}, 
		{maxX, 0.0f, maxZ}, 
		{0.0f, maxY, maxZ}, 
		{0.0f, maxY, maxZ}, 
		{maxX, maxY, maxZ}, 
		{maxX, maxY, maxZ}, 
		{maxX, maxY, maxZ}, 
		{maxX, maxY, maxZ}
	};

	for(int i=0;i<19;i++)
	{
		node.left = (leftList[i] << 2) | axis[i];
		node.right = rightList[i] << (axis[i] == 3 ? 0 : 2);
		node.min.setX(minList[i][0]);
		node.min.setY(minList[i][1]);
		node.min.setZ(minList[i][2]);
		node.max.setX(maxList[i][0]);
		node.max.setY(maxList[i][1]);
		node.max.setZ(maxList[i][2]);
		fwrite(&node, sizeof(BVHNode), 1, fpNode);
	}

	fprintf(fpMat, "newmtl white\n");
	fprintf(fpMat, "\tKd 0.8 0.8 0.8\n");
	fprintf(fpMat, "\tKa 0 0 0\n");
	fprintf(fpMat, "newmtl greed\n");
	fprintf(fpMat, "\tKd 0.05 0.8 0.05\n");
	fprintf(fpMat, "\tKa 0 0 0\n");
	fprintf(fpMat, "newmtl red\n");
	fprintf(fpMat, "\tKd 0.8 0.05 0.05\n");
	fprintf(fpMat, "\tKa 0 0 0\n");

	fclose(fpVert);
	fclose(fpTri);
	fclose(fpNode);
	fclose(fpMat);
}
