#include "psh/psh.h"

#include "Voxelize.h"
#include <io.h>
#include "Files.h"
#include "OptionManager.h"
#include <hash_map>
#include <vector>

int g_maxDepth;
int g_maxDepth2;
int g_maxLowDepth;

Voxelize::Voxelize()
	: m_fp(0), m_vertPtr(0), m_triPtr(0), m_nodePtr(0), m_indexMap(0), m_PCAOctree(0), m_leafVoxelMat(0)
{
}

Voxelize::~Voxelize()
{
	if(m_fp) fclose(m_fp);
	if(m_vertPtr) delete m_vertPtr;
	if(m_triPtr) delete m_triPtr;
	if(m_nodePtr) delete m_nodePtr;
	if(m_indexMap) delete[] m_indexMap;
	if(m_PCAOctree) delete[] m_PCAOctree;
	if(m_leafVoxelMat) delete[] m_leafVoxelMat;
	m_vertPtr = 0;
	m_triPtr = 0;
	m_nodePtr = 0;
}

AABB Voxelize::computeSubBox(int x, int y, int z, const AABB &box)
{
	AABB subBox;
	float boxDelta = (box.max.x() - box.min.x()) / N;
	subBox.min = box.min + (Vector3(x, y, z)*boxDelta);
	subBox.max = subBox.min + Vector3(boxDelta);
	return subBox;
}

bool Voxelize::isIntersect(const AABB &box, const Triangle &tri)
{
	Vector3 vert[3] = {m_vert[tri.p[0]].v, m_vert[tri.p[1]].v, m_vert[tri.p[2]].v};
	return isIntersect(box, vert);
}

bool Voxelize::isIntersect(const AABB &box, Vector3 *vert)
{
	/*======================== X-tests ========================*/
#define AXISTEST_X01(a, b, fa, fb)			   \
	p0 = a*v0[1] - b*v0[2];			       	   \
	p2 = a*v2[1] - b*v2[2];			       	   \
	if(p0<p2) {min=p0; max=p2;} else {min=p2; max=p0;} \
	rad = fa * boxHalfSize.e[1] + fb * boxHalfSize.e[2];   \
	if(min>rad + m_intersectEpsilon || max<-rad - m_intersectEpsilon) return 0;

#define AXISTEST_X2(a, b, fa, fb)			   \
	p0 = a*v0[1] - b*v0[2];			           \
	p1 = a*v1[1] - b*v1[2];			       	   \
	if(p0<p1) {min=p0; max=p1;} else {min=p1; max=p0;} \
	rad = fa * boxHalfSize.e[1] + fb * boxHalfSize.e[2];   \
	if(min>rad + m_intersectEpsilon || max<-rad - m_intersectEpsilon) return 0;

	/*======================== Y-tests ========================*/
#define AXISTEST_Y02(a, b, fa, fb)			   \
	p0 = -a*v0[0] + b*v0[2];		      	   \
	p2 = -a*v2[0] + b*v2[2];	       	       	   \
	if(p0<p2) {min=p0; max=p2;} else {min=p2; max=p0;} \
	rad = fa * boxHalfSize.e[0] + fb * boxHalfSize.e[2];   \
	if(min>rad + m_intersectEpsilon || max<-rad - m_intersectEpsilon) return 0;

#define AXISTEST_Y1(a, b, fa, fb)			   \
	p0 = -a*v0[0] + b*v0[2];		      	   \
	p1 = -a*v1[0] + b*v1[2];	     	       	   \
	if(p0<p1) {min=p0; max=p1;} else {min=p1; max=p0;} \
	rad = fa * boxHalfSize.e[0] + fb * boxHalfSize.e[2];   \
	if(min>rad + m_intersectEpsilon || max<-rad - m_intersectEpsilon) return 0;

	/*======================== Z-tests ========================*/

#define AXISTEST_Z12(a, b, fa, fb)			   \
	p1 = a*v1[0] - b*v1[1];			           \
	p2 = a*v2[0] - b*v2[1];			       	   \
	if(p2<p1) {min=p2; max=p1;} else {min=p1; max=p2;} \
	rad = fa * boxHalfSize.e[0] + fb * boxHalfSize.e[1];   \
	if(min>rad + m_intersectEpsilon || max<-rad - m_intersectEpsilon) return 0;

#define AXISTEST_Z0(a, b, fa, fb)			   \
	p0 = a*v0[0] - b*v0[1];				   \
	p1 = a*v1[0] - b*v1[1];			           \
	if(p0<p1) {min=p0; max=p1;} else {min=p1; max=p0;} \
	rad = fa * boxHalfSize.e[0] + fb * boxHalfSize.e[1];   \
	if(min>rad + m_intersectEpsilon || max<-rad - m_intersectEpsilon) return 0;

#define FINDMINMAX(x0,x1,x2,min,max) \
	min = max = x0;   \
	if(x1<min) min=x1;\
	if(x1>max) max=x1;\
	if(x2<min) min=x2;\
	if(x2>max) max=x2;

	Vector3 boxCenter = 0.5f*(box.min + box.max);
	Vector3 boxHalfSize = 0.5f*(box.max - box.min);
	if((vert[0] - vert[1]).maxAbsComponent() == 0.0f ||
		(vert[1] - vert[2]).maxAbsComponent() == 0.0f ||
		(vert[2] - vert[0]).maxAbsComponent() == 0.0f)
	{
		return false;
	}

	Vector3 v0, v1, v2;
	Vector3 e0, e1, e2;
	Vector3 normal;
	float min,max,d,p0,p1,p2,rad,fex,fey,fez;  

	v0 = vert[0] - boxCenter;
	v1 = vert[1] - boxCenter;
	v2 = vert[2] - boxCenter;

	e0 = v1 - v0;
	e1 = v2 - v1;
	e2 = v0 - v2;

	fex = fabs(e0.x());
	fey = fabs(e0.y());
	fez = fabs(e0.z());
	AXISTEST_X01(e0[2], e0[1], fez, fey);
	AXISTEST_Y02(e0[2], e0[0], fez, fex);
	AXISTEST_Z12(e0[1], e0[0], fey, fex);

	fex = fabs(e1[0]);
	fey = fabs(e1[1]);
	fez = fabs(e1[2]);
	AXISTEST_X01(e1[2], e1[1], fez, fey);
	AXISTEST_Y02(e1[2], e1[0], fez, fex);
	AXISTEST_Z0(e1[1], e1[0], fey, fex);

	fex = fabs(e2[0]);
	fey = fabs(e2[1]);
	fez = fabs(e2[2]);
	AXISTEST_X2(e2[2], e2[1], fez, fey);
	AXISTEST_Y1(e2[2], e2[0], fez, fex);
	AXISTEST_Z12(e2[1], e2[0], fey, fex);

	FINDMINMAX(v0[0],v1[0],v2[0],min,max);
	if(min>boxHalfSize[0] + m_intersectEpsilon || max<-boxHalfSize[0] - m_intersectEpsilon) return false;

	FINDMINMAX(v0[1],v1[1],v2[1],min,max);
	if(min>boxHalfSize[1] + m_intersectEpsilon || max<-boxHalfSize[1] - m_intersectEpsilon) return false;

	FINDMINMAX(v0[2],v1[2],v2[2],min,max);
	if(min>boxHalfSize[2] + m_intersectEpsilon || max<-boxHalfSize[2] - m_intersectEpsilon) return false;

	/*
	normal = cross(e0, e1);
	d = -dot(normal, v0);

	int q;
	Vector3 vmin,vmax;
	for(q=0;q<=2;q++)
	{
		if(normal[q]>0.0f)
		{
			vmin[q]=-boxHalfSize[q];
			vmax[q]=boxHalfSize[q];
		}
		else
		{
			vmin[q]=boxHalfSize[q];
			vmax[q]=-boxHalfSize[q];
		}
	}
	if(dot(normal,vmin)+d>0.0f) 
		return false;
	if(dot(normal,vmax)+d>=0.0f) 
		return true;

	return false;
	*/
	return true;
}

bool Voxelize::isIntersect(const AABB &a, const BVHNode *b)
{
	if(a.max.e[0] < b->min.e[0] - m_intersectEpsilon || a.min.e[0] > b->max.e[0] + m_intersectEpsilon) return false;
	if(a.max.e[1] < b->min.e[1] - m_intersectEpsilon || a.min.e[1] > b->max.e[1] + m_intersectEpsilon) return false;
	if(a.max.e[2] < b->min.e[2] - m_intersectEpsilon || a.min.e[2] > b->max.e[2] + m_intersectEpsilon) return false;
	return true;
}

bool Voxelize::isIntersect(const AABB &a)
{
	int stack[100];
	int stackPtr = 1;
	stack[0] = 0;
	int axis;
	unsigned int lChild;
	int side;

	const BVHNode *currentNode = &m_node[0];
	Vector3 boxCenter = 0.5f*(a.min + a.max);

	while(true)
	{
		if(isIntersect(a, currentNode))
		{
			if(!ISLEAF(currentNode))
			{
				lChild = GETLEFTCHILD(currentNode);

				axis = AXIS(currentNode);

				side = boxCenter.e[axis] < (0.5f*(currentNode->min + currentNode->max)).e[axis];

				stack[stackPtr] = side + lChild;
				currentNode =  &m_node[(side^1) + lChild];

				++stackPtr;
				continue;
			}
			else
			{
				for(int i=0;i<GETCHILDCOUNT(currentNode);i++)
				{
					const Triangle &tri = m_tri[currentNode->indexOffset+i];
					if(tri.p[0] == tri.p[1] || tri.p[1] == tri.p[2] || tri.p[2] == tri.p[0]) continue;
					if(isIntersect(a, tri)) 
						return true;
				}
			}
		}

		if(--stackPtr == 0) break;

		currentNode = &m_node[stack[stackPtr]];
	}

	return false;
}

void Voxelize::getIntersectVoxels(const BVHNode *tree, const Triangle &tri, std::vector<int> &list)
{
	int stack[100];
	int stackPtr = 1;
	stack[0] = 0;
	int axis;
	unsigned int lChild;
	int side;

	const BVHNode *currentNode = &tree[0];
	Vector3 triCenter = .33333333334*(m_vert[tri.p[0]].v + m_vert[tri.p[1]].v + m_vert[tri.p[2]].v);

	while(true)
	{
		if(isIntersect(AABB(currentNode->min, currentNode->max), tri))
		{
			if(!ISLEAF(currentNode))
			{
				lChild = GETLEFTCHILD(currentNode);

				axis = AXIS(currentNode);

				side = triCenter.e[axis] < (0.5f*(currentNode->min + currentNode->max)).e[axis];

				stack[stackPtr] = side + lChild;
				currentNode =  &tree[(side^1) + lChild];

				++stackPtr;
				continue;
			}
			else
			{
				for(int i=0;i<GETCHILDCOUNT(currentNode);i++)
				{
					const OOCVoxel &voxel = m_oocVoxelList[currentNode->indexOffset+i];
					if(isIntersect(voxel.rootBB, tri)) 
						list.push_back(currentNode->indexOffset+i);
				}
			}
		}

		if(--stackPtr == 0) break;

		currentNode = &tree[stack[stackPtr]];
	}
}

bool Voxelize::depthTest(int depth, const AABB &bb)
{
	if(depth < g_maxDepth) return false;
	if(depth >= g_maxDepth2) return true;
	static AABB manualBB(Vector3(-100, 200, -100), Vector3(250, 300, 100));
	//static AABB manualBB(Vector3(-600, 180, -50), Vector3(-300, 400, 200));
	if(!manualBB.isIn(bb.min)) return true;
	if(!manualBB.isIn(bb.max)) return true;
	return false;
}

int Voxelize::createOctreeNode(FILE *fp, int parentIndex, int myIndex, int &lastIndex, const AABB &box, int depth, bool generateOOC, bool *isOOCPart)
{
	int childIndex[N][N][N];
	AABB childBox[N][N][N];
	bool childIsOOCPart[N][N][N] = {0, };
	int curIndex = myIndex*N*N*N;

	//int voxelSize = N << (g_maxDepth-1);
	//unsigned char intChar[4];

	bool maxDepth = false;

	for(int x=0;x<N;x++)
	{
		for(int y=0;y<N;y++)
		{
			for(int z=0;z<N;z++)
			{
				Voxel voxel;
				voxel.clear();
				//voxel.parentIndex = parentIndex;

				childBox[x][y][z] = computeSubBox(x, y, z, box);
				childIndex[x][y][z] = 0;

				if(isIntersect(childBox[x][y][z]))
				{
					//if(depthTest(depth, childBox[x][y][z]))
					if(	(generateOOC && g_maxDepth == depth) ||
						(!generateOOC && g_maxDepth2 == depth))
					{
						maxDepth = true;
						/*
						int gx, gy, gz;
						Vector3 pos = (childBox[x][y][z].min + (0.5f*m_voxelDelta)) - m_BB.min;
						gx = (int)(pos[0] / m_voxelDelta[0]);
						gy = (int)(pos[1] / m_voxelDelta[1]);
						gz = (int)(pos[2] / m_voxelDelta[2]);
						*/

						voxel.setLeaf();
						/*
						memcpy(intChar, &curIndex, sizeof(int));
						m_indexMap[gx + gy*voxelSize + gz*voxelSize*voxelSize + 0*voxelSize*voxelSize*(__int64)voxelSize] = intChar[0];
						m_indexMap[gx + gy*voxelSize + gz*voxelSize*voxelSize + 1*voxelSize*voxelSize*(__int64)voxelSize] = intChar[1];
						m_indexMap[gx + gy*voxelSize + gz*voxelSize*voxelSize + 2*voxelSize*voxelSize*(__int64)voxelSize] = intChar[2];
						m_indexMap[gx + gy*voxelSize + gz*voxelSize*voxelSize + 3*voxelSize*voxelSize*(__int64)voxelSize] = intChar[3];
						//m_psh.push(HashElem<int>(Position(gx, gy, gz), curIndex));
						*/

						setGeomBitmap(voxel, childBox[x][y][z]);
					}
					else
					{
						voxel.setChildIndex(++lastIndex);
						childIndex[x][y][z] = voxel.getChildIndex();
					}
				}
				
				_fseeki64(fp, (__int64)curIndex*sizeof(Voxel)+sizeof(OctreeHeader), SEEK_SET);
				fwrite(&voxel, sizeof(Voxel), 1, fp);
				curIndex++;
			}
		}
	}

	int numVoxels = N*N*N;
	if(maxDepth) return numVoxels;

	curIndex = myIndex*N*N*N;
	bool hasOOCPart = false;
	for(int x=0;x<N;x++)
		for(int y=0;y<N;y++)
			for(int z=0;z<N;z++)
			{
				if(childIndex[x][y][z])
				{
					numVoxels += createOctreeNode(fp, curIndex, childIndex[x][y][z], lastIndex, childBox[x][y][z], depth+1, generateOOC, &childIsOOCPart[x][y][z]);
					hasOOCPart |= childIsOOCPart[x][y][z];
				}
				curIndex++;
			}

	curIndex = myIndex*N*N*N;
	if(hasOOCPart)
	{
		for(int x=0;x<N;x++)
			for(int y=0;y<N;y++)
				for(int z=0;z<N;z++)
				{
					if(childIndex[x][y][z] && !childIsOOCPart[x][y][z])
					{
						m_oocVoxelList.push_back(OOCVoxel(curIndex, depth+1, childBox[x][y][z]));
					}
					curIndex++;
				}
	}

	//if(generateOOC && numVoxels > 1000)
	if(generateOOC && depth <= g_maxDepth2-g_maxLowDepth)
	{
		if(!hasOOCPart)
		{
			m_oocVoxelList.push_back(OOCVoxel(parentIndex, depth, box));
		}
		if(isOOCPart)
			*isOOCPart = true;
	}

	return numVoxels;
}

void Voxelize::setGeomBitmap(Voxel &voxel, const AABB &box)
{
	AABB childBox1;
	AABB childBox2;
	// level 1
	for(int x1=0,offset1=0;x1<N;x1++)
	{
		for(int y1=0;y1<N;y1++)
		{
			for(int z1=0;z1<N;z1++,offset1++)
			{
				voxel.geomBitmap[offset1] = 0;

				childBox1 = computeSubBox(x1, y1, z1, box);
				if(isIntersect(childBox1))
				{
					// level 2
					for(int x2=0,offset2=0;x2<N;x2++)
					{
						for(int y2=0;y2<N;y2++)
						{
							for(int z2=0;z2<N;z2++,offset2++)
							{
								childBox2 = computeSubBox(x2, y2, z2, childBox1);

								if(isIntersect(childBox2))
								{
									voxel.geomBitmap[offset1] |= (1u << offset2);
								}
							}
						}
					}
				}
			}
		}
	}
}

int Voxelize::createOctreeNode()
{
	Vector3 center = 0.5f*(m_node[0].min + m_node[0].max);
	Vector3 size = m_node[0].max - m_node[0].min;
	float targetHalfSize = 0.5f*size.maxComponent();

	m_BB.min = Vector3(center.x() - targetHalfSize, center.y() - targetHalfSize, center.z() - targetHalfSize);
	m_BB.max = Vector3(center.x() + targetHalfSize, center.y() + targetHalfSize, center.z() + targetHalfSize);

	m_lastIndex = 0;

	OctreeHeader header;
	header.dim = N;
	header.maxDepth = g_maxDepth;
	header.min = m_BB.min;
	header.max = m_BB.max;
	fwrite(&header, sizeof(OctreeHeader), 1, m_fp);

	int voxelSize = N << (g_maxDepth-1);
	m_voxelDelta = (m_BB.max - m_BB.min) / voxelSize;
	m_intersectEpsilon = 0.0f;//(m_BB.max - m_BB.min).maxAbsComponent() * 0.0001f;

	//if(m_indexMap) delete[] m_indexMap;
	//m_indexMap = new unsigned char[voxelSize*voxelSize*(__int64)voxelSize*4];
	//memset(m_indexMap, 255, voxelSize*voxelSize*(__int64)voxelSize*4);

	Voxel::initD(m_BB);
	int numVoxels = createOctreeNode(m_fp, -1, 0, m_lastIndex, m_BB, 1, g_maxDepth != g_maxDepth2);

	/*
	cimg_library::CImg<unsigned char> data(m_indexMap, voxelSize, voxelSize, voxelSize, 4);
	PSH psh(data);
	psh.perform();
	psh.display();
	//psh.save("hash.txt", "offset.txt");
	*/
	return numVoxels;
}

int Voxelize::createOctreeNode(FILE *fp, const OOCVoxel &voxel)
{
	int lastIndex = 0;

	OctreeHeader header;
	header.dim = N;
	header.maxDepth = g_maxDepth2;
	header.min = voxel.rootBB.min;
	header.max = voxel.rootBB.max;
	fwrite(&header, sizeof(OctreeHeader), 1, fp);

	int voxelSize = N << (g_maxDepth2-1);
	m_voxelDelta = (voxel.rootBB.max - voxel.rootBB.min) / voxelSize;
	m_intersectEpsilon = 0.0f;//(voxel.rootBB.max - voxel.rootBB.min).maxAbsComponent() * 0.0001f;

	int numVoxels = createOctreeNode(fp, -1, 0, lastIndex, voxel.rootBB, voxel.startDepth, false);
	return numVoxels;
}

void Voxelize::setGeomLOD(int index, const AABB &bb)
{
	Vector3 center, extents[3];
	m_PCAOctree[index].ComputePC(center, extents);
	Vector3 corners[4];
	m_PCAOctree[index].Get4Corners(corners, (Vector3*)&bb.min);
	m_octree[index].setNorm(m_PCAOctree[index].GetGeometricNormal());
	m_octree[index].setD(dot(corners[0], m_octree[index].getNorm()));
	//m_octree[index].setD(dot(0.5f*(bb.min + bb.max), m_octree[index].getNorm()));
	/*
	m_octree[index].corners[0] = corners[0];
	m_octree[index].corners[1] = corners[1];
	m_octree[index].corners[2] = corners[2];
	m_octree[index].corners[3] = corners[3];
	*/
}

COOCPCAwoExtent Voxelize::computeGeomLOD(const AABB &bb, int index)
{
	int childIndex = index * N * N * N;

	COOCPCAwoExtent pca;
	for(int x=0;x<N;x++)
		for(int y=0;y<N;y++)
			for(int z=0;z<N;z++)
			{
				AABB subBox = computeSubBox(x, y, z, bb);
				if(m_octree[childIndex].hasChild())
				{
					m_PCAOctree[childIndex] = computeGeomLOD(subBox, m_octree[childIndex].getChildIndex());
					setGeomLOD(childIndex, subBox);
					pca = pca + m_PCAOctree[childIndex];
				}
				else if(m_octree[childIndex].isLeaf())
				{
					setGeomLOD(childIndex, subBox);
					pca = pca + m_PCAOctree[childIndex];
				}
				childIndex++;
			}

	return pca;
}

VoxelMaterialExtra Voxelize::computeMaterialLOD(int index)
{
	int childIndex = index * N * N * N;

	VoxelMaterialExtra materialExtra;
	for(int x=0;x<N;x++)
		for(int y=0;y<N;y++)
			for(int z=0;z<N;z++)
			{
				if(m_octree[childIndex].hasChild())
				{
					VoxelMaterialExtra childMaterialExtra = computeMaterialLOD(m_octree[childIndex].getChildIndex());
					m_octree[childIndex].setMat(childMaterialExtra.getKd(), childMaterialExtra.getKs(), childMaterialExtra.getD(), childMaterialExtra.getNs());
					materialExtra = materialExtra + childMaterialExtra;
				}
				else if(m_octree[childIndex].isLeaf())
				{
					m_leafVoxelMat[childIndex].normailize();
					materialExtra = materialExtra + m_leafVoxelMat[childIndex];
					m_octree[childIndex].setMat(m_leafVoxelMat[childIndex].getKd(), m_leafVoxelMat[childIndex].getKs(), m_leafVoxelMat[childIndex].getD(), m_leafVoxelMat[childIndex].getNs());
				}
				else	// empty voxel
				{
					materialExtra = materialExtra + VoxelMaterialExtra::getTransparentMaterial();
				}
				childIndex++;
			}

	materialExtra.normailize();
	return materialExtra;
}

float Voxelize::triArea(std::vector<Vector3> &verts, int pos)
{
	return cross(verts[pos+1] - verts[pos], verts[pos+2] - verts[pos]).length() * 0.5f;
}

void Voxelize::applyTri(int index, const AABB &bb, Vector3 *vert, const Vector3 &norm, const NewMaterial &material)
{
	const Voxel &voxel = m_octree[index];
	if(voxel.isLeaf())
	{
		m_PCAOctree[index].InsertTriangle(vert, norm);
		m_leafVoxelMat[index].addMaterial(material);
		return;
	}
	if(!voxel.hasChild()) return;

	int childIndex = voxel.getChildIndex() * N * N * N;

	for(int x=0;x<N;x++)
		for(int y=0;y<N;y++)
			for(int z=0;z<N;z++)
			{
				AABB subBox = computeSubBox(x, y, z, bb);
				if(isIntersect(subBox, vert))
					applyTri(childIndex, subBox, vert, norm, material);
				childIndex++;
			}
}

void Voxelize::computeLOD(const char *fileName, int startDepth)
{
	m_octree.load(fileName, startDepth);
	m_voxelDelta = m_octree.m_voxelDelta;

	m_PCAOctree = new COOCPCAwoExtent[m_octree.m_numVoxels];
	m_leafVoxelMat = new VoxelMaterialExtra[m_octree.m_numVoxels];

	// compute PCA for leaf voxels
	int numTris = (int)(m_tri.m_fileSize.QuadPart / sizeof(Triangle));

	int maxIndex = m_octree.m_voxelDelta.indexOfMaxComponent();
	float areaLimit = m_voxelDelta.e[maxIndex] * max(m_octree.m_voxelDelta.e[(maxIndex + 1)%3], m_octree.m_voxelDelta.e[(maxIndex + 2)%3]);
	rgb col;

	std::vector<Vector3> tesselatedTris;

	Vector3 absTriN;
	Vector3 up(0.0f, 1.0f, 0.0f);

	if(m_matList.size() == 0)
		m_matList.push_back(NewMaterial());

	for(int i=0;i<numTris;i++)
	{
		const Triangle &tri = m_tri[i];
		if(tri.p[0] == tri.p[1] || tri.p[1] == tri.p[2] || tri.p[2] == tri.p[0]) continue;

		const NewMaterial &material = m_matList[m_matList.size() == 1 ? 0 : tri.material];

		tesselatedTris.clear();
		tesselatedTris.push_back(m_vert[tri.p[0]].v);
		tesselatedTris.push_back(m_vert[tri.p[1]].v);
		tesselatedTris.push_back(m_vert[tri.p[2]].v);

		// tesselated triangle
		for(int pos=0;pos<tesselatedTris.size();pos+=3)
		{
			if(triArea(tesselatedTris, pos) > 0.5*areaLimit && tesselatedTris.size() < 100)
			{
				Vector3 v1 = tesselatedTris[pos+1];
				Vector3 v2 = tesselatedTris[pos+2];
				Vector3 newV1 = 0.5f*(tesselatedTris[pos+0] + tesselatedTris[pos+1]);
				Vector3 newV2 = 0.5f*(tesselatedTris[pos+1] + tesselatedTris[pos+2]);
				Vector3 newV3 = 0.5f*(tesselatedTris[pos+2] + tesselatedTris[pos+0]);
				tesselatedTris[pos+1] = newV1;
				tesselatedTris[pos+2] = newV3;

				tesselatedTris.push_back(newV1);
				tesselatedTris.push_back(v1);
				tesselatedTris.push_back(newV2);

				tesselatedTris.push_back(newV2);
				tesselatedTris.push_back(v2);
				tesselatedTris.push_back(newV3);

				tesselatedTris.push_back(newV1);
				tesselatedTris.push_back(newV2);
				tesselatedTris.push_back(newV3);
				pos = 0;
			}
		}

		//printf("1 -> %.1f\n", tesselatedTris.size() / 3.0f);
		for(int pos=0;pos<tesselatedTris.size();pos+=3)
		{
			Vector3 vert[3] = {tesselatedTris[pos], tesselatedTris[pos+1], tesselatedTris[pos+2]};

			Octree::Position minPos(INT_MAX, INT_MAX, INT_MAX), maxPos(0, 0, 0);
			for(int j=0;j<3;j++)
			{
				Octree::Position pos = m_octree.getPosition(vert[j]);
				minPos.x = min(minPos.x, pos.x); minPos.y = min(minPos.y, pos.y); minPos.z = min(minPos.z, pos.z);
				maxPos.x = max(maxPos.x, pos.x); maxPos.y = max(maxPos.y, pos.y); maxPos.z = max(maxPos.z, pos.z);
			}

			minPos.x = max(0, minPos.x-1);
			minPos.y = max(0, minPos.y-1);
			minPos.z = max(0, minPos.z-1);
			maxPos.x = min(N << (g_maxDepth-1), maxPos.x+1);
			maxPos.y = min(N << (g_maxDepth-1), maxPos.y+1);
			maxPos.z = min(N << (g_maxDepth-1), maxPos.z+1);

			for(int x=minPos.x;x<=maxPos.x;x++)
				for(int y=minPos.y;y<=maxPos.y;y++)
					for(int z=minPos.z;z<=maxPos.z;z++)
					{
						Octree::Position pos(x, y, z);

						Octree::Hash::iterator it = m_octree.m_hash.find(pos);
						if(it == m_octree.m_hash.end()) continue;

						AABB bb = m_octree.computeBox(pos);
						if(!isIntersect(bb, vert)) continue;

						absTriN = dot(tri.n, up) > 0 ? tri.n : -tri.n;
						//m_PCAOctree[m_octree.m_hash[pos]].InsertTriangle(vert, absTriN, col, m_octree.m_hash[pos]);
						applyTri(m_octree.m_hash[pos], bb, vert, absTriN, material);
						m_octree[m_octree.m_hash[pos]].m = tri.material;
					}

		}
		/*
		// insert each tesselated triangle into voxel which contain first vertex, (not except method)
		for(int pos=0;pos<tesselatedTris.size();pos+=3)
		{
			Octree::Position voxelPos = m_octree.getPosition(tesselatedTris[pos]);
			Octree::Hash::iterator it = m_octree.m_hash.find(voxelPos);
			if(it == m_octree.m_hash.end()) continue;

			Vector3 vert[3] = {tesselatedTris[pos], tesselatedTris[pos+1], tesselatedTris[pos+2]};
			m_PCAOctree[m_octree.m_hash[voxelPos]].InsertTriangle(vert, tri.n, col);
		}
		*/

		/*
		Vector3 vert[3] = {m_vert[tri.p[0]].v, m_vert[tri.p[1]].v, m_vert[tri.p[2]].v};

		Octree::Position minPos(INT_MAX, INT_MAX, INT_MAX), maxPos(0, 0, 0);
		for(int j=0;j<3;j++)
		{
			Octree::Position pos = m_octree.getPosition(vert[j]);
			minPos.x = min(minPos.x, pos.x); minPos.y = min(minPos.y, pos.y); minPos.z = min(minPos.z, pos.z);
			maxPos.x = max(maxPos.x, pos.x); maxPos.y = max(maxPos.y, pos.y); maxPos.z = max(maxPos.z, pos.z);
		}

		for(int x=minPos.x;x<=maxPos.x;x++)
			for(int y=minPos.y;y<=maxPos.y;y++)
				for(int z=minPos.z;z<=maxPos.z;z++)
				{
					Octree::Position pos(x, y, z);

					Octree::Hash::iterator it = m_octree.m_hash.find(pos);
					if(it == m_octree.m_hash.end()) continue;

					if(!isIntersect(m_octree.computeBox(pos), tri)) continue;

					m_PCAOctree[m_octree.m_hash[pos]].InsertTriangle(vert, tri.n, col);
				}
		*/
	}

	// compute PCA for LOD voxels, bottom up
	computeGeomLOD(AABB(m_octree.getHeader().min,m_octree.getHeader().max), 0);

	// compute avg materials for LOD voxels, bottom up
	computeMaterialLOD(0);

	m_octree.save(fileName);

	delete[] m_PCAOctree;
	delete[] m_leafVoxelMat;
	m_PCAOctree = 0;
	m_leafVoxelMat = 0;
}

int Voxelize::Do(const char *filepath, int maxDepth, int maxDepth2, int maxLowDepth)
{
	/*
	char fileName[256];
	OctreeHeader h[17];
	for(int i=0;i<17;i++)
	{
		sprintf_s(fileName, 255, "voxel_%d.ooc", i);
		FILE *fp;
		fopen_s(&fp, fileName, "rb");
		fread(&h[i], sizeof(OctreeHeader), 1, fp);
		fclose(fp);
	}
	*/

	TimerValue start, end;
	start.set();

	float elapsedHours;
	int elapsedMinOfHour;
	double elapsedMinOfHourFrac;
	OptionManager *opt = OptionManager::getSingletonPtr();

	g_maxDepth = maxDepth > 0 ? maxDepth : MAX_DEPTH;
	g_maxDepth2 = maxDepth2 > 0 ? maxDepth2 : MAX_DEPTH2;
	g_maxLowDepth = maxLowDepth > 0 ? maxLowDepth : MAX_LOW_DEPTH;

	char fileNameVoxel[MAX_PATH];
	char fileNameVert[MAX_PATH];
	char fileNameTri[MAX_PATH];
	char fileNameNode[MAX_PATH];
	char fileNameMaterial[MAX_PATH];

	sprintf(fileNameVoxel, "%s/default_voxel.ooc", filepath);
	sprintf(fileNameVert, "%s/vertex.ooc", filepath);
	sprintf(fileNameTri, "%s/tris.ooc", filepath);
	sprintf(fileNameNode, "%s/BVH.node", filepath);
	sprintf(fileNameMaterial, "%s/material.mtl", filepath);

	m_vertPtr = new OOC_FILE_CLASS<Vertex>(fileNameVert, 
							1024*1024*128,
							1024*1024*4);
	m_triPtr = new OOC_FILE_CLASS<Triangle>(fileNameTri, 
							1024*1024*256,
							1024*1024*4);
	m_nodePtr = new OOC_FILE_CLASS<BVHNode>(fileNameNode, 
							1024*1024*512,
							1024*1024*4);

	if(!m_vertPtr || !m_triPtr || !m_nodePtr)
	{
		cout << "File mapping failed!" << endl;
		return 0;
	}

	m_vert = *m_vertPtr;
	m_tri = *m_triPtr;
	m_node = *m_nodePtr;

	if(!loadMaterialFromMTL(fileNameMaterial, m_matList))
	{
		cout << "Couldn't find material file : " << fileNameMaterial << endl;
	}

	fopen_s(&m_fp, fileNameVoxel, "wb");

	TimerValue startUpper, endUpper;
	startUpper.set();
	printf("%d voxels are generated\n", createOctreeNode());
	endUpper.set();

	elapsedMinOfHourFrac = modf((endUpper - startUpper)/(float)(60*60), &elapsedHours);
	elapsedMinOfHour = elapsedMinOfHourFrac * 60.0;
	cout << "Voxelization (upper level) ended, time = " << (endUpper - startUpper) << "s (" << (int)elapsedHours << " h, " << elapsedMinOfHour << " min)" << endl;

	fclose(m_fp);
	m_fp = 0;

	cout << "Compute LOD" << endl;
	computeLOD(fileNameVoxel);

	int sizeVoxelList = (int)m_oocVoxelList.size();
	if(sizeVoxelList > 0)
		oocVoxelize(filepath);

	end.set();
	
	elapsedMinOfHourFrac = modf((end - start)/(float)(60*60), &elapsedHours);
	elapsedMinOfHour = elapsedMinOfHourFrac * 60.0;
	cout << "Voxelization ended, time = " << (end - start) << "s (" << (int)elapsedHours << " h, " << elapsedMinOfHour << " min)" << endl;

	return 1;
}

string Trim(const string& s)
{

	size_t f,e ;

	if (s.length() == 0)
		return s;

	if (s.c_str()[0] == 10)
		return "";

	f = s.find_first_not_of(" \t\r\n");
	e = s.find_last_not_of(" \t\r\n");

	if (f == string::npos)
		return "";
	return string(s,f,e-f+1);
}

bool Voxelize::loadMaterialFromMTL(const char *fileName, NewMaterialList &matList)
{
	FILE *fp;
	errno_t err;

	char workingDirectory[MAX_PATH];
	strcpy_s(workingDirectory, MAX_PATH-1, fileName);
	// remove last entry of prompt
	for(int i=strlen(workingDirectory)-1;i>=0;i--)
	{
		if(workingDirectory[i] == '/' || workingDirectory[i] == '\\')
		{
			workingDirectory[i] = 0;
			break;
		}
		workingDirectory[i] = 0;
	}


	if(err = fopen_s(&fp, fileName, "r")) return false;

	// parse MTL file
	char currentLine[500];
	NewMaterial mat;
	bool isFirstMat = true;
	while(fgets(currentLine, 499, fp))
	{
		if(strstr(currentLine, "newmtl"))
		{
			string curMatName = currentLine+7;
			curMatName = Trim(curMatName);

			if(!isFirstMat) 
			{
				matList.push_back(mat);
				mat.setDefault();
			}

			mat.setName(curMatName.c_str());

			isFirstMat = false;

			continue;
		}

		string curStr = currentLine;
		curStr = Trim(curStr);
		const char *curC = curStr.c_str();

		Vector3 tmpReflectance;

		if(strstr(curC, "#")) continue;		// skip comments

		if(strstr(curC, "Ka") == curC)
		{
			sscanf(curC, "Ka %f %f %f", 
				&tmpReflectance.e[0], 
				&tmpReflectance.e[1], 
				&tmpReflectance.e[2]);
			mat.setMatKa(tmpReflectance);
		}
		if(strstr(curC, "Kd") == curC)
		{
			sscanf(curC, "Kd %f %f %f", 
				&tmpReflectance.e[0], 
				&tmpReflectance.e[1], 
				&tmpReflectance.e[2]);
			mat.setMatKd(tmpReflectance);
		}
		if(strstr(curC, "Ks") == curC)
		{
			sscanf(curC, "Ks %f %f %f", 
				&tmpReflectance.e[0], 
				&tmpReflectance.e[1], 
				&tmpReflectance.e[2]);
			mat.setMatKs(tmpReflectance);
		}

		float tmpFloat;
		if(strstr(curC, "Ns") == curC)
		{
			sscanf(curC, "Ns %f", &tmpFloat);
			mat.setMat_Ns(tmpFloat);
		}
		if(strstr(curC, "d") == curC)
		{
			sscanf(curC, "d %f", &tmpFloat);
			mat.setMat_d(tmpFloat);
		}

		int tmpInt;
		if(strstr(curC, "illum") == curC)
		{
			sscanf(curC, "illum %d", &tmpInt);
			mat.setMat_illum(tmpInt);
		}

		if(strstr(curC, "map_Ka") == curC)
		{
		}

		if(strstr(curC, "map_Kd") == curC)
		{
		}

		if(strstr(curC, "map_bump") == curC)
		{
		}
	}
	matList.push_back(mat);
	fclose(fp);
	return true;
}

#include <string>
string toString(unsigned int v)
{
	string res;
	char tmp[2];
	tmp[1]=0x0;
	if (v != 0) {
		while (v != 0) {
			tmp[0]='0'+(char)(v%10);
			res.insert(0, &tmp[0]);
			v /= 10;
		}
	}
	else res = "0";
	return res;
}
string toString(unsigned int v, int l, char fill)
{
	string res = toString(v);
	char tmp[2];
	tmp[1] = 0x0;
	tmp[0] = fill;
	while (res.length() < l) 
		res.insert(0,&tmp[0]);
	return res;
}
STD ostream* warningOut = &(STD cerr);
#include "Files.h"
#include "BufferedOutputs.h"

bool Voxelize::Subdivide(BVHNode *tree, const AABB &rootBB, int &numNodes, TriangleIndexList *triIDs, unsigned int left, unsigned int right, unsigned int myIndex, unsigned int nextIndex, int depth)
{
	BVHNode *node = &tree[myIndex];
	int maxNumTrisPerLeaf = 1;

	// find biggest axis:
	Vector3 diff = rootBB.max - rootBB.min;
	if(myIndex != 0) diff = node->max - node->min;
	int biggestaxis = diff.indexOfMaxComponent();
	float split_pt = .5 * diff[biggestaxis] + rootBB.min[biggestaxis];
	if(myIndex != 0) split_pt = .5 * diff[biggestaxis] + node->min[biggestaxis];
	else {node->min = rootBB.min; node->max = rootBB.max;}

	// compute average primitive location:
	int tsz = right - left + 1;
	float avgloc;
	unsigned int curLeft = left, 
		         curRight = right;
	for (int count = 0; count < tsz; count++) 
	{
		const Triangle &tri = m_tri[triIDs->at(curLeft)];

		avgloc = m_vert[tri.p[0]].v.e[biggestaxis]; 
		avgloc += m_vert[tri.p[1]].v.e[biggestaxis]; 
		avgloc += m_vert[tri.p[2]].v.e[biggestaxis]; 
		avgloc *=.33333333334;

		if (avgloc < split_pt) {
			curLeft++;
		} else { // swap with last unprocessed element
			unsigned int temp = triIDs->at(curLeft);
			triIDs->at(curLeft) = triIDs->at(curRight);
			triIDs->at(curRight) = temp;			

			curRight--;
		}
	}	

	unsigned int numLeft = curLeft - left;
	
	// special case: subdivision did not work out, just go half/half
	if (numLeft == 0 || numLeft == tsz) {
		numLeft = tsz/2;		
	}
	
	node->children = ((nextIndex) * BVHNODE_BYTES >> 3) | biggestaxis;
	node->children2 = ((nextIndex+1) * BVHNODE_BYTES >> 3);

	numNodes += 2;

	float BB_min_limit[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
	float BB_max_limit[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};

	BVHNode *lChild = &tree[GETLEFTCHILD(node)];
	BVHNode *rChild = &tree[GETRIGHTCHILD(node)];
	// only one triangle left, make this a leaf:
	if (numLeft <= maxNumTrisPerLeaf) {
		unsigned int count = numLeft;
		lChild->indexCount = MAKECHILDCOUNT(count);
		lChild->indexOffset = triIDs->at(left);
		const Triangle &tri = m_tri[triIDs->at(left)];

//		setBB(lChild->min, lChild->max, GETVERTEX(tri.p[0]));
//		updateBB(lChild->min, lChild->max, GETVERTEX(tri.p[1]));
//		updateBB(lChild->min, lChild->max, GETVERTEX(tri.p[2]));

		// get real (global) triangle indice
		lChild->min.set(BB_min_limit);
		lChild->max.set(BB_max_limit);
		for(int i=0;i<count;i++)
		{
			const Triangle &tri = m_tri[triIDs->at(left+i)];
			updateBB(lChild->min, lChild->max, m_vert[tri.p[0]].v);
			updateBB(lChild->min, lChild->max, m_vert[tri.p[1]].v);
			updateBB(lChild->min, lChild->max, m_vert[tri.p[2]].v);	
		}
	}	
	else { 
		lChild->min.set(BB_min_limit);
		lChild->max.set(BB_max_limit);
		for (unsigned int index = left; index <= (left+numLeft-1); index++) {
			const Triangle &tri = m_tri[triIDs->at(index)];
			updateBB(lChild->min, lChild->max, m_vert[tri.p[0]].v);
			updateBB(lChild->min, lChild->max, m_vert[tri.p[1]].v);
			updateBB(lChild->min, lChild->max, m_vert[tri.p[2]].v);
		}
		
//		testWrite(lChild, node);

		Subdivide(tree, rootBB, numNodes, triIDs, left, left+numLeft-1, nextIndex, nextIndex + 2, depth + 1);
	}

	// only one triangle left, make this a leaf:
	if ((tsz - numLeft) <= maxNumTrisPerLeaf) {
		unsigned int count = tsz - numLeft;
		rChild->indexCount = MAKECHILDCOUNT(count);
		rChild->indexOffset = triIDs->at(left+numLeft);
		const Triangle &tri = m_tri[triIDs->at(left+numLeft)];	

		// get real (global) triangle indices
		rChild->min.set(BB_min_limit);
		rChild->max.set(BB_max_limit);
		for(int i=0;i<count;i++)
		{
			const Triangle &tri = m_tri[triIDs->at(left+numLeft+i)];
			updateBB(rChild->min, rChild->max, m_vert[tri.p[0]].v);
			updateBB(rChild->min, rChild->max, m_vert[tri.p[1]].v);
			updateBB(rChild->min, rChild->max, m_vert[tri.p[2]].v);	
		}
	}	
	else { 
		rChild->min.set(BB_min_limit);
		rChild->max.set(BB_max_limit);
		for (unsigned int index = left+numLeft; index <= right; index++) {
			const Triangle &tri = m_tri[triIDs->at(index)];

			updateBB(rChild->min, rChild->max, m_vert[tri.p[0]].v);
			updateBB(rChild->min, rChild->max, m_vert[tri.p[1]].v);
			updateBB(rChild->min, rChild->max, m_vert[tri.p[2]].v);
		}
		
		Subdivide(tree, rootBB, numNodes, triIDs, left+numLeft, right, nextIndex + 1, numNodes, depth + 1);
	}
	return true;
}

void Voxelize::buildBVH(int numTris, const AABB &bb, const char *fileName)
{
	if(numTris == 1)
	{
		BVHNode tree;

		tree.indexCount = MAKECHILDCOUNT(1);
		tree.indexOffset = 0;
		const Triangle &tri = m_tri[0];
		float BB_min_limit[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
		float BB_max_limit[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
		tree.min.set(BB_min_limit);
		tree.max.set(BB_max_limit);
		updateBB(tree.min, tree.max, m_vert[tri.p[0]].v);
		updateBB(tree.min, tree.max, m_vert[tri.p[1]].v);
		updateBB(tree.min, tree.max, m_vert[tri.p[2]].v);	

		FILE *fp;
		fopen_s(&fp, fileName, "wb");
		fwrite(&tree, sizeof(BVHNode), 1, fp);
		fclose(fp);
		return;
	}

	TriangleIndexList *leftlist[2];
	leftlist[0] = new TriangleIndexList(numTris);

	for (int i = 0; i < numTris; i++) 
	{
		const Triangle &tri = m_tri[i];
		(*leftlist[0])[i] = i;
	}

	BVHNode *tree = new BVHNode[numTris*2 - 1];

	for (int i = 0; i < numTris; i++) {	
		(*leftlist[0])[i] = i;
	}
		
	int numNodes = 1;
	Subdivide(tree, bb, numNodes, leftlist[0], 0, numTris-1, 0, 1 , 0);

	FILE *fp;
	fopen_s(&fp, fileName, "wb");
	fwrite(tree, sizeof(BVHNode), numTris*2-1, fp);
	fclose(fp);

	delete leftlist[0];
	delete[] tree;
}

void Voxelize::buildVoxelBVH(const char *fileName, const AABB &rootBB)
{
	LogManager *log = LogManager::getSingletonPtr();
	char output[500];

	int numTris = (int)(m_tri.m_fileSize.QuadPart / sizeof(Triangle));
		
	//
	// build tree:
	//
	buildBVH(numTris, rootBB, fileName);
}

void Voxelize::buildOOCVoxelBSPTree()
{
	int numOOCVoxel = m_oocVoxelList.size();
	std::vector<int> *leftlist[2];
	leftlist[0] = new std::vector<int>(m_oocVoxelList.size());

	m_oocVoxelBSPtree = new BVHNode[numOOCVoxel*2 - 1];

	for (int i = 0; i < numOOCVoxel; i++) {	
		(*leftlist[0])[i] = i;
	}
		
	int numNodes = 1;
	SubdivideBSPTree(m_oocVoxelBSPtree, m_BB, numNodes, leftlist[0], 0, numOOCVoxel-1, 0, 1 , 0);

	delete leftlist[0];
}

bool Voxelize::SubdivideBSPTree(BVHNode *tree, const AABB &rootBB, int &numNodes, std::vector<int> *voxelIDs, unsigned int left, unsigned int right, unsigned int myIndex, unsigned int nextIndex, int depth)
{
	BVHNode *node = &tree[myIndex];

	const int maxNumTrisPerLeaf = 1;

	// find biggest axis:
	Vector3 diff = rootBB.max - rootBB.min;
	if(myIndex != 0) diff = node->max - node->min;
	int biggestaxis = diff.indexOfMaxComponent();
	float split_pt = .5 * diff[biggestaxis] + rootBB.min[biggestaxis];
	if(myIndex != 0) split_pt = .5 * diff[biggestaxis] + node->min[biggestaxis];
	else {node->min = rootBB.min; node->max = rootBB.max;}

	// compute average primitive location:
	int tsz = right - left + 1;
	float avgloc;
	unsigned int curLeft = left, 
		         curRight = right;
	for (int count = 0; count < tsz; count++) 
	{
		const OOCVoxel &voxel = m_oocVoxelList[voxelIDs->at(curLeft)];

		avgloc = 0.5f*(voxel.rootBB.min + voxel.rootBB.max).e[biggestaxis];

		if (avgloc < split_pt) {
			curLeft++;
		} else { // swap with last unprocessed element
			unsigned int temp = voxelIDs->at(curLeft);
			voxelIDs->at(curLeft) = voxelIDs->at(curRight);
			voxelIDs->at(curRight) = temp;			

			curRight--;
		}
	}	

	unsigned int numLeft = curLeft - left;
	
	// special case: subdivision did not work out, just go half/half
	if (numLeft == 0 || numLeft == tsz) {
		numLeft = tsz/2;		
	}
	
	node->children = ((nextIndex) * BVHNODE_BYTES >> 3) | biggestaxis;
	node->children2 = ((nextIndex+1) * BVHNODE_BYTES >> 3);

	numNodes += 2;

	float BB_min_limit[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
	float BB_max_limit[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};

	BVHNode *lChild = &tree[GETLEFTCHILD(node)];
	BVHNode *rChild = &tree[GETRIGHTCHILD(node)];
	// only one voxel left, make this a leaf:
	if (numLeft <= maxNumTrisPerLeaf) {
		unsigned int count = numLeft;
		lChild->indexCount = MAKECHILDCOUNT(count);
		lChild->indexOffset = voxelIDs->at(left);
		const OOCVoxel &voxel = m_oocVoxelList[voxelIDs->at(left)];

		lChild->min.set(BB_min_limit);
		lChild->max.set(BB_max_limit);

		updateBB(lChild->min, lChild->max, voxel.rootBB.min);
		updateBB(lChild->min, lChild->max, voxel.rootBB.max);
	}	
	else { 
		lChild->min.set(BB_min_limit);
		lChild->max.set(BB_max_limit);
		for (unsigned int index = left; index <= (left+numLeft-1); index++) {
			const OOCVoxel &voxel = m_oocVoxelList[voxelIDs->at(index)];
			updateBB(lChild->min, lChild->max, voxel.rootBB.min);
			updateBB(lChild->min, lChild->max, voxel.rootBB.max);
		}
		
		SubdivideBSPTree(tree, rootBB, numNodes, voxelIDs, left, left+numLeft-1, nextIndex, nextIndex + 2, depth + 1);
	}

	// only one voxel left, make this a leaf:
	if ((tsz - numLeft) <= maxNumTrisPerLeaf) {
		unsigned int count = tsz - numLeft;
		rChild->indexCount = MAKECHILDCOUNT(count);
		rChild->indexOffset = voxelIDs->at(left+numLeft);
		const OOCVoxel &voxel = m_oocVoxelList[voxelIDs->at(left+numLeft)];

		// get real (global) triangle indices
		rChild->min.set(BB_min_limit);
		rChild->max.set(BB_max_limit);
		updateBB(rChild->min, rChild->max, voxel.rootBB.min);
		updateBB(rChild->min, rChild->max, voxel.rootBB.max);
	}	
	else { 
		rChild->min.set(BB_min_limit);
		rChild->max.set(BB_max_limit);
		for (unsigned int index = left+numLeft; index <= right; index++) {
			const OOCVoxel &voxel = m_oocVoxelList[voxelIDs->at(index)];

			updateBB(rChild->min, rChild->max, voxel.rootBB.min);
			updateBB(rChild->min, rChild->max, voxel.rootBB.max);
		}
		
		SubdivideBSPTree(tree, rootBB, numNodes, voxelIDs, left+numLeft, right, nextIndex + 1, numNodes, depth + 1);
	}
	return true;
}


void Voxelize::oocVoxelize(const char *filePath)
{
	OptionManager *opt = OptionManager::getSingletonPtr();
	int sizeVoxelList = (int)m_oocVoxelList.size();
	printf("generate OOC voxels (%d voxels)\n", sizeVoxelList);

	char fileName[256];


	TimerValue startBin, endBin;
	TimerValue startBuild, endBuild;
	TimerValue startMerge, endMerge;

	float elapsedHours;
	int elapsedMinOfHour;
	double elapsedMinOfHourFrac;

	// bining
	startBin.set();
	printf("Assign triangles to each voxel...\n");
	BufferedOutputs<Triangle> **pTrisList = new BufferedOutputs<Triangle> *[sizeVoxelList];
	for(int i=0;i<sizeVoxelList;i++)
	{
		sprintf_s(fileName, 255, "%s/tri_%d.ooc", filePath, i);
		pTrisList[i] = new BufferedOutputs<Triangle>(fileName, 1000);
		pTrisList[i]->clear();
	}

	int numTris = (int)(m_tri.m_fileSize.QuadPart / sizeof(Triangle));
	int numProcessedTri = 0;
	int numMaxThreads = 64;

	typedef stdext::hash_map<int, std::vector<Triangle> > AssignMap;
	typedef AssignMap::iterator AssignMapIt;
	AssignMap *assignMap = new AssignMap[numMaxThreads];
	std::vector<int> *listIntersect = new std::vector<int>[numMaxThreads];

	buildOOCVoxelBSPTree();
	while(numProcessedTri < numTris)
	{
		int numCurTri = min(1000000, numTris - numProcessedTri);
		//printf("%d tris are assigned.\n", numProcessedTri);
		for(int i=0;i<numMaxThreads;i++)
			assignMap[i].clear();

#		pragma omp parallel for shared(numProcessedTri) schedule(dynamic)
		for(int i=0;i<numCurTri;i++)
		{
			const Triangle &tri = m_tri[i+numProcessedTri];
			int threadID = omp_get_thread_num();

#			if 0
			for(int j=0;j<sizeVoxelList;j++)
			{
				const OOCVoxel &oocVoxel = m_oocVoxelList[j];
				if(isIntersect(oocVoxel.rootBB, tri))
				{
					assignMap[threadID][j].push_back(tri);
					//pTrisList[j]->appendElement(tri);
				}
			}
#			else
			std::vector<int> &list = listIntersect[threadID];
			list.clear();
			getIntersectVoxels(m_oocVoxelBSPtree, tri, list);
			for(int j=0;j<list.size();j++)
				assignMap[threadID][list[j]].push_back(tri);
#			endif
		}

		for(int i=0;i<numMaxThreads;i++)
		{
			AssignMap &map = assignMap[i];
			for(AssignMapIt it=map.begin();it!=map.end();it++)
			{
				const std::vector<Triangle> &tris = it->second;
				for(int j=0;j<tris.size();j++)
				{
					pTrisList[it->first]->appendElement(tris[j]);
				}
			}
		}

		numProcessedTri += numCurTri;
	}

	delete[] m_oocVoxelBSPtree;
	delete[] assignMap;
	delete[] listIntersect;
	
	for(int i=0;i<sizeVoxelList;i++)
	{
		delete pTrisList[i];
	}
	delete[] pTrisList;

	endBin.set();

	Progression prog("OOCVoxel construction", sizeVoxelList, 100);

	startBuild.set();
#	if 0
	for(int i=0;i<sizeVoxelList;i++)
	{
		const OOCVoxel &oocVoxel = m_oocVoxelList[i];

		sprintf_s(fileName, 255, "tri_%d.ooc", i);

		OOC_FILE_CLASS<Triangle> *m_localTriPtr = new OOC_FILE_CLASS<Triangle>(fileName, 
								1024*1024*opt->getOptionAsInt("ooc", "maxCacheMemTrisMB", 256),
								1024*opt->getOptionAsInt("ooc", "cacheEntrySizeTrisKB", 1024*4));


		m_tri = *m_localTriPtr;

		sprintf_s(fileName, 255, "BVH_%d.ooc", i);

		buildVoxelBVH(fileName, oocVoxel.rootBB);

		OOC_FILE_CLASS<BVHNode> *m_localNodePtr = new OOC_FILE_CLASS<BVHNode>(fileName, 
								1024*1024*opt->getOptionAsInt("ooc", "maxCacheMemTreeNodesMB", 512),
								1024*opt->getOptionAsInt("ooc", "cacheEntrySizeTreeNodesKB", 1024*4));

		m_node = *m_localNodePtr;

		sprintf_s(fileName, 255, "voxel_%d.ooc", i);
		FILE *fp;
		fopen_s(&fp, fileName, "wb");
		createOctreeNode(fp, m_oocVoxelList[i]);
		fclose(fp);
		computeLOD(fileName, m_oocVoxelList[i].startDepth);

		delete m_localTriPtr;
		delete m_localNodePtr;
		prog.step();
	}
#	else
	Voxelize *voxellizes = new Voxelize[numMaxThreads];
#	pragma omp parallel for
	for(int i=0;i<sizeVoxelList;i++)
	{
		const OOCVoxel &oocVoxel = m_oocVoxelList[i];

		Voxelize &voxellize = voxellizes[omp_get_thread_num()];
		char fileName[256];

		voxellize.m_matList = m_matList;
		voxellize.m_vert = m_vert;

		sprintf_s(fileName, 255, "%s/tri_%d.ooc", filePath, i);

		OOC_FILE_CLASS<Triangle> *localTriPtr = new OOC_FILE_CLASS<Triangle>(fileName, 1024*1024*256, 1024*1024*4);


		voxellize.m_tri = *localTriPtr;

		sprintf_s(fileName, 255, "%s/BVH_%d.ooc", filePath, i);

		voxellize.buildVoxelBVH(fileName, oocVoxel.rootBB);

		OOC_FILE_CLASS<BVHNode> *localNodePtr = new OOC_FILE_CLASS<BVHNode>(fileName, 1024*1024*512, 1024*1024*4);

		voxellize.m_node = *localNodePtr;

		sprintf_s(fileName, 255, "%s/voxel_%d.ooc", filePath, i);
		FILE *fp;
		fopen_s(&fp, fileName, "wb");
		voxellize.createOctreeNode(fp, m_oocVoxelList[i]);
		fclose(fp);
		voxellize.computeLOD(fileName, m_oocVoxelList[i].startDepth);

		delete localTriPtr;
		delete localNodePtr;
#		pragma omp critical
		prog.step();
	}

	delete[] voxellizes;
#	endif
	endBuild.set();

	startMerge.set();
	m_tri = *m_triPtr;
	m_node = *m_nodePtr;

	char fileNameOOCVoxelHeader[MAX_PATH];
	char fileNameOOCVoxel[MAX_PATH];
	sprintf(fileNameOOCVoxelHeader, "%s/default_OOCVoxel.hdr", filePath);
	sprintf(fileNameOOCVoxel, "%s/default_OOCVoxel.ooc", filePath);
	FILE *fpHeader, *fpOOC;
	fopen_s(&fpHeader, fileNameOOCVoxelHeader, "wb");
	fopen_s(&fpOOC, fileNameOOCVoxel, "wb");

	int offset = 0;
	for(int i=0;i<sizeVoxelList;i++)
	{
		OOCVoxel &header = m_oocVoxelList[i];
		header.offset = offset;

		sprintf_s(fileName, 255, "%s/voxel_%d.ooc", filePath, i);
		FILE *fpVoxel;
		OctreeHeader temp;
		fopen_s(&fpVoxel, fileName, "rb");
		fread(&temp, sizeof(OctreeHeader), 1, fpVoxel);
		int size = (int)filelength(fileno(fpVoxel)) - sizeof(OctreeHeader);
		int numVoxels = size / sizeof(Voxel);
		Voxel *buf = new Voxel[numVoxels];
		numVoxels = fread(buf, sizeof(Voxel), numVoxels, fpVoxel);
		numVoxels = fwrite(buf, sizeof(Voxel), numVoxels, fpOOC);
		header.numVoxels = numVoxels;
		offset += numVoxels;
			
		delete[] buf;
		fclose(fpVoxel);
		fwrite(&header, sizeof(OOCVoxel), 1, fpHeader);
	}

	for(int i=0;i<sizeVoxelList;i++)
	{
		sprintf_s(fileName, 255, "%s/voxel_%d.ooc", filePath, i);
		unlink(fileName);
		sprintf_s(fileName, 255, "%s/tri_%d.ooc", filePath, i);
		unlink(fileName);
		sprintf_s(fileName, 255, "%s/BVH_%d.ooc", filePath, i);
		unlink(fileName);
	}

	fclose(fpHeader);
	fclose(fpOOC);

	endMerge.set();

	elapsedMinOfHourFrac = modf((endBin - startBin)/(float)(60*60), &elapsedHours);
	elapsedMinOfHour = elapsedMinOfHourFrac * 60.0;
	cout << "Binning time = " << (endBin - startBin) << "s (" << (int)elapsedHours << " h, " << elapsedMinOfHour << " min)" << endl;

	elapsedMinOfHourFrac = modf((endBuild - startBuild)/(float)(60*60), &elapsedHours);
	elapsedMinOfHour = elapsedMinOfHourFrac * 60.0;
	cout << "Building time = " << (endBuild - startBuild) << "s (" << (int)elapsedHours << " h, " << elapsedMinOfHour << " min)" << endl;

	elapsedMinOfHourFrac = modf((endMerge - startMerge)/(float)(60*60), &elapsedHours);
	elapsedMinOfHour = elapsedMinOfHourFrac * 60.0;
	cout << "Merging time = " << (endMerge - startMerge) << "s (" << (int)elapsedHours << " h, " << elapsedMinOfHour << " min)" << endl;
}
