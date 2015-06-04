#include "StdAfx.h"
#include "BVH.h"
#include "RangeDecoder_File.h"
#include "positionquantizer_new.h"
#include "integercompressor_new.h"

CLODMetric LODMetric;
float g_MaxAllowModifier;
int g_NumTraversed = 0;
int g_NumIntersected = 0;
int g_tempMax = 0;

#if HIERARCHY_TYPE == TYPE_BVH
void BVH::drawNode(BSPArrayTreeNodePtr node, Vector3 min, Vector3 max, int depth, unsigned int subObjectId) {
	extern int bbStep;
	extern int bbDepthStep;

	while(!nodeQueue.empty()) nodeQueue.pop();
	queueElem elem;
	elem.node = node;
	elem.depth = 0;
	nodeQueue.push(elem);
	while(true)
	{
//	if(bbDepthStep > elem.depth) bbStep++;
	if(bbStep <= curStep) return;
	curStep++;

//	if(bbDepthStep < elem.depth) bbDepthStep = elem.depth;

	if(nodeQueue.empty()) return;
	elem = nodeQueue.front();
	BSPArrayTreeNodePtr node = elem.node;
	nodeQueue.pop();
	if(ISLEAF(node)) continue;
	min = node->min;
	max = node->max;

	ModelInstance *subObject = &objectList[subObjectId];
	BSPArrayTreeNodePtr children = GETNODE(subObject->tree,GETLEFTCHILD(node));
	BSPArrayTreeNodePtr children2 = GETNODE(subObject->tree,GETRIGHTCHILD(node));

	switch(elem.depth%3)
	{
	case 0 : 
		glColor4f(0.9f, 0, 0, 0.5f);
		break;
	case 1 : 
		glColor4f(0, 0.9f, 0, 0.5f);
		break;
	case 2 : 
		glColor4f(0, 0, 0.9f, 0.5f);
		break;
	}

	elem.node = children;
	elem.depth = elem.depth + 1;
	nodeQueue.push(elem);
	elem.node = children2;
	nodeQueue.push(elem);

	float x1, y1, z1, x2, y2, z2;
	x1 = min.e[0];
	y1 = min.e[1];
	z1 = min.e[2];
	x2 = max.e[0];
	y2 = max.e[1];
	z2 = max.e[2];
	glBegin(GL_LINE_LOOP);
		glVertex3f(x1, y1, z1);
		glVertex3f(x2, y1, z1);
		glVertex3f(x2, y2, z1);
		glVertex3f(x1, y2, z1);
		glVertex3f(x1, y1, z1);
	glEnd();
	glBegin(GL_LINE_LOOP);
		glVertex3f(x1, y1, z2);
		glVertex3f(x2, y1, z2);
		glVertex3f(x2, y2, z2);
		glVertex3f(x1, y2, z2);
		glVertex3f(x1, y1, z2);
	glEnd();
	glBegin(GL_LINES);
		glVertex3f(x1, y1, z1);
		glVertex3f(x1, y1, z2);
	glEnd();
	glBegin(GL_LINES);
		glVertex3f(x2, y1, z1);
		glVertex3f(x2, y1, z2);
	glEnd();
	glBegin(GL_LINES);
		glVertex3f(x1, y2, z1);
		glVertex3f(x1, y2, z2);
	glEnd();
	glBegin(GL_LINES);
		glVertex3f(x2, y2, z1);
		glVertex3f(x2, y2, z2);
	glEnd();

	}
	/*
	// Inner node ? Then recurse
	if (ISNOLEAF(node)) {		
		drawNode(children, children->min, children->max, depth + 1, subObjectId);
		drawNode(children2, children2->min, children2->max, depth + 1, subObjectId);
	}
	*/
}


void BVH::GLdrawTree(Ray &viewer, unsigned int subObjectId)
{
	ModelInstance *subObject = &objectList[subObjectId];
	glMatrixMode( GL_PROJECTION );
	glLoadIdentity();
	glFrustum(-1,1, -1,1, 2, 4000 );
	glMatrixMode( GL_MODELVIEW );

	glLoadIdentity();
	Vector3 lookAt = viewer.origin() + viewer.direction();
	gluLookAt(viewer.origin().x(), viewer.origin().y(), viewer.origin().z(),  
		lookAt.x(), lookAt.y(), lookAt.z(), 
		0, 1, 0);

	// clear image
	glClearColor(0,0,0,1);
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

	// set OpenGL state
	glEnable(GL_DEPTH_TEST);		
	glDisable(GL_TEXTURE_2D);	
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_BLEND);

	// wire color
	glColor4f(0.9, 0.1, 0.1, 0.7);

	curStep = 0;
	// recursive draw of BSP tree
	drawNode(GETNODE(subObject->tree,0), objectBB[0], objectBB[1], 0, subObjectId);

	// restore state
	glDisable(GL_BLEND);
	glDisable(GL_DEPTH_TEST);	
	glEnable(GL_TEXTURE_2D);
}

void BVH::printTree(bool dumpTree, const char *LoggerName, unsigned int subObjectId)
{
}

void BVH::printHighLevelTree(bool dumpTree, const char *LoggerName)
{
}

void BVH::buildHighLevelTree()
{
}
int BVH::RayTriIntersect(const Ray &ray, ModelInstance *object, BSPArrayTreeNodePtr node, HitPointPtr hitPoint, float tmax)  
{
	float point[2];
	float vdot, vdot2;
	float alpha, beta;
	float t, u0, v0, u1, v1, u2, v2;

	unsigned int triID = *MAKEIDX_PTR(object->indexlist,GETIDXOFFSET(node));
	const Triangle &tri = GETTRI(object, triID);
	assert(tri.i1 <= 2);
	assert(tri.i2 <= 2);

	// is ray parallel to plane or a back face ?
	vdot = dot(ray.direction(), tri.n);		
	//if (vdot > -(sign*EPSILON))
	//	return false;

	// find parameter t of ray -> intersection point
	vdot2 = dot(ray.origin(),tri.n);
	t = (tri.d - vdot2) / vdot;

	// if either too near or further away than a previous hit, we stop
	if (t < -INTERSECT_EPSILON || t > (tmax + INTERSECT_EPSILON))
		return false;

	// intersection point with plane
	point[0] = ray.data[0].e[tri.i1] + ray.data[1].e[tri.i1] * t;
	point[1] = ray.data[0].e[tri.i2] + ray.data[1].e[tri.i2] * t;

	// begin barycentric intersection algorithm 
	const Vector3 &tri_p0 = GETVERTEX(object,tri.p[0]); 
	float p0_1 = tri_p0.e[tri.i1], p0_2 = tri_p0.e[tri.i2]; 
	u0 = point[0] - p0_1; 
	v0 = point[1] - p0_2; 
	const Vector3 &tri_p1 = GETVERTEX(object,tri.p[1]); 
	u1 = tri_p1[tri.i1] - p0_1; 
	v1 = tri_p1[tri.i2] - p0_2; 
	const Vector3 &tri_p2 = GETVERTEX(object,tri.p[2]); 
	u2 = tri_p2[tri.i1] - p0_1; 
	v2 = tri_p2[tri.i2] - p0_2;

	beta = (v0 * u1 - u0 * v1) / (v2 * u1 - u2 * v1);
	//if (beta < 0 || beta > 1)
	if (beta < -TRI_INTERSECT_EPSILON || beta > 1 + TRI_INTERSECT_EPSILON)
		return false;
	alpha = (u0 - beta * u2) / u1;	

	// not in triangle ?	
	if (alpha < -TRI_INTERSECT_EPSILON || (alpha + beta) > 1.0f + TRI_INTERSECT_EPSILON)
		return false;

	// we have a hit:	
	hitPoint->alpha = alpha;  // .. and barycentric coords
	hitPoint->beta  = beta;

	// Fill hitpoint structure:
	//
	#ifdef _USE_TRI_MATERIALS
	hitPoint->m = tri.material;
	#else
	hitPoint->m = defaultMaterial;
	#endif
	hitPoint->t = t;
	hitPoint->triIdx = triID;

	hitPoint->objectPtr = object;
	hitPoint->m_TraveledDist = t;

	//hitPoint->objectPtr = subObject;
	#ifdef USE_LOD
	hitPoint->m_TraveledDist = t;
	#endif

	#ifdef _USE_VERTEX_NORMALS
	// interpolate vertex normals..
	hitPoint->n = tri.normals[0] + hitPoint->alpha * tri.normals[1] + hitPoint->beta * tri.normals[2];
	#else
	hitPoint->n = tri.n;
	#endif

	if (vdot > 0.0f)
		hitPoint->n *= -1.0f;

	#ifdef _USE_TEXTURING
	// interpolate tex coords..
	hitPoint->uv = tri.uv[0] + hitPoint->alpha * tri.uv[1] + hitPoint->beta * tri.uv[2];
	#endif

	// hitpoint:
	hitPoint->x = ray.pointAtParameter(t);
	
	return true;	
}

FORCEINLINE bool BVH::RayBoxIntersect(const Ray& r, Vector3 &min, Vector3 &max, float *returnMin, float *returnMax)  {
	float interval_min = -FLT_MAX;
	float interval_max = FLT_MAX;
	Vector3 pp[2];

	pp[0] = min;
	pp[1] = max;

	float t0 = (pp[r.posneg[3]].e[0] - r.data[0].e[0]) * r.data[2].e[0];
	float t1 = (pp[r.posneg[0]].e[0] - r.data[0].e[0]) * r.data[2].e[0];
	if (t0 > interval_min) interval_min = t0;
	if (t1 < interval_max) interval_max = t1;
	if (interval_min > interval_max) return false;

	t0 = (pp[r.posneg[4]].e[1] - r.data[0].e[1]) * r.data[2].e[1];
	t1 = (pp[r.posneg[1]].e[1] - r.data[0].e[1]) * r.data[2].e[1];
	if (t0 > interval_min) interval_min = t0;
	if (t1 < interval_max) interval_max = t1;
	if (interval_min > interval_max) return false;

	t0 = (pp[r.posneg[5]].e[2] - r.data[0].e[2]) * r.data[2].e[2];
	t1 = (pp[r.posneg[2]].e[2] - r.data[0].e[2]) * r.data[2].e[2];
	if (t0 > interval_min) interval_min = t0;
	if (t1 < interval_max) interval_max = t1;

	*returnMin = interval_min;
	*returnMax = interval_max;
	return (interval_min <= interval_max);
}

void testTraverse(ModelInstance *object, unsigned int cur, int *numTraverse, int depth, int *maxDepth, float *avgDepth, int *numLeaf)
{
	*numTraverse = *numTraverse + 1;
	BSPArrayTreeNodePtr curNode = GETNODE(object->tree, cur);

	if(*maxDepth < depth) *maxDepth = depth;
	if(ISLEAF(curNode)) 
	{
		*avgDepth = ((*avgDepth)*(*numLeaf) + depth)/((*numLeaf)+1);
		*numLeaf = *numLeaf + 1;
		return ;
	}

	unsigned int lChild = GETLEFTCHILD(curNode);
	unsigned int rChild = GETRIGHTCHILD(curNode);

	testTraverse(object, lChild, numTraverse, depth+1, maxDepth, avgDepth, numLeaf);
	testTraverse(object, rChild, numTraverse, depth+1, maxDepth, avgDepth, numLeaf);
}

int BVH::RayTreeIntersectRecursive(Ray &ray, HitPointPtr hit, unsigned int index, float traveledDist)
{
	ModelInstance *object = &objectList[0];
	ray.transform(&object->translate_world);
	int numTraverse = 0;
	int maxDepth = 0;
	float avgDepth = 0;
	int numLeaf = 0;
//	testTraverse(object, 0, &numTraverse, 0, &maxDepth, &avgDepth, &numLeaf);

	BSPArrayTreeNodePtr currentNode = GETNODE(object->tree, index);
	float min, max;
	bool hasHit = false;
	hit->t = FLT_MAX;
	bool hitTest = RayBoxIntersect(ray, currentNode->min, currentNode->max, &min, &max);
	if ( hitTest && min < hit->t && max > 0.0f)
	{
		if(ISLEAF(currentNode))
		{
			return RayTriIntersect(ray, object, currentNode, hit, min(max, hit->t));
		}
		hasHit = hasHit || RayTreeIntersectRecursive(ray, hit, GETLEFTCHILD(currentNode), traveledDist);
		hasHit = hasHit || RayTreeIntersectRecursive(ray, hit, GETRIGHTCHILD(currentNode), traveledDist);
	}
	return hasHit;
}

/*
int BVH::RayTreeIntersect(Ray &ray, HitPointPtr hit, float traveledDist)
{
	return RayTreeIntersectRecursive(ray, hit, 0, traveledDist);
}
*/

#define TYPE_NO_COMPRESS 1
#define TYPE_COMPRESS1 2
#define COMPRESS_TYPE TYPE_COMPRESS1

int BVH::RayTreeIntersect(Ray &ray, HitPointPtr hit, float traveledDist)
{
	static int NumInter = 0;
	
	ModelInstance *object = &objectList[0];
	ray.transform(&object->translate_world);
	/*
	int numTraverse = 0;
	int maxDepth = 0;
	float avgDepth = 0;
	int numLeaf = 0;
	testTraverse(object, 0, &numTraverse, 0, &maxDepth, &avgDepth, &numLeaf);
	*/
	static StackElem *stack = (StackElem *)_aligned_malloc(MAXBVHDEPTH * sizeof(StackElem), 16);		

	int stackPtr;
	BSPArrayTreeNodePtr currentNode, parentNode;
	int CurrentDepth = 0;
	float min, max;	
	bool hasHit = false, HasHitRootBV = false;
	hit->t = FLT_MAX;	

	stack[0].index = 0;
	stackPtr = 1;	

	#if COMPRESS_TYPE == TYPE_NO_COMPRESS
	currentNode = GETNODE(object->tree, 0);
	#endif
	#if COMPRESS_TYPE == TYPE_COMPRESS1
	Vector3 pMin, pMax;
	Vector3 cMin, cMax;

	currentNode = getNode(0);
	cMin = currentNode->min + pMin;
	cMax = currentNode->max + pMax;
	#endif

	// traverse BVH tree:
	while (1) {
		// is current node intersected and also closer than previous hit?

		#if COMPRESS_TYPE == TYPE_NO_COMPRESS
		bool hitTest = RayBoxIntersect(ray, currentNode->min, currentNode->max, &min, &max);
		#endif
		#if COMPRESS_TYPE == TYPE_COMPRESS1
		bool hitTest = RayBoxIntersect(ray, cMin, cMax, &min, &max);
		#endif
		
		if ( hitTest && min < hit->t && max > 0.0f) {

			// is inner node?
			if (!ISLEAF(currentNode)) {
				// Store ordered children
				//BSPArrayTreeNodePtr child = GETNODE(object->tree, GETLEFTCHILD(currentNode));

				//BSPArrayTreeNodePtr thisNode = currentNode;
				//BSPArrayTreeNodePtr lChild = GETNODE(object->tree, GETLEFTCHILD(currentNode));
				//BSPArrayTreeNodePtr rChild = GETNODE(object->tree, GETRIGHTCHILD(currentNode));
				//	assert(currentNode->min <= lChild->min);
				//	assert(currentNode->max >= lChild->max);
				//	assert(currentNode->min <= rChild->min);
				//	assert(currentNode->max >= rChild->max);
				int axis = AXIS(currentNode);
				unsigned int farChild = GETLEFTCHILD(currentNode) + (ray.posneg[axis] << 2);
				unsigned int nearChild = GETLEFTCHILD(currentNode) + ((ray.posneg[axis] ^ 1) << 2);

				#if COMPRESS_TYPE == TYPE_NO_COMPRESS
				stack[stackPtr].index = farChild;
				currentNode = GETNODE(object->tree, nearChild);
				#endif

				#if COMPRESS_TYPE == TYPE_COMPRESS1
				pMin = cMin;
				pMax = cMax;
				float mid = 0.5 * (cMax.e[axis] - cMin.e[axis]) + cMin.e[axis];

				stack[stackPtr].index = farChild;
				stack[stackPtr].pMin = pMin;
				stack[stackPtr].pMax = pMax;

				if(ray.posneg[axis])
				{
					pMax.e[axis] = mid;
					stack[stackPtr].pMin.e[axis] = mid;
				}
				else
				{
					pMin.e[axis] = mid;
					stack[stackPtr].pMax.e[axis] = mid;
				}

				currentNode = getNode(nearChild);

				cMin = currentNode->min + pMin;
				cMax = currentNode->max + pMax;
				#endif

				stackPtr++;
				continue;
			}
			else {
				// is leaf node:
				// intersect with current node's members
				bool tempHit = RayTriIntersect(ray, object, currentNode, hit, min(max, hit->t));
				hasHit = tempHit || hasHit;
				if(hasHit) break;
			}
		}
		if (--stackPtr == 0) break;
		
		// fetch next node from stack
		#if COMPRESS_TYPE == TYPE_NO_COMPRESS
		currentNode = GETNODE(object->tree, stack[stackPtr].index);
		#endif

		#if COMPRESS_TYPE == TYPE_COMPRESS1
		currentNode = getNode(stack[stackPtr].index);
		pMin = stack[stackPtr].pMin;
		pMax = stack[stackPtr].pMax;
		cMin = currentNode->min + pMin;
		cMax = currentNode->max + pMax;
		#endif

		// traversal ends when stack empty
//		if (currentNode == NULL)
//			break;
	}

	//_aligned_free(stack);

	// return hit status
	return hasHit;
}
void BVH::initialize(const char *filename)
{
	fp = fopen(filename, "rb");
	loadClusterTable();
}
void BVH::finalize()
{
	if(fp)
		fclose(fp);
}

static PositionQuantizerNew* pq;
static IntegerCompressorNew* ic[3];

static void decompressVertexPosition(float* n)
{
  if (pq)
  {
    int* qn = (int*)n;
    for (int i = 0; i < 3; i++)
    {
      qn[i] = ic[i]->DecompressNone();
    }
  }
  /*
  else
  {
    for (int i = 0; i < 3; i++)
    {
      n[i] = fc[i]->DecompressNone();
    }
  }
  */
}

static void decompressVertexPosition(const float* l, float* n)
{
  if (pq)
  {
    const int* ql = (const int*)l;
    int* qn = (int*)n;
    for (int i = 0; i < 3; i++)
    {
      qn[i] = ic[i]->DecompressLast(ql[i]);
    }
  }
  /*
  else
  {
    for (int i = 0; i < 3; i++)
    {
      n[i] = fc[i]->DecompressLast(l[i]);
    }
  }
  */
}

BSPArrayTreeNodePtr BVH::getNode(unsigned int index)
{
	unsigned int CN = ((index << 3) >> nodesPerClusterPower) >> BSPTREENODESIZEPOWER;
	if(clusterTable[CN].PCN == -1)
	{
		// Page Fault
		long DCO = clusterTable[CN].DCO;

		// Victim selection policy : Random
		int PCN;
		if(emptyList.empty())
		{
			PCN = rand()%maxNumPCN;
			for(int i=0;i<numClusters;i++)
			{
				if(clusterTable[i].PCN == PCN)
				{
					clusterTable[i].PCN = -1;
					break;
				}
			}
		}
		else
		{
			PCN = emptyList.front();
			emptyList.pop();
		}

		clusterTable[CN].PCN = PCN;

		fseek(fp, DCO, SEEK_SET);

		RangeDecoder *rd_geom = new RangeDecoderFile(fp);
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

		ic[0]->SetupDecompressor(rd_geom);
		ic[1]->SetupDecompressor(rd_geom);
		ic[2]->SetupDecompressor(rd_geom);
*/

		for(int i=0;i<nodesPerCluster;i++)
		{
			BSPArrayTreeNodePtr node = (BSPArrayTreeNodePtr)(physicalMemory + (PCN*nodesPerCluster*sizeof(BSPArrayTreeNode) + i*sizeof(BSPArrayTreeNode)));
			
			node->children = rd_geom->decodeInt();
			node->children2 = rd_geom->decodeInt();
			node->min.e[0] = rd_geom->decodeFloat();
			node->min.e[1] = rd_geom->decodeFloat();
			node->min.e[2] = rd_geom->decodeFloat();
			node->max.e[0] = rd_geom->decodeFloat();
			node->max.e[1] = rd_geom->decodeFloat();
			node->max.e[2] = rd_geom->decodeFloat();
		}

		rd_geom->done();
		delete rd_geom;
		/*
		ic[0]->FinishDecompressor();
		ic[1]->FinishDecompressor();
		ic[2]->FinishDecompressor();

		delete pq;
		delete ic[0];
		delete ic[1];
		delete ic[2];
		*/
	}
	long offset = (index << 3) & offsetMask;
	return (BSPArrayTreeNodePtr)(physicalMemory + (clusterTable[CN].PCN*nodesPerCluster*sizeof(BSPArrayTreeNode) + offset));
}

#include <math.h>
#include <time.h>
int BVH::loadClusterTable()
{
	fread(&nodesPerCluster, sizeof(unsigned int), 1, fp);
	fread(&numNodes, sizeof(unsigned int), 1, fp);
	fread(&numClusters, sizeof(unsigned int), 1, fp);
	fread(bb_min_f, sizeof(float), 3, fp);
	fread(bb_max_f, sizeof(float), 3, fp);

	clusterTable = new ClusterTableEntry[numClusters];

	for(int i=0;i<numClusters;i++)
	{
		long offset;
		fread(&offset, sizeof(long), 1, fp);
		clusterTable[i].PCN = -1;
		clusterTable[i].DCO = offset;
	}

	nodesPerClusterPower = log((double)nodesPerCluster)/log(2.0);
	int offsetPower = log((double)(nodesPerCluster*sizeof(BSPArrayTreeNode)))/log(2.0);
	assert(pow(2.0, (int)nodesPerClusterPower) == nodesPerCluster);
	assert(pow(2.0, offsetPower) == nodesPerCluster*sizeof(BSPArrayTreeNode));
	offsetMask = 0;
	for(int i=0;i<offsetPower;i++)
	{
		unsigned int bit = 1 << i;
		offsetMask |= bit;
	}

	OptionManager *opt = OptionManager::getSingletonPtr();
	maxNumPCN = (opt->getOptionAsInt("raytracing", "maxUsePhysicalMemory", 10))*1024*1024/(nodesPerCluster * sizeof(BSPArrayTreeNode));

	// allocate physical memory
	physicalMemory = new char[nodesPerCluster*maxNumPCN*sizeof(BSPArrayTreeNode)];

	for(int i=0;i<maxNumPCN;i++)
	{
		emptyList.push(i);
	}
	srand((unsigned int)time(NULL));
	return 1;
}
#endif