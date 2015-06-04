#include "StdAfx.h"
#include <time.h>
#include "BVH.h"
//#include "stopwatch.hpp"
#include "BBox.h"

CLODMetric LODMetric;
float g_MaxAllowModifier;
int g_NumTraversed = 0;
int g_NumIntersected = 0;
int g_tempMax = 0;
extern Stopwatch **TTriangle;

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

int numFrame = 0;
int g_numCrackedTri = 0;

int BVH::RayTriIntersect(const Ray &ray, ModelInstance *object, BSPArrayTreeNodePtr node, HitPointPtr hitPoint, float tmax)  
{
	float point[2];
	float vdot, vdot2;
	float alpha, beta;
	float t, u0, v0, u1, v1, u2, v2;

	if(object->useRACM)
	{
#ifdef _USE_RACM
	unsigned int triID = GETIDXOFFSET(node);//*MAKEIDX_PTR(object->indexlist,GETIDXOFFSET(node));
	const COutTriangle &tri = ((CCompressedMesh*)(object->pRACM))->GetTri(triID);

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
	const CGeomVertex &tri_p0 = object->pRACM->GetVertex((int)tri.m_c[0].m_v);
	const CGeomVertex &tri_p1 = object->pRACM->GetVertex((int)tri.m_c[1].m_v);
	const CGeomVertex &tri_p2 = object->pRACM->GetVertex((int)tri.m_c[2].m_v);

	float p0_1 = tri_p0[tri.i1], p0_2 = tri_p0[tri.i2]; 
	u0 = point[0] - p0_1; 
	v0 = point[1] - p0_2; 
	u1 = tri_p1[tri.i1] - p0_1; 
	v1 = tri_p1[tri.i2] - p0_2; 
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
	// below code has BUG!!! if we use tri material info, then fix it.
	#ifdef _USE_TRI_MATERIALS
	//hitPoint->m = tri.material;
	hitPoint->m = 0;
	#else
	hitPoint->m = 0;
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
	hitPoint->n.e[0] = tri.n.x;
	hitPoint->n.e[1] = tri.n.y;
	hitPoint->n.e[2] = tri.n.z;
	#endif

	if (vdot > 0.0f)
		hitPoint->n *= -1.0f;

	#ifdef _USE_TEXTURING
	// interpolate tex coords..
	hitPoint->uv = tri.uv[0] + hitPoint->alpha * tri.uv[1] + hitPoint->beta * tri.uv[2];
	#endif

	// hitpoint:
	hitPoint->x = ray.pointAtParameter(t);
	#endif
	}
	else
	{
	unsigned int triID = GETIDXOFFSET(node);//*MAKEIDX_PTR(object->indexlist,GETIDXOFFSET(node));
//	if(triID > numFrame * 10) return false;
//	unsigned int triID = GETIDXOFFSET(node);

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
	const Vector3 &tri_p1 = GETVERTEX(object,tri.p[1]); 
	const Vector3 &tri_p2 = GETVERTEX(object,tri.p[2]); 
	float p0_1 = tri_p0.e[tri.i1], p0_2 = tri_p0.e[tri.i2]; 
	u0 = point[0] - p0_1; 
	v0 = point[1] - p0_2; 
	u1 = tri_p1[tri.i1] - p0_1; 
	v1 = tri_p1[tri.i2] - p0_2; 
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
	//hitPoint->m = defaultMaterial;
	hitPoint->m = 0;
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
	}
	return true;	
}

FORCEINLINE bool BVH::RayBoxIntersect(const Ray& r, Vector3 &min, Vector3 &max, float &interval_min, float &interval_max)  {
	interval_min = -FLT_MAX;
	interval_max = FLT_MAX;

	Vector3 box[2];

	box[0] = min;
	box[1] = max;

	float t0 = (box[r.posneg[3]].e[0] - r.data[0].e[0]) * r.data[2].e[0];
	float t1 = (box[r.posneg[0]].e[0] - r.data[0].e[0]) * r.data[2].e[0];

	interval_min = max(t0, interval_min);
	interval_max = min(t1, interval_max);

	if (interval_min > interval_max) return false;

	t0 = (box[r.posneg[4]].e[1] - r.data[0].e[1]) * r.data[2].e[1];
	t1 = (box[r.posneg[1]].e[1] - r.data[0].e[1]) * r.data[2].e[1];

	interval_min = max(t0, interval_min);
	interval_max = min(t1, interval_max);

	if (interval_min > interval_max) return false;

	t0 = (box[r.posneg[5]].e[2] - r.data[0].e[2]) * r.data[2].e[2];
	t1 = (box[r.posneg[2]].e[2] - r.data[0].e[2]) * r.data[2].e[2];

	interval_min = max(t0, interval_min);
	interval_max = min(t1, interval_max);

	return (interval_min <= interval_max);
}

void testTraverse(ModelInstance *object, unsigned int cur, unsigned int parent, int *numTraverse, int depth, int *maxDepth, float *avgDepth, int *numLeaf)
{
	*numTraverse = *numTraverse + 1;
	BSPArrayTreeNodePtr curNode = GETNODE(object->tree, cur);
	BSPArrayTreeNodePtr parentNode = NULL;
	if(parent != -1)
		parentNode = GETNODE(object->tree, parent);

	if(*maxDepth < depth) *maxDepth = depth;
	if(ISLEAF(curNode)) 
	{
		*avgDepth = ((*avgDepth)*(*numLeaf) + depth)/((*numLeaf)+1);
		*numLeaf = *numLeaf + 1;
		return ;
	}

	unsigned int lChild = GETLEFTCHILD(curNode);
	unsigned int rChild = GETRIGHTCHILD(curNode);

	testTraverse(object, lChild, cur, numTraverse, depth+1, maxDepth, avgDepth, numLeaf);
	testTraverse(object, rChild, cur, numTraverse, depth+1, maxDepth, avgDepth, numLeaf);
}

int BVH::RayTreeIntersectRecursive(Ray &ray, HitPointPtr hit, unsigned int index, float traveledDist)
{
	ModelInstance *object = &objectList[0];

	//ray.transform(&object->translate_world);

	int numTraverse = 0;
	int maxDepth = 0;
	float avgDepth = 0;
	int numLeaf = 0;
//	testTraverse(object, 0, &numTraverse, 0, &maxDepth, &avgDepth, &numLeaf);

	BSPArrayTreeNodePtr currentNode = GETNODE(object->tree, index);
	float min, max;
	bool hasHit = false;
	//hit->t = FLT_MAX;

	bool hitTest = RayBoxIntersect(ray, currentNode->min, currentNode->max, min, max);

	if ( hitTest /*&& min < hit->t && max > 0.0f*/)
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

int g_NumTraversedNodes = 0;

int BVH::RayTreeIntersect(ModelInstance* object, Ray &ray, HitPointPtr hit, float traveledDist)
{
	/*
	int numTraverse = 0;
	int maxDepth = 0;
	float avgDepth = 0;
	int numLeaf = 0;
	testTraverse(object, 0, -1, &numTraverse, 0, &maxDepth, &avgDepth, &numLeaf);
	*/

	/*
	// Test decompression time (1000 clusters)
	Stopwatch tDecompression("Decompression time");
	for(int i=0;i<1000;i++)
	{
		GETNODE(object->tree, (4096*i) << 2);
	}
	cout << tDecompression << endl;
	exit(1);
	*/

	//static StackElem *stack = (StackElem *)_aligned_malloc(MAXBVHDEPTH * sizeof(StackElem), 16);		
	StackElem stack[MAXBVHDEPTH];

	int stackPtr;
	BSPArrayTreeNodePtr currentNode, parentNode;
	int CurrentDepth = 0;
	float min, max;	
	bool hasHit = false, HasHitRootBV = false;

	stack[0].index = 0;
	stackPtr = 1;

	currentNode = GETNODE(object->tree, 0);

	Ray temp = ray;

	// if we have animation
	if (object->sizeTransformMatList) {	
		// Note!!!
		// Ray -> must do the transeform using this object's transformation matrix	
		ray.transform(&object->invTransformMatList[ object->indexCurrentTransform ]);	
	}

	// Note!!!
	// We should calculate error bound depend on model size. Because it prevent secondary ray's reflection to only next tri.
	// We will anneal this later.
	Vector3 diff_vec = object->bb[1] - object->bb[0];
	float error_bound = 0.05f * diff_vec.maxAbsComponent();

	// traverse BVH tree:
	while (1) {
		// is current node intersected and also closer than previous hit?
		bool hitTest = RayBoxIntersect(ray, currentNode->min, currentNode->max, min, max);

		g_NumTraversedNodes++;

		if ( hitTest && min < hit->t && max > error_bound) {

			// is inner node?
			if (!ISLEAF(currentNode)) {
				// Store ordered children
				BSPArrayTreeNodePtr lChild = GETNODE(object->tree, GETLEFTCHILD(currentNode));
				BSPArrayTreeNodePtr rChild = GETNODE(object->tree, GETRIGHTCHILD(currentNode));				

				int axis = AXIS(currentNode);

				unsigned int farChild = ray.posneg[axis] ? GETRIGHTCHILD(currentNode) : GETLEFTCHILD(currentNode);
				unsigned int nearChild = ray.posneg[axis] ? GETLEFTCHILD(currentNode) : GETRIGHTCHILD(currentNode);

				stack[stackPtr].index = farChild;
				currentNode = GETNODE(object->tree, nearChild);

				stackPtr++;
				continue;
			}
			else {
				// is leaf node:
				// intersect with current node's members
				#ifdef _USE_OPENMP
				int threadNum = omp_get_thread_num();
				#else
				int threadNum = 0;
				#endif
				TTriangle[threadNum]->Start();
				int tempHit = RayTriIntersect(ray, object, currentNode, hit, min(max, hit->t));
				TTriangle[threadNum]->Stop();

				hasHit = tempHit || hasHit;

			}
		}
		if (--stackPtr == 0) break;
		
		// fetch next node from stack
		currentNode = GETNODE(object->tree, stack[stackPtr].index);
	}

	// Note !!!!!
	// 1. Ray -> restore
	// 2. Hitpoint -> Must do the inverse transform

	ray = temp;
	if (object->sizeTransformMatList && hasHit) {
		hit->x = object->transformMatList[ object->indexCurrentTransform ] * hit->x;
		int idx = ray.direction().indexOfMaxComponent();

		// calculate t value to reach the ray's target
		hit->t = (hit->x.e[idx] - ray.data[0].e[idx]) / ray.direction().e[idx];
		hit->t -= BSP_EPSILON;

		// NOTE!
		// Assume.
		// The normal vector of a hit point transforms the direction of the world coordinate using only rotation elements
		hit->n = transformVec(object->transformMatList[ object->indexCurrentTransform ],  hit->n);
	}	

	return hasHit;
}

//int BVH::isVisible(const ModelInstance* object, Ray& ray, const Vector3 &origin, const Vector3& target)
//{
//	static int NumInter = 0;
//	static StackElem *stack = (StackElem *)_aligned_malloc(MAXBVHDEPTH * sizeof(StackElem), 16);		
//
//	int stackPtr;
//	BSPArrayTreeNodePtr currentNode, parentNode;
//	int CurrentDepth = 0;
//	float min, max, closest_t = FLT_MAX;	
//	bool hasHit = false, HasHitRootBV = false;
//
//	stack[0].index = 0;
//	stackPtr = 1;
//
//	currentNode = GETNODE(object->tree, 0);
//
//	Ray temp = ray;
//
//	// if we have animation
//	if (object->sizeTransformMatList) {	
//		// Note!!!
//		// Ray -> must do the transeform using this object's transformation matrix	
//		ray.transform(&object->invTransformMatList[ object->indexCurrentTransform ]);	
//	}
//
//
//	// traverse BVH tree:
//	while (1) {
//		// is current node intersected and also closer than previous hit?
//		bool hitTest = RayBoxIntersect(ray, currentNode->min, currentNode->max, min, max);
//
//		g_NumTraversedNodes++;
//
//		if ( hitTest && min < closest_t && max > 0.0f) {
//
//			// is inner node?
//			if (!ISLEAF(currentNode)) {
//				// Store ordered children
//				BSPArrayTreeNodePtr lChild = GETNODE(object->tree, GETLEFTCHILD(currentNode));
//				BSPArrayTreeNodePtr rChild = GETNODE(object->tree, GETRIGHTCHILD(currentNode));				
//
//				int axis = AXIS(currentNode);
//
//				unsigned int farChild = ray.posneg[axis] ? GETRIGHTCHILD(currentNode) : GETLEFTCHILD(currentNode);
//				unsigned int nearChild = ray.posneg[axis] ? GETLEFTCHILD(currentNode) : GETRIGHTCHILD(currentNode);
//
//				stack[stackPtr].index = farChild;
//				currentNode = GETNODE(object->tree, nearChild);
//
//				stackPtr++;
//				continue;
//			}
//			else {
//				// is leaf node:
//				// intersect with current node's members
//				int tempHit = RayTriIntersect(ray, object, currentNode, hit, min(max, hit->t));
//
//				hasHit = tempHit || hasHit;
//
//			}
//		}
//		if (--stackPtr == 0) break;
//		
//		// fetch next node from stack
//		currentNode = GETNODE(object->tree, stack[stackPtr].index);
//	}
//
//	// Note !!!!!
//	// Ray -> must do the inverse transform (optional)
//	// Hitpoint -> Must do the inverse transform
//
//	if (object->sizeTransformMatList && hasHit) {
//
//		hit->x = object->transformMatList[ object->indexCurrentTransform ] * hit->x;
//
//		int idx = ray.direction().indexOfMaxComponent();
//		// calculate t value to reach the ray's target
//		hit->t = (hit->x.e[idx] - ray.data[0].e[idx]) / ray.direction().e[idx];
//		hit->t -= BSP_EPSILON;
//
//		hit->n = object->transformMatList[ object->indexCurrentTransform ] * hit->n;
//	}	
//	
//	ray = temp;
//
//	return hasHit;
//}

void BVH::initialize(char *filepath)
{
}
void BVH::finalize()
{
}

/************************************************************************/
/* ´ö¼ö                                                                 */
/************************************************************************/
bool BVH::isOverlab( BVH* target ){
	BSPArrayTreeNodePtr currentNode, targetNode ;
	BBox *box1, *box2, *box3, *box4 ;
	bool result ;

	if ( target == NULL )
		return false ;

	currentNode = GETNODE( objectList->tree, 0 ) ;			// read root of this
	targetNode	= GETNODE( target->objectList->tree, 0 ) ;	// read root of target

	box1 = new BBox( GETNODE( objectList->tree, GETLEFTCHILD(currentNode) )->min , GETNODE( objectList->tree, GETLEFTCHILD(targetNode) )->max ) ;
	box2 = new BBox( GETNODE( objectList->tree, GETRIGHTCHILD(currentNode) )->min , GETNODE( objectList->tree, GETRIGHTCHILD(targetNode) )->max ) ;
	*box1 += *box2 ;

	box3 = new BBox( GETNODE( target->objectList->tree, GETLEFTCHILD(currentNode) )->min , GETNODE( target->objectList->tree, GETLEFTCHILD(targetNode) )->max ) ;
	box4 = new BBox( GETNODE( target->objectList->tree, GETRIGHTCHILD(currentNode) )->min , GETNODE( target->objectList->tree, GETRIGHTCHILD(targetNode) )->max ) ;
	(*box3) += (*box4) ;

	result = box1->overlaps( *box3 ) ;

	delete box1 ;
	delete box2 ;
	delete box3 ;
	delete box4 ;

	return result ;
}

#endif