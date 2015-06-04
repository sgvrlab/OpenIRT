#include "stdafx.h"
#include "HighLevelTree.h"
#include "qsplit.h"
#include "Tri_Tri_intersect.h"

HighLevelTree::HighLevelTree(ModelInstance *objectList, unsigned int numObjects) 
{
	pObjectList = new ModelInstance*[numObjects];

	for (unsigned int i = 0; i < numObjects; ++i)
		pObjectList[i] = &objectList[i];

	this->numObjects = numObjects;
}

HighLevelTree::~HighLevelTree() 
{	
	if (pObjectList)
		delete [] pObjectList;

	deleteTree(root);
}

void HighLevelTree::deleteTree(SimpleNode* node)
{
	if (node->left)
		deleteTree(node->left);
	if (node->right)
		deleteTree(node->right);

	delete node;
}

SimpleNode* HighLevelTree::buildBranch(ModelInstance** pObjectList, int obj_size, int axis) 
{
	if (obj_size == 0)
		return NULL;

	SimpleNode* newNode = new SimpleNode();

	// leaf node...
	if (obj_size == 1) {	
		newNode->obj = pObjectList[0];
		newNode->left = NULL;
		newNode->right = NULL;
		newNode->bbox.setValue(pObjectList[0]->transformedBB);
		return newNode;
	}
	if (obj_size == 2) {
		newNode->obj = NULL;
		newNode->bbox = surround(pObjectList[0]->transformedBB, pObjectList[1]->transformedBB);

		newNode->left = new SimpleNode();
		newNode->left->obj = pObjectList[0];
		newNode->left->bbox.setValue(pObjectList[0]->transformedBB);
		newNode->left->left = NULL;
		newNode->left->right = NULL;

		newNode->right = new SimpleNode();
		newNode->right->obj = pObjectList[1];
		newNode->right->bbox.setValue(pObjectList[1]->transformedBB);
		newNode->right->left = NULL;
		newNode->right->right = NULL;
		return newNode;
	}

	// internal node...
	// find the midpoint of the bounding box to use as a qsplit pivot
	newNode->obj = NULL;
	newNode->bbox.setValue(pObjectList[0]->transformedBB);
	for (int i = 1; i < obj_size; ++i)
		newNode->bbox = surround(newNode->bbox.pp, pObjectList[i]->transformedBB);

	Vector3 pivot = (newNode->bbox.getmax() + newNode->bbox.getmin()) / 2.0f;

	// now split according to correct axis
	int mid_point = qsplit(pObjectList, obj_size, pivot[axis], axis);

	// create a new bounding volume
	newNode->left = buildBranch(pObjectList, mid_point, (axis+1)%3);
	newNode->right = buildBranch(&pObjectList[mid_point], obj_size - mid_point, (axis+1)%3);

	return newNode;
}

//
// we calculate the bounding box of objects using object's transformation info.
void HighLevelTree::calcObjTransfBB()
{
	// Note!
	// We must all check 8 cases because the box can rotate any direction.
	ModelInstance* model;
	for (int i = 0; i < numObjects; ++i) {
		model = pObjectList[i];

		if (model->sizeTransformMatList == 0) {
			model->transformedBB[0] = model->bb[0];
			model->transformedBB[1] = model->bb[1];
			continue;
		}

		model->indexCurrentTransform++;

		if (model->indexCurrentTransform < model->sizeTransformMatList) {
			Vector3 sideVertex[8];
			Vector3 a = model->bb[0];
			Vector3 b = model->bb[1];

			sideVertex[0] = Vector3( a.x(), a.y(), a.z() );
			sideVertex[1] = Vector3( a.x(), a.y(), b.z() );
			sideVertex[2] = Vector3( a.x(), b.y(), a.z() );
			sideVertex[3] = Vector3( a.x(), b.y(), b.z() );
			sideVertex[4] = Vector3( b.x(), a.y(), a.z() );
			sideVertex[5] = Vector3( b.x(), a.y(), b.z() );
			sideVertex[6] = Vector3( b.x(), b.y(), a.x() );
			sideVertex[7] = Vector3( b.x(), b.y(), b.z() );

			model->transformedBB[0] = Vector3(FLT_MAX, FLT_MAX, FLT_MAX);
			model->transformedBB[1] = Vector3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
	
			Vector3 tx;
			for (int k = 0; k < 8; ++k) {
				tx = (model->transformMatList[ model->indexCurrentTransform] ) * sideVertex[k];

				if (model->transformedBB[0].x() > tx.x())
					model->transformedBB[0].setX( tx.x());
				if (model->transformedBB[0].y() > tx.y())
					model->transformedBB[0].setY( tx.y());
				if (model->transformedBB[0].z() > tx.z())
					model->transformedBB[0].setZ( tx.z());

				if (model->transformedBB[1].x() < tx.x())
					model->transformedBB[1].setX( tx.x());
				if (model->transformedBB[1].y() < tx.y())
					model->transformedBB[1].setY( tx.y());
				if (model->transformedBB[1].z() < tx.z())
					model->transformedBB[1].setZ( tx.z());

			}

			//model->transformedBB[0] = (model->transformMatList[model->indexCurrentTransform]) * model->bb[0];
			//model->transformedBB[1] = (model->transformMatList[model->indexCurrentTransform]) * model->bb[1];

			//if (model->transformedBB[0].x() > model->transformedBB[1].x())
			//	swap(model->transformedBB[0].e[0], model->transformedBB[1].e[0]);
			//if (model->transformedBB[0].y() > model->transformedBB[1].y())
			//	swap(model->transformedBB[0].e[1], model->transformedBB[1].e[1]);
			//if (model->transformedBB[0].z() > model->transformedBB[1].z())
			//	swap(model->transformedBB[0].e[2], model->transformedBB[1].e[2]);

		}
		else
			model->indexCurrentTransform = model->sizeTransformMatList - 1;
	}
}


void HighLevelTree::buildTree() 
{	
	calcObjTransfBB();

	// make the dummy root
	root = new SimpleNode();

	// find the midpoint of the bounding box to use as a qsplit pivot
	root->bbox.setValue(pObjectList[0]->transformedBB);

	if (numObjects == 1) {
		root->obj = pObjectList[0];
		root->left = NULL;
		root->right = NULL;
		return;
	}

	for (int i = 1; i < numObjects; ++i) {
		root->bbox = surround(root->bbox.pp, pObjectList[i]->transformedBB);
	}

	Vector3 pivot = (root->bbox.getmax() + root->bbox.getmin()) / 2.0f;

	int mid_point = qsplit(pObjectList, numObjects, pivot.x(), 0);

	// create a new bounding Volume
	root->obj = NULL;
	root->left = buildBranch(pObjectList, mid_point, 1);
	root->right = buildBranch(&pObjectList[mid_point], numObjects - mid_point, 1);
}

BBox HighLevelTree::surround(const Vector3* b1, const Vector3* b2) 
{
	return BBox (
		Vector3(b1[0].x() < b2[0].x() ? b1[0].x() : b2[0].x(),
				b1[0].y() < b2[0].y() ? b1[0].y() : b2[0].y(),
				b1[0].z() < b2[0].z() ? b1[0].z() : b2[0].z() ), 
				Vector3(b1[1].x() > b2[1].x() ? b1[1].x() : b2[1].x(),
						b1[1].y() > b2[1].y() ? b1[1].y() : b2[1].y(),
						b1[1].z() > b2[1].z() ? b1[1].z() : b2[1].z() )); 
}

void HighLevelTree::intersectTest(Ray& ray, HitPointPtr hitPoint, SimpleNode* currentNode, float TraveledDist, bool* hasHit)
{	
	float result_min, result_max;
	bool ishit = currentNode->bbox.rayIntersect(ray, &result_min, &result_max);
	
	if (ishit) {

		// type1. inner node
		if (currentNode->obj == NULL) {
			if (currentNode->left != NULL) 
				intersectTest(ray, hitPoint, currentNode->left, TraveledDist, hasHit);
			if (currentNode->right != NULL)
				intersectTest(ray, hitPoint, currentNode->right, TraveledDist, hasHit);
		}

		// type2. leaf node
		else {
			Hitpoint tempHit;
			tempHit.t = FLT_MAX;

			bool isHit = false;
			
			isHit = currentNode->obj->bvh->RayTreeIntersect(currentNode->obj, ray, &tempHit, TraveledDist);
			
			if (isHit && (tempHit.t < hitPoint->t)) {
				*hitPoint = tempHit;
				*hasHit = true;
			}
		}
	}
}

bool HighLevelTree::isVisible(SimpleNode* currentNode, const Vector3 &lightPos, HitPointPtr shadowHitPoint, bool* hasHit)
{
	Vector3 dir = lightPos - shadowHitPoint->x;
	dir.makeUnitVector();
	Ray ray(shadowHitPoint->x, dir);

	int idx = dir.indexOfMaxComponent();
	float target_t = (lightPos.e[idx] - shadowHitPoint->x.e[idx]) / dir.e[idx];
	target_t -= BSP_EPSILON;

	bool isvisible = true;
	float TraveledDist = 0.0f;

	intersectTest(ray, shadowHitPoint, currentNode, TraveledDist, hasHit);

	if ((*hasHit) && (shadowHitPoint->t < target_t))
		isvisible = false;

	return isvisible;
}

/************************************************************************/
/* 덕수. Collision Detection                                            */
/************************************************************************/
void HighLevelTree::collisionDetection( void ){
	BBox *tempBox1, *tempBox2 ;

	tempBox1 = new BBox() ;
	tempBox2 = new BBox() ;

	for ( int i = 0 ; i < numObjects ; i++ ){
		for ( int j = i+1 ; j < numObjects ; j++ ){

			tempBox1->setValue(pObjectList[i]->transformedBB) ;
			tempBox2->setValue(pObjectList[j]->transformedBB) ;

			// colliding objects pair detect
			if ( tempBox1->overlaps( *tempBox2 ) ){
				cout << "collide [ " << i << " , " << j << " ]" << endl ;
				// Processing collision detection between collied objects
				objectCollideTest( pObjectList[i], pObjectList[j] ) ;
			}
		}
	}

	// check collision for each pair

	delete tempBox1 ;
	delete tempBox2 ;

}

using namespace std ;
bool HighLevelTree::objectCollideTest( ModelInstance *mdl1 ,ModelInstance *mdl2 ){

	stack<BSPArrayTreeNodePtr> checkStack ;
	BSPArrayTreeNodePtr node1, node2 ;

	unsigned int tri1_ID, tri2_ID ;
	Vector3 v0, v1, v2, u0, u1, u2 ;
	float a0[3], a1[3], a2[3], b0[3], b1[3], b2[3] ;

	Matrix matMdl1( mdl1->transformMatList[mdl1->indexCurrentTransform] ) ;
	Matrix matMdl2( mdl2->transformMatList[mdl2->indexCurrentTransform] ) ;

	// push root node
	checkStack.push( GETNODE( mdl1->tree, 0 ) ) ;
	checkStack.push( GETNODE( mdl2->tree, 0 ) ) ;

	// counter for test
	unsigned int count = 0, culling = 0, trueTriTest = 0 ;

	while ( true ) {
		// pop a set of node to test
		if ( checkStack.size() == 0 )
			break ;

		// The order of pop and push is important
		// pop order and push order must have reverse order
		node2 = checkStack.top() ;
		checkStack.pop();
		node1 = checkStack.top() ;
		checkStack.pop();

		if ( nodeCollide( node1, node2, matMdl1, matMdl2 ) ){

			if ( ISLEAF(node1) && ISLEAF(node2 ) ){

				tri1_ID = *MAKEIDX_PTR(mdl1->indexlist,GETIDXOFFSET(node1));
				tri2_ID = *MAKEIDX_PTR(mdl2->indexlist,GETIDXOFFSET(node2));

				// 변환 행렬 적용

				if(!mdl1->useRACM)
				{
					Triangle tri1 = GETTRI(mdl1, tri1_ID);
					v0 = matMdl1*GETVERTEX(mdl1,tri1.p[0]);
					v1 = matMdl1*GETVERTEX(mdl1,tri1.p[1]);
					v2 = matMdl1*GETVERTEX(mdl1,tri1.p[2]);
				}
				else
				{
					#ifdef _USE_RACM
					const COutTriangle &tri1 = mdl1->pRACM->GetTri(tri1_ID);
					const CGeomVertex &tempV0 = mdl1->pRACM->GetVertex(tri1.m_c[0].m_v);
					const CGeomVertex &tempV1 = mdl1->pRACM->GetVertex(tri1.m_c[1].m_v);
					const CGeomVertex &tempV2 = mdl1->pRACM->GetVertex(tri1.m_c[2].m_v);
					v0 = matMdl1*_Vector4(Vector3(tempV0.x, tempV0.y, tempV0.z));
					v1 = matMdl1*_Vector4(Vector3(tempV1.x, tempV1.y, tempV1.z));
					v2 = matMdl1*_Vector4(Vector3(tempV2.x, tempV2.y, tempV2.z));
					#endif
				}

				if(!mdl2->useRACM)
				{
					Triangle tri2 = GETTRI(mdl2, tri2_ID);
					u0 = matMdl2*GETVERTEX(mdl2,tri2.p[0]);
					u1 = matMdl2*GETVERTEX(mdl2,tri2.p[1]);
					u2 = matMdl2*GETVERTEX(mdl2,tri2.p[2]);
				}
				else
				{
					#ifdef _USE_RACM
					const COutTriangle &tri2 = mdl2->pRACM->GetTri(tri2_ID);
					const CGeomVertex &tempU0 = mdl2->pRACM->GetVertex(tri2.m_c[0].m_v);
					const CGeomVertex &tempU1 = mdl2->pRACM->GetVertex(tri2.m_c[1].m_v);
					const CGeomVertex &tempU2 = mdl2->pRACM->GetVertex(tri2.m_c[2].m_v);
					u0 = matMdl2*_Vector4(Vector3(tempU0.x, tempU0.y, tempU0.z));
					u1 = matMdl2*_Vector4(Vector3(tempU1.x, tempU1.y, tempU1.z));
					u2 = matMdl2*_Vector4(Vector3(tempU2.x, tempU2.y, tempU2.z));
					#endif
				}


				// float [3] type으로 저장
				a0[0] = v0.e[0] ; a0[1] = v0.e[1] ; a0[2] = v0.e[2] ;
				a1[0] = v1.e[0] ; a1[1] = v1.e[1] ; a0[2] = v1.e[2] ;
				a2[0] = v2.e[0] ; a2[1] = v2.e[1] ; a2[2] = v2.e[2] ;

				b0[0] = u0.e[0] ; b0[1] = u0.e[1] ; b0[2] = u0.e[2] ;
				b1[0] = u1.e[0] ; b1[1] = u1.e[1] ; b0[2] = u1.e[2] ;
				b2[0] = u2.e[0] ; b2[1] = u2.e[1] ; b2[2] = u2.e[2] ;

				// Triangle Triangle Intersection Test
				if ( NoDivTriTriIsect(a0, a1, a2, b0, b1, b2) )
					trueTriTest++ ;

				count++ ;

			} else if ( ISLEAF(node1) ){

				checkStack.push( node1 ) ;
				checkStack.push( GETNODE(mdl2->tree, GETLEFTCHILD(node2)) ) ;

				checkStack.push( node1 ) ;
				checkStack.push( GETNODE(mdl2->tree, GETRIGHTCHILD(node2)) ) ;

			} else if ( ISLEAF(node2) ){

				checkStack.push( GETNODE(mdl1->tree, GETLEFTCHILD(node1)) ) ;
				checkStack.push( node2 ) ;

				checkStack.push( GETNODE(mdl1->tree, GETRIGHTCHILD(node1)) ) ;
				checkStack.push( node2 ) ;

			} else {
				checkStack.push( GETNODE(mdl1->tree, GETLEFTCHILD(node1)) ) ;
				checkStack.push( GETNODE(mdl2->tree, GETLEFTCHILD(node2)) ) ;

				checkStack.push( GETNODE(mdl1->tree, GETLEFTCHILD(node1)) ) ;
				checkStack.push( GETNODE(mdl2->tree, GETRIGHTCHILD(node2)) ) ;

				checkStack.push( GETNODE(mdl1->tree, GETRIGHTCHILD(node1)) ) ;
				checkStack.push( GETNODE(mdl2->tree, GETLEFTCHILD(node2)) ) ;

				checkStack.push( GETNODE(mdl1->tree, GETRIGHTCHILD(node1) ) ) ;
				checkStack.push( GETNODE(mdl2->tree, GETRIGHTCHILD(node2) ) ) ;
			}
		} else {
			// count culling
			culling++ ;
		}
	}

	cout << "***********************" << endl ;
	cout << "Tri_Tri Test " << count << endl ;
	cout << "True Tri_Tri Test " << trueTriTest << endl ;
	cout << "Culling " << culling << endl ;
	cout << "***********************" << endl ;

	if ( trueTriTest == 0 )
		return false ;

	FILE *fp ;
	fp = fopen("collide_time.txt","a") ;
	fprintf(fp,"%d ", mdl1->indexCurrentTransform ) ;
	fclose(fp) ;

	return true ;
}

bool HighLevelTree::nodeCollide( BSPArrayTreeNodePtr n1, BSPArrayTreeNodePtr n2, Matrix m1, Matrix m2 ){

	BBox tempBox1( m1*(n1->max), m1*(n1->min), 1 ) ;
	BBox tempBox2( m2*(n2->max), m2*(n2->min), 1 ) ;

	if ( tempBox1.overlaps( tempBox2 ) )
		return true ;

	return false ;
}