#ifndef COMMON_HIGH_LEVEL_TREE_H
#define COMMON_HIGH_LEVEL_TREE_H

#include <stack>
#include "common.h"
#include "BBox.h"

typedef struct SImpleNodeType {
	BBox bbox;
	ModelInstance* obj;
	struct SImpleNodeType* left;
	struct SImpleNodeType* right;
}SimpleNode;


/**
 * Bounding volume hierarchy class.
 * 
 */
class HighLevelTree {

public:
	ModelInstance	**pObjectList;	
	int				numObjects;
	SimpleNode*		root;

public:
	HighLevelTree(ModelInstance *objectList, unsigned int numObjects);
	~HighLevelTree();
	void buildTree();
	SimpleNode* buildBranch(ModelInstance** pObjectList, int obj_size, int axis);
	void intersectTest(Ray& ray, HitPointPtr hitPoint, SimpleNode* currentNode, float TraveledDist, bool* hasHit);
	bool isVisible(SimpleNode* currentNode, const Vector3 &lightPos, HitPointPtr shadowHitPoint, bool* hasHit);
private:
	void deleteTree(SimpleNode* node);
	BBox surround(const Vector3* b1, const Vector3* b2);
	void calcObjTransfBB();

	/************************************************************************/
	/* ´ö¼ö. Collision Detection                                            */
	/************************************************************************/
public :
	void collisionDetection( void ) ;
private:
	bool objectCollideTest( ModelInstance *mdl1 ,ModelInstance *mdl2 ) ;
private:
	bool nodeCollide( BSPArrayTreeNodePtr n1, BSPArrayTreeNodePtr n2, Matrix m1, Matrix m2 ) ;
};


#endif