/********************************************************************
	created:	2011/08/13
	file path:	d:\Projects\Redering\OpenIRT\include
	file base:	SceneNode
	file ext:	h
	author:		Tae-Joon Kim (tjkim.kaist@gmail.com)
	
	comment:	Node class of scene graph
*********************************************************************/

#pragma once

#include <vector>
#include "Model.h"
#include "Matrix.h"

namespace irt
{

class SceneNode
{
public:
	Matrix matrix;
	std::vector<SceneNode*> *childs;
	SceneNode *parent;
	char name[256];

	Model *model;
	AABB nodeBB;
	
	SceneNode();
	SceneNode(const char *name, SceneNode *parent, Model* model = NULL, Matrix *matrix = NULL);
	~SceneNode(void);

	void clear();
	void set(const char *name, SceneNode *parent = NULL, Model* model = NULL, Matrix *matrix = NULL);
	bool hasChilds() {return childs != NULL;}
	SceneNode* addChild(const char *name, Model *model, Matrix *matrix = NULL);

	Matrix getTransformedMatrix();
	void updateBB();
	void updateBB(const AABB &bb, bool updateParents = false);
	static void updateBBWithVertex(AABB &bb, const Vector3 &vert);
};

};