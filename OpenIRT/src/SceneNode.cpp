#include <string>
#include <float.h>
#include "SceneNode.h"

using namespace irt;

SceneNode::SceneNode() : childs(0)
{
	clear();
}


SceneNode::SceneNode(const char *name, SceneNode *parent, Model *model, Matrix *matrix) : childs(0)
{
	clear();

	set(name, parent, model, matrix);
}

SceneNode::~SceneNode(void)
{
	clear();
}

void SceneNode::clear()
{
	if(hasChilds())
	{
		for(size_t i=0;i<childs->size();i++)
			delete childs->at(i);
		delete childs;
		childs = NULL;
	}

	parent = NULL;
	model = NULL;
	nodeBB.min.set(FLT_MAX, FLT_MAX, FLT_MAX);
	nodeBB.max.set(-FLT_MAX, -FLT_MAX, -FLT_MAX);
	name[0] = 0;

	matrix = identityMatrix();
}

void SceneNode::set(const char *name, SceneNode *parent, Model* model, Matrix *matrix)
{
	strcpy_s(this->name, 256, name);
	this->parent = parent;
	this->model = model;

	if(matrix)
	{
		this->matrix = *matrix;
	}

	if(model)
	{
		AABB bb = model->getModelBB();
		Matrix transMat = getTransformedMatrix();
		Matrix invTransMat = transMat;
		invTransMat.invert();

		model->setTransfMatrix(transMat);
		model->setInvTransfMatrix(invTransMat);

		if(transMat != identityMatrix())
		{
			model->updateTransformedBB(bb, transMat);
		}

		model->setName(name);
		updateBB(bb, true);
	}

}

SceneNode* SceneNode::addChild(const char *name, Model *model, Matrix *matrix)
{
	if(!hasChilds())
		childs = new std::vector<SceneNode*>;
	SceneNode *childNode = new SceneNode(name, this, model, matrix);
	childs->push_back(childNode);
	return childNode;
}

Matrix SceneNode::getTransformedMatrix()
{
	if(!parent)
		return matrix;

	return parent->getTransformedMatrix() * matrix;
}

void SceneNode::updateBB()
{
	if(hasChilds())
	{
		for(size_t i=0;i<childs->size();i++)
		{
			childs->at(i)->updateBB();
			updateBB(childs->at(i)->nodeBB, false);
		}
	}
}

void SceneNode::updateBB(const AABB &bb, bool updateParents)
{
	nodeBB.min.setX(min(nodeBB.min.x(), bb.min.x()));
	nodeBB.min.setY(min(nodeBB.min.y(), bb.min.y()));
	nodeBB.min.setZ(min(nodeBB.min.z(), bb.min.z()));
	nodeBB.max.setX(max(nodeBB.max.x(), bb.max.x()));
	nodeBB.max.setY(max(nodeBB.max.y(), bb.max.y()));
	nodeBB.max.setZ(max(nodeBB.max.z(), bb.max.z()));

	if(parent && updateParents)
		parent->updateBB(nodeBB, updateParents);
}

void SceneNode::updateBBWithVertex(AABB &bb, const Vector3 &vert)
{
	bb.min.setX(min(bb.min.x(), vert.x()));
	bb.min.setY(min(bb.min.y(), vert.y()));
	bb.min.setZ(min(bb.min.z(), vert.z()));
	bb.max.setX(max(bb.max.x(), vert.x()));
	bb.max.setY(max(bb.max.y(), vert.y()));
	bb.max.setZ(max(bb.max.z(), vert.z()));
}