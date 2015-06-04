/********************************************************************
	created:	2011/12/13
	file path:	d:\Projects\Redering\OpenIRT\include
	file base:	TReX
	file ext:	h
	author:		Tae-Joon Kim (tjkim.kaist@gmail.com)
	
	comment:	Frustum culling
*********************************************************************/

#pragma once

#include "Plane.h"
#include "Camera.h"
#include <vector>

namespace irt
{

template<class ModelType>
class FrustumCulling
{
protected:
	Camera *m_camera;
	ModelType *m_model;

	Plane m_plane[6];

	typedef std::vector<BVHNode> BVHList;
public:
	BVHList m_BVHList;
	FrustumCulling(void) {}
	virtual ~FrustumCulling(void) {}
	FrustumCulling(Camera *camera, ModelType *model, float x1, float y1, float x2, float y2, float zNear, float zFar)
	{
		init(camera, model, x1, y1, x2, y2, zNear, zFar);
	}

	void init(Camera *camera, ModelType *model, float x1, float y1, float x2, float y2, float zNear, float zFar)
	{
		m_camera = camera;
		m_model = model;

		Ray frustumRay[4];
		camera->getRayWithOrigin(frustumRay[0], x1, y1);
		camera->getRayWithOrigin(frustumRay[1], x2, y1);
		camera->getRayWithOrigin(frustumRay[2], x2, y2);
		camera->getRayWithOrigin(frustumRay[3], x1, y2);

		m_plane[0].set(frustumRay[0].origin(), frustumRay[0].origin()+frustumRay[0].direction(), frustumRay[0].origin()+frustumRay[1].direction());
		m_plane[1].set(frustumRay[1].origin(), frustumRay[1].origin()+frustumRay[1].direction(), frustumRay[1].origin()+frustumRay[2].direction());
		m_plane[2].set(frustumRay[2].origin(), frustumRay[2].origin()+frustumRay[2].direction(), frustumRay[2].origin()+frustumRay[3].direction());
		m_plane[3].set(frustumRay[3].origin(), frustumRay[3].origin()+frustumRay[3].direction(), frustumRay[3].origin()+frustumRay[0].direction());

		Vector3 n = camera->getCenter()-camera->getEye();
		n.makeUnitVector();
		m_plane[4].set(-n, dot(camera->getEye()+(n*zNear), -n));
		m_plane[5].set(n, dot(camera->getEye()+(n*zFar), n));
	}

	int frustumBoxIntersect(int numPlane, const Plane plane[], const Vector3 &min, const Vector3 &max)
	{
		int ret = 1;	// 0 : intersect, 1 : inside, 2 : outside, 3 : far
		Vector3 vmin, vmax; 

		for(int i=0;i<numPlane;i++) { 
			// X axis
			if(plane[i].n.e[0] > 0) {
				vmin.e[0] = min.e[0];
				vmax.e[0] = max.e[0];
			} else {
				vmin.e[0] = max.e[0];
				vmax.e[0] = min.e[0];
			}
			// Y axis
			if(plane[i].n.e[1] > 0) {
				vmin.e[1] = min.e[1];
				vmax.e[1] = max.e[1];
			} else {
				vmin.e[1] = max.e[1];
				vmax.e[1] = min.e[1];
			}
			// Z axis
			if(plane[i].n.e[2] > 0) {
				vmin.e[2] = min.e[2];
				vmax.e[2] = max.e[2];
			} else {
				vmin.e[2] = max.e[2];
				vmax.e[2] = min.e[2];
			}
			if(dot(plane[i].n, vmin) - plane[i].d > 0)
				return i == 5 ? 3 : 2;
			if(dot(plane[i].n, vmax) - plane[i].d >= 0)
				ret = 0;
		}
		return ret;
	}

	void cullBVH(int maxDepth)
	{
		m_BVHList.clear();

		ModelType *model = m_model;
		BVHNode *node = model->getBV(model->getRootIdx());
		BVHNode newNode = *node;
		newNode.left = 1 << 2;
		newNode.right = 2 << 2;

		printf("Original BVH = %d\n", (2 << maxDepth)*2-1);

		if(frustumBoxIntersect(6, m_plane, node->min, node->max) < 2)
		{
			m_BVHList.push_back(newNode);
			cullBVH(model, node, 0, 0, maxDepth);
			//printf("Culled BVH level 1 = %d\n", m_BVHList.size());
			restructure(model, 0);
			refit(model, 0);
			//int num = 0;
			//draw(model, 0, 0, num);
			//printf("Culled BVH level 2 = %d\n", num);
		}
	}

	void cullBVH(ModelType *model, BVHNode *node, int index, int depth, int maxDepth)
	{
		if(depth == maxDepth)
		{
			return;
		}

		BVHNode *leftNode = model->getBV(model->getLeftChildIdx(node));
		BVHNode *rightNode = model->getBV(model->getRightChildIdx(node));

		int leftIntersect = frustumBoxIntersect(6, m_plane, leftNode->min, leftNode->max);
		int rightIntersect = frustumBoxIntersect(6, m_plane, rightNode->min, rightNode->max);

		int size = (int)m_BVHList.size();
		m_BVHList[index].left = (size << 2) | 0x1;
		m_BVHList[index].right = ((size+1) << 2) | ((leftIntersect < 2) << 1) | (rightIntersect < 2);
		m_BVHList.push_back(*leftNode);
		m_BVHList.push_back(*rightNode);

		BVHNode &lNode = m_BVHList[size];
		BVHNode &rNode = m_BVHList[size+1];

		lNode.left = (lNode.left >> 2) << 2;
		rNode.left = (rNode.left >> 2) << 2;

		if(!model->isLeaf(leftNode) && leftIntersect < 2)
			cullBVH(model, leftNode, size, depth+1, maxDepth);
		if(!model->isLeaf(rightNode) && rightIntersect < 2)
			cullBVH(model, rightNode, size+1, depth+1, maxDepth);
	}

	void restructure(ModelType *model, int index)
	{
		BVHNode &node = m_BVHList[index];
		if(!(node.left & 0x1)) return;

		int leftState = (node.right & 0x2) >> 1;
		int rightState = node.right & 0x1;

		if(leftState) restructure(model, node.left >> 2);
		if(rightState) restructure(model, node.right >> 2);

		if(leftState == 0) node = m_BVHList[node.right >> 2];
		else if(rightState == 0) node = m_BVHList[node.left >> 2];
	}

	void refit(ModelType *model, int index)
	{
		BVHNode &node = m_BVHList[index];

		if(!(node.left & 0x1)) return;

		refit(model, node.left >> 2);
		refit(model, node.right >> 2);

		const BVHNode &leftNode = m_BVHList[node.left >> 2];
		const BVHNode &rightNode = m_BVHList[node.right >> 2];

		node.min.e[0] = leftNode.min.e[0] < rightNode.min.e[0] ? leftNode.min.e[0] : rightNode.min.e[0];
		node.min.e[1] = leftNode.min.e[1] < rightNode.min.e[1] ? leftNode.min.e[1] : rightNode.min.e[1];
		node.min.e[2] = leftNode.min.e[2] < rightNode.min.e[2] ? leftNode.min.e[2] : rightNode.min.e[2];
		node.max.e[0] = leftNode.max.e[0] > rightNode.max.e[0] ? leftNode.max.e[0] : rightNode.max.e[0];
		node.max.e[1] = leftNode.max.e[1] > rightNode.max.e[1] ? leftNode.max.e[1] : rightNode.max.e[1];
		node.max.e[2] = leftNode.max.e[2] > rightNode.max.e[2] ? leftNode.max.e[2] : rightNode.max.e[2];

		node.left = ((node.left >> 2) << 2) | (node.max - node.min).indexOfMaxComponent();
		node.right = ((node.right >> 2) << 2) | 0x1;	// finalized
	}

	void draw(ModelType *model, int index, int depth, int &num)
	{
		BVHNode &node = m_BVHList[index];

		if(!(node.right & 0x1)) return;

		extern void drawBB(const AABB &bb, bool fill);
		switch(depth % 3)
		{
		case 0: glColor3f(1.0f, 0.0f, 0.0f); break;
		case 1: glColor3f(0.0f, 1.0f, 0.0f); break;
		case 2: glColor3f(0.0f, 0.0f, 1.0f); break;
		}
		drawBB(AABB(node.min, node.max), false);
		num++;

		draw(model, node.left >> 2, depth+1, num);
		draw(model, node.right >> 2, depth+1, num);
	}
};

};