/********************************************************************
	created:	2010/02/08
	file path:	d:\Projects\Redering\OpenIRT\include
	file base:	BVHNode
	file ext:	h
	author:		Tae-Joon Kim (tjkim@tclab.kaist.ac.kr)
	
	comment:	BVHNode class for use in ray tracer
*********************************************************************/

#pragma once

#include "BV.h"

/**
* Condensed structure for a BVH node.
*
* left and right point to the child nodes for inner nodes. 
* For leaf nodes, left holds number of triangles which is 
* assigned to the leaf nodes. right points to the triangle list.
* Additionally, the first two low-order bits 
* of the pointer are used for determining the type of the node:
*  3 - Leaf
*  0 - Split by X plane
*  1 - Split by Y plane
*  2 - Split by Z plane
* This is possible because the structure should be aligned to
* 32bit boundaries in memory, so the respective bits would be
* zero anyway.
*
* min and max have x, y, and z extents of a bounding box
*/

namespace irt
{

class BVHNode {
public:
	unsigned int left;
	unsigned int right;
	Vector3 min;
	Vector3 max;
};

};