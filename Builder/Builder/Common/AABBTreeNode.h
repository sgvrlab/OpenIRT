#ifndef COMMON_AABBTREENODE_H
#define COMMON_AABBTREENODE_H

#include <vector>
#include "common.h"
#include "aabbox.hpp"

struct AABBTreeNode {
public:
  
	// builds a tree in top down fashion
	float construct(int depth, ModelInstance *object, TriangleIndexList &triIDs);	

	// destroys this subtree
	void destroy();
	void destroyAccumulate(TriangleIndexList &triIDs);
	
	/**
	 * Updates the bounding boxes in the tree without changing
	 * the tree's structure in the process.
	 **/
	float updateBB_indexed(ModelInstance *object, float &SARatioIncrease);	
	float update_indexed_threaded(ModelInstance *object, float &SARatioIncrease);

	/**
	 * Helper functions that encapsulate access to the compressed 
	 * information inside this node.
	 **/
	FORCEINLINE AABBTreeNode *getLeftChild() { return (AABBTreeNode *)((unsigned int)lChild & ~3); }
	FORCEINLINE AABBTreeNode *getRightChild() { return (AABBTreeNode *)((unsigned int)lChild & ~3) + 1; }
	FORCEINLINE int getTriID() { return leafTriID >> 2; }
	FORCEINLINE int isLeaf() { return (leafTriID & 3) == 3; }
	FORCEINLINE int getSplitAxis() { return leafTriID & 3; }

	/**
	 * Does some (weak) spatial ordering. The purpose is to find out which child
	 * is the 'near' and which the 'far' node bases on the ray's precomputed direction
	 * information. The split axis of this inner node is taken as the determining
	 * axis.
	 **/
	FORCEINLINE void storeOrderedChildren(Ray &ray, AABBTreeNode **nearNode, AABBTreeNode **farNode) {
		AABBTreeNode *child = getLeftChild();
		int axis = getSplitAxis();
		*farNode = child + ray.posneg[axis];
		*nearNode = child + (ray.posneg[axis] ^ 1);
	}

	FORCEINLINE void storeOrderedChildren(SIMDRay &rays, AABBTreeNode **nearNode, AABBTreeNode **farNode) {
		AABBTreeNode *child = getLeftChild();
		int axis = getSplitAxis();
		*farNode = child + rays.rayChildOffsets[axis];
		*nearNode = child + (rays.rayChildOffsets[axis] ^ 1);
	}
	
	// node bounding box
	aabbox _bbox;

	/**
	 * Compact representation of several members:
	 *  - lower 2 bits:
	 *     0-2: split axis
	 *       3: node is leaf
	 *  - upper 30 bits:
	 *     - for inner node: address to first child 
	 *       (second child is in consecutive location)
	 *     - for leaf node: triangle index (shifted left by two bits)
	 **/
	union {
		AABBTreeNode *lChild;	
		unsigned int leafTriID;
	};
	

	// TODO: make this float difference between bbMin/Max to test
	// for overlap?
	float saRatio;
};

#endif 