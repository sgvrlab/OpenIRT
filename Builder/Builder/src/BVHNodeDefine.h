#ifndef BVH_NODE_DEFINE_H
#define BVH_NODE_DEFINE_H

#include "Vector3.h"

#define BVHNODE_BYTES 32
typedef union BSPArrayTreeNode_t {
	struct { // inner node
		unsigned int children;
		unsigned int children2;
		Vector3 min;
		Vector3 max;
	};
	struct { // leaf node
		unsigned int indexCount;
		unsigned int indexOffset;
		Vector3 min;
		Vector3 max;
	};
	struct { // inner node with RLOD
		unsigned int children;
		unsigned int lodindex;
		Vector3 min;
		Vector3 max;
	};
	struct { // leaf node with RLOD
		unsigned int triIndex;
		unsigned int lodindex;
		Vector3 min;
		Vector3 max;
	};
} BSPArrayTreeNode, *BSPArrayTreeNodePtr;
#define BSPTREENODESIZE 32
#endif // BVH_NODE_DEFINE_H