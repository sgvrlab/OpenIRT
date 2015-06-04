#ifndef BVH_NODE_DEFINEQ_H
#define BVH_NODE_DEFINEQ_H

#include "Vector3.h"

#define BVHNODEQ_BYTES 32
typedef union BSPArrayTreeNodeQ_t {
	struct { // inner node
		unsigned int children;
		unsigned int children2;
		int min[3];
		int max[3];
	};
	struct { // leaf node
		unsigned int indexCount;
		unsigned int indexOffset;
		int min[3];
		int max[3];
	};
} BSPArrayTreeNodeQ, *BSPArrayTreeNodeQPtr;
#define BSPTREENODEQSIZE 32
#endif // BVH_NODE_DEFINE_H