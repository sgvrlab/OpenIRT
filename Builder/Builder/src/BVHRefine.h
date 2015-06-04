#pragma once
#include "BVHNodeDefine.h"
#include "positionquantizer_new.h"

class BVHRefine
{
public:
	BVHRefine(void);
public:
	~BVHRefine(void);
public:
	FILE *fpo, *fpd;
	bool refine(const char* filename);
	bool refineRecursive(unsigned int idx, Vector3 &min, Vector3 &max);
	PositionQuantizerNew* pq;
	FORCEINLINE void updateBB(Vector3 &min, Vector3 &max, Vector3 &vec)
	{
		min.e[0] = ( min.e[0] < vec.e[0] ) ? min.e[0] : vec.e[0];
		min.e[1] = ( min.e[1] < vec.e[1] ) ? min.e[1] : vec.e[1];
		min.e[2] = ( min.e[2] < vec.e[2] ) ? min.e[2] : vec.e[2];

		max.e[0] = ( max.e[0] > vec.e[0] ) ? max.e[0] : vec.e[0];
		max.e[1] = ( max.e[1] > vec.e[1] ) ? max.e[1] : vec.e[1];
		max.e[2] = ( max.e[2] > vec.e[2] ) ? max.e[2] : vec.e[2];
	}
};
