#pragma once

#include "Model.h"

namespace irt
{

// currently, supports only triangular mesh.
class BVHBuilder
{
public:
	BVHBuilder(void);
	~BVHBuilder(void);

	void init(Model *mesh);
	void clear(void);

	bool build(void);

	static bool build(Model *mesh);

protected:
	Model *m_mesh;

	void updateBB(Vector3 &min, Vector3 &max, const Vector3 &vec);
	bool subDivide(unsigned int *triIDs, unsigned int left, unsigned int right, unsigned int myIndex = 0, unsigned int nextIndex = 1, int depth = 0);
};

};