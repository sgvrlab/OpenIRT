#ifndef AREA_LIGHT_H_
#define AREA_LIGHT_H_

// Note!!!
// This class has many constraints. Therefore we will refine this later.
// We assume only area light that has a box shape.

#include "common.h"

#define INSIDE_LIGHT 1
#define OUTSIDE_LIGHT 2

class AreaLight
{
public:
	Vector3 pos;
	float width_x;
	float width_z; 
	rgb color;
	int numGrid; // numGrid^2, eg. 2*2, 4*4
	int type;

	AreaLight(const Vector3& _pos, const float _width_x, const float _width_z, const int _numGrid, const int typeLight);
	~AreaLight() {}

	void generateAreaLight(LightList* lightList);
	void setColor(const rgb& c);
};

#endif