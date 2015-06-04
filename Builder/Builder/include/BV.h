#ifndef BV_H
#define BV_H
/********************************************************************
	created:	2011/08/01
	file path:	d:\Projects\Redering\OpenIRT\include
	file base:	BV
	file ext:	h
	author:		Tae-Joon Kim (tjkim.kaist@gmail.com)
	
	comment:	Classes for various bounding volumes
*********************************************************************/

#include "Vector3.h"

#ifndef fminf
#define fminf(a,b) (((a) < (b)) ? (a) : (b))
#endif

#ifndef fmaxf
#define fmaxf(a,b) (((a) > (b)) ? (a) : (b))
#endif

class BV {
	virtual void update(const Vector3 &vertex) {}
};

class AABB : public BV {
public:
	Vector3 min;
	Vector3 max;

	AABB()
	{
		min = Vector3(FLT_MAX);
		max = Vector3(-FLT_MAX);
	}

	AABB(const Vector3 &min, const Vector3 &max)
	{
		this->min = min;
		this->max = max;
	}

	bool isIn(const Vector3 &vertex)
	{
		if(vertex.x() < min.x()) return false;
		if(vertex.y() < min.y()) return false;
		if(vertex.z() < min.z()) return false;
		if(vertex.x() > max.x()) return false;
		if(vertex.y() > max.y()) return false;
		if(vertex.z() > max.z()) return false;
		return true;
	}

	virtual void update(const Vector3 &vertex)
	{
		min.setX(fminf(min.x(), vertex.x()));
		min.setY(fminf(min.y(), vertex.y()));
		min.setZ(fminf(min.z(), vertex.z()));
		max.setX(fmaxf(max.x(), vertex.x()));
		max.setY(fmaxf(max.y(), vertex.y()));
		max.setZ(fmaxf(max.z(), vertex.z()));
	}
};

#endif