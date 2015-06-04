/********************************************************************
	created:	2009/04/16
	file path:	d:\Projects\Redering\OpenIRT\include
	file base:	Ray
	file ext:	h
	author:		Tae-Joon Kim (tjkim@tclab.kaist.ac.kr)
	
	comment:	Ray class, a ray is consist of origin and direction
				as inputs and inverse direction which will be 
				calculated. Direction signs are also included.
*********************************************************************/
#pragma once

#include "Vector3.h"
#include "Matrix.h"

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

namespace irt
{

class Ray
{
// Member variables
public:
	Vector3 data[3];
	int posneg[6];

// Member functions
public:
	Ray(void) {}
	~Ray(void) {}
	Vector3 origin() const {return data[0];}
	Vector3 direction() const {return data[1];}
	Vector3 invDirection() const {return data[2];}

	void setOrigin(const Vector3 &origin) {data[0] = origin;}
	void setDirection(const Vector3 &direction)
	{
		data[1] = direction;

		data[2] = Vector3(1.0f / direction.e[0], 1.0f / direction.e[1], 1.0f / direction.e[2]);

		posneg[0] =  (data[1].e[0] >= 0 ? 1 : 0);
		posneg[3] = posneg[0] ^ 1;

		posneg[1] =  (data[1].e[1] >= 0 ? 1 : 0);
		posneg[4] = posneg[1] ^ 1;  

		posneg[2] =  (data[1].e[2] >= 0 ? 1 : 0);
		posneg[5] = posneg[2] ^ 1;
	}

	void set(const Vector3 &origin, const Vector3 &direction)
	{
		setOrigin(origin);
		setDirection(direction);
	}

	bool inline boxIntersect(const Vector3 &minBB, const Vector3 &maxBB, float &t0, float &t1) const
	{
		t0 = ((posneg[0] ? minBB.x() : maxBB.x()) - data[0].x()) * data[2].x();
		t1 = ((posneg[0] ? maxBB.x() : minBB.x()) - data[0].x()) * data[2].x();

		t0 = max(((posneg[1] ? minBB.y() : maxBB.y()) - data[0].y()) * data[2].y(), t0);
		t1 = min(((posneg[1] ? maxBB.y() : minBB.y()) - data[0].y()) * data[2].y(), t1);

		t0 = max(((posneg[2] ? minBB.z() : maxBB.z()) - data[0].z()) * data[2].z(), t0);
		t1 = min(((posneg[2] ? maxBB.z() : minBB.z()) - data[0].z()) * data[2].z(), t1);

		return (t0 <= t1);
	}

	void inline transform(const Matrix &matrix)
	{
		data[0] = transformLoc(matrix, data[0]);
		data[1] = transformVec(matrix, data[1]);

		data[2] = Vector3(1.0f / data[1].e[0], 1.0f / data[1].e[1], 1.0f / data[1].e[2]);

		posneg[0] =  (data[1].e[0] >= 0 ? 1 : 0);
		posneg[3] = posneg[0] ^ 1;

		posneg[1] =  (data[1].e[1] >= 0 ? 1 : 0);
		posneg[4] = posneg[1] ^ 1;  

		posneg[2] =  (data[1].e[2] >= 0 ? 1 : 0);
		posneg[5] = posneg[2] ^ 1;
	}
};

};