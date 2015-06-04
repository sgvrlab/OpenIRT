/********************************************************************
	created:	2010/08/26
	file path:	d:\Projects\Redering\OpenIRT\include
	file base:	Parallelogram
	file ext:	h
	author:		Tae-Joon Kim (tjkim@tclab.kaist.ac.kr)
	
	comment:	Parallelogram class for use in ray tracer and geometry representation
*********************************************************************/

#pragma once

#include "Vector3.h"
#include "random.h"

namespace irt
{

class Parallelogram {
public:
	Vector3 corner;			
	Vector3 v1;
	Vector3 v2;
	Vector3 normal;

	inline Vector3 sample(unsigned int prevRnd) const
	{
		return corner + v1 * rnd(prevRnd) + v2 * rnd(prevRnd);
	}
};

};