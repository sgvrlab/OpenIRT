/********************************************************************
	created:	2011/12/07
	file path:	d:\Projects\Redering\OpenIRT\include
	file base:	Plane
	file ext:	h
	author:		Tae-Joon Kim (tjkim.kaist@gmail.com)
	
	comment:	Plane class
*********************************************************************/

#pragma once

#include "Vector3.h"

namespace irt
{

class Plane {
public:
	Vector3 n;			    // normal vector (normalized)
	float d;				// d from plane equation

	Plane() {}
	Plane(const Vector3 &n, float d) : n(n), d(d) {}
	Plane(const Vector3 v[])
	{
		set(v);
	}

	void set(const Vector3 &n, float d)
	{
		this->n = n;
		this->d = d;
	}

	void set(const Vector3 v[])
	{
		n = cross(v[1] - v[0], v[2] - v[0]);
		n.makeUnitVector();
		d = dot(v[0], n);
	}

	void set(const Vector3 &v0, const Vector3 &v1, const Vector3 &v2)
	{
		n = cross(v1 - v0, v2 - v0);
		n.makeUnitVector();
		d = dot(v0, n);
	}
};

};