#ifndef BOXHEADER_H
#define BOXHEADER_H

#include "Vector3.h"

/** a box */ 
struct Box
{
	Vector3 p_min;
	Vector3 p_max;
	/// construct empty box
	Box()                                { clear(); }
	/// clear the box
	void clear() {
		p_min = Vector3(FLT_MAX,FLT_MAX,FLT_MAX);
		p_max = Vector3(-FLT_MAX,-FLT_MAX,-FLT_MAX);
	}
	/// add a point to the box
	void addPoint(const Vector3& p)
	{
		for (int c=0; c<3; ++c) {
			if (p[c] > p_max[c]) p_max[c] = p[c];
			if (p[c] < p_min[c]) p_min[c] = p[c];
		}
	}
	/// return the vector of extend
	Vector3 getExtent() const { return p_max-p_min; }
	/// return the maximum extent in one dimension
	float getMaxExtent() const { return getExtent().maxComponent(); }
	/// return the volume of the box
	float getVolume() const { 
		Vector3 ext = getExtent();
		return ext[0] * ext[1] * ext[2]; 
	}
	/// return the surface area of the box
	float getSurfaceArea() const { Vector3 e = getExtent(); return (e[0]*(e[1]+e[2])+e[1]*e[2])*2; }
	/// output to stream
//	friend STD ostream& operator << (STD ostream& os, const Box& b) { return os << b.p_min << "->" << b.p_max; }
};

#endif