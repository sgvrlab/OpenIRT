#ifndef POINT_H
#define POINT_H

#include "Defines.h"

/** 3d point */
template <class Coord>
struct Point
{
	/// store the coordinates
	Coord coords[3];
	/// construct point (0,0,0)
	Point() { setValue(0); }
	/// construct point (c,c,c)
	Point(Coord c) { setPoint(c,c,c); }
	/// construct point (c0,c1,c2)
	Point(Coord c0, Coord c1, Coord c2) { setPoint(c0,c1,c2); }
	/// copy constructor
	template <class Coord2>
	Point(const Point<Coord2>& p) { setPoint(p[0],p[1],p[2]); }
	/// copy constructor
	Point(const Point<Coord>& p) { setPoint(p[0],p[1],p[2]); }
	/// set the point 
	void setPoint(Coord c0, Coord c1, Coord c2) { coords[0] = c0; coords[1] = c1; coords[2] = c2; }
	/// set all coordinates to the maximal available values
	void setMax();
	/// set all coordinates to the minimal available values
	void setMin();
	/// set all coordinates to c
	void setValue(Coord c) { coords[0] = coords[1] = coords[2] = c; }
	/// return the coordinates with the maximum value
	Coord getMaxCoord() const { return max(max(coords[0],coords[1]),coords[2]); }
	/// return the coordinates with the maximum value
	Coord getMinCoord() const { return min(min(coords[0],coords[1]),coords[2]); }
	/// return the i-th coordinate
	Coord& operator[] (int ci) { return coords[ci]; }
	/// return the i-th coordinate
	const Coord& operator[] (int ci) const { return coords[ci]; }
	/// return a new point equal to *this - p
	Point<Coord> operator - (const Point<Coord>& p) const {	return Point<Coord>(coords[0]-p[0],coords[1]-p[1],coords[2]-p[2]); }
	/// return a new point equal to *this + p
	Point<Coord> operator + (const Point<Coord>& p) const {	return Point<Coord>(coords[0]+p[0],coords[1]+p[1],coords[2]+p[2]); }
	/// return a new point equal to (*this)*f
	Point<Coord> operator * (Coord f) const				  {	return Point<Coord>(coords[0]*f,coords[1]*f,coords[2]*f); }
	/// multiply
	template <class Coord2>
		Point<Coord>& operator *= (const Point<Coord2>& p) { coords[0]*=p[0]; coords[1]*=p[1]; coords[2]*=p[2]; return *this; }
	/// divide
	template <class Coord2>
		Point<Coord>& operator /= (const Point<Coord2>& p) { coords[0]/=p[0]; coords[1]/=p[1]; coords[2]/=p[2]; return *this; }
	/// return the product of the coordinates
	Coord prod() const { return coords[0]*coords[1]*coords[2]; }
	/// output to stream
	friend STD ostream& operator << (STD ostream& os, const Point<Coord>& p) { return os << "[" << p.coords[0] << "," << p.coords[1] << "," << p.coords[2] << "]"; }
};


#endif