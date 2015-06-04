#ifndef COMMON_TRIANGLE_H
#define COMMON_TRIANGLE_H
/********************************************************************
	created:	2004/10/04
	created:	4.10.2004   15:11
	filename: 	c:\MSDev\MyProjects\Renderer\Common\Triangle.h
	file base:	Triangle
	file ext:	h
	author:		Christian Lauterbach (lauterb@informatik.uni-bremen.de)
	
	purpose:	Triangle class for use in ray tracer and geometry representation
*********************************************************************/

/**
 * Main triangle structure. Has normal coordinates and everything
 * related to shading. All other information related to shading
 * will be in the IntersectionTriangle structure below.
 */

#include "Vector3.h"

typedef struct Triangle_t {
	unsigned int p[3];		// vertex indices
	Vector3 n;			    // normal vector (normalized)
	float d;				// d from plane equation
	unsigned char  i1,i2;	// planes to be projected to

#ifdef _USE_TRI_MATERIALS
	unsigned short material;	// Index of material in list
#endif

#ifdef _USE_VERTEX_NORMALS
	Vector3 normals[3];		// Vertex normals
#endif

#ifdef _USE_TEXTURING
	Vector2 uv[3];			// Tex coords for each vertex
#endif

} Triangle, *TrianglePtr;

typedef std::vector<Triangle> TriangleList;	
typedef TriangleList::iterator TriangleListIterator;

typedef std::vector<unsigned int> TriangleIndexList;
typedef TriangleIndexList::iterator TriangleIndexListIterator;

typedef struct MinimalTriangle_t {
	Vector3 *p[3];			// pointer to vertices
} MinimalTriangle, *MinimalTrianglePtr;

/**
 * Triangle structure especially for intersecting with the
 * triangle with Badouel's algorithm (see Graphics Gems I, p. 390 - 393).
 * Size : 48 Bytes (~ 1,5 CPU cache lines on IA32)
 */
typedef struct IntersectionTriangle_t {	
	// Members with fixed meaning:
	//
	float n[3];			// normal
	float d;			// d from plane equation
	float p[2];			// first point of triangle projected to plane
	int i1, i2;			// projection plane indices

	float u1inv;		// 1 / u1

	// Members with meaning changing depending on u1case
	// 
	float precalc1;		// u1 / (v2*u1 - u2*v1)
	float precalc2;		// v1 / (v2*u1 - u2*v1)
	float precalc3;		// u2
} IntersectionTriangle, *IntersectionTrianglePtr;

typedef std::vector<IntersectionTriangle> IntersectionTriangleList;	
typedef IntersectionTriangleList::iterator IntersectionTriangleListIterator;

inline ostream &operator<<(ostream &os, const Triangle &t) {
	os << "(" << t.p[0] << ") (" << t.p[1] << ") (" << t.p[2] << ")\n";
	return os;
}

inline ostream &operator<<(ostream &os, const IntersectionTriangle &t) {
	os << "n = (" << t.n[0] << ", " << t.n[1] << ", " << t.n[2] << ") d = " << t.d << endl;
	os << " Projection to plane " << (3 - t.i1 - t.i2) << ", Point on plane is (" << t.p[0] << ", " << t.p[1] << ")" << endl;
	if (t.u1inv == 0.0f)
		os << " Case 1: (u1inv==0) - ";
	else
		os << " Case 2: (u1inv = " << t.u1inv << " - ";
	os << "pc1= " << t.precalc1 << " pc2= " << t.precalc2 << " pc3= " << t.precalc3 << endl;
	return os;
}

#endif