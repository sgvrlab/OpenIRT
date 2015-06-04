/********************************************************************
	created:	2010/02/08
	file path:	d:\Projects\Redering\OpenIRT\include
	file base:	Triangle
	file ext:	h
	author:		Tae-Joon Kim (tjkim@tclab.kaist.ac.kr)
	
	comment:	Triangle class for use in ray tracer and geometry representation
*********************************************************************/

#pragma once

/**
 * Main triangle structure. Has normal coordinates and everything
 * related to shading. All other information related to shading
 * will be in the IntersectionTriangle structure below.
 */

#include "Vector3.h"

namespace irt
{

class Triangle {
public:
	unsigned int p[3];		// vertex indices
	unsigned char  i1,i2;	// planes to be projected to
	unsigned short material;	// Index of material in list
	Vector3 n;			    // normal vector (normalized)
	float d;				// d from plane equation
};

class SimpTriangle {
public:
	unsigned int p[3];		// vertex indices
};

};