/********************************************************************
	created:	2009/03/05
	filename: 	Vertex.h
	file base:	Vertex
	file ext:	h
	author:		Kim Tae-Joon (tjkim@tclab.kaist.ac.kr)
	
	purpose:	Vertex class for use in ray tracer and geometry representation
*********************************************************************/

#pragma once

#include "Vector2.h"
#include "Vector3.h"

/**
 * Main vertex structure
 */

namespace irt
{

class Vertex {
public:
	Vector3 v;				// vertex geometry
	float dummy1;
	Vector3 n;				// normal vector
	float dummy2;
	Vector3 c;				// color
	float dummy3;
	Vector2 uv;				// Texture coordinate
	unsigned char dummy[8];
};

class SimpVertex {
public:
	Vector3 v;
};

};