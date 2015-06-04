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
#include "Vector3.h"

/**
 * Main triangle structure for OOC use
 */
typedef struct Triangle_t {
	unsigned int p[3];		// vertex indices
	Vector3 n;			    // normal vector (normalized)
	float d;				// d from plane equation
	unsigned char  i1,i2;	// planes to be projected to	

//#ifdef _USE_TRI_MATERIALS
	unsigned short material;	// Index of material in list
//#endif

#ifdef _USE_VERTEX_NORMALS
	Vector3 normals[3];		// Vertex normals
#endif

#ifdef _USE_TEXTURING
	Vector2 uv[3];			// Tex coords for each vertex
#endif

} Triangle, *TrianglePtr;


#endif