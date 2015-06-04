/********************************************************************
	created:	2009/06/07
	file path:	d:\Projects\Redering\OpenIRT\include
	file base:	HitPointInfo
	file ext:	h
	author:		Tae-Joon Kim (tjkim@tclab.kaist.ac.kr)
	
	comment:	HitPointInfo class, has information of hit point
				between a ray and an object.
*********************************************************************/
#pragma once

#include "Vector2.h"
#include "Vector3.h"
#include "Vector4.h"

namespace irt
{

class Model;

class HitPointInfo
{
// Member variables
public:
	float t;			// parameter of ray
	Vector3 n;			// surface normal at hitpoint
	float alpha, beta;	// barycentric coordinates of triangle
	unsigned int m;		// material index
	Vector2 uv;			// texture coordinate
	Model *modelPtr;
	int tri;
	Vector3 x;

// Member functions
public:
	HitPointInfo(void);
	~HitPointInfo(void);
};

class AugHitPointInfo : public HitPointInfo
{
// Member variables
public:
	bool hasHit;
	int pixelX, pixelY;
	Vector3 hitPoint;
};


/**
 * A collection of 4 hitpoints as created by tracing a ray/particle 
 * through the scene.
 */
class SIMDHitpoint {
public:
	Vector4 t;		 // Params of rays
	Vector4 x[3];	 // Coordinates ([0] is x, [1] y, [2] z)
	Vector4 n[3];	 // Surface normal at hitpoints (like coordinates)
	Vector4 alpha;   // barycentric coordinates of triangles
	Vector4 beta; 
	__declspec(align(16)) unsigned int m[4]; // Rerefences to material
	__declspec(align(16)) Vector4 u;		 // tex coords
	__declspec(align(16)) Vector4 v;		 // tex coords
	__declspec(align(16)) int triIdx[4];	 // triangle indices
	Model *modelPtr[4];	// Pointer to the object that this ray hit (when using instanced objects)

};

};