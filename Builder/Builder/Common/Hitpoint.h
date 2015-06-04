#ifndef COMMON_HITPOINT_H
#define COMMON_HITPOINT_H

#include "Vector2.h"
#include "Vector3.h"
#include "Vector4.h"

//class Material;
class ModelInstance;

/**
 * A hitpoint as created by tracing a ray/particle 
 * through the scene (trace())
 */
typedef struct Hitpoint_t {
	float t;		 // Param of ray
	Vector3 x;		// Coordinate
	Vector3 n;		// Surface normal at hitpoint
	float alpha,	// barycentric coordinates of triangle 
		  beta; 
	unsigned int m; // Rerefence to material
	Vector2 uv;		// tex coords
	int triIdx;		// Triangle index
	ModelInstance *objectPtr;	// Index of sub object

	// sungeui start -----------
	int m_HitLODIdx;		// first bit indicate intersecting triangle
							// other bits are LODIdx
	float m_ErrBnd;			// error bound when the ray hits the LOD.
	float m_TraveledDist;	// the distance that all the previous ray traveled;
							
	// sungeui end -------------

} Hitpoint, *HitPointPtr;

/**
* A collection of 4 hitpoints as created by tracing a ray/particle 
* through the scene.
*/
typedef struct SIMDHitpoint_t {
	float t[4];		 // Params of rays
	Vector4 x[3];	 // Coordinates ([0] is x, [1] y, [2] z)
	Vector4 n[3];	 // Surface normal at hitpoints (like coordinates)
	__declspec(align(16)) float alpha[4];  // barycentric coordinates of triangles
	__declspec(align(16)) float beta[4]; 
	__declspec(align(16)) unsigned int m[4]; // Rerefences to material
	__declspec(align(16)) Vector2 uv[4];	 // tex coords
	__declspec(align(16)) int triIdx[4];	 // triangle indices
	__declspec(align(16)) float m_ErrBnd[4];		// error bound when the ray hits the LOD.
	__declspec(align(16)) float m_TraveledDist[4];	// the distance that all the previous ray traveled;
	int m_HitLODIdx[4];		// first bit indicate intersecting triangle
	ModelInstance *objectPtr[4];	// Pointer to the object that this ray hit (when using instanced objects)

} SIMDHitpoint, *SIMDHitPointPtr;


#endif