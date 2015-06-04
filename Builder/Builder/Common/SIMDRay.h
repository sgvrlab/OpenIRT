#ifndef COMMON_SIMDRAY_H
#define COMMON_SIMDRAY_H

/********************************************************************
	created:	2004/10/22
	created:	22:10:2004   16:52
	filename: 	c:\MSDev\MyProjects\Renderer\Common\SIMDRay.h
	file path:	c:\MSDev\MyProjects\Renderer\Common
	file base:	SIMDRay
	file ext:	h
	author:		Christian Lauterbach (lauterb@informatik.uni-bremen.de)
	
	purpose:	Optimized Ray class for 4 rays at a time, uses SSE.
*********************************************************************/

#include "Vector3.h"
#include "Vector4.h"

/**
 * Optimized Ray class for 4 rays at a time. Due to SSE
 * loading efficiency, the ray components are stored row-major,
 * that ist, the first 4 floats are the first components of all
 * four rays, the second 4 floats are the second components and
 * so on. This allows us to load them efficiently.
 *
 * All arrays are aligned to 16 Byte boundaries in order to load
 * them efficiently with SSE instructions. That is why we store
 * 4x4 arrays although 4x3 would suffice.
 */
class SIMDRay  {
public:

	/**
	 * Constructors:
	 */ 
	SIMDRay() { /*previousTransform = &identityTransform;*/ }
	SIMDRay(const Vector3& o1, const Vector3& d1, 
			const Vector3& o2, const Vector3& d2,
			const Vector3& o3, const Vector3& d3,
			const Vector3& o4, const Vector3& d4) 
	{ 
		for (int i = 0; i < 3; i++) {
			origin[i][0] = o1[i];
			origin[i][1] = o2[i];
			origin[i][2] = o3[i];
			origin[i][3] = o4[i];

			direction[i][0] = d1[i];
			direction[i][1] = d2[i];
			direction[i][2] = d3[i];
			direction[i][3] = d4[i];
		}
		setInvDirections();
	}

	SIMDRay(const Vector4& o1, const Vector4& d1, 
			const Vector4& o2, const Vector4& d2,
			const Vector4& o3, const Vector4& d3,
			const Vector4& o4, const Vector4& d4) 
	{ 
		for (int i = 0; i < 4; i++) {
			origin[i][0] = o1[i];
			origin[i][1] = o2[i];
			origin[i][2] = o3[i];
			origin[i][3] = o4[i];

			direction[i][0] = d1[i];
			direction[i][1] = d2[i];
			direction[i][2] = d3[i];
			direction[i][3] = d4[i];
		}
		setInvDirections();
	}

	SIMDRay(const Vector3 o[4], const Vector3 d[4]) {
		for (int i = 0; i < 3; i++) {
			origin[i][0] = o[0][i];
			origin[i][1] = o[1][i];
			origin[i][2] = o[2][i];
			origin[i][3] = o[3][i];

			direction[i][0] = d[0][i];
			direction[i][1] = d[1][i];
			direction[i][2] = d[2][i];
			direction[i][3] = d[3][i];
		}
		setInvDirections();
	}

	SIMDRay(const Vector4 o[4], const Vector4 d[4]) {
		for (int i = 0; i < 4; i++) {
			origin[i][0] = o[0][i];
			origin[i][1] = o[1][i];
			origin[i][2] = o[2][i];
			origin[i][3] = o[3][i];

			direction[i][0] = d[0][i];
			direction[i][1] = d[1][i];
			direction[i][2] = d[2][i];
			direction[i][3] = d[3][i];
		}
		setInvDirections();
	}
	
	/**
	 * Copy constructor
	 */
	SIMDRay(const SIMDRay& r) {
		*this = r;
	}

	void setInvDirections() {
		for (int i = 0; i < 3; i++) {		
			invdirection[i][0] = 1.0f / direction[i][0];
			invdirection[i][1] = 1.0f / direction[i][1];
			invdirection[i][2] = 1.0f / direction[i][2];
			invdirection[i][3] = 1.0f / direction[i][3];
		}
				
		rayChildOffsets[0] = (direction[0][0] >= 0.0f)?1:0;
		rayChildOffsets[1] = (direction[1][0] >= 0.0f)?1:0;
		rayChildOffsets[2] = (direction[2][0] >= 0.0f)?1:0;

		//previousTransform = &identityTransform;
	}

	Vector3 getOrigin(int num) const {
		return Vector3(origin[0][num], origin[1][num], origin[2][num]);
	}


	Vector3 getDirection(int num) const {
		return Vector3(direction[0][num], direction[1][num], direction[2][num]);
	}

	Vector3 getInvDirection(int num) const {
		return Vector3(invdirection[0][num], invdirection[1][num], invdirection[2][num]);
	}

	/**
	 * Set one origin point
	 */
	void setOrigin(const Vector3& v, int num) {
		origin[0][num] = v[0];
		origin[1][num] = v[1];
		origin[2][num] = v[2];		
	}
	void setOrigin(const Vector4& v, int num) {
		origin[0][num] = v[0];
		origin[1][num] = v[1];
		origin[2][num] = v[2];
		origin[3][num] = v[3];
	}

	/**
	 * Set all origin points. v must be a Vector3[4].
	 */
	void setOrigins(const Vector3 *v) {
		origin[0][0] = v[0][0];
		origin[0][1] = v[1][0];
		origin[0][2] = v[2][0];
		origin[0][3] = v[3][0];

		origin[1][0] = v[0][1];
		origin[1][1] = v[1][1];
		origin[1][2] = v[2][1];
		origin[1][3] = v[3][1];

		origin[2][0] = v[0][2];
		origin[2][1] = v[1][2];
		origin[2][2] = v[2][2];
		origin[2][3] = v[3][2];

		origin[3][0] = 0.0f;
		origin[3][1] = 0.0f;
		origin[3][2] = 0.0f;
		origin[3][3] = 0.0f;
	}

	/**
	* Set all origin points. v must be a Vector4[4].
	*/
	void setOrigins(const Vector4 *v) {
		origin[0][0] = v[0][0];
		origin[0][1] = v[1][0];
		origin[0][2] = v[2][0];
		origin[0][3] = v[3][0];

		origin[1][0] = v[0][1];
		origin[1][1] = v[1][1];
		origin[1][2] = v[2][1];
		origin[1][3] = v[3][1];

		origin[2][0] = v[0][2];
		origin[2][1] = v[1][2];
		origin[2][2] = v[2][2];
		origin[2][3] = v[3][2];
	}

	/**
	 * Set one direction vector
	 */
	void setDirection(const Vector3& v, const int num) 
	{
		direction[0][num] = v[0];
		direction[1][num] = v[1];
		direction[2][num] = v[2];
		setInvDirections();
	}

	/**
	 * Set all direction vectors (v must hold 4 Vector3s !)
	 */
	void setDirections(const Vector3 *v) 
	{
		direction[0][0] = v[0][0];
		direction[0][1] = v[1][0];
		direction[0][2] = v[2][0];
		direction[0][3] = v[3][0];

		direction[1][0] = v[0][1];
		direction[1][1] = v[1][1];
		direction[1][2] = v[2][1];
		direction[1][3] = v[3][1];

		direction[2][0] = v[0][2];
		direction[2][1] = v[1][2];
		direction[2][2] = v[2][2];
		direction[2][3] = v[3][2];

		setInvDirections();
	}

	

	/**
	 * Get point on one Ray at value t
	 */	
	Vector3 pointAtParameter(float t, int num) const 
	{ 
		float r[3];
		r[0] = origin[0][num] + t*direction[0][num];
		r[1] = origin[1][num] + t*direction[1][num];
		r[2] = origin[2][num] + t*direction[2][num];
		return Vector3(r);
	}

	/**
	 * Get point on all Rays at 4 t values 
	 * (t must be float[4], store must be Vector4[3] !)
	 */
	void pointAtParameters(float *t, Vector4 *store) const 
	{ 
		__m128 t4 = _mm_load_ps(t);
		_mm_store_ps(store[0].e, _mm_add_ps(_mm_load_ps(origin[0]), _mm_mul_ps(_mm_load_ps(direction[0]), t4)));
		_mm_store_ps(store[1].e, _mm_add_ps(_mm_load_ps(origin[1]), _mm_mul_ps(_mm_load_ps(direction[1]), t4)));
		_mm_store_ps(store[2].e, _mm_add_ps(_mm_load_ps(origin[2]), _mm_mul_ps(_mm_load_ps(direction[2]), t4)));
		
		/*
		_asm {			
			push edi;
			push esi;

			// t values
			mov esi, t;
			// load t into xmm1
			movaps xmm1, [esi];

			// origins
			mov edi, this;			
			// directions
			mov esi, this;
			add esi, 40h;
			// output 
			mov ecx, store;

			// Coordinate X

			// load origin into xmm2
			movaps xmm2, [edi];
			// load direction into xmm0			
			movaps xmm0, [esi];
			// direction * t
			mulps  xmm0, xmm1;
			// + origin
			addps  xmm0, xmm2;
			// store result
			movaps [ecx],xmm0;
			// increase pointers
			add esi, 16;
			add edi, 16;
			add ecx, 16;

			// Coordinate Y

			// load origin into xmm2
			movaps xmm2, [edi];
			// load direction into xmm0			
			movaps xmm0, [esi];
			// direction * t
			mulps  xmm0, xmm1;
			// + origin
			addps  xmm0, xmm2;
			// store result
			movaps [ecx],xmm0;
			// increase pointers
			add esi, 16;
			add edi, 16;
			add ecx, 16;

			// Coordinate Z
			// load origin into xmm2
			movaps xmm2, [edi];
			// load direction into xmm0			
			movaps xmm0, [esi]
			// direction * t
			mulps  xmm0, xmm1;
			// + origin
			addps  xmm0, xmm2;
			// store result
			movaps [ecx],xmm0;							

			pop esi;
			pop edi;
		}	*/
	}	

	FORCEINLINE void transform(RayTransform *newTransform) {				
		//_mm_store_ps(origin[0], _mm_sub_ps(_mm_load_ps(origin[0]), _mm_load1_ps(&newTransform->e[0])));
		//_mm_store_ps(origin[1], _mm_sub_ps(_mm_load_ps(origin[1]), _mm_load1_ps(&newTransform->e[1])));
		//_mm_store_ps(origin[2], _mm_sub_ps(_mm_load_ps(origin[2]), _mm_load1_ps(&newTransform->e[2])));		
		//previousTransform = newTransform;
	}

	FORCEINLINE void undoTransform() {		
		//_mm_store_ps(origin[0], _mm_add_ps(_mm_load_ps(origin[0]), _mm_load1_ps(&previousTransform->e[0])));
		//_mm_store_ps(origin[1], _mm_add_ps(_mm_load_ps(origin[1]), _mm_load1_ps(&previousTransform->e[1])));
		//_mm_store_ps(origin[2], _mm_add_ps(_mm_load_ps(origin[2]), _mm_load1_ps(&previousTransform->e[2])));		
		//previousTransform = &identityTransform;		
	}

	FORCEINLINE void undoTransformPoint(Vector4 *point) {
		//_mm_store_ps(point[0].e, _mm_add_ps(_mm_load_ps(point[0].e), _mm_load1_ps(&previousTransform->e[0])));
		//_mm_store_ps(point[1].e, _mm_add_ps(_mm_load_ps(point[1].e), _mm_load1_ps(&previousTransform->e[1])));
		//_mm_store_ps(point[2].e, _mm_add_ps(_mm_load_ps(point[2].e), _mm_load1_ps(&previousTransform->e[2])));		
	}

	// Origins
	__declspec(align(16)) float origin[4][4];
	// Directions
	__declspec(align(16)) float direction[4][4];
	// Inverted directions
	__declspec(align(16)) float invdirection[4][4];
	__declspec(align(16)) unsigned int rayChildOffsets[4];

	// points to a previous transform on this ray or NULL if never transformed
	RayTransform *previousTransform;

};

inline ostream &operator<<(ostream &os, const SIMDRay &r) {
	os << "1: (" << r.getOrigin(0) << ") + t("  << r.getInvDirection(0) << ")\n";
	os << "2: (" << r.getOrigin(1) << ") + t("  << r.getInvDirection(1) << ")\n";
	os << "3: (" << r.getOrigin(2) << ") + t("  << r.getInvDirection(2) << ")\n";
	os << "4: (" << r.getOrigin(3) << ") + t("  << r.getInvDirection(3) << ")\n";
	return os;
}

/**
 * Checks the direction signs of the SIMDRay. 
 *
 * Returns true if all directions match, false otherwise.
 */
FORCEINLINE int directionsMatch(SIMDRay &rays) {	
	int sign_x = _mm_movemask_ps(_mm_load_ps(rays.direction[0]));
	int sign_y = _mm_movemask_ps(_mm_load_ps(rays.direction[1]));
	int sign_z = _mm_movemask_ps(_mm_load_ps(rays.direction[2]));
	return (((sign_x == 0) || (sign_x == 15)) && ((sign_y == 0) || (sign_y == 15)) && ((sign_z == 0) || (sign_z == 15)));
}

/**
* Makes 4 reflection rays from a SIMD hitpoint and an incoming SIMD ray saves them in
* the SIMDRay pointer rays.
* The method assumes all pointers are 16-byte aligned !
*/
FORCEINLINE int makeReflectionRay(SIMDRay *rays, const SIMDRay *incomingrays, SIMDHitpoint *hit, int mask) {		
	__declspec(align(16)) static const unsigned int maskLUTable[16][4] = { 
		{ 0x00000000, 0x00000000, 0x00000000, 0x00000000 },	// 0
		{ 0xffffffff, 0x00000000, 0x00000000, 0x00000000 }, // 1
		{ 0x00000000, 0xffffffff, 0x00000000, 0x00000000 }, // 2
		{ 0xffffffff, 0xffffffff, 0x00000000, 0x00000000 }, // 3
		{ 0x00000000, 0x00000000, 0xffffffff, 0x00000000 }, // 4
		{ 0xffffffff, 0x00000000, 0xffffffff, 0x00000000 }, // 5
		{ 0x00000000, 0xffffffff, 0xffffffff, 0x00000000 }, // 6
		{ 0xffffffff, 0xffffffff, 0xffffffff, 0x00000000 }, // 7
		{ 0x00000000, 0x00000000, 0x00000000, 0xffffffff },	// 8
		{ 0xffffffff, 0x00000000, 0x00000000, 0xffffffff }, // 9
		{ 0x00000000, 0xffffffff, 0x00000000, 0xffffffff }, // 10
		{ 0xffffffff, 0xffffffff, 0x00000000, 0xffffffff }, // 11
		{ 0x00000000, 0x00000000, 0xffffffff, 0xffffffff }, // 12
		{ 0xffffffff, 0x00000000, 0xffffffff, 0xffffffff }, // 13
		{ 0x00000000, 0xffffffff, 0xffffffff, 0xffffffff }, // 14
		{ 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff }, // 15
	};

	/**
	* Vector3 hit_p = Vector3(hitpoint.x[0][i], hitpoint.x[1][i], hitpoint.x[2][i]);
	Vector3 hit_n = Vector3(hitpoint.n[0][i], hitpoint.n[1][i], hitpoint.n[2][i]);

	rgb reflectionColor;
	Vector3 reflectionDirection = reflect(hit_p - ray.getOrigin(i) , hit_n);
	reflectionDirection.makeUnitVector();
	Ray reflectionRay(hit_p, reflectionDirection);

	*/

	__m128 hit_x = _mm_load_ps(hit->x[0].e);
	__m128 hit_y = _mm_load_ps(hit->x[1].e);
	__m128 hit_z = _mm_load_ps(hit->x[2].e);

	__m128 origin_x = _mm_load_ps(incomingrays->origin[0]);
	__m128 origin_y = _mm_load_ps(incomingrays->origin[1]);
	__m128 origin_z = _mm_load_ps(incomingrays->origin[2]);

	// save ray origin = hitpoint
	_mm_store_ps(rays->origin[0], hit_x);
	_mm_store_ps(rays->origin[1], hit_y);
	_mm_store_ps(rays->origin[2], hit_z);
	
	// incoming direction
	__m128 indir_x = _mm_sub_ps(hit_x, origin_x);
	__m128 indir_y = _mm_sub_ps(hit_y, origin_y);
	__m128 indir_z = _mm_sub_ps(hit_z, origin_z);

	// load normals
	__m128 n_x = _mm_load_ps(hit->n[0].e);
	__m128 n_y = _mm_load_ps(hit->n[1].e);
	__m128 n_z = _mm_load_ps(hit->n[2].e);


	//
	// Goal:
	// reflection = in - normal * (2 * dot(in, normal));
	//

	// (a)
	// calculate dot(in, normal)
	//

	__m128 dot = _mm_add_ps(_mm_add_ps(_mm_mul_ps(n_x, indir_x),_mm_mul_ps(n_y, indir_y)), _mm_mul_ps(n_z, indir_z));
	
	// (b)
	// normal * dot(in, normal)
	//
	n_x = _mm_mul_ps(n_x, dot);
	n_y = _mm_mul_ps(n_y, dot);
	n_z = _mm_mul_ps(n_z, dot);

	// (c) 
	// in - 2*normal (subtract twice so we don't have to
	// multiply by 2)
	//

	__m128 reflect_x = _mm_sub_ps(_mm_sub_ps(indir_x, n_x), n_x);
	__m128 reflect_y = _mm_sub_ps(_mm_sub_ps(indir_y, n_y), n_y);
	__m128 reflect_z = _mm_sub_ps(_mm_sub_ps(indir_z, n_z), n_z);
	
	//
	// Now normalize the vector:
	//

	dot = _mm_add_ps(_mm_add_ps(_mm_mul_ps(reflect_x, reflect_x),_mm_mul_ps(reflect_y, reflect_y)), _mm_mul_ps(reflect_z, reflect_z));
	__m128 invlength = _mm_rcp_ps(_mm_sqrt_ps(dot));

	reflect_x = _mm_mul_ps(reflect_x, invlength);
	reflect_y = _mm_mul_ps(reflect_y, invlength);
	reflect_z = _mm_mul_ps(reflect_z, invlength);

	//
	// last, store results in new ray :
	//

	_mm_store_ps(rays->direction[0], reflect_x);
	_mm_store_ps(rays->direction[1], reflect_y);
	_mm_store_ps(rays->direction[2], reflect_z);
	_mm_store_ps(rays->invdirection[0], _mm_rcp_ps(reflect_x));
	_mm_store_ps(rays->invdirection[1], _mm_rcp_ps(reflect_y));
	_mm_store_ps(rays->invdirection[2], _mm_rcp_ps(reflect_z));


	/*
	__asm {
		push edi;
		push esi;

		mov edi, incomingrays;
		mov eax, mask;
		mov esi, hit;
		mov ecx, rays;

		// register layout:
		// xmm0 - xmm2: hitpoint.x
		// xmm3 - xmm5: rays.origin

		// load origins and target into registers		
		movaps xmm0, [esi]SIMDHitpoint.x[0];	// x coordinates of hitpoints
		movaps xmm1, [esi]SIMDHitpoint.x[16];	// y coordinates of hitpoints
		movaps xmm2, [esi]SIMDHitpoint.x[32];	// z coordinates of hitpoints
		movaps xmm3, [edi]SIMDRay.origin[0];	// x origins of incoming rays
		movaps xmm4, [edi]SIMDRay.origin[16];	// y origins of incoming rays
		movaps xmm5, [edi]SIMDRay.origin[32];	// z origins of incoming rays

		// save origins to ray
		movaps [ecx]SIMDRay.origin, xmm0;
		movaps [ecx]SIMDRay.origin[16], xmm1;
		movaps [ecx]SIMDRay.origin[32], xmm2;

		// indirection = hitpoint - origin
		subps xmm0, xmm3;
		subps xmm1, xmm4;
		subps xmm2, xmm5;

		// load normal vectors at hitpoints
		movaps xmm3, [esi]SIMDHitpoint.n[0];	// x coordinates of normals
		movaps xmm4, [esi]SIMDHitpoint.n[16];	// y coordinates of normals
		movaps xmm5, [esi]SIMDHitpoint.n[32];	// z coordinates of normals

		//
		// Goal:
		// reflection = in - normal * (2 * dot(in, normal));
		//

		// (a)
		// calculate dot(in, normal)
		//
		movaps xmm6, xmm0;		
		mulps xmm6, xmm3;	// n.x * in.x
		movaps xmm7, xmm1;
		mulps xmm7, xmm4;	// n.y * in.y
		addps xmm6, xmm7;
		movaps xmm7, xmm2;
		mulps xmm7, xmm5;	// n.z * in.z
		addps xmm6, xmm7;	// xmm6 = dot(in, normal)

		// (b)
		// normal * dot(in, normal)
		//
		mulps xmm3, xmm6;	// n.x * dot
		mulps xmm4, xmm6;	// n.y * dot
		mulps xmm5, xmm6;	// n.z * dot

		// (c) 
		// in - 2*normal (subtract twice so we don't have to
		// multiply by 2)
		//
		subps xmm0, xmm3;
		subps xmm1, xmm4;
		subps xmm2, xmm5;
		subps xmm0, xmm3;
		subps xmm1, xmm4;
		subps xmm2, xmm5;

		//
		// xmm0 - xmm2 holds the reflection vector !
		//
		// Now normalize the vector:
		//

		movaps xmm3, xmm0;  // backup reflection to xmm3-xmm5
		movaps xmm4, xmm1;
		movaps xmm5, xmm2;

		mulps xmm0, xmm0;	// x*x
		mulps xmm1, xmm1;	// y*y
		mulps xmm2, xmm2;	// z*z
		addps xmm0, xmm1;
		addps xmm0, xmm2;	// x*x + y*y + z*z
		sqrtps xmm0, xmm0;	// length

		divps xmm3, xmm0;	// direction.x / length
		divps xmm4, xmm0;	// direction.y / length
		divps xmm5, xmm0;	// direction.z / length

		//
		// last, store results in new ray :
		//

		movaps [ecx]SIMDRay.direction[0], xmm3;		// ray.direction.x
		movaps [ecx]SIMDRay.direction[16], xmm4;	// ray.direction.y
		movaps [ecx]SIMDRay.direction[32], xmm5;	// ray.direction.z
		rcpps xmm1, xmm3;	// inverted direction.x
		rcpps xmm2, xmm4;	// inverted direction.y
		rcpps xmm6, xmm5;	// inverted direction.z
		movaps [ecx]SIMDRay.invdirection[0], xmm1;
		movaps [ecx]SIMDRay.invdirection[16], xmm2;
		movaps [ecx]SIMDRay.invdirection[32], xmm6;		

		pop esi;
		pop edi;
	}	*/

	rays->rayChildOffsets[0] = (rays->direction[0][0] >= 0.0f)?1:0;
	rays->rayChildOffsets[1] = (rays->direction[1][0] >= 0.0f)?1:0;
	rays->rayChildOffsets[2] = (rays->direction[2][0] >= 0.0f)?1:0;

	return mask;
}

/**
* Makes 4 shadow rays from a SIMD hitpoint and a light source (for use
* in isVisible()). The method assumes all pointers are 16-byte aligned !
* The dot product between the surface normal and the normalized light
* direction is returned in angle.
*/

#ifdef _USE_AREA_SHADOWS
FORCEINLINE int makeShadowRay(SIMDRay *rays, SIMDHitpoint *hit, Vector3 &lightPos, Vector3 &lightNormal, float *tmax, float *angle, int mask) {		
#else
FORCEINLINE int makeShadowRay(SIMDRay *rays, SIMDHitpoint *hit, Vector3 &lightPos, float *tmax, float *angle, int mask) {		
#endif

	// table for 8-bit mask -> 128-bit SIMD mask
	__declspec(align(16)) static const unsigned int maskLUTable[16][4] = { 
		{ 0x00000000, 0x00000000, 0x00000000, 0x00000000 },	// 0
		{ 0xffffffff, 0x00000000, 0x00000000, 0x00000000 }, // 1
		{ 0x00000000, 0xffffffff, 0x00000000, 0x00000000 }, // 2
		{ 0xffffffff, 0xffffffff, 0x00000000, 0x00000000 }, // 3
		{ 0x00000000, 0x00000000, 0xffffffff, 0x00000000 }, // 4
		{ 0xffffffff, 0x00000000, 0xffffffff, 0x00000000 }, // 5
		{ 0x00000000, 0xffffffff, 0xffffffff, 0x00000000 }, // 6
		{ 0xffffffff, 0xffffffff, 0xffffffff, 0x00000000 }, // 7
		{ 0x00000000, 0x00000000, 0x00000000, 0xffffffff },	// 8
		{ 0xffffffff, 0x00000000, 0x00000000, 0xffffffff }, // 9
		{ 0x00000000, 0xffffffff, 0x00000000, 0xffffffff }, // 10
		{ 0xffffffff, 0xffffffff, 0x00000000, 0xffffffff }, // 11
		{ 0x00000000, 0x00000000, 0xffffffff, 0xffffffff }, // 12
		{ 0xffffffff, 0x00000000, 0xffffffff, 0xffffffff }, // 13
		{ 0x00000000, 0xffffffff, 0xffffffff, 0xffffffff }, // 14
		{ 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff }, // 15
	};	

	__m128 epsilon4 = _mm_set1_ps(0.2f); //_mm_set1_ps(1.00f);

	__m128 light_x = _mm_load1_ps(&lightPos.e[0]);
	__m128 light_y = _mm_load1_ps(&lightPos.e[1]);
	__m128 light_z = _mm_load1_ps(&lightPos.e[2]);

	/*
	// ray origin = light position
	_mm_store_ps(rays->origin[0], light_x);
	_mm_store_ps(rays->origin[1], light_y);
	_mm_store_ps(rays->origin[2], light_z);

	// ray direction = hitpoints - light position
	__m128 dir_x = _mm_sub_ps(_mm_load_ps(hit->x[0].e), light_x);
	__m128 dir_y = _mm_sub_ps(_mm_load_ps(hit->x[1].e), light_y);
	__m128 dir_z = _mm_sub_ps(_mm_load_ps(hit->x[2].e), light_z);
	*/

	__m128 hit_x = _mm_load1_ps(hit->x[0].e);
	__m128 hit_y = _mm_load1_ps(hit->x[1].e);
	__m128 hit_z = _mm_load1_ps(hit->x[2].e);

	// ray origin = light position
	_mm_store_ps(rays->origin[0], hit_x);
	_mm_store_ps(rays->origin[1], hit_y);
	_mm_store_ps(rays->origin[2], hit_z);

	// ray direction = hitpoints - light position
	__m128 dir_x = _mm_sub_ps(light_x, hit_x);
	__m128 dir_y = _mm_sub_ps(light_y, hit_y);
	__m128 dir_z = _mm_sub_ps(light_z, hit_z);

	// get length of direction vector
	__m128 dir_len = _mm_sqrt_ps( _mm_add_ps(_mm_add_ps(_mm_mul_ps(dir_x, dir_x), 
		_mm_mul_ps(dir_y, dir_y)), 
		_mm_mul_ps(dir_z, dir_z)));
	// calculate reciprocal of direction length
	__m128 dir_len_inv = _mm_rcp_ps(dir_len);

	// multiply with reciprocal to normalize:
	dir_x = _mm_mul_ps(dir_x, dir_len_inv);
	dir_y = _mm_mul_ps(dir_y, dir_len_inv);
	dir_z = _mm_mul_ps(dir_z, dir_len_inv);

	// store directions:
	_mm_store_ps(rays->direction[0], dir_x);
	_mm_store_ps(rays->direction[1], dir_y);
	_mm_store_ps(rays->direction[2], dir_z);

	// and reciprocals of directions:
	_mm_store_ps(rays->invdirection[0], _mm_rcp_ps(dir_x));
	_mm_store_ps(rays->invdirection[1], _mm_rcp_ps(dir_y));
	_mm_store_ps(rays->invdirection[2], _mm_rcp_ps(dir_z));

	// length of direction is our max value, subtract EPSILON
	_mm_store_ps(tmax, _mm_sub_ps(dir_len, epsilon4));

	// 
	// calculate dot(direction, hitpoint.n) to see whether
	// the light is behind the hitpoint and does not need a
	// shadow ray.
	//

	// load hitpoint.n
	__m128 nx = _mm_load_ps(hit->n[0].e);
	__m128 ny = _mm_load_ps(hit->n[1].e);
	__m128 nz = _mm_load_ps(hit->n[2].e);

	// calculate dot product
	__m128 dot = _mm_add_ps(_mm_add_ps(_mm_mul_ps(nx, dir_x),
		_mm_mul_ps(ny, dir_y)),
		_mm_mul_ps(nz, dir_z));

	// invert and store in angle
	dot = _mm_mul_ps(dot, _mm_set1_ps(1.0f));
	_mm_store_ps(angle, dot);

	// light is visible where dot > 0
	mask &= _mm_movemask_ps(_mm_cmpge_ps(dot, _mm_setzero_ps()));

	rays->rayChildOffsets[0] = (rays->direction[0][0] >= 0.0f)?1:0;
	rays->rayChildOffsets[1] = (rays->direction[1][0] >= 0.0f)?1:0;
	rays->rayChildOffsets[2] = (rays->direction[2][0] >= 0.0f)?1:0;

	return mask;
}

#endif
