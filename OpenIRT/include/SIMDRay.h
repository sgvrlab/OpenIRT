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

#pragma once

#include "defines.h"
#include "Vector3.h"
#include "Vector4.h"
#include "Matrix.h"
#include "CommonOptions.h"

#define ALL_RAYS 15
/*
//#define PACKET_EPSILON 1.0f
#define PACKET_EPSILON 0.05f
*/
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
namespace irt
{

class SIMDRay  {
public:

	/**
	 * Constructors:
	 */ 
	SIMDRay() { previousTransform = &identityTransform; }
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

	FORCEINLINE void setInvDirections() {
		const __m128 direction_x = _mm_load_ps(direction[0]);
		const __m128 direction_y = _mm_load_ps(direction[1]);
		const __m128 direction_z = _mm_load_ps(direction[2]);

		_mm_store_ps(invdirection[0], _mm_rcp_ps(direction_x));
		_mm_store_ps(invdirection[1], _mm_rcp_ps(direction_y));
		_mm_store_ps(invdirection[2], _mm_rcp_ps(direction_z));

		rayChildOffsets[0] = (direction[0][0] >= 0.0f)?1:0;
		rayChildOffsets[1] = (direction[1][0] >= 0.0f)?1:0;
		rayChildOffsets[2] = (direction[2][0] >= 0.0f)?1:0;

		previousTransform = &identityTransform;		

		#ifdef USE_LOD
		// compute largest absolute direction value across axes
		const __m128 maxVals = _mm_max_ps(direction_x, _mm_max_ps(direction_y, direction_z));
		const __m128 minVals = _mm_mul_ps( _mm_min_ps(direction_x, _mm_min_ps(direction_y, direction_z)), _mm_set1_ps(-1.0f) );
		_mm_store_ps(maxDirections, _mm_max_ps(maxVals, minVals));
		#endif
	}

	Vector3 getOrigin(int num) const {
		return Vector3(origin[0][num], origin[1][num], origin[2][num]);
	}


	Vector3 getDirection(int num) const {
		return Vector3(direction[0][num], direction[1][num], direction[2][num]);
	}

	/**
	 * Set one origin point
	 */
	FORCEINLINE void setOrigin(const Vector3& v, int num) {
		origin[0][num] = v[0];
		origin[1][num] = v[1];
		origin[2][num] = v[2];		
	}
	FORCEINLINE void setOrigin(const Vector4& v, int num) {
		origin[0][num] = v[0];
		origin[1][num] = v[1];
		origin[2][num] = v[2];
		origin[3][num] = v[3];
	}

	/**
	 * Set all origin points. v must be a Vector3[4].
	 */
	FORCEINLINE void setOrigins(const Vector3 *v) {
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


	FORCEINLINE void setAllOrigins(const Vector3 v) {
		origin[0][0] = v[0];
		origin[0][1] = v[0];
		origin[0][2] = v[0];
		origin[0][3] = v[0];

		origin[1][0] = v[1];
		origin[1][1] = v[1];
		origin[1][2] = v[1];
		origin[1][3] = v[1];

		origin[2][0] = v[2];	
		origin[2][1] = v[2];	
		origin[2][2] = v[2];	
		origin[2][3] = v[2];	
	}

	/**
	* Set all origin points. v must be a Vector4[4].
	*/
	FORCEINLINE void setOrigins(const Vector4 *v) {
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
	FORCEINLINE void setDirection(const Vector3& v, const int num) 
	{
		direction[0][num] = v[0];
		direction[1][num] = v[1];
		direction[2][num] = v[2];
		setInvDirections();
	}

	/**
	 * Set all direction vectors (v must hold 4 Vector3s !)
	 */
	FORCEINLINE void setDirections(const Vector3 *v) 
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
	FORCEINLINE Vector3 pointAtParameter(float t, int num) const 
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
	FORCEINLINE void pointAtParameters(const Vector4 &t, Vector4 *store) const 
	{ 		
		store[0].e4 = _mm_add_ps(_mm_load_ps(origin[0]), _mm_mul_ps(_mm_load_ps(direction[0]), t.e4));
		store[1].e4 = _mm_add_ps(_mm_load_ps(origin[1]), _mm_mul_ps(_mm_load_ps(direction[1]), t.e4));
		store[2].e4 = _mm_add_ps(_mm_load_ps(origin[2]), _mm_mul_ps(_mm_load_ps(direction[2]), t.e4));
	}	

	FORCEINLINE void transform(RayTransform *newTransform) {				
		_mm_store_ps(origin[0], _mm_sub_ps(_mm_load_ps(origin[0]), _mm_load1_ps(&newTransform->e[0])));
		_mm_store_ps(origin[1], _mm_sub_ps(_mm_load_ps(origin[1]), _mm_load1_ps(&newTransform->e[1])));
		_mm_store_ps(origin[2], _mm_sub_ps(_mm_load_ps(origin[2]), _mm_load1_ps(&newTransform->e[2])));		
		previousTransform = newTransform;
	}

	FORCEINLINE void undoTransform() {		
		_mm_store_ps(origin[0], _mm_add_ps(_mm_load_ps(origin[0]), _mm_load1_ps(&previousTransform->e[0])));
		_mm_store_ps(origin[1], _mm_add_ps(_mm_load_ps(origin[1]), _mm_load1_ps(&previousTransform->e[1])));
		_mm_store_ps(origin[2], _mm_add_ps(_mm_load_ps(origin[2]), _mm_load1_ps(&previousTransform->e[2])));		
		previousTransform = &identityTransform;		
	}

	FORCEINLINE void undoTransformPoint(Vector4 *point) {
		_mm_store_ps(point[0].e, _mm_add_ps(_mm_load_ps(point[0].e), _mm_load1_ps(&previousTransform->e[0])));
		_mm_store_ps(point[1].e, _mm_add_ps(_mm_load_ps(point[1].e), _mm_load1_ps(&previousTransform->e[1])));
		_mm_store_ps(point[2].e, _mm_add_ps(_mm_load_ps(point[2].e), _mm_load1_ps(&previousTransform->e[2])));		
	}

	/**
	  * Intersects a ray bundle with a bounding box and writes the min and max t values at which
	  * the rays touch the bounding box. Returns a mask which rays hit the box
	  */
	FORCEINLINE int RayBoxIntersect(const Vector3 bb[2], SIMDVec4 &min, SIMDVec4 &max) const {
		
		// kd-tree: no checking of ray directions, assumed to be matching.
		#if 0//ACCELERATION_STRUCTURE == ACCELERATION_KDTREE
		/**
		 * SSE: intersect all 4 rays against the scene bounding box.
		 * Essentially, this tests the distances of each ray to the min and
		 * max plane of the axis. If we get a minimal value that is larger than
		 * the maximal, then the ray will cut the plane behind the viewer or
		 * behind the scene, so we can drop out of the intersection calculation
		 * immediately, since the ray will not intersect the scene.
		 * 
		 * The intersection will continue if at least one of the rays hits the
		 * bounding box.
		*/
		const __m128 bbEpsilon = _mm_set1_ps(PACKET_EPSILON);

		__m128 origins = _mm_load_ps(origin[0]);
		__m128 directions = _mm_load_ps(invdirection[0]);

		__m128 bbMin = _mm_set1_ps(bb[rayChildOffsets[0] ^ 1].e[0]);
		__m128 bbMax = _mm_set1_ps(bb[rayChildOffsets[0]].e[0]);

		// float t0 = (pp[r.posneg[3]].e[0] - r.data[0].e[0]) * r.data[2].e[0];
		min.v4 = _mm_mul_ps(_mm_sub_ps(bbMin, origins), directions);

		// float t1 = (pp[r.posneg[0]].e[0] - r.data[0].e[0]) * r.data[2].e[0];
		max.v4 = _mm_mul_ps(_mm_sub_ps(bbMax, origins), directions);		
		
		origins = _mm_load_ps(origin[1]);
		directions = _mm_load_ps(invdirection[1]);

		bbMin = _mm_set1_ps(bb[rayChildOffsets[1] ^ 1].e[1]);
		bbMax = _mm_set1_ps(bb[rayChildOffsets[1]].e[1]);

		// float t0 = (pp[r.posneg[3]].e[0] - r.data[0].e[0]) * r.data[2].e[0];
		__m128 t0 = _mm_mul_ps(_mm_sub_ps(bbMin, origins), directions);

		// float t1 = (pp[r.posneg[0]].e[0] - r.data[0].e[0]) * r.data[2].e[0];
		__m128 t1 = _mm_mul_ps(_mm_sub_ps(bbMax, origins), directions);

		//
		// if (t0 > interval_min) interval_min = t0;
		//
		min.v4 = _mm_max_ps(t0, min.v4);

		//
		// if (t1 < interval_max) interval_max = t1;
		//
		max.v4 = _mm_min_ps(t1, max.v4);

		// test for early termination (if none of the rays hit the bb)
		//
		// if (min[rnum] > max[rnum]) 
		//		continue;
		//
		int miss = _mm_movemask_ps(_mm_cmpgt_ps(min.v4, _mm_add_ps(max.v4, bbEpsilon)));
		if (miss == ALL_RAYS)
			return 0;

		origins = _mm_load_ps(origin[2]);
		directions = _mm_load_ps(invdirection[2]);

		bbMin = _mm_set1_ps(bb[rayChildOffsets[2] ^ 1].e[2]);
		bbMax = _mm_set1_ps(bb[rayChildOffsets[2]].e[2]);

		// float t0 = (pp[r.posneg[3]].e[0] - r.data[0].e[0]) * r.data[2].e[0];
		t0 = _mm_mul_ps(_mm_sub_ps(bbMin, origins), directions);

		// float t1 = (pp[r.posneg[0]].e[0] - r.data[0].e[0]) * r.data[2].e[0];
		t1 = _mm_mul_ps(_mm_sub_ps(bbMax, origins), directions);

		//
		// if (t0 > interval_min) interval_min = t0;
		//
		min.v4 = _mm_max_ps(t0, min.v4);

		//
		// if (t1 < interval_max) interval_max = t1;
		//
		max.v4 = _mm_min_ps(t1, max.v4);

		// test for early termination (if none of the rays hit the bb)
		//
		// if (min[rnum] > max[rnum]) 
		//		continue;
		//
		miss |= _mm_movemask_ps(_mm_or_ps( _mm_cmpgt_ps(min.v4, _mm_add_ps(max.v4, bbEpsilon)),
			                               _mm_cmple_ps(max.v4, _mm_setzero_ps()) ));
		if (miss == ALL_RAYS)
			return 0;

		// ray hit the box, store min/max values
		min.v4 = _mm_max_ps(min.v4, _mm_set1_ps(PACKET_EPSILON));

		return miss ^ 15;

		#else // if ACCELERATION_STRUCTURE != ACCELERATION_KDTREE:
		// BVH-based: non-coherent ray packet possible, so directions may vary 

		/**
		* SSE: intersect all 4 rays against the scene bounding box.
		* Essentially, this tests the distances of each ray to the min and
		* max plane of the axis. If we get a minimal value that is larger than
		* the maximal, then the ray will cut the plane behind the viewer or
		* behind the scene, so we can drop out of the intersection calculation
		* immediately, since the ray will not intersect the scene.
		* 
		* The intersection will continue if at least one of the rays hits the
		* bounding box.
		*/
		
		const __m128 bbEpsilon = _mm_set1_ps(PACKET_EPSILON);

		__m128 origins = _mm_load_ps(origin[0]);
		__m128 invdirections = _mm_load_ps(invdirection[0]);

		__m128 allMins = _mm_set1_ps(bb[0].e[0]);
		__m128 allMaxs = _mm_set1_ps(bb[1].e[0]);

		__m128 maxSelector = _mm_cmpge_ps(invdirections, _mm_setzero_ps());

		__m128 bbMin = _mm_or_ps(_mm_and_ps(maxSelector, allMins), _mm_andnot_ps(maxSelector, allMaxs));
		__m128 bbMax = _mm_or_ps(_mm_andnot_ps(maxSelector, allMins), _mm_and_ps(maxSelector, allMaxs));

		// float t0 = (pp[r.posneg[3]].e[0] - r.data[0].e[0]) * r.data[2].e[0];
		min.v4 = _mm_mul_ps(_mm_sub_ps(bbMin, origins), invdirections);

		// float t1 = (pp[r.posneg[0]].e[0] - r.data[0].e[0]) * r.data[2].e[0];
		max.v4 = _mm_mul_ps(_mm_sub_ps(bbMax, origins), invdirections);

		origins = _mm_load_ps(origin[1]);
		invdirections = _mm_load_ps(invdirection[1]);

		allMins = _mm_set1_ps(bb[0].e[1]);
		allMaxs = _mm_set1_ps(bb[1].e[1]);

		maxSelector = _mm_cmpge_ps(invdirections, _mm_setzero_ps());

		bbMin = _mm_or_ps(_mm_and_ps(maxSelector, allMins), _mm_andnot_ps(maxSelector, allMaxs));
		bbMax = _mm_or_ps(_mm_andnot_ps(maxSelector, allMins), _mm_and_ps(maxSelector, allMaxs));

		// float t0 = (pp[r.posneg[3]].e[0] - r.data[0].e[0]) * r.data[2].e[0];
		__m128 t0 = _mm_mul_ps(_mm_sub_ps(bbMin, origins), invdirections);

		// float t1 = (pp[r.posneg[0]].e[0] - r.data[0].e[0]) * r.data[2].e[0];
		__m128 t1 = _mm_mul_ps(_mm_sub_ps(bbMax, origins), invdirections);

		//
		// if (t0 > interval_min) interval_min = t0;
		//
		min.v4 = _mm_max_ps(t0, min.v4);

		//
		// if (t1 < interval_max) interval_max = t1;
		//
		max.v4 = _mm_min_ps(t1, max.v4);

		// test for early termination (if none of the rays hit the bb)
		//
		// if (min[rnum] > max[rnum]) 
		//		continue;
		//
		
		int miss = _mm_movemask_ps(_mm_cmpgt_ps(min.v4, _mm_add_ps(max.v4, bbEpsilon)));
		if (miss == ALL_RAYS)
			return 0;	
		
		origins = _mm_load_ps(origin[2]);
		invdirections = _mm_load_ps(invdirection[2]);	
		allMins = _mm_set1_ps(bb[0].e[2]);
		allMaxs = _mm_set1_ps(bb[1].e[2]);

		maxSelector = _mm_cmpge_ps(invdirections, _mm_setzero_ps());

		bbMin = _mm_or_ps(_mm_and_ps(maxSelector, allMins), _mm_andnot_ps(maxSelector, allMaxs)); 
		bbMax = _mm_or_ps(_mm_andnot_ps(maxSelector, allMins), _mm_and_ps(maxSelector, allMaxs)); 

		// float t0 = (pp[r.posneg[3]].e[0] - r.data[0].e[0]) * r.data[2].e[0];
		t0 = _mm_mul_ps(_mm_sub_ps(bbMin, origins), invdirections);

		// float t1 = (pp[r.posneg[0]].e[0] - r.data[0].e[0]) * r.data[2].e[0];
		t1 = _mm_mul_ps(_mm_sub_ps(bbMax, origins), invdirections);

		//
		// if (t0 > interval_min) interval_min = t0;
		//
		min.v4 = _mm_max_ps(t0, min.v4);

		//
		// if (t1 < interval_max) interval_max = t1;
		//
		max.v4 = _mm_min_ps(t1, max.v4);

		// test for early termination (if none of the rays hit the bb)
		//
		// if (min[rnum] > max[rnum]) 
		//		continue;
		//
		miss |= _mm_movemask_ps(_mm_cmpgt_ps(min.v4, _mm_add_ps(max.v4, bbEpsilon)));				

		// ray probably hit the box, store min/max values
		//min.v4 = _mm_max_ps(min.v4, _mm_set1_ps(PACKET_EPSILON));

		return miss ^ ALL_RAYS;
		#endif		
	}


	template <bool directionsMatch>
	FORCEINLINE int RayBoxIntersectLocal(const Vector3 bb[2] , SIMDVec4 &min, SIMDVec4 &max, const __m128 &maxT) const {
		const __m128 bbEpsilon = _mm_set1_ps(PACKET_EPSILON);

		/**
		 * SSE: intersect all 4 rays against the scene bounding box.
		 * Essentially, this tests the distances of each ray to the min and
		 * max plane of the axis. If we get a minimal value that is larger than
		 * the maximal, then the ray will cut the plane behind the viewer or
		 * behind the scene, so we can drop out of the intersection calculation
		 * immediately, since the ray will not intersect the scene.
		 * 
		 * The intersection will continue if at least one of the rays hits the
		 * bounding box.
		 */

		// BVH-based: non-coherent ray packet possible, so directions may vary 
		if (directionsMatch) {
			// x-axis:
			__m128 invdirections = _mm_load_ps(invdirection[0]);

			__m128 bbMin = _mm_set1_ps(bb[rayChildOffsets[0] ^ 1].e[0]);
			__m128 bbMax = _mm_set1_ps(bb[rayChildOffsets[0]    ].e[0]);

			//__m128 bbMin = _mm_shuffle_ps(bb_min, bb_min, _MM_SHUFFLE(0,0,0,0));
			//__m128 bbMax = _mm_shuffle_ps(bb_max, bb_max, _MM_SHUFFLE(0,0,0,0));
			
			// float t0 = (pp[r.posneg[3]].e[0] - r.data[0].e[0]) * r.data[2].e[0];
			min.v4 = _mm_mul_ps(bbMin, invdirections);

			// float t1 = (pp[r.posneg[0]].e[0] - r.data[0].e[0]) * r.data[2].e[0];
			max.v4 = _mm_mul_ps(bbMax, invdirections);

			// y-axis
			invdirections = _mm_load_ps(invdirection[1]);
			//bbMin = _mm_shuffle_ps(bb_min, bb_min, _MM_SHUFFLE(1,1,1,1));
			//bbMax = _mm_shuffle_ps(bb_max, bb_max, _MM_SHUFFLE(1,1,1,1));
			
			bbMin = _mm_set1_ps(bb[rayChildOffsets[1] ^ 1].e[1]);
			bbMax = _mm_set1_ps(bb[rayChildOffsets[1]    ].e[1]);
						
			// float t0 = (pp[r.posneg[3]].e[0] - r.data[0].e[0]) * r.data[2].e[0];
			__m128 t0 = _mm_mul_ps(bbMin, invdirections);

			// float t1 = (pp[r.posneg[0]].e[0] - r.data[0].e[0]) * r.data[2].e[0];
			__m128 t1 = _mm_mul_ps(bbMax, invdirections);

			//
			// if (t0 > interval_min) interval_min = t0;
			//
			min.v4 = _mm_max_ps(t0, min.v4);

			//
			// if (t1 < interval_max) interval_max = t1;
			//
			max.v4 = _mm_min_ps(t1, max.v4);

			// test for early termination (if none of the rays hit the bb)
			//
			// if (min[rnum] > max[rnum]) 
			//		continue;
			//
			
			int miss = _mm_movemask_ps(_mm_or_ps(_mm_cmpgt_ps(min.v4, maxT), _mm_cmpgt_ps(min.v4, _mm_add_ps(max.v4, bbEpsilon))));
			if (miss == ALL_RAYS)
				return 0;
			
			invdirections = _mm_load_ps(invdirection[2]);
			//bbMin = _mm_shuffle_ps(bb_min, bb_min, _MM_SHUFFLE(2,2,2,2));
			//bbMax = _mm_shuffle_ps(bb_max, bb_max, _MM_SHUFFLE(2,2,2,2));
			
			bbMin = _mm_set1_ps(bb[rayChildOffsets[2] ^ 1].e[2]);
			bbMax = _mm_set1_ps(bb[rayChildOffsets[2]    ].e[2]);

			// float t0 = (pp[r.posneg[3]].e[0] - r.data[0].e[0]) * r.data[2].e[0];
			t0 = _mm_mul_ps(bbMin, invdirections);

			// float t1 = (pp[r.posneg[0]].e[0] - r.data[0].e[0]) * r.data[2].e[0];
			t1 = _mm_mul_ps(bbMax, invdirections);

			//
			// if (t0 > interval_min) interval_min = t0;
			//
			min.v4 = _mm_max_ps(t0, min.v4);

			//
			// if (t1 < interval_max) interval_max = t1;
			//
			max.v4 = _mm_min_ps(t1, max.v4);

			// test for early termination (if none of the rays hit the bb)
			//
			// if (min[rnum] > max[rnum]) 
			//		continue;
			//
			miss |= _mm_movemask_ps(_mm_or_ps( _mm_or_ps(_mm_cmpgt_ps(min.v4, _mm_add_ps(max.v4, bbEpsilon)),
														 _mm_cmple_ps(max.v4, _mm_setzero_ps())),
											   _mm_cmpgt_ps(min.v4, maxT)));
			return miss ^ ALL_RAYS;	
		}
		else {
		
			// x-axis:
			__m128 invdirections = _mm_load_ps(invdirection[0]);
			__m128 allMins = _mm_set1_ps(bb[0].e[0]);
			__m128 allMaxs = _mm_set1_ps(bb[1].e[0]);

			__m128 maxSelector = _mm_cmpge_ps(invdirections, _mm_setzero_ps());
			__m128 bbMin = _mm_or_ps(_mm_and_ps(maxSelector, allMins), _mm_andnot_ps(maxSelector, allMaxs)); 
			__m128 bbMax = _mm_or_ps(_mm_andnot_ps(maxSelector, allMins), _mm_and_ps(maxSelector, allMaxs)); 

			// float t0 = (pp[r.posneg[3]].e[0] - r.data[0].e[0]) * r.data[2].e[0];
			min.v4 = _mm_mul_ps(bbMin, invdirections);

			// float t1 = (pp[r.posneg[0]].e[0] - r.data[0].e[0]) * r.data[2].e[0];
			max.v4 = _mm_mul_ps(bbMax, invdirections);

			// y-axis
			invdirections = _mm_load_ps(invdirection[1]);
			allMins = _mm_set1_ps(bb[0].e[1]);
			allMaxs = _mm_set1_ps(bb[1].e[1]);

			// select min/max component of box depending on ray direction
			maxSelector = _mm_cmpge_ps(invdirections, _mm_setzero_ps());
			bbMin = _mm_or_ps(_mm_and_ps(maxSelector, allMins), _mm_andnot_ps(maxSelector, allMaxs));
			bbMax = _mm_or_ps(_mm_andnot_ps(maxSelector, allMins), _mm_and_ps(maxSelector, allMaxs));

			// float t0 = (pp[r.posneg[3]].e[0] - r.data[0].e[0]) * r.data[2].e[0];
			__m128 t0 = _mm_mul_ps(bbMin, invdirections);

			// float t1 = (pp[r.posneg[0]].e[0] - r.data[0].e[0]) * r.data[2].e[0];
			__m128 t1 = _mm_mul_ps(bbMax, invdirections);

			//
			// if (t0 > interval_min) interval_min = t0;
			//
			min.v4 = _mm_max_ps(t0, min.v4);

			//
			// if (t1 < interval_max) interval_max = t1;
			//
			max.v4 = _mm_min_ps(t1, max.v4);

			// test for early termination (if none of the rays hit the bb)
			//
			// if (min[rnum] > max[rnum]) 
			//		continue;
			//
			
			int miss = _mm_movemask_ps(_mm_or_ps(_mm_cmpgt_ps(min.v4, maxT), _mm_cmpgt_ps(min.v4, _mm_add_ps(max.v4, bbEpsilon))));
			if (miss == ALL_RAYS)
				return 0;
			
			invdirections = _mm_load_ps(invdirection[2]);
			allMins = _mm_set1_ps(bb[0].e[2]);
			allMaxs = _mm_set1_ps(bb[1].e[2]);

			maxSelector = _mm_cmpge_ps(invdirections, _mm_setzero_ps());

			bbMin = _mm_or_ps(_mm_and_ps(maxSelector, allMins), _mm_andnot_ps(maxSelector, allMaxs)); 
			bbMax = _mm_or_ps(_mm_andnot_ps(maxSelector, allMins), _mm_and_ps(maxSelector, allMaxs)); 

			// float t0 = (pp[r.posneg[3]].e[0] - r.data[0].e[0]) * r.data[2].e[0];
			t0 = _mm_mul_ps(bbMin, invdirections);

			// float t1 = (pp[r.posneg[0]].e[0] - r.data[0].e[0]) * r.data[2].e[0];
			t1 = _mm_mul_ps(bbMax, invdirections);

			//
			// if (t0 > interval_min) interval_min = t0;
			//
			min.v4 = _mm_max_ps(t0, min.v4);

			//
			// if (t1 < interval_max) interval_max = t1;
			//
			max.v4 = _mm_min_ps(t1, max.v4);

			// test for early termination (if none of the rays hit the bb)
			//
			// if (min[rnum] > max[rnum]) 
			//		continue;
			//
			miss |= _mm_movemask_ps(_mm_or_ps( _mm_or_ps(_mm_cmpgt_ps(min.v4, _mm_add_ps(max.v4, bbEpsilon)),
														 _mm_cmple_ps(max.v4, _mm_setzero_ps())),
											   _mm_cmpgt_ps(min.v4, maxT)));
			return miss ^ ALL_RAYS;	
		}
	}


	FORCEINLINE void makeRays(Vector3 &corner, Vector3 &across, Vector3 &up, Vector3 &center, SIMDVec4 &a, SIMDVec4 &b) 
	{		
		__m128 corner4 = _mm_load1_ps(&corner.e[0]);
		__m128 across4 = _mm_load1_ps(&across.e[0]);
		__m128 up4 = _mm_load1_ps(&up.e[0]);
		__m128 center4 = _mm_load1_ps(&center.e[0]);

		// target = corner + across*a[i] + up*b[i];
		// direction = target - center
		__m128 directionx = _mm_sub_ps(_mm_add_ps(corner4, _mm_add_ps(_mm_mul_ps(a.v4,across4), _mm_mul_ps(b.v4, up4))),center4);

		corner4 = _mm_load1_ps(&corner.e[1]);
		across4 = _mm_load1_ps(&across.e[1]);
		up4 = _mm_load1_ps(&up.e[1]);
		center4 = _mm_load1_ps(&center.e[1]);

		// target = corner + across*a[i] + up*b[i];
		// direction = target - center
		__m128 directiony = _mm_sub_ps(_mm_add_ps(corner4, _mm_add_ps(_mm_mul_ps(a.v4,across4), _mm_mul_ps(b.v4, up4))),center4);

		corner4 = _mm_load1_ps(&corner.e[2]);
		across4 = _mm_load1_ps(&across.e[2]);
		up4 = _mm_load1_ps(&up.e[2]);
		center4 = _mm_load1_ps(&center.e[2]);

		// target = corner + across*a[i] + up*b[i];
		// direction = target - center
		__m128 directionz = _mm_sub_ps(_mm_add_ps(corner4, _mm_add_ps(_mm_mul_ps(a.v4,across4), _mm_mul_ps(b.v4,up4))),center4);

		// make unit vector:
		// get length of direction vector
		__m128 dir_len_inv = _mm_rsqrt_ps( _mm_add_ps(_mm_add_ps(_mm_mul_ps(directionx, directionx), 
																 _mm_mul_ps(directiony, directiony)), 
																 _mm_mul_ps(directionz, directionz)));

		// multiply with reciprocal to normalize:
		directionx = _mm_mul_ps(directionx, dir_len_inv);
		directiony = _mm_mul_ps(directiony, dir_len_inv);
		directionz = _mm_mul_ps(directionz, dir_len_inv);

		// store directions:
		_mm_store_ps(direction[0], directionx);
		_mm_store_ps(direction[1], directiony);
		_mm_store_ps(direction[2], directionz);

		// and reciprocals of directions:
		_mm_store_ps(invdirection[0], _mm_rcp_ps(directionx));
		_mm_store_ps(invdirection[1], _mm_rcp_ps(directiony));
		_mm_store_ps(invdirection[2], _mm_rcp_ps(directionz));

		rayChildOffsets[0] = (direction[0][0] >= 0.0f)?1:0;
		rayChildOffsets[1] = (direction[1][0] >= 0.0f)?1:0;
		rayChildOffsets[2] = (direction[2][0] >= 0.0f)?1:0;

		#ifdef USE_LOD
		// compute largest absolute direction value across axes
		const __m128 maxVals = _mm_max_ps(directionx, _mm_max_ps(directiony, directionz));
		const __m128 minVals = _mm_mul_ps( _mm_min_ps(directionx, _mm_min_ps(directiony, directionz)), _mm_set1_ps(-1.0f) );
		_mm_store_ps(maxDirections, _mm_max_ps(maxVals, minVals));
		#endif
	}


	FORCEINLINE void makeRaysWithOrigin(Vector3 &corner, Vector3 &across, Vector3 &up, Vector3 &center, SIMDVec4 &a, SIMDVec4 &b) 
	{
		_mm_store_ps(origin[0], _mm_load1_ps(&center.e[0]));
		_mm_store_ps(origin[1], _mm_load1_ps(&center.e[1]));
		_mm_store_ps(origin[2], _mm_load1_ps(&center.e[2]));

		makeRays(corner, across, up, center, a, b);
	}

	FORCEINLINE int directionsMatch() {	
		int sign_x = _mm_movemask_ps(_mm_load_ps(direction[0]));
		int sign_y = _mm_movemask_ps(_mm_load_ps(direction[1]));
		int sign_z = _mm_movemask_ps(_mm_load_ps(direction[2]));
		return (((sign_x == 0) || (sign_x == 15)) && ((sign_y == 0) || (sign_y == 15)) && ((sign_z == 0) || (sign_z == 15)));
	}

	// Origins
	__declspec(align(16)) float origin[4][4];
	// Directions
	__declspec(align(16)) float direction[4][4];
	// Inverted directions
	__declspec(align(16)) float invdirection[4][4];
	__declspec(align(16)) unsigned int rayChildOffsets[4];	

	#ifdef USE_LOD
	__declspec(align(16)) float maxDirections[4];
	#endif

	// points to a previous transform on this ray or NULL if never transformed
	RayTransform *previousTransform;
};

inline std::ostream &operator<<(std::ostream &os, const SIMDRay &r) {
	os << "1: (" << r.getOrigin(0) << ") + t("  << r.getDirection(0) << ")\n";
	os << "2: (" << r.getOrigin(1) << ") + t("  << r.getDirection(1) << ")\n";
	os << "3: (" << r.getOrigin(2) << ") + t("  << r.getDirection(2) << ")\n";
	os << "4: (" << r.getOrigin(3) << ") + t("  << r.getDirection(3) << ")\n";
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

	__m128 hit_x = _mm_load_ps(hit->x[0].e);
	__m128 hit_y = _mm_load_ps(hit->x[1].e);
	__m128 hit_z = _mm_load_ps(hit->x[2].e);

	__m128 origin_x = _mm_load_ps(incomingrays->origin[0]);
	__m128 origin_y = _mm_load_ps(incomingrays->origin[1]);
	__m128 origin_z = _mm_load_ps(incomingrays->origin[2]);	
	
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

	// extrude the reflection ray a bit for numerical stability
	__m128 reflect_eps = _mm_set1_ps(PACKET_EPSILON);
	hit_x = _mm_add_ps(_mm_mul_ps(reflect_x, reflect_eps), hit_x);
	hit_y = _mm_add_ps(_mm_mul_ps(reflect_y, reflect_eps), hit_y);
	hit_z = _mm_add_ps(_mm_mul_ps(reflect_z, reflect_eps), hit_z);

	// save ray origin = hitpoint
	_mm_store_ps(rays->origin[0], hit_x);
	_mm_store_ps(rays->origin[1], hit_y);
	_mm_store_ps(rays->origin[2], hit_z);
	
	rays->rayChildOffsets[0] = (rays->direction[0][0] >= 0.0f)?1:0;
	rays->rayChildOffsets[1] = (rays->direction[1][0] >= 0.0f)?1:0;
	rays->rayChildOffsets[2] = (rays->direction[2][0] >= 0.0f)?1:0;

	#ifdef USE_LOD
	// compute largest absolute direction value across axes
	const __m128 maxVals = _mm_max_ps(reflect_x, _mm_max_ps(reflect_y, reflect_z));
	const __m128 minVals = _mm_mul_ps( _mm_min_ps(reflect_x, _mm_min_ps(reflect_y, reflect_z)), _mm_set1_ps(-1.0f) );
	_mm_store_ps(rays->maxDirections, _mm_max_ps(maxVals, minVals));
	#endif

	return mask;
}

/**
* Makes 4 shadow rays from a SIMD hitpoint and a light source (for use
* in isVisible()). The method assumes all pointers are 16-byte aligned !
* The dot product between the surface normal and the normalized light
* direction is returned in angle.
*/
FORCEINLINE int makeShadowRay(SIMDRay *rays, SIMDHitpoint *hit, Vector3 &lightPos, float *tmax, SIMDVec4 &angle, int mask) {
	__m128 light_x = _mm_set1_ps(lightPos.e[0]);
	__m128 light_y = _mm_set1_ps(lightPos.e[1]);
	__m128 light_z = _mm_set1_ps(lightPos.e[2]);

	__m128 hit_x = _mm_load_ps(hit->x[0].e);
	__m128 hit_y = _mm_load_ps(hit->x[1].e);
	__m128 hit_z = _mm_load_ps(hit->x[2].e);

	// ray origin = light
	_mm_store_ps(rays->origin[0], light_x);
	_mm_store_ps(rays->origin[1], light_y);
	_mm_store_ps(rays->origin[2], light_z);

	// ray direction = light position - hitpoints
	__m128 dir_x = _mm_sub_ps(hit_x, light_x);
	__m128 dir_y = _mm_sub_ps(hit_y, light_y);
	__m128 dir_z = _mm_sub_ps(hit_z, light_z);

	// get length of direction vector
	__m128 dirDot = _mm_dot3_ps(dir_x, dir_y, dir_z, dir_x, dir_y, dir_z);

	// distance to light
	__m128 dir_len_inv = _mm_rsqrt_ps(dirDot);

	// calculate reciprocal of direction length
	__m128 dir_len = _mm_rcp_ps(dir_len_inv); //_mm_rsqrt_ps(dirDot);

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
	dir_len = _mm_mul_ps(dir_len, _mm_set1_ps(0.999f));	
	_mm_store_ps(tmax, dir_len);

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
	angle.v4 = _mm_dot3_ps(dir_x, dir_y, dir_z, nx, ny, nz);

	// invert and store in angle
	angle.v4 = _mm_mul_ps(angle.v4, _mm_set1_ps(-1.0f));	

	// light is visible where dot > 0
	mask &= _mm_movemask_ps(_mm_cmpge_ps(angle.v4, _mm_setzero_ps()));

	rays->rayChildOffsets[0] = (rays->direction[0][0] >= 0.0f)?1:0;
	rays->rayChildOffsets[1] = (rays->direction[1][0] >= 0.0f)?1:0;
	rays->rayChildOffsets[2] = (rays->direction[2][0] >= 0.0f)?1:0;

	#ifdef USE_LOD
	// compute largest absolute direction value across axes
	const __m128 maxVals = _mm_max_ps(dir_x, _mm_max_ps(dir_y, dir_z));
	const __m128 minVals = _mm_mul_ps( _mm_min_ps(dir_x, _mm_min_ps(dir_y, dir_z)), _mm_set1_ps(-1.0f) );
	_mm_store_ps(rays->maxDirections, _mm_max_ps(maxVals, minVals));
	#endif

	return mask;
}

FORCEINLINE int makeAreaShadowRay(SIMDRay *rays, SIMDHitpoint *hit, int i, float *target_x, float *target_y, float *target_z, float *tmax, SIMDVec4 &angle, int mask) {
	__m128 hit_x = _mm_set1_ps(hit->x[0][i]);
	__m128 hit_y = _mm_set1_ps(hit->x[1][i]);
	__m128 hit_z = _mm_set1_ps(hit->x[2][i]);

	// ray direction = light position - hitpoints
	__m128 dir_x = _mm_sub_ps(_mm_load_ps(target_x), hit_x);
	__m128 dir_y = _mm_sub_ps(_mm_load_ps(target_y), hit_y);
	__m128 dir_z = _mm_sub_ps(_mm_load_ps(target_z), hit_z);

	// get length of direction vector
	__m128 dirDot = _mm_dot3_ps(dir_x, dir_y, dir_z, dir_x, dir_y, dir_z);

	// distance to light
	__m128 dir_len_inv = _mm_rsqrt_ps(dirDot);

	// calculate reciprocal of direction length
	__m128 dir_len = _mm_rcp_ps(dir_len_inv); //_mm_rsqrt_ps(dirDot);

	// multiply with reciprocal to normalize:
	dir_x = _mm_mul_ps(dir_x, dir_len_inv);
	dir_y = _mm_mul_ps(dir_y, dir_len_inv);
	dir_z = _mm_mul_ps(dir_z, dir_len_inv);

	// ray origin = hitpoint
	_mm_store_ps(rays->origin[0], hit_x);
	_mm_store_ps(rays->origin[1], hit_y);
	_mm_store_ps(rays->origin[2], hit_z);	
	
	// store directions:
	_mm_store_ps(rays->direction[0], dir_x);
	_mm_store_ps(rays->direction[1], dir_y);
	_mm_store_ps(rays->direction[2], dir_z);

	// and reciprocals of directions:
	_mm_store_ps(rays->invdirection[0], _mm_rcp_ps(dir_x));
	_mm_store_ps(rays->invdirection[1], _mm_rcp_ps(dir_y));
	_mm_store_ps(rays->invdirection[2], _mm_rcp_ps(dir_z));

	// length of direction is our max value, subtract EPSILON	
	dir_len = _mm_mul_ps(dir_len, _mm_set1_ps(0.999f));	
	_mm_store_ps(tmax, dir_len);

	// 
	// calculate dot(direction, hitpoint.n) to see whether
	// the light is behind the hitpoint and does not need a
	// shadow ray.
	//

	// load hitpoint.n
	__m128 nx = _mm_set1_ps(hit->n[0][i]);
	__m128 ny = _mm_set1_ps(hit->n[1][i]);
	__m128 nz = _mm_set1_ps(hit->n[2][i]);

	// calculate dot product
	angle.v4 = _mm_dot3_ps(dir_x, dir_y, dir_z, nx, ny, nz);
	
	// light is visible where dot > 0
	mask &= _mm_movemask_ps(_mm_cmpge_ps(angle.v4, _mm_setzero_ps()));

	rays->rayChildOffsets[0] = (rays->direction[0][0] >= 0.0f)?1:0;
	rays->rayChildOffsets[1] = (rays->direction[1][0] >= 0.0f)?1:0;
	rays->rayChildOffsets[2] = (rays->direction[2][0] >= 0.0f)?1:0;

	#ifdef USE_LOD
	// compute largest absolute direction value across axes
	const __m128 maxVals = _mm_max_ps(dir_x, _mm_max_ps(dir_y, dir_z));
	const __m128 minVals = _mm_mul_ps( _mm_min_ps(dir_x, _mm_min_ps(dir_y, dir_z)), _mm_set1_ps(-1.0f) );
	_mm_store_ps(rays->maxDirections, _mm_max_ps(maxVals, minVals));
	#endif

	return mask;
}


FORCEINLINE int makeAreaShadowRay(SIMDRay *rays, const Vector3 &hit_pos, const Vector3 &hit_n, float *target_x, float *target_y, float *target_z, float *tmax, SIMDVec4 &angle, int mask) {
	__m128 hit_x = _mm_set1_ps(hit_pos.e[0]);
	__m128 hit_y = _mm_set1_ps(hit_pos.e[1]);
	__m128 hit_z = _mm_set1_ps(hit_pos.e[2]);

	// ray direction = light position - hitpoints
	__m128 dir_x = _mm_sub_ps(_mm_load_ps(target_x), hit_x);
	__m128 dir_y = _mm_sub_ps(_mm_load_ps(target_y), hit_y);
	__m128 dir_z = _mm_sub_ps(_mm_load_ps(target_z), hit_z);

	// get length of direction vector
	__m128 dirDot = _mm_dot3_ps(dir_x, dir_y, dir_z, dir_x, dir_y, dir_z);

	// distance to light
	__m128 dir_len_inv = _mm_rsqrt_ps(dirDot);

	// calculate reciprocal of direction length
	__m128 dir_len = _mm_rcp_ps(dir_len_inv); //_mm_rsqrt_ps(dirDot);

	// multiply with reciprocal to normalize:
	dir_x = _mm_mul_ps(dir_x, dir_len_inv);
	dir_y = _mm_mul_ps(dir_y, dir_len_inv);
	dir_z = _mm_mul_ps(dir_z, dir_len_inv);	

	// ray origin = hitpoint
	_mm_store_ps(rays->origin[0], hit_x);
	_mm_store_ps(rays->origin[1], hit_y);
	_mm_store_ps(rays->origin[2], hit_z);	

	// store directions:
	_mm_store_ps(rays->direction[0], dir_x);
	_mm_store_ps(rays->direction[1], dir_y);
	_mm_store_ps(rays->direction[2], dir_z);

	// and reciprocals of directions:
	_mm_store_ps(rays->invdirection[0], _mm_rcp_ps(dir_x));
	_mm_store_ps(rays->invdirection[1], _mm_rcp_ps(dir_y));
	_mm_store_ps(rays->invdirection[2], _mm_rcp_ps(dir_z));

	// length of direction is our max value, subtract EPSILON	
	dir_len = _mm_mul_ps(dir_len, _mm_set1_ps(0.999f));	
	_mm_store_ps(tmax, dir_len);

	// 
	// calculate dot(direction, hitpoint.n) to see whether
	// the light is behind the hitpoint and does not need a
	// shadow ray.
	//

	// load hitpoint.n
	__m128 nx = _mm_set1_ps(hit_n.e[0]);
	__m128 ny = _mm_set1_ps(hit_n.e[1]);
	__m128 nz = _mm_set1_ps(hit_n.e[2]);

	// calculate dot product
	angle.v4 = _mm_dot3_ps(dir_x, dir_y, dir_z, nx, ny, nz);

	// light is visible where dot > 0
	mask &= _mm_movemask_ps(_mm_cmpge_ps(angle.v4, _mm_setzero_ps()));

	rays->rayChildOffsets[0] = (rays->direction[0][0] >= 0.0f)?1:0;
	rays->rayChildOffsets[1] = (rays->direction[1][0] >= 0.0f)?1:0;
	rays->rayChildOffsets[2] = (rays->direction[2][0] >= 0.0f)?1:0;

	#ifdef USE_LOD
	// compute largest absolute direction value across axes
	const __m128 maxVals = _mm_max_ps(dir_x, _mm_max_ps(dir_y, dir_z));
	const __m128 minVals = _mm_mul_ps( _mm_min_ps(dir_x, _mm_min_ps(dir_y, dir_z)), _mm_set1_ps(-1.0f) );
	_mm_store_ps(rays->maxDirections, _mm_max_ps(maxVals, minVals));
	#endif

	return mask;
}


FORCEINLINE int computeLightAngles(SIMDHitpoint *hit, Vector3 &lightPos, SIMDVec4 &angle, int mask) {

	__m128 light_x = _mm_load1_ps(&lightPos.e[0]);
	__m128 light_y = _mm_load1_ps(&lightPos.e[1]);
	__m128 light_z = _mm_load1_ps(&lightPos.e[2]);

	__m128 hit_x = _mm_load_ps(hit->x[0].e);
	__m128 hit_y = _mm_load_ps(hit->x[1].e);
	__m128 hit_z = _mm_load_ps(hit->x[2].e);

	// ray direction = light position - hitpoints
	__m128 dir_x = _mm_sub_ps(hit_x, light_x);
	__m128 dir_y = _mm_sub_ps(hit_y, light_y);
	__m128 dir_z = _mm_sub_ps(hit_z, light_z);

	// get length of direction vector
	__m128 dirDot = _mm_dot3_ps(dir_x, dir_y, dir_z, dir_x, dir_y, dir_z);		
	
	// calculate reciprocal of direction length
	__m128 dir_len_inv = _mm_rsqrt_ps(dirDot);

	// multiply with reciprocal to normalize:
	dir_x = _mm_mul_ps(dir_x, dir_len_inv);
	dir_y = _mm_mul_ps(dir_y, dir_len_inv);
	dir_z = _mm_mul_ps(dir_z, dir_len_inv);

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
	angle.v4 = _mm_dot3_ps(dir_x, dir_y, dir_z, nx, ny, nz);
		
	// invert and store in angle
	angle.v4 = _mm_mul_ps(angle.v4, _mm_set1_ps(-1.0f));	

	// light is visible where dot > 0
	mask &= _mm_movemask_ps(_mm_cmpge_ps(angle.v4, _mm_setzero_ps()));

	return mask;
}

};