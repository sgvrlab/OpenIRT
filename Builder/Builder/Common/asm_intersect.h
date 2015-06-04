/********************************************************************
	created:	2004/10/22
	created:	22:10:2004   21:07
	filename: 	c:\MSDev\MyProjects\Renderer\Common\asm_intersect.h
	file path:	c:\MSDev\MyProjects\Renderer\Common
	file base:	asmintersect
	file ext:	h
	author:		Christian Lauterbach (lauterb@informatik.uni-bremen.de)
	
	purpose:	Assembler triangle intersection routines. If VISIBILITY_ONLY
				is defined, all hitpoint calculations etc. are omitted, only
				the hitmask is returned (used by visibility determination
				where only true or false needs to be returned, not a hit point)
*********************************************************************/


int count = GETCHILDCOUNT(objList);     // Number of triangles
int idxList = GETIDXOFFSET(objList);	// Triangle index start
int newHitMask = 0;					    // Bitmask for newly hit triangles

#ifdef _SIMD_SHOW_STATISTICS
_debug_LeafIntersectCount++;

// if hitMask is already set, then this is an overhead intersection
// DOES NOT WORK AT THE MOMENT (because hitMask is OR'ed with ~activeMask,
// so this reports way too high numbers)
if (hitMask > 0)
	_debug_LeafIntersectOverhead++;
#endif	

#ifdef _USE_SIMD_RAYTRACING

//
// Static constants, used below
// 

// TODO: fix culling sign in here?

__declspec(align(16)) static const float signedEpsilon[4] = { -EPSILON, -EPSILON, -EPSILON, -EPSILON };
__declspec(align(16)) static const float epsilonNear[4]   = { -INTERSECT_EPSILON, -INTERSECT_EPSILON, -INTERSECT_EPSILON, -INTERSECT_EPSILON };
__declspec(align(16)) static const float epsilonTri[4]   = { TRI_INTERSECT_EPSILON, TRI_INTERSECT_EPSILON, TRI_INTERSECT_EPSILON, TRI_INTERSECT_EPSILON };
__declspec(align(16)) static const float floatOne[4]      = { 1.0f + TRI_INTERSECT_EPSILON, 1.0f + TRI_INTERSECT_EPSILON, 1.0f + TRI_INTERSECT_EPSILON, 1.0f + TRI_INTERSECT_EPSILON};	

// lookup table for mapping a bit mask (4 bits) to a SSE bit mask (4*32 bits)
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


// For each triangle in this node:
for (int i=0; i<count; i++, idxList++) {
	//assert(idxList < treeStats.sumTris);		
	unsigned int triID = *MAKEIDX_PTR(subObject->indexlist,idxList);		
	const Triangle &tri = GETTRI(subObject,triID);
	assert(tri.i1 <= 2);
	assert(tri.i2 <= 2);
	
	#ifdef _SIMD_SHOW_STATISTICS
	_debug_LeafTriIntersectCount++;	
	#endif	
	
	/**
	 * SSE Code for triangle intersection.
	 *
	 * This intersects a triangle with all 4 rays in the SIMDRay in
	 * parallel. If a ray is already terminated (as signified by the
	 * bit set in hitMask), it intersected with the rest, but results
	 * will not be entered.
	 */

	//
	// compute dot(direction, n)
	//

	__m128 nx = _mm_load1_ps(&tri.n.e[0]);
	__m128 ny = _mm_load1_ps(&tri.n.e[1]);
	__m128 nz = _mm_load1_ps(&tri.n.e[2]);
	__m128 dx = _mm_load_ps(rays.direction[0]);
	__m128 dy = _mm_load_ps(rays.direction[1]);
	__m128 dz = _mm_load_ps(rays.direction[2]);
	__m128 vdot = _mm_add_ps(_mm_add_ps(_mm_mul_ps(dx, nx), _mm_mul_ps(dy, ny)), _mm_mul_ps(dz, nz));

	//
	// if (vdot > -EPSILON)
	//     continue;			
	//
	
	int testHitMask = _mm_movemask_ps(_mm_cmpgt_ps(vdot, _mm_set1_ps(-EPSILON))) | hitValue;
	if (testHitMask == ALL_RAYS)
		continue;

	//
	// compute second dot product:
	// vdot2 = dot(ray.origin(),tri.n);
	//

	__m128 ox = _mm_load_ps(rays.origin[0]);
	__m128 oy = _mm_load_ps(rays.origin[1]);
	__m128 oz = _mm_load_ps(rays.origin[2]);
	__m128 vdot2 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(ox, nx), _mm_mul_ps(oy, ny)), _mm_mul_ps(oz, nz));

	//
	// compute t value of intersection:
	// t = (tri.d - vdot2) / vdot;
	//

	__m128 t = _mm_div_ps(_mm_sub_ps(_mm_load1_ps(&tri.d), vdot2), vdot);
	
	//
	// if either too near or further away than a previous hit, we stop
	//
	// if (t < -INTERSECT_EPSILON || t > (tmax + INTERSECT_EPSILON))
	//   	continue;
	//

	__m128 intersectEpsilon = _mm_set1_ps(INTERSECT_EPSILON);
	testHitMask |= _mm_movemask_ps(_mm_or_ps(_mm_cmplt_ps(t,_mm_sub_ps(_mm_load_ps(tmins), intersectEpsilon)), 
		                                     _mm_cmpgt_ps(t,_mm_add_ps(_mm_load_ps(tmaxs), intersectEpsilon))));
	if (testHitMask == ALL_RAYS)
		continue;

	//
	// find intersection point with 2D plane
	//

	//
	// point[0] = ray.data[0].e[tri.i1] + ray.data[1].e[tri.i1] * t;
	// point[1] = ray.data[0].e[tri.i2] + ray.data[1].e[tri.i2] * t;
	//

	__m128 p0 = _mm_add_ps( _mm_mul_ps(_mm_load_ps(rays.direction[tri.i1]), t), _mm_load_ps(rays.origin[tri.i1]));
	__m128 p1 = _mm_add_ps( _mm_mul_ps(_mm_load_ps(rays.direction[tri.i2]), t), _mm_load_ps(rays.origin[tri.i2]));


	// begin barycentric intersection algorithm 
	const Vector3 &tri_p0 = GETVERTEX(subObject,tri.p[0]); 
	float p0_1 = tri_p0.e[tri.i1], p0_2 = tri_p0.e[tri.i2]; 
	__m128 t0 = _mm_load1_ps(&p0_1);
	__m128 t1 = _mm_load1_ps(&p0_2);		
	__m128 u0 = _mm_sub_ps(p0, t0);
	__m128 v0 = _mm_sub_ps(p1, t1);

	const Vector3 &tri_p1 = GETVERTEX(subObject,tri.p[1]); 	
	__m128 u1 = _mm_sub_ps(_mm_load1_ps(&tri_p1.e[tri.i1]), t0);
	__m128 v1 = _mm_sub_ps(_mm_load1_ps(&tri_p1.e[tri.i2]), t1);

	const Vector3 &tri_p2 = GETVERTEX(subObject,tri.p[2]); 
	__m128 u2 = _mm_sub_ps(_mm_load1_ps(&tri_p2.e[tri.i1]), t0);
	__m128 v2 = _mm_sub_ps(_mm_load1_ps(&tri_p2.e[tri.i2]), t1);

	// beta = (v0 * u1 - u0 * v1) / (v2 * u1 - u2 * v1);
	__m128 beta = _mm_div_ps(_mm_sub_ps(_mm_mul_ps(v0, u1),_mm_mul_ps(u0,v1)), _mm_sub_ps(_mm_mul_ps(v2,u1),_mm_mul_ps(u2,v1)));

	//
	// if (beta < -TRI_INTERSECT_EPSILON || beta > 1 + TRI_INTERSECT_EPSILON)
	//	continue;
	//


	testHitMask |= _mm_movemask_ps(_mm_or_ps(_mm_cmplt_ps(beta, _mm_set1_ps(-SIMDTRI_INTERSECT_EPSILON)),
											 _mm_cmpgt_ps(beta, _mm_set1_ps(1.0f + SIMDTRI_INTERSECT_EPSILON))));
	if (testHitMask == ALL_RAYS)
		continue;

	// alpha = (u0 - beta * u2) / u1;
	__m128 alpha = _mm_div_ps(_mm_sub_ps(u0, _mm_mul_ps(beta, u2)), u1);

	// not in triangle ?
	// if (alpha < -TRI_INTERSECT_EPSILON || (alpha + beta) > 1.0f + TRI_INTERSECT_EPSILON)
	testHitMask |= _mm_movemask_ps(_mm_or_ps(_mm_cmplt_ps(alpha, _mm_set1_ps(-TRI_INTERSECT_EPSILON)), 
											 _mm_cmpgt_ps(_mm_add_ps(alpha, beta), _mm_set1_ps(1.0f + TRI_INTERSECT_EPSILON))));
	if (testHitMask == ALL_RAYS)
		continue;
	
	//
	// Ladies and Gentlemen, we have (at least one) winner !
	//

	// newHitMask = bits for rays not hit before
	newHitMask |= (~testHitMask & ~hitValue) & 15;

	__m128 hitMask = _mm_load_ps((float *)maskLUTable[newHitMask]);
		
	#ifdef VISIBILITY_ONLY		
	
	//
	// save intersection t values:
	// (change only new ones!)
	//

	_mm_store_ps(intersect_t, _mm_or_ps(_mm_and_ps(t, hitMask), _mm_andnot_ps(hitMask, _mm_load_ps(intersect_t))));
	
	#else
	
	// max t values:
	_mm_store_ps(tmaxs, _mm_or_ps(_mm_and_ps(hitMask, t), _mm_andnot_ps(hitMask, _mm_load_ps(tmaxs))) );

	// tri indices:
	_mm_store_ps((float *)obj->triIdx, _mm_or_ps(_mm_and_ps(hitMask, _mm_load1_ps((float *)&triID)), _mm_andnot_ps(hitMask, _mm_load_ps((float *)obj->triIdx))) );

	// alpha:
	_mm_store_ps(obj->alpha, _mm_or_ps(_mm_and_ps(hitMask, alpha), _mm_andnot_ps(hitMask, _mm_load_ps(obj->alpha))) );

	// beta:
	_mm_store_ps(obj->beta, _mm_or_ps(_mm_and_ps(hitMask, beta), _mm_andnot_ps(hitMask, _mm_load_ps(obj->beta))) );

	#endif
	
}

#endif