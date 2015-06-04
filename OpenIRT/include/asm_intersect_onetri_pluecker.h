/**
 * Pluecker-coordinate triangle intersection for one 2x2 ray
 * and one triangle. Assumes precomputed information stored
 * in:
 *  - v0cross_x, v0cross_y, v0cross_z
 *  - v1cross_x, v1cross_y, v1cross_z
 *  - v2cross_x, v2cross_y, v2cross_z
 *  - trin_{x,y,z}
 *  - nominator
 */
	register unsigned int newHitMask = 0;
	
	const __m128 dx = _mm_load_ps(rays.direction[0]);
	const __m128 dy = _mm_load_ps(rays.direction[1]);
	const __m128 dz = _mm_load_ps(rays.direction[2]);	

	// test each ray against each of the triangles edges
	const __m128 v0d = _mm_dot3_ps(v0cross_x, v0cross_y, v0cross_z, dx, dy, dz);
	const __m128 v1d = _mm_dot3_ps(v1cross_x, v1cross_y, v1cross_z, dx, dy, dz);
	const __m128 v2d = _mm_dot3_ps(v2cross_x, v2cross_y, v2cross_z, dx, dy, dz);

	// generate hitmask of rays that are inside the triangle
	// (which is if the signs of the three edge tests match)
	register const __m128 zero = _mm_setzero_ps();

	#ifdef SKIP_INSIDE_TEST

	register __m128 hitMask = *(__m128 *)&_mm_cmpeq_epi32(*(__m128i *)&zero, *(__m128i *)&zero);

	#else

	#ifdef BACKFACE_CULL
	// back-face culling:
	register __m128 hitMask = _mm_and_ps(_mm_cmpge_ps(v0d, zero), _mm_and_ps(_mm_cmpge_ps(v1d, zero), _mm_cmpge_ps(v2d, zero)));  // either all > 0	
	#else
	// no back-face culling:
	register __m128 v1d_mask = _mm_cmpge_ps(v1d, zero);
	register __m128 hitMask = *(__m128 *)&_mm_and_si128(_mm_cmpeq_epi32(*(__m128i *)&_mm_cmpge_ps(v0d, zero), *(__m128i *)&v1d_mask), 
		                                                _mm_cmpeq_epi32(*(__m128i *)&v1d_mask, *(__m128i *)&_mm_cmpge_ps(v2d, zero)));
	#endif	
	
	// at least one ray is inside?
	if (_mm_movemask_ps(hitMask)) 
	#endif
	{

		// ray distance to triangle plane
		register __m128 dist = _mm_mul_ps(nominator, _mm_rcp_ps(_mm_dot3_ps(trin_x, trin_y, trin_z, dx, dy, dz)));

		// test whether in front of ray origin and previous hit
		hitMask = _mm_and_ps(_mm_cmple_ps(dist, rayPacket.hitpoints[r].t.e4), hitMask);
		hitMask = _mm_and_ps(_mm_cmpge_ps(dist, _mm_set1_ps(INTERSECT_EPSILON)), hitMask);

		newHitMask = _mm_movemask_ps(hitMask);

		// did any of the rays intersect?
		if (newHitMask) {
			
			#ifdef VISIBILITY_ONLY

			// hit originating triangle? then set the target to this distance
			const __m128 sameHitMask = *(__m128 *)&_mm_cmpeq_epi32(_mm_set1_epi32(triID), oldHitTriIdx.v4);
			dist = _mm_or_ps(_mm_and_ps(sameHitMask, target_t4.v4), _mm_andnot_ps(sameHitMask, dist));
			
			// max t values:
			tmaxs.v4 = _mm_or_ps(_mm_and_ps(hitMask, dist), _mm_andnot_ps(hitMask, tmaxs.v4));
						
			#else

			// normalize barycentric coordinates
			const __m128 vol = _mm_rcp_ps(_mm_add_ps(_mm_add_ps(v0d, v1d), v2d));
			const __m128 alpha = _mm_mul_ps(v2d, vol);
			const __m128 beta  = _mm_mul_ps(v1d, vol);

			#ifdef USE_VERTEX_NORMALS
			//
			// interpolate vertex normals..
			//
			__m128 vN0x = _mm_load1_ps(&vertNormals[0].e[0]);
			__m128 vN0y = _mm_load1_ps(&vertNormals[0].e[1]);
			__m128 vN0z = _mm_load1_ps(&vertNormals[0].e[2]);
			__m128 newNx = _mm_add_ps(vN0x, _mm_add_ps(_mm_mul_ps(beta, _mm_sub_ps(_mm_load1_ps(&vertNormals[2].e[0]), vN0x)), _mm_mul_ps(alpha, _mm_sub_ps(_mm_load1_ps(&vertNormals[1].e[0]), vN0x))));			
			__m128 newNy = _mm_add_ps(vN0y, _mm_add_ps(_mm_mul_ps(beta, _mm_sub_ps(_mm_load1_ps(&vertNormals[2].e[1]), vN0y)), _mm_mul_ps(alpha, _mm_sub_ps(_mm_load1_ps(&vertNormals[1].e[1]), vN0y))));			
			__m128 newNz = _mm_add_ps(vN0z, _mm_add_ps(_mm_mul_ps(beta, _mm_sub_ps(_mm_load1_ps(&vertNormals[2].e[2]), vN0z)), _mm_mul_ps(alpha, _mm_sub_ps(_mm_load1_ps(&vertNormals[1].e[2]), vN0z))));	

			#else

			// only one normal per surface
			__m128 newNx = trin_x;
			__m128 newNy = trin_y;
			__m128 newNz = trin_z;

			#endif

			#ifndef BACKFACE_CULL
			// test for back facing triangle
			const __m128 vdot2 = _mm_cmpgt_ps(_mm_dot3_ps(dx, dy, dz, trin_x, trin_y, trin_z), _mm_setzero_ps());
			const __m128 normalSigns = _mm_andnot_ps(*(__m128 *)&_mm_srli_epi32(*(__m128i *)& vdot2, 1), vdot2);

			newNx = _mm_xor_ps(newNx, normalSigns);
			newNy = _mm_xor_ps(newNy, normalSigns);
			newNz = _mm_xor_ps(newNz, normalSigns);			
			#endif

			#ifdef SKIP_INSIDE_TEST
			if (newHitMask == ALL_RAYS)
				#ifdef USE_DIRECT_MAT_ID
				updateAllHitpoints(modelPtr, rays, hit, matID, triID, alpha, beta, dist, newNx, newNy, newNz
#					ifdef USE_TEXTURING
					, vertTextures
#					endif
				);
				#else
				updateAllHitpoints(modelPtr, rays, hit, tri, triID, alpha, beta, dist, newNx, newNy, newNz
#					ifdef USE_TEXTURING
					, vertTextures
#					endif
				);
				#endif
			else
				#ifdef USE_DIRECT_MAT_ID
				updateHitpoints(modelPtr, rays, hit, matID, triID, hitMask, alpha, beta, dist, newNx, newNy, newNz
#					ifdef USE_TEXTURING
					, vertTextures
#					endif
				);
				#else
				updateHitpoints(modelPtr, rays, hit, tri, triID, hitMask, alpha, beta, dist, newNx, newNy, newNz
#					ifdef USE_TEXTURING
					, vertTextures
#					endif
				);
				#endif
			#else
				#ifdef USE_DIRECT_MAT_ID
			updateHitpoints(modelPtr, rays, hit, matID, triID, hitMask, alpha, beta, dist, newNx, newNy, newNz
#				ifdef USE_TEXTURING
				, vertTextures
#				endif
				);
				#else
			updateHitpoints(modelPtr, rays, hit, tri, triID, hitMask, alpha, beta, dist, newNx, newNy, newNz
#				ifdef USE_TEXTURING
				, vertTextures
#				endif
				);
				#endif
			#endif

			#endif // if !VISIBILITY_ONLY
		}
	}
