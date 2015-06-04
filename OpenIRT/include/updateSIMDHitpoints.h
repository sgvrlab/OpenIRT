#pragma once

FORCEINLINE void updateHitpoints(Model *modelPtr, const SIMDRay &rays, SIMDHitpoint *hit, 
								 const Triangle &tri, 
								 const unsigned int triID, const __m128 &hitMask, 
								 const __m128 &alpha, const __m128 &beta, const __m128 &t,
								 const __m128 &newNx, const __m128 &newNy, const __m128 &newNz
#								 ifdef USE_TEXTURING
								 , const Vector2 *vertTextures
#								 endif
								 ) 
{
	// max t values:			
	hit->t.e4 = _mm_or_ps(_mm_and_ps(hitMask, t), _mm_andnot_ps(hitMask, hit->t.e4));

	// max t values for whole packet:
	//const __m128 newMaxT = _mm_max_ps(rayPacket.maxIntersectT, newT);
	//rayPacket.maxIntersectT =  _mm_or_ps(_mm_and_ps(hitMask, newMaxT), _mm_andnot_ps(hitMask, rayPacket.maxIntersectT));
	
	// tri indices:
	_mm_store_si128((__m128i *)hit->triIdx, _mm_or_si128(_mm_and_si128(   *(__m128i *)&hitMask, _mm_set1_epi32(triID)), 
				                                         _mm_andnot_si128(*(__m128i *)&hitMask, _mm_load_si128((__m128i *)hit->triIdx))) );

	// object number:
	_mm_store_ps((float *)hit->modelPtr, _mm_or_ps(_mm_and_ps(hitMask, _mm_load1_ps((float *)&modelPtr)), _mm_andnot_ps(hitMask, _mm_load_ps((float *)hit->modelPtr))) );		

	// material ID:
	unsigned int tempMat = tri.material; // needed because tri.material is unsigned short and we load 4 bytes						
	_mm_store_si128((__m128i *)hit->m, _mm_or_si128(_mm_and_si128(   *(__m128i *)&hitMask, _mm_set1_epi32(tempMat)), 
				                       _mm_andnot_si128(*(__m128i *)&hitMask, _mm_load_si128((__m128i *)hit->m))) );

	// alpha:
	hit->alpha.e4 = _mm_or_ps(_mm_and_ps(hitMask, alpha), _mm_andnot_ps(hitMask, hit->alpha.e4) );

	// beta:
	hit->beta.e4 = _mm_or_ps(_mm_and_ps(hitMask, beta), _mm_andnot_ps(hitMask, hit->beta.e4) );

	// store normal:
	hit->n[0].e4 = _mm_or_ps(_mm_and_ps(hitMask, newNx), _mm_andnot_ps(hitMask, hit->n[0].e4));
	hit->n[1].e4 = _mm_or_ps(_mm_and_ps(hitMask, newNy), _mm_andnot_ps(hitMask, hit->n[1].e4));
	hit->n[2].e4 = _mm_or_ps(_mm_and_ps(hitMask, newNz), _mm_andnot_ps(hitMask, hit->n[2].e4));

	#ifdef USE_TEXTURING
	__m128 gamma = _mm_sub_ps(_mm_sub_ps(_mm_set1_ps(1.0f), alpha), beta);

	__m128 newU = _mm_add_ps( _mm_add_ps(_mm_mul_ps(gamma, _mm_set1_ps(vertTextures[0].e[0])), 
										 _mm_mul_ps(alpha, _mm_set1_ps(vertTextures[1].e[0]))), 							  
							  _mm_mul_ps(beta,  _mm_set1_ps(vertTextures[2].e[0])));
	__m128 newV = _mm_add_ps( _mm_add_ps(_mm_mul_ps(gamma, _mm_set1_ps(vertTextures[0].e[1])), 
										 _mm_mul_ps(alpha, _mm_set1_ps(vertTextures[1].e[1]))), 							  
							  _mm_mul_ps(beta,  _mm_set1_ps(vertTextures[2].e[1])));

		
	//
	// interpolate tex coords:
	//
	// hit->u[rnum] = tri.uv[0][0] + hit->alpha[rnum] * tri.uv[1][0] + hit->beta[rnum] * tri.uv[2][0];
	//__m128 newU = _mm_add_ps(_mm_set1_ps(tri.uv[0].e[0]), _mm_add_ps(_mm_mul_ps(alpha, _mm_set1_ps(tri.uv[1].e[0])), 
	//																 _mm_mul_ps(beta,  _mm_set1_ps(tri.uv[2].e[0]))));
	// hit->v[rnum] = tri.uv[0][1] + hit->alpha[rnum] * tri.uv[1][1] + hit->beta[rnum] * tri.uv[2][1];
	//__m128 newV = _mm_add_ps(_mm_set1_ps(tri.uv[0].e[1]), _mm_add_ps(_mm_mul_ps(alpha, _mm_set1_ps(tri.uv[1].e[1])), 
	//													  	 	     _mm_mul_ps(beta,  _mm_set1_ps(tri.uv[2].e[1]))));

	hit->u.e4 = _mm_or_ps(_mm_and_ps(hitMask, newU), _mm_andnot_ps(hitMask, _mm_load_ps(hit->u.e)));
	hit->v.e4 = _mm_or_ps(_mm_and_ps(hitMask, newV), _mm_andnot_ps(hitMask, _mm_load_ps(hit->v.e)));	
	#endif	

	// store hitpoint
	rays.pointAtParameters(hit->t, hit->x);
}

FORCEINLINE void updateAllHitpoints(Model *modelPtr, const SIMDRay &rays, SIMDHitpoint *hit, 
									const Triangle &tri, 
									const unsigned int triID, 
									const __m128 &alpha, const __m128 &beta, const __m128 &t,
									const __m128 &newNx, const __m128 &newNy, const __m128 &newNz
#									ifdef USE_TEXTURING
									, const Vector2 *vertTextures
#									endif
									) {
	// max t values:			
	hit->t.e4 = t;

	// max t values for whole packet:
	//const __m128 newMaxT = _mm_max_ps(rayPacket.maxIntersectT, newT);
	//rayPacket.maxIntersectT =  _mm_or_ps(_mm_and_ps(hitMask, newMaxT), _mm_andnot_ps(hitMask, rayPacket.maxIntersectT));
	
	// tri indices:
	_mm_store_si128((__m128i *)hit->triIdx, _mm_set1_epi32(triID));

	// object number:
	//_mm_store_ps((float *)hit->modelPtr, _mm_load1_ps((float *)&modelPtr));		
	hit->modelPtr[0] = hit->modelPtr[1] = hit->modelPtr[2] = hit->modelPtr[3] = modelPtr;

	// material ID:
	unsigned int tempMat = tri.material; // needed because tri.material is unsigned short and we load 4 bytes						
	_mm_store_si128((__m128i *)hit->m, _mm_set1_epi32(tempMat));

	// alpha:
	hit->alpha.e4 = alpha;

	// beta:
	hit->beta.e4 = beta;

	// store normal:
	hit->n[0].e4 = newNx;
	hit->n[1].e4 = newNy;
	hit->n[2].e4 = newNz;

	#ifdef USE_TEXTURING
	__m128 gamma = _mm_sub_ps(_mm_sub_ps(_mm_set1_ps(1.0f), alpha), beta);

	__m128 newU = _mm_add_ps( _mm_add_ps(_mm_mul_ps(gamma, _mm_set1_ps(vertTextures[0].e[0])), 
										 _mm_mul_ps(alpha, _mm_set1_ps(vertTextures[1].e[0]))), 							  
							  _mm_mul_ps(beta,  _mm_set1_ps(vertTextures[2].e[0])));
	__m128 newV = _mm_add_ps( _mm_add_ps(_mm_mul_ps(gamma, _mm_set1_ps(vertTextures[0].e[1])), 
										 _mm_mul_ps(alpha, _mm_set1_ps(vertTextures[1].e[1]))), 							  
							  _mm_mul_ps(beta,  _mm_set1_ps(vertTextures[2].e[1])));
	/*
	//
	// interpolate tex coords:
	//
	// hit->u[rnum] = tri.uv[0][0] + hit->alpha[rnum] * tri.uv[1][0] + hit->beta[rnum] * tri.uv[2][0];
	__m128 newU = _mm_add_ps(_mm_set1_ps(tri.uv[0].e[0]), _mm_add_ps(_mm_mul_ps(alpha, _mm_set1_ps(tri.uv[1].e[0])), 
																	 _mm_mul_ps(beta,  _mm_set1_ps(tri.uv[2].e[0]))));
	// hit->v[rnum] = tri.uv[0][1] + hit->alpha[rnum] * tri.uv[1][1] + hit->beta[rnum] * tri.uv[2][1];
	__m128 newV = _mm_add_ps(_mm_set1_ps(tri.uv[0].e[1]), _mm_add_ps(_mm_mul_ps(alpha, _mm_set1_ps(tri.uv[1].e[1])), 
														  	 	     _mm_mul_ps(beta,  _mm_set1_ps(tri.uv[2].e[1]))));
	*/

	hit->u.e4 = newU;
	hit->v.e4 = newV;	
	#endif	

	// store hitpoint
	rays.pointAtParameters(hit->t, hit->x);
}

FORCEINLINE void updateHitpoints(Model *modelPtr, const SIMDRay &rays, SIMDHitpoint *hit, 
								 unsigned int tempMat,
								 const unsigned int triID, const __m128 &hitMask, 
								 const __m128 &alpha, const __m128 &beta, const __m128 &t,
								 const __m128 &newNx, const __m128 &newNy, const __m128 &newNz
#								 ifdef USE_TEXTURING
								 , const Vector2 *vertTextures
#								 endif
								 ) 
{
	// max t values:			
	hit->t.e4 = _mm_or_ps(_mm_and_ps(hitMask, t), _mm_andnot_ps(hitMask, hit->t.e4));

	// max t values for whole packet:
	//const __m128 newMaxT = _mm_max_ps(rayPacket.maxIntersectT, newT);
	//rayPacket.maxIntersectT =  _mm_or_ps(_mm_and_ps(hitMask, newMaxT), _mm_andnot_ps(hitMask, rayPacket.maxIntersectT));
	
	// tri indices:
	_mm_store_si128((__m128i *)hit->triIdx, _mm_or_si128(_mm_and_si128(   *(__m128i *)&hitMask, _mm_set1_epi32(triID)), 
				                                         _mm_andnot_si128(*(__m128i *)&hitMask, _mm_load_si128((__m128i *)hit->triIdx))) );

	// object number:
	//_mm_store_ps((float *)hit->modelPtr, _mm_or_ps(_mm_and_ps(hitMask, _mm_load1_ps((float *)&modelPtr)), _mm_andnot_ps(hitMask, _mm_load_ps((float *)hit->modelPtr))) );		
	hit->modelPtr[0] = hit->modelPtr[1] = hit->modelPtr[2] = hit->modelPtr[3] = modelPtr;

	// material ID:
	_mm_store_si128((__m128i *)hit->m, _mm_or_si128(_mm_and_si128(   *(__m128i *)&hitMask, _mm_set1_epi32(tempMat)), 
				                       _mm_andnot_si128(*(__m128i *)&hitMask, _mm_load_si128((__m128i *)hit->m))) );

	// alpha:
	hit->alpha.e4 = _mm_or_ps(_mm_and_ps(hitMask, alpha), _mm_andnot_ps(hitMask, hit->alpha.e4) );

	// beta:
	hit->beta.e4 = _mm_or_ps(_mm_and_ps(hitMask, beta), _mm_andnot_ps(hitMask, hit->beta.e4) );

	// store normal:
	hit->n[0].e4 = _mm_or_ps(_mm_and_ps(hitMask, newNx), _mm_andnot_ps(hitMask, hit->n[0].e4));
	hit->n[1].e4 = _mm_or_ps(_mm_and_ps(hitMask, newNy), _mm_andnot_ps(hitMask, hit->n[1].e4));
	hit->n[2].e4 = _mm_or_ps(_mm_and_ps(hitMask, newNz), _mm_andnot_ps(hitMask, hit->n[2].e4));

	#ifdef USE_TEXTURING
	__m128 gamma = _mm_sub_ps(_mm_sub_ps(_mm_set1_ps(1.0f), alpha), beta);

	__m128 newU = _mm_add_ps( _mm_add_ps(_mm_mul_ps(gamma, _mm_set1_ps(vertTextures[0].e[0])), 
										 _mm_mul_ps(alpha, _mm_set1_ps(vertTextures[1].e[0]))), 							  
							  _mm_mul_ps(beta,  _mm_set1_ps(vertTextures[2].e[0])));
	__m128 newV = _mm_add_ps( _mm_add_ps(_mm_mul_ps(gamma, _mm_set1_ps(vertTextures[0].e[1])), 
										 _mm_mul_ps(alpha, _mm_set1_ps(vertTextures[1].e[1]))), 							  
							  _mm_mul_ps(beta,  _mm_set1_ps(vertTextures[2].e[1])));

		
	//
	// interpolate tex coords:
	//
	// hit->u[rnum] = tri.uv[0][0] + hit->alpha[rnum] * tri.uv[1][0] + hit->beta[rnum] * tri.uv[2][0];
	//__m128 newU = _mm_add_ps(_mm_set1_ps(tri.uv[0].e[0]), _mm_add_ps(_mm_mul_ps(alpha, _mm_set1_ps(tri.uv[1].e[0])), 
	//																 _mm_mul_ps(beta,  _mm_set1_ps(tri.uv[2].e[0]))));
	// hit->v[rnum] = tri.uv[0][1] + hit->alpha[rnum] * tri.uv[1][1] + hit->beta[rnum] * tri.uv[2][1];
	//__m128 newV = _mm_add_ps(_mm_set1_ps(tri.uv[0].e[1]), _mm_add_ps(_mm_mul_ps(alpha, _mm_set1_ps(tri.uv[1].e[1])), 
	//													  	 	     _mm_mul_ps(beta,  _mm_set1_ps(tri.uv[2].e[1]))));

	hit->u.e4 = _mm_or_ps(_mm_and_ps(hitMask, newU), _mm_andnot_ps(hitMask, _mm_load_ps(hit->u.e)));
	hit->v.e4 = _mm_or_ps(_mm_and_ps(hitMask, newV), _mm_andnot_ps(hitMask, _mm_load_ps(hit->v.e)));	
	#endif	

	// store hitpoint
	rays.pointAtParameters(hit->t, hit->x);
}

FORCEINLINE void updateAllHitpoints(Model *modelPtr, const SIMDRay &rays, SIMDHitpoint *hit, 
									unsigned int tempMat, 
									const unsigned int triID, 
									const __m128 &alpha, const __m128 &beta, const __m128 &t,
									const __m128 &newNx, const __m128 &newNy, const __m128 &newNz
#									ifdef USE_TEXTURING
									, const Vector2 *vertTextures
#									endif
									) {
	// max t values:			
	hit->t.e4 = t;

	// max t values for whole packet:
	//const __m128 newMaxT = _mm_max_ps(rayPacket.maxIntersectT, newT);
	//rayPacket.maxIntersectT =  _mm_or_ps(_mm_and_ps(hitMask, newMaxT), _mm_andnot_ps(hitMask, rayPacket.maxIntersectT));
	
	// tri indices:
	_mm_store_si128((__m128i *)hit->triIdx, _mm_set1_epi32(triID));

	// object number:
	//_mm_store_ps((float *)hit->modelPtr, _mm_load1_ps((float *)&modelPtr));		
	hit->modelPtr[0] = hit->modelPtr[1] = hit->modelPtr[2] = hit->modelPtr[3] = modelPtr;

	// material ID:
	_mm_store_si128((__m128i *)hit->m, _mm_set1_epi32(tempMat));

	// alpha:
	hit->alpha.e4 = alpha;

	// beta:
	hit->beta.e4 = beta;

	// store normal:
	hit->n[0].e4 = newNx;
	hit->n[1].e4 = newNy;
	hit->n[2].e4 = newNz;

	#ifdef USE_TEXTURING
	__m128 gamma = _mm_sub_ps(_mm_sub_ps(_mm_set1_ps(1.0f), alpha), beta);

	__m128 newU = _mm_add_ps( _mm_add_ps(_mm_mul_ps(gamma, _mm_set1_ps(vertTextures[0].e[0])), 
										 _mm_mul_ps(alpha, _mm_set1_ps(vertTextures[1].e[0]))), 							  
							  _mm_mul_ps(beta,  _mm_set1_ps(vertTextures[2].e[0])));
	__m128 newV = _mm_add_ps( _mm_add_ps(_mm_mul_ps(gamma, _mm_set1_ps(vertTextures[0].e[1])), 
										 _mm_mul_ps(alpha, _mm_set1_ps(vertTextures[1].e[1]))), 							  
							  _mm_mul_ps(beta,  _mm_set1_ps(vertTextures[2].e[1])));
	/*
	//
	// interpolate tex coords:
	//
	// hit->u[rnum] = tri.uv[0][0] + hit->alpha[rnum] * tri.uv[1][0] + hit->beta[rnum] * tri.uv[2][0];
	__m128 newU = _mm_add_ps(_mm_set1_ps(tri.uv[0].e[0]), _mm_add_ps(_mm_mul_ps(alpha, _mm_set1_ps(tri.uv[1].e[0])), 
																	 _mm_mul_ps(beta,  _mm_set1_ps(tri.uv[2].e[0]))));
	// hit->v[rnum] = tri.uv[0][1] + hit->alpha[rnum] * tri.uv[1][1] + hit->beta[rnum] * tri.uv[2][1];
	__m128 newV = _mm_add_ps(_mm_set1_ps(tri.uv[0].e[1]), _mm_add_ps(_mm_mul_ps(alpha, _mm_set1_ps(tri.uv[1].e[1])), 
														  	 	     _mm_mul_ps(beta,  _mm_set1_ps(tri.uv[2].e[1]))));
	*/

	hit->u.e4 = newU;
	hit->v.e4 = newV;	
	#endif	

	// store hitpoint
	rays.pointAtParameters(hit->t, hit->x);
}