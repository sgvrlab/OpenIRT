/********************************************************************
	created:	2004/10/22
	created:	22:10:2004   21:19
	filename: 	c:\MSDev\MyProjects\Renderer\Common\asm_traversetree.h
	file path:	c:\MSDev\MyProjects\Renderer\Common
	file base:	asm_traversetree
	file ext:	h
	author:		Christian Lauterbach (lauterb@informatik.uni-bremen.de)
	
	purpose:	Assembler code in SSE for traversing the BSP tree with
	            a SIMD ray.
*********************************************************************/

#define ALL_RAYS 15
int hitValue = 0;

__declspec(align(16)) SIMDVec4 min;
__declspec(align(16)) SIMDVec4 max;

#ifndef VISIBILITY_ONLY
#ifdef USE_LOD
__m128 LODBaseDistance = _mm_load_ps(traveledDistance);
#endif

#else

SIMDVec4 intersect_t;
intersect_t.v[0] = FLT_MAX;
intersect_t.v[1] = FLT_MAX;
intersect_t.v[2] = FLT_MAX;
intersect_t.v[3] = FLT_MAX;
#ifdef USE_LOD
__m128 LODBaseDistance = _mm_setzero_ps();
#endif

#endif

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

// only compile this if needed in case it does not work
// with other settings that change internal structures
#ifdef _USE_SIMD_RAYTRACING

int stackPtr;
int threadID = omp_get_thread_num();
__declspec(align(16)) StackElem *stack = stacks[omp_get_thread_num()];
unsigned int nearChild, farChild, currentOffset, activeRays = ALL_RAYS;
BSPArrayTreeNodePtr currentNode;

#ifdef _SIMD_SHOW_STATISTICS
_debug_TreeIntersectCount++;
#endif

#ifdef VISIBILITY_ONLY
hitValue = initialMask ^ ALL_RAYS;
#else
hitValue = 0;
#endif

// sungeui start -------------
#ifndef VISIBILITY_ONLY
hits->m_HitLODIdx[0] = 0;
hits->m_HitLODIdx[1] = 0;
hits->m_HitLODIdx[2] = 0;
hits->m_HitLODIdx[3] = 0;	
#else
SIMDHitpoint hit;
SIMDHitpoint *hits = &hit;
#endif
// sungeui end ----------------

unsigned int subObjectId;
initializeObjectIntersection(threadID);
while (ModelInstance *subObject = getNextObjectIntersection(rays, threadID, min, max)) {	

/*
#ifndef TRAVERSE_OFFSET
	hitValue = RayBoxIntersect(rays, subObject->bb, min, max);
#else
	min.v4 = _mm_set1_ps(EPSILON);
	max.v4 = _mm_set1_ps(FLT_MAX);
#endif
	

// did any ray hit the bounding box ?
#if !defined(VISIBILITY_ONLY) && !defined(TRAVERSE_OFFSET)
if (hitValue) { // yes, so trace them all	
#endif
*/
	min.v4 = _mm_max_ps(min.v4, _mm_set1_ps(EPSILON));

	activeRays = hitValue ^ 15;
	currentOffset = INT_MAX;
	stack[0].node = NULL;
	stackPtr = 1;

	if (g_Verbose) {
		int foo = 1;
	}

	#ifdef TRAVERSE_OFFSET
	currentNode = GETNODE(subObject->tree, startOffset);
	#else
	currentNode = GETNODE(subObject->tree, 0);
	#endif

	__m128 epsilon = _mm_set1_ps(BSP_EPSILON);

	if (g_Verbose)
		cout << rays << endl;

	// traverse BSP tree:	
 	while (currentOffset != 0) {

		if (g_Verbose)
			cout << "Start traverse at " << currentOffset << ", hitValue=" << hitValue << endl;

		/**
		 * SSE Part 2 version 3.0 :) = calculate distance to splitting plane.
		 *
		 * This will calculate the parameter t to hit the splitting plane of the
		 * current node for all 4 rays in parallel. If the distance is smaller than
		 * 0 or larger than max, then the ray hits the plane before/after the bounding
		 * box, that is, we only need to trace the near Node further. If the distance
		 * is smaller than min, we only need to trace the far Node. Otherwise, we first
		 * trace the near, then the far node. 
		 * To unroll the recursion, a stack is used. If we need to trace 2 nodes (near,
		 * then far), the far node is pushed on the stack and will be traced after the
		 * near node has been evaluated.
		 */
		
		int axis;
		__m128 activeMask = _mm_load_ps((float *)maskLUTable[activeRays & ~hitValue]);
		
		while (ISNOLEAF(currentNode)) {
			axis = AXIS(currentNode);

			#ifdef _SIMD_SHOW_STATISTICS			
			_debug_NodeIntersections++;
			if (hitValue > 0)
				_debug_NodeIntersectionsOverhead++;
			#endif

			// sungeui start -----------------------------
			g_NumTraversed++;

			#ifdef USE_LOD
			if ( HAS_LOD(currentNode->lodIndex) ) {

				// Note: later, I need to put this info. in kd-node to remove unnecessary
				//		 data access
				int QuanIdx = GET_ERR_QUANTIZATION_IDX(currentNode->lodIndex);
				assert (QuanIdx >= 0 && QuanIdx < (1<< ERR_BITs));

				// get error bounds for quantized error
				float ErrBnd = GETLODERROR(subObject,QuanIdx);	
				// ... and load into SIMD register
				__m128 errBnd4 = _mm_load1_ps(&ErrBnd);	
				
				#ifndef VISIBILITY_ONLY
				// evaluate SIMD metric:
				__m128 lodMetric4 = _mm_mul_ps(_mm_add_ps(min.v4, LODBaseDistance), _mm_load1_ps(&g_MaxAllowModifier));
				// test if below error bounds for *any* of the rays:
				int testMask = _mm_movemask_ps(_mm_cmplt_ps(errBnd4, lodMetric4)) & activeRays & ~hitValue;

				
				if (g_Verbose) {
					cout << "Intersect with LOD, testMask=" << testMask << ", activeRays="<< activeRays << " hitValue=" << hitValue << endl;
					cout << "ErrBnd=(" << ErrBnd << ") < ";
					float temp[4];
					_mm_store_ps(temp, lodMetric4);
					cout << temp[0] << " " << temp[1] << " " << temp[2] << " " << temp[3] << endl;
				}

				if (testMask) {
					hitValue |= testMask & RayLODIntersect(rays, subObject, currentNode, hits, max, min, testMask ^ ALL_RAYS);

					if (g_Verbose)
						cout << "LOD result mask=" << hitValue << endl;

					if (hitValue == ALL_RAYS) {
						// all rays terminated, early abort
						return hitValue;
					}
						
					if (hitValue == activeRays)
						goto LOD_END;	// skip further traversal for this subtree

					activeRays &= ~hitValue;
					activeMask = _mm_load_ps((float *)maskLUTable[activeRays & ~hitValue]);
				}
				#else

				// test if below error bounds for *any* of the rays:
				int testMask = _mm_movemask_ps(_mm_cmplt_ps(errBnd4, _mm_load_ps(ErrBndForDirLight))) & activeRays & ~hitValue;
				unsigned int idxList = GET_REAL_IDX(currentNode->lodIndex);
				if (testMask) { // || idxList == hitLODIdx[0]) { /// TODO: compare SIMD-wise

					hitValue |= testMask & RayLODIntersectTarget(rays, subObject, currentNode, hits, max, min, intersect_t, hitValue | (testMask ^ ALL_RAYS));
					if (hitValue == ALL_RAYS) {
						return _mm_movemask_ps(_mm_cmpge_ps(intersect_t.v4, _mm_load_ps(tmax)));						
					}					

					goto LOD_END;	// skip further traversal for this subtree
															

				}
				#endif	

			}
			#endif

			const __m128 splitCoord = _mm_load1_ps(&currentNode->splitcoord);
			const __m128 rayOrigin = _mm_load_ps(rays.origin[axis]);
			const __m128 invDirection = _mm_load_ps(rays.invdirection[axis]);

			if (g_Verbose) {
					float temp[4];
					_mm_store_ps(temp, splitCoord);
					cout << "split: " << temp[0] << " " << temp[1] << " " << temp[2] << " " << temp[3] << endl;
					_mm_store_ps(temp, rayOrigin);
					cout << "orig: " << temp[0] << " " << temp[1] << " " << temp[2] << " " << temp[3] << endl;
					_mm_store_ps(temp, invDirection);
					cout << "invDir: " << temp[0] << " " << temp[1] << " " << temp[2] << " " << temp[3] << endl;
				}


			//
			// calculate dist to splitting plane:
			// dist = (currentNode->splitcoord - origin[currentAxis]) * invdirection[currentAxis];
			//

			__m128 dist = _mm_mul_ps(_mm_sub_ps(splitCoord, rayOrigin), invDirection);

			if (g_Verbose) {
					cout << "Axis " << axis << " with splitcoord = " << currentNode->splitcoord << endl;
					float temp[4];
					_mm_store_ps(temp, dist);
					cout << temp[0] << " " << temp[1] << " " << temp[2] << " " << temp[3] << endl;
					_mm_store_ps(temp, min.v4);
					cout << temp[0] << " " << temp[1] << " " << temp[2] << " " << temp[3] << endl;
					_mm_store_ps(temp, max.v4);
					cout << temp[0] << " " << temp[1] << " " << temp[2] << " " << temp[3] << endl;
				}

			// assume far child first
			int childSelect = rays.rayChildOffsets[axis];

			// need to take far child?
			if (_mm_movemask_ps(_mm_and_ps(_mm_cmpgt_ps(dist, _mm_sub_ps(min.v4, epsilon)), activeMask)) == 0) {			
				currentNode = GETNODE(subObject->tree,GETCHILDNUM(currentNode, childSelect));		
				
				if (g_Verbose) 
					cout << "Take far node only!" << endl;
				continue;
			} 
			// or just near child?
			else if (_mm_movemask_ps(_mm_and_ps(_mm_cmplt_ps(dist, _mm_add_ps(max.v4, epsilon)), activeMask)) == 0) { 						
				// node = near node
				currentNode = GETNODE(subObject->tree,GETCHILDNUM(currentNode, childSelect ^ 1));

				if (g_Verbose) 
					cout << "Take near node only!" << endl;
				
				continue;
			} 
			else { // need to intersect both children:				
				const __m128 mask2 = _mm_cmpgt_ps(dist, _mm_sub_ps(min.v4, epsilon));
				const __m128 mask3 = _mm_cmplt_ps(dist, _mm_add_ps(max.v4, epsilon));

				if (g_Verbose) 
					cout << "Take both nodes." << endl;
				

				stack[stackPtr].node = GETCHILDNUM(currentNode, childSelect);
				stack[stackPtr].max.v4 = max.v4;
				
				// update max values:
				max.v4 = _mm_or_ps( _mm_and_ps( mask3, dist ), _mm_andnot_ps( mask3, max.v4 ) );				
								
				stack[stackPtr].min.v4 = _mm_or_ps( _mm_and_ps( mask2, dist ), _mm_andnot_ps( mask2, min.v4 ) );
				stack[stackPtr].mask = _mm_movemask_ps(activeMask);
				max.v4 = dist;
				
				stackPtr++;
				
				activeMask = _mm_cmplt_ps( min.v4, max.v4 );
				activeRays = _mm_movemask_ps(activeMask);

				// select near node
				currentNode = GETNODE(subObject->tree,GETCHILDNUM(currentNode, childSelect ^ 1));	
				
			}
		}

		activeRays = _mm_movemask_ps(activeMask);
		axis = AXIS(currentNode);
		
		bool needcontinue = GETCHILDCOUNT(currentNode) && activeRays > 0;
		if (needcontinue) {
			
			#ifndef VISIBILITY_ONLY


			hitValue |= RayObjIntersect(rays, subObject, currentNode, hits, max.v, min.v, hitValue | (activeRays ^ 15)) & activeRays;

			// test: all rays hit something ?
			if ((hitValue & ALL_RAYS) == ALL_RAYS) { // yes, return ! 
				return hitValue;	
			}

			#else

			hitValue |= RayObjIntersectTarget(rays, subObject, currentNode, max.v, min.v, intersect_t.v, hitValue | (activeRays ^ 15)) & activeRays;

			// test: all rays hit something ?
			if ((hitValue & ALL_RAYS) == ALL_RAYS) { // yes, return !				
				// test intersection values against target values
				// return (intersect_t >= target_t);	
				return _mm_movemask_ps(_mm_cmpge_ps(intersect_t.v4, _mm_load_ps(tmax)));
			}

			#endif

		}

		#ifdef USE_LOD
		LOD_END:	
		#endif
		
		// no, at least one ray needs to be traced further

		// get far child from stack
		stackPtr--;
		currentOffset = stack[stackPtr].node;
		currentNode = GETNODE(subObject->tree,currentOffset);		
		min.v4 = stack[stackPtr].min.v4;
		max.v4 = stack[stackPtr].max.v4;
		activeRays = stack[stackPtr].mask;
		
	}
	
//#if !defined(VISIBILITY_ONLY) && !defined(TRAVERSE_OFFSET)
}
//#endif

#endif