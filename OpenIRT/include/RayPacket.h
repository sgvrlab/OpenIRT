#pragma once

#include "HitpointInfo.h"
#include "BV.h"
#include "SIMDRay.h"

#define RayPacketTemplate template <int nRays, bool directionsMatch, bool sameOrigin, bool hasCornerRays> 
#define RayPacketT RayPacket<nRays,directionsMatch,sameOrigin,hasCornerRays>

namespace irt
{

RayPacketTemplate
class RayPacket
{
public:	
	RayPacket() {
		numRaysHit = 0;
		memset(rayHasHit, 0, sizeof(char)*nRays);		

		for (int r = 0; r < nRays; r++) {
			hitpoints[r].t[0] = FLT_MAX;
			hitpoints[r].t[1] = FLT_MAX;
			hitpoints[r].t[2] = FLT_MAX;
			hitpoints[r].t[3] = FLT_MAX;
		}
	}
	
	/**
	 * Generate primary ray packet from camera information.
	 */
	void setupForPrimaryRays(Vector3 origin, Vector3 cornerLU, Vector3 across, Vector3 up, 
		                     int raysHorizontal, int raysVertical,
						     float start_a, float start_b,
						     float delta_a, float delta_b,
							 float jitter[][2]) {
		
	    //
	    // Part 1: make eye rays
        //

		this->origin = origin;

		// init SIMDRay origins
		register __m128 xcoords = _mm_set1_ps(this->origin[0]);
		register __m128 ycoords = _mm_set1_ps(this->origin[1]);
		register __m128 zcoords = _mm_set1_ps(this->origin[2]);		
		
		if (hasCornerRays) {
			_mm_store_ps(cornerRays.origin[0], xcoords);
			_mm_store_ps(cornerRays.origin[1], ycoords);
			_mm_store_ps(cornerRays.origin[2], zcoords);
		}

		// store in all rays
		for (int r = 0; r < nRays; r++) {
			_mm_store_ps(rays[r].origin[0], xcoords);
			_mm_store_ps(rays[r].origin[1], ycoords);
			_mm_store_ps(rays[r].origin[2], zcoords);
		}

		// initialize real rays:
		int rayNum = 0;
		SIMDVec4 current_a, current_b;
		float delta_a_2 = 2.0f * delta_a;
		float delta_b_2 = 2.0f * delta_b;

		int realRayNum = 0;
		SIMDVec4 final_a, final_b;

		current_b.v[0] = start_b;
		current_b.v[1] = start_b;
		current_b.v[2] = start_b + delta_b;
		current_b.v[3] = start_b + delta_b;

		for (int y = 0; y < raysVertical; y++) {
			current_a.v[0] = start_a;
			current_a.v[1] = start_a + delta_a;
			current_a.v[2] = start_a;
			current_a.v[3] = start_a + delta_a;
			
			for (int x = 0; x < raysHorizontal; x++) {
				//rays[rayNum].makeRays(cornerLU, across, up, origin, current_a, current_b);
				final_a.v[0] = current_a.v[0] + jitter[realRayNum][0]*delta_a;
				final_b.v[0] = current_b.v[0] + jitter[realRayNum++][1]*delta_b;
				final_a.v[1] = current_a.v[1] + jitter[realRayNum][0]*delta_a;
				final_b.v[1] = current_b.v[1] + jitter[realRayNum++][1]*delta_b;
				final_a.v[2] = current_a.v[2] + jitter[realRayNum][0]*delta_a;
				final_b.v[2] = current_b.v[2] + jitter[realRayNum++][1]*delta_b;
				final_a.v[3] = current_a.v[3] + jitter[realRayNum][0]*delta_a;
				final_b.v[3] = current_b.v[3] + jitter[realRayNum++][1]*delta_b;

				rays[rayNum].makeRays(cornerLU, across, up, origin, final_a, final_b);
				current_a.v4 = _mm_add_ps(current_a.v4, _mm_load1_ps(&delta_a_2));
				rayNum++;
			}

			current_b.v4 = _mm_add_ps(current_b.v4, _mm_load1_ps(&delta_b_2));
		}

		// 
		// Part 2: make frustum
		// 

		Vector3 dirs[4];
		/*
		// left upper
		dirs[0] = rays[0].getDirection(0);
		// right upper
		dirs[1] = rays[raysHorizontal-1].getDirection(1);
		// left lower
		dirs[2] = rays[raysHorizontal*(raysVertical-1)].getDirection(2);
		// right lower
		dirs[3] = rays[raysHorizontal*raysVertical - 1].getDirection(3);
		*/
		for(int i=0;i<4;i++)
		{
			Vector2 pos;
			switch(i)
			{
			case 0: pos.e[0] = start_a; pos.e[1] = start_b; break;
			case 1: pos.e[0] = start_a + delta_a_2*raysHorizontal; pos.e[1] = start_b; break;
			case 2: pos.e[0] = start_a; pos.e[1] = start_b + delta_b_2*raysVertical; break;
			case 3: pos.e[0] = start_a + delta_a_2*raysHorizontal; pos.e[1] = start_b + delta_b_2*raysVertical; break;
			}
			// target = corner + across*a[i] + up*b[i];
			// direction = target - center
			dirs[i] = (cornerLU + across*pos.e[0] + up*pos.e[1]) - origin;
		}

		if (hasCornerRays) {
			// also set in ray packet representing corner
			cornerRays.setDirections(dirs);
			int directionSigns = ((dirs[0][0] > 0)?1:0) | 
				                 ((dirs[0][1] > 0)?2:0) | 
								 ((dirs[0][2] > 0)?4:0);
			directionMask = _mm_load_ps((float *)maskLUTable[directionSigns]);
		}
			
		setFrustumFromCorners(dirs[0], dirs[1], dirs[2], dirs[3]);		
	}

	void setupForPrimaryRays(Vector3 origin, Vector3 cornerLU, Vector3 across, Vector3 up, 
		                     int raysHorizontal, int raysVertical,
							 float pos[][2]) {
		
	    //
	    // Part 1: make eye rays
        //

		this->origin = origin;

		// init SIMDRay origins
		register __m128 xcoords = _mm_set1_ps(this->origin[0]);
		register __m128 ycoords = _mm_set1_ps(this->origin[1]);
		register __m128 zcoords = _mm_set1_ps(this->origin[2]);		
		
		if (hasCornerRays) {
			_mm_store_ps(cornerRays.origin[0], xcoords);
			_mm_store_ps(cornerRays.origin[1], ycoords);
			_mm_store_ps(cornerRays.origin[2], zcoords);
		}

		// store in all rays
		for (int r = 0; r < nRays; r++) {
			_mm_store_ps(rays[r].origin[0], xcoords);
			_mm_store_ps(rays[r].origin[1], ycoords);
			_mm_store_ps(rays[r].origin[2], zcoords);
		}

		// initialize real rays:
		int rayNum = 0;
		SIMDVec4 current_a, current_b;

		int realRayNum = 0;

		for (int y = 0; y < raysVertical; y++) {
			for (int x = 0; x < raysHorizontal; x++) {
				current_a.v[0] = pos[realRayNum][0];
				current_b.v[0] = pos[realRayNum++][1];
				current_a.v[1] = pos[realRayNum][0];
				current_b.v[1] = pos[realRayNum++][1];
				current_a.v[2] = pos[realRayNum][0];
				current_b.v[2] = pos[realRayNum++][1];
				current_a.v[3] = pos[realRayNum][0];
				current_b.v[3] = pos[realRayNum++][1];
				rays[rayNum].makeRays(cornerLU, across, up, origin, current_a, current_b);
				rayNum++;
			}
		}

		// 
		// Part 2: make frustum
		// 

		Vector3 dirs[4];

		// left upper
		dirs[0] = rays[0].getDirection(0);
		// right upper
		dirs[1] = rays[raysHorizontal-1].getDirection(1);
		// left lower
		dirs[2] = rays[raysHorizontal*(raysVertical-1)].getDirection(2);
		// right lower
		dirs[3] = rays[raysHorizontal*raysVertical - 1].getDirection(3);

		if (hasCornerRays) {
			// also set in ray packet representing corner
			cornerRays.setDirections(dirs);
			int directionSigns = ((dirs[0][0] > 0)?1:0) | 
				                 ((dirs[0][1] > 0)?2:0) | 
								 ((dirs[0][2] > 0)?4:0);
			directionMask = _mm_load_ps((float *)maskLUTable[directionSigns]);
		}
			
		setFrustumFromCorners(dirs[0], dirs[1], dirs[2], dirs[3]);		
	}

	/**
	 * Generate frustum information for rays that were saved in this ray packet,
	 * assuming that all rays share the same origin
	 */
	FORCEINLINE bool setupFrustumFromRays() {		
		// AABB for origins and ray destinations
		AABB4 rayOrigins, rayDestinations;

		// for each ray in packet:
		// update bounding box
		for (int r = firstHitRay; r < nRays; r++) {
			__m128 d[4];
			d[0] = _mm_load_ps(rays[r].direction[0]);
			d[1] = _mm_load_ps(rays[r].direction[1]);
			d[2] = _mm_load_ps(rays[r].direction[2]);
			d[3] = _mm_setzero_ps();
			_MM_TRANSPOSE4_PS(d[0], d[1], d[2], d[3]);
			
			if ((rayHasHit[r] & 1) == 0)
				rayDestinations.Update(d[0]);
			if ((rayHasHit[r] & 2) == 0)
				rayDestinations.Update(d[1]);
			if ((rayHasHit[r] & 4) == 0)
				rayDestinations.Update(d[2]);
			if ((rayHasHit[r] & 8) == 0)
				rayDestinations.Update(d[3]);
		}
		
		origin = rays[firstHitRay].getOrigin(0);

		Vector3 dirLU, dirRU, dirLB, dirRB;		
		int axis;
		Vector3 axisDistance;
		for (axis = 0; axis < 3; axis++) {
			float min = rayDestinations.GetBBMin()[axis];
			float max = rayDestinations.GetBBMax()[axis];
			if ((min > 0.0f && max > 0.0f) || (min < 0.0f && max < 0.0f)) {
				axisDistance[axis] = (min > 0.0f)?min:fabs(max);
			}
			else
				axisDistance[axis] = 0.0f;
		}
		
		axis = axisDistance.indexOfMaxComponent();

		// this cannot be a valid frustum!
		// (angles of directions of this frustum would be > PI)
		if (axisDistance[0] == 0.0f && axisDistance[1] == 0.0f && axisDistance[2] == 0.0f)
			return false;

		//int minmaxSelect = (rayDestinations.GetBBMin().e[axis] > 0.0f)?0:1;

		Vector3 n_topbottom, n_leftright;
		switch(axis) 
		{
			case 0:
				// sweep plane around n=(0 1 0) for left/right and n=(0 0 1) for top/bottom
				n_leftright = Vector3(0,1,0);
				n_topbottom = Vector3(0,0,1);
				break;
				
			case 1:
				// sweep plane around n=(0 0 1) for left/right and n=(1 0 0) for top/bottom
				n_leftright = Vector3(0,0,1);
				n_topbottom = Vector3(1,0,0);
				break;

			case 2:
				// sweep plane around n=(0 1 0) for left/right and n=(1 0 0) for top/bottom
				n_leftright = Vector3(0,1,0);
				n_topbottom = Vector3(1,0,0);
				break;
		}

		Vector3 bestTop = rayDestinations.GetBBMin(),
				bestBottom = rayDestinations.GetBBMin(),
				bestLeft = rayDestinations.GetBBMin(),
				bestRight = rayDestinations.GetBBMin();

		// for each point of the AABB (except first, since we 
		// initialized with that):
		for (int i = 1; i < 8; i++) {
			Vector3 point(rayDestinations.bb[i & 1].v[0], rayDestinations.bb[(i>>1) & 1].v[1], rayDestinations.bb[(i >> 2) & 1].v[2]);
			Vector3 n_up = cross(n_topbottom, point);
			Vector3 n_right = cross(n_leftright, point);

			if (dot(bestTop, n_up) > 0)
				bestTop = point;
			if (dot(bestBottom, n_up) < 0)
				bestBottom = point;
			if (dot(bestLeft, n_right) < 0)
				bestLeft = point;
			if (dot(bestRight, n_right) > 0)
				bestRight = point;
		}

		// top
		Vector3 normals[4];
		normals[0] = cross(bestTop, n_topbottom);
		// bottom
		normals[1] = cross(n_topbottom, bestBottom);
		// left
		normals[2] = cross(n_leftright, bestLeft);
		// right
		normals[3] = cross(bestRight, n_leftright);

		for (int axis = 0; axis < 3; axis++) {	// axis	
			for (int i = 0; i < 4; i++) {		// plane
				plane_normals[0][axis].v[i] = max(normals[i][axis], 0.0f);	
				plane_normals[1][axis].v[i] = min(normals[i][axis], 0.0f);
			}
		}
		
		return true;

	}


	FORCEINLINE bool setupFrustumFromRaysGeneralOrigin() {		
		// AABB for origins and ray destinations
		aabbox rayOrigins, rayDestinations;

		// for each ray in packet:
		// update bounding box
		for (int r = firstHitRay; r < nRays; r++) {

			switch (rayHasHit[r]) {
				case 0:
					rayDestinations.SetBB(rays[r].direction[0][0], rays[r].direction[1][0], rays[r].direction[2][0]);
					rayDestinations.Update(rays[r].direction[0][1], rays[r].direction[1][1], rays[r].direction[2][1]);
					rayDestinations.Update(rays[r].direction[0][2], rays[r].direction[1][2], rays[r].direction[2][2]);
					rayDestinations.Update(rays[r].direction[0][3], rays[r].direction[1][3], rays[r].direction[2][3]);

					rayOrigins.SetBB(rays[r].origin[0][0], rays[r].origin[1][0], rays[r].origin[2][0]);
					rayOrigins.Update(rays[r].origin[0][1], rays[r].origin[1][1], rays[r].origin[2][1]);
					rayOrigins.Update(rays[r].origin[0][2], rays[r].origin[1][2], rays[r].origin[2][2]);
					rayOrigins.Update(rays[r].origin[0][3], rays[r].origin[1][3], rays[r].origin[2][3]);
					break;
				case 1:
					rayDestinations.SetBB(rays[r].direction[0][1], rays[r].direction[1][1], rays[r].direction[2][1]);
					rayDestinations.Update(rays[r].direction[0][2], rays[r].direction[1][2], rays[r].direction[2][2]);
					rayDestinations.Update(rays[r].direction[0][3], rays[r].direction[1][3], rays[r].direction[2][3]);

					rayOrigins.SetBB(rays[r].origin[0][1], rays[r].origin[1][1], rays[r].origin[2][1]);
					rayOrigins.Update(rays[r].origin[0][2], rays[r].origin[1][2], rays[r].origin[2][2]);
					rayOrigins.Update(rays[r].origin[0][3], rays[r].origin[1][3], rays[r].origin[2][3]);
					break;
				case 2:
					rayDestinations.SetBB(rays[r].direction[0][0], rays[r].direction[1][0], rays[r].direction[2][0]);
					rayDestinations.Update(rays[r].direction[0][2], rays[r].direction[1][2], rays[r].direction[2][2]);
					rayDestinations.Update(rays[r].direction[0][3], rays[r].direction[1][3], rays[r].direction[2][3]);

					rayOrigins.SetBB(rays[r].origin[0][0], rays[r].origin[1][0], rays[r].origin[2][0]);
					rayOrigins.Update(rays[r].origin[0][2], rays[r].origin[1][2], rays[r].origin[2][2]);
					rayOrigins.Update(rays[r].origin[0][3], rays[r].origin[1][3], rays[r].origin[2][3]);
					break;
				case 3: 
					rayDestinations.SetBB(rays[r].direction[0][2], rays[r].direction[1][2], rays[r].direction[2][2]);
					rayDestinations.Update(rays[r].direction[0][3], rays[r].direction[1][3], rays[r].direction[2][3]);

					rayOrigins.SetBB(rays[r].origin[0][2], rays[r].origin[1][2], rays[r].origin[2][2]);
					rayOrigins.Update(rays[r].origin[0][3], rays[r].origin[1][3], rays[r].origin[2][3]);
					break;
				case 4:
					rayDestinations.SetBB(rays[r].direction[0][0], rays[r].direction[1][0], rays[r].direction[2][0]);
					rayDestinations.Update(rays[r].direction[0][1], rays[r].direction[1][1], rays[r].direction[2][1]);
					rayDestinations.Update(rays[r].direction[0][3], rays[r].direction[1][3], rays[r].direction[2][3]);

					rayOrigins.SetBB(rays[r].origin[0][0], rays[r].origin[1][0], rays[r].origin[2][0]);
					rayOrigins.Update(rays[r].origin[0][1], rays[r].origin[1][1], rays[r].origin[2][1]);
					rayOrigins.Update(rays[r].origin[0][3], rays[r].origin[1][3], rays[r].origin[2][3]);
					break;
				case 5:
					rayDestinations.SetBB(rays[r].direction[0][1], rays[r].direction[1][1], rays[r].direction[2][1]);
					rayDestinations.Update(rays[r].direction[0][3], rays[r].direction[1][3], rays[r].direction[2][3]);

					rayOrigins.SetBB(rays[r].origin[0][1], rays[r].origin[1][1], rays[r].origin[2][1]);
					rayOrigins.Update(rays[r].origin[0][3], rays[r].origin[1][3], rays[r].origin[2][3]);
					break;
				case 6:
					rayDestinations.SetBB(rays[r].direction[0][0], rays[r].direction[1][0], rays[r].direction[2][0]);
					rayDestinations.Update(rays[r].direction[0][3], rays[r].direction[1][3], rays[r].direction[2][3]);

					rayOrigins.SetBB(rays[r].origin[0][0], rays[r].origin[1][0], rays[r].origin[2][0]);
					rayOrigins.Update(rays[r].origin[0][3], rays[r].origin[1][3], rays[r].origin[2][3]);
					break;
				case 7:
					rayDestinations.SetBB(rays[r].direction[0][3], rays[r].direction[1][3], rays[r].direction[2][3]);

					rayOrigins.SetBB(rays[r].origin[0][3], rays[r].origin[1][3], rays[r].origin[2][3]);
					break;
				case 8:
					rayDestinations.SetBB(rays[r].direction[0][0], rays[r].direction[1][0], rays[r].direction[2][0]);
					rayDestinations.Update(rays[r].direction[0][1], rays[r].direction[1][1], rays[r].direction[2][1]);
					rayDestinations.Update(rays[r].direction[0][2], rays[r].direction[1][2], rays[r].direction[2][2]);

					rayOrigins.SetBB(rays[r].origin[0][0], rays[r].origin[1][0], rays[r].origin[2][0]);
					rayOrigins.Update(rays[r].origin[0][1], rays[r].origin[1][1], rays[r].origin[2][1]);
					rayOrigins.Update(rays[r].origin[0][2], rays[r].origin[1][2], rays[r].origin[2][2]);
					break;
				case 9:
					rayDestinations.SetBB(rays[r].direction[0][1], rays[r].direction[1][1], rays[r].direction[2][1]);
					rayDestinations.Update(rays[r].direction[0][2], rays[r].direction[1][2], rays[r].direction[2][2]);

					rayOrigins.SetBB(rays[r].origin[0][1], rays[r].origin[1][1], rays[r].origin[2][1]);
					rayOrigins.Update(rays[r].origin[0][2], rays[r].origin[1][2], rays[r].origin[2][2]);
					break;
				case 10:
					rayDestinations.SetBB(rays[r].direction[0][0], rays[r].direction[1][0], rays[r].direction[2][0]);
					rayDestinations.Update(rays[r].direction[0][2], rays[r].direction[1][2], rays[r].direction[2][2]);

					rayOrigins.SetBB(rays[r].origin[0][0], rays[r].origin[1][0], rays[r].origin[2][0]);
					rayOrigins.Update(rays[r].origin[0][2], rays[r].origin[1][2], rays[r].origin[2][2]);
					break;
				case 11:
					rayDestinations.SetBB(rays[r].direction[0][2], rays[r].direction[1][2], rays[r].direction[2][2]);

					rayOrigins.SetBB(rays[r].origin[0][2], rays[r].origin[1][2], rays[r].origin[2][2]);
					break;
				case 12:
					rayDestinations.SetBB(rays[r].direction[0][0], rays[r].direction[1][0], rays[r].direction[2][0]);
					rayDestinations.Update(rays[r].direction[0][1], rays[r].direction[1][1], rays[r].direction[2][1]);

					rayOrigins.SetBB(rays[r].origin[0][0], rays[r].origin[1][0], rays[r].origin[2][0]);
					rayOrigins.Update(rays[r].origin[0][1], rays[r].origin[1][1], rays[r].origin[2][1]);

					break;
				case 13:
					rayDestinations.SetBB(rays[r].direction[0][1], rays[r].direction[1][1], rays[r].direction[2][1]);

					rayOrigins.SetBB(rays[r].origin[0][1], rays[r].origin[1][1], rays[r].origin[2][1]);
					break;
				case 14:
					rayDestinations.SetBB(rays[r].direction[0][0], rays[r].direction[1][0], rays[r].direction[2][0]);

					rayOrigins.SetBB(rays[r].origin[0][0], rays[r].origin[1][0], rays[r].origin[2][0]);
					break;					
			}			
		}

		// TODO: works only for common ray origin!
		origin = rays[firstHitRay].getOrigin(0);

		Vector3 dirLU, dirRU, dirLB, dirRB;		
		int axis;
		Vector3 axisDistance;
		for (axis = 0; axis < 3; axis++) {
			float min = rayDestinations.GetBBMin()[axis];
			float max = rayDestinations.GetBBMax()[axis];
			if ((min > 0.0f && max > 0.0f) || (min < 0.0f && max < 0.0f)) {
				axisDistance[axis] = (min > 0.0f)?min:fabs(max);
			}
			else
				axisDistance[axis] = 0.0f;
		}

		axis = axisDistance.indexOfMaxComponent();

		// this cannot be a valid frustum!
		// (angles of directions of this frustum would be > PI)
		if (axisDistance[0] == 0.0f && axisDistance[1] == 0.0f && axisDistance[2] == 0.0f)
			return false;

		Vector3 n_topbottom, n_leftright;
		switch(axis) 
		{
		case 0:
			// sweep plane around n=(0 1 0) for left/right and n=(0 0 1) for top/bottom
			n_leftright = Vector3(0,1,0);
			n_topbottom = Vector3(0,0,1);
			break;

		case 1:
			// sweep plane around n=(0 0 1) for left/right and n=(1 0 0) for top/bottom
			n_leftright = Vector3(0,0,1);
			n_topbottom = Vector3(1,0,0);
			break;

		case 2:
			// sweep plane around n=(0 1 0) for left/right and n=(1 0 0) for top/bottom
			n_leftright = Vector3(0,1,0);
			n_topbottom = Vector3(1,0,0);
			break;
		}

		Vector3 bestTop = rayDestinations.bb[0],
			bestBottom = rayDestinations.bb[0],
			bestLeft = rayDestinations.bb[0],
			bestRight = rayDestinations.bb[0];

		// for each point of the destination AABB (except first, since we 
		// initialized with that):
		for (int i = 1; i < 8; i++) {
			Vector3 point(rayDestinations.bb[i & 1].x(), rayDestinations.bb[(i>>1) & 1].y(), rayDestinations.bb[(i >> 2) & 1].z());
			Vector3 n_up = cross(n_topbottom, point);
			Vector3 n_right = cross(n_leftright, point);

			if (dot(bestTop, n_up) > 0)
				bestTop = point;
			if (dot(bestBottom, n_up) < 0)
				bestBottom = point;
			if (dot(bestLeft, n_right) < 0)
				bestLeft = point;
			if (dot(bestRight, n_right) > 0)
				bestRight = point;
		}

		// top
		Vector3 normals[4];
		normals[0] = cross(bestTop, n_topbottom);
		// bottom
		normals[1] = cross(n_topbottom, bestBottom);
		// left
		normals[2] = cross(n_leftright, bestLeft);
		// right
		normals[3] = cross(bestRight, n_leftright);

		for (int axis = 0; axis < 3; axis++) {	// axis	
			for (int i = 0; i < 4; i++) {		// plane
				plane_normals[0][axis].v[i] = max(normals[i][axis], 0.0f);	
				plane_normals[1][axis].v[i] = min(normals[i][axis], 0.0f);
			}
		}

		return true;

	}

	FORCEINLINE void setFrustumFromCorners(Vector3 &dirLU, Vector3 &dirRU, Vector3 &dirLB, Vector3 &dirRB) {
		Vector3 normals[4];
		int i;

		// top
		normals[0] = cross(dirRU, dirLU);
		// bottom
		normals[1] = cross(dirLB, dirRB);
		// left
		normals[2] = cross(dirLU, dirLB);
		// right
		normals[3] = cross(dirRB, dirRU);

		for (int axis = 0; axis < 3; axis++) {	// axis	
			for (i = 0; i < 4; i++) {			// plane
				plane_normals[0][axis].v[i] = max(normals[i][axis], 0.0f);	
				plane_normals[1][axis].v[i] = min(normals[i][axis], 0.0f);
			}			
		}
	}

	/**
	 * Intersects the Ray Packet with an axis-aligned box defined by the two
	 * vectors. Returns index of first intersecting ray in packet if there is an intersection.
	 **/
	FORCEINLINE int intersectWithBox(const Vector3 bb_orig[2], int firstActiveRay) const {		
		SIMDVec4 min, max;

		if (sameOrigin) {
			// need to transform AABB to frustum coordinate space because
			// we assume that d=0 for all normals in frustum			
			
			register Vector3 bb[2];
			bb[0] = bb_orig[0] - origin;
			bb[1] = bb_orig[1] - origin;

			/*
			const __m128 bb0 = _mm_sub_ps(bb_orig[0].e4, origin.e4);
			const __m128 bb1 = _mm_sub_ps(bb_orig[1].e4, origin.e4);
			
			const __m128 bb_min = _mm_or_ps(_mm_and_ps(directionMask, bb0), _mm_andnot_ps(directionMask, bb1));
			const __m128 bb_max = _mm_or_ps(_mm_andnot_ps(directionMask, bb0), _mm_and_ps(directionMask, bb1));								
			*/
			
			// test first ray:
			if (rays[firstActiveRay].RayBoxIntersectLocal<directionsMatch>(bb, min, max, hitpoints[firstActiveRay].t.e4))
			{
				return firstActiveRay;
			}

			
			// test frustum:
			if (!intersectFrustumWithBoxLocal(bb)) 
				return nRays; // if frustum does not hit, early exit			

			// first ray did not hit, try whole packet		
			for (int r = firstActiveRay+1; r < nRays; r++) {
				if (rays[r].RayBoxIntersectLocal<directionsMatch>(bb, min, max, hitpoints[r].t.e4))				
				{				
						return r;
				}
			}

			// did not hit
			return nRays;

			/*
				// need to transform AABB to frustum coordinate space because
				// we assume that d=0 for all normals in frustum
				register Vector4 bb[2];
				bb[0] = bb_orig[0] - origin;
				bb[1] = bb_orig[1] - origin;			
				
				// test first ray:
				if (rays[firstActiveRay].RayBoxIntersectLocal<directionsMatch>(bb, min, max, hitpoints[firstActiveRay].t.e4))
				{
					return firstActiveRay;						
				}

				// test frustum:
				if (!intersectFrustumWithBoxLocal(bb)) 
					return nRays; // if frustum does not hit, early exit			

				// first ray did not hit, try whole packet		
				for (int r = firstActiveRay+1; r < nRays; r++) {
					if (rays[r].RayBoxIntersectLocal<directionsMatch>(bb, min, max, hitpoints[r].t.e4))				
					{				
							return r;
					}
				}

				// did not hit
				return nRays;
			*/
		}
		else {
			//
			// not same origin, need to do full traversal:
			//

			for (int r = firstActiveRay; r < nRays; r++) {
				if (rays[r].RayBoxIntersect(bb_orig, min, max))
				{				
						return r;
				}
			}

			return nRays;
		}
	}
		
	/**
	 * Intersects the frustum with an axis-aligned box defined by the two
	 * points. Returns true if there is an intersection.
	 **/
	FORCEINLINE bool intersectFrustumWithBox(const Vector3 bb[2]) const {		
		// need to transform AABB to frustum coordinate space because
		// we assume that d=0 for all normals below
		Vector3 bb_min = bb[0] - origin;
		Vector3 bb_max = bb[1] - origin;

		// intersect the axis-aligned box with all four frustum faces
		// in parallel, see explanation in multi-level ray tracing paper		

		__m128 bmin = _mm_set1_ps(bb_min[0]);
		__m128 bmax = _mm_set1_ps(bb_max[0]);
		bmin = _mm_mul_ps(bmin, plane_normals[0][0].v4);
		bmax = _mm_mul_ps(bmax, plane_normals[1][0].v4);
		__m128 nplane = _mm_add_ps(bmin, bmax);
		
		bmin = _mm_set1_ps(bb_min[1]);
		bmax = _mm_set1_ps(bb_max[1]);
		bmin = _mm_mul_ps(bmin, plane_normals[0][1].v4);
		bmax = _mm_mul_ps(bmax, plane_normals[1][1].v4);
		bmin = _mm_add_ps(bmin, bmax);
		nplane = _mm_add_ps(nplane, bmin);	
		bmin = _mm_set1_ps(bb_min[2]);
		bmax = _mm_set1_ps(bb_max[2]);
		bmin = _mm_mul_ps(bmin, plane_normals[0][2].v4);
		bmax = _mm_mul_ps(bmax, plane_normals[1][2].v4);
		bmin = _mm_add_ps(bmin, bmax);
		nplane = _mm_add_ps(nplane, bmin);	

		const int signs = _mm_movemask_ps( nplane );

		// if signs == ALL_RAYS, then the box intersects with this frustum/beam
		// because we could not find an dividing plane
		return signs == ALL_RAYS;		
	}

	/**
	 * Intersects the frustum with an axis-aligned box defined by the two
	 * points. Returns true if there is an intersection.
	 * (this version assumes that the origin has already been subtracted 
	 *	from the bounding box coordinates!)
	 **/
	FORCEINLINE bool intersectFrustumWithBoxLocal(const Vector3 bb[2]) const {				
		// intersect the axis-aligned box with all four frustum faces
		// in parallel, see explanation in multi-level ray tracing paper		
		
		//__m128 bmin = _mm_shuffle_ps(bb_min, bb_min, _MM_SHUFFLE(0,0,0,0)); 
		//__m128 bmax = _mm_shuffle_ps(bb_max, bb_max, _MM_SHUFFLE(0,0,0,0));//
		__m128 bmin = _mm_set1_ps(bb[0][0]);
		__m128 bmax = _mm_set1_ps(bb[1][0]);
		bmin = _mm_mul_ps(bmin, plane_normals[0][0].v4);
		bmax = _mm_mul_ps(bmax, plane_normals[1][0].v4);
		__m128 nplane = _mm_add_ps(bmin, bmax);
		
		//bmin = _mm_shuffle_ps(bb_min, bb_min, _MM_SHUFFLE(1,1,1,1)); //_mm_set1_ps(bb[0][0]);
		//bmax = _mm_shuffle_ps(bb_max, bb_max, _MM_SHUFFLE(1,1,1,1));//_mm_set1_ps(bb[1][0]);
		
		bmin = _mm_set1_ps(bb[0][1]);
		bmax = _mm_set1_ps(bb[1][1]);
		bmin = _mm_mul_ps(bmin, plane_normals[0][1].v4);
		bmax = _mm_mul_ps(bmax, plane_normals[1][1].v4);
		bmin = _mm_add_ps(bmin, bmax);
		nplane = _mm_add_ps(nplane, bmin);	
		//bmin = _mm_shuffle_ps(bb_min, bb_min, _MM_SHUFFLE(2,2,2,2)); //_mm_set1_ps(bb[0][0]);
		//bmax = _mm_shuffle_ps(bb_max, bb_max, _MM_SHUFFLE(2,2,2,2));//_mm_set1_ps(bb[1][0]);		
		bmin = _mm_set1_ps(bb[0][2]);
		bmax = _mm_set1_ps(bb[1][2]);
		bmin = _mm_mul_ps(bmin, plane_normals[0][2].v4);
		bmax = _mm_mul_ps(bmax, plane_normals[1][2].v4);
		bmin = _mm_add_ps(bmin, bmax);
		nplane = _mm_add_ps(nplane, bmin);	

		const int signs = _mm_movemask_ps( nplane );

		// if signs == ALL_RAYS, then the box intersects with this frustum/beam
		// because we could not find an dividing plane
		return signs == ALL_RAYS;		
	}

	/**
	 * Intersects the beam with an axis-aligned plane defined by pos[axis]=coord
	 * and returns true when the bounding rectangle on the plane is outside the node
	 * bounding box.
	 */
	FORCEINLINE void intersectWithskdNode(int axis, float coord[2], const Vector3 bb[2], bool &hit1, bool &hit2, __m128 &t1, __m128&t2) {
		static int axes[3][2] = { {1,2}, {0,2}, {0,1} }; // LUT to other axes by primary axis
		int i1 = axes[axis][0];
		int i2 = axes[axis][1];

		// calculate ray distances t to the plane:
		__m128 origins = _mm_load_ps(cornerRays.origin[axis]);
		__m128 directions = _mm_load_ps(cornerRays.invdirection[axis]);

		//const __m128 coord1 = _mm_set1_ps(coord[0]);
		//const __m128 coord2 = _mm_set1_ps(coord[1]);

		//__m128 maxSelector = _mm_cmpge_ps(directions, _mm_setzero_ps());
		//const __m128 coord_near = _mm_or_ps(_mm_and_ps(maxSelector, coord1), _mm_andnot_ps(maxSelector, coord2)); 
		//const __m128 coord_far  = _mm_or_ps(_mm_andnot_ps(maxSelector, coord1), _mm_and_ps(maxSelector, coord2)); 
		
		const __m128 coord_near = _mm_set1_ps(coord[cornerRays.rayChildOffsets[axis] ^ 1]);
		const __m128 coord_far = _mm_set1_ps(coord[cornerRays.rayChildOffsets[axis]]);

		t1 = _mm_mul_ps(_mm_sub_ps(coord_near, origins), directions);		
		t2 = _mm_mul_ps(_mm_sub_ps(coord_far, origins), directions);

		__m128 hitpoints1, hitpoints2, 
			   minvals, maxvals, 
			   directionsigns, compare_min, compare_max;

		// calculate hitpoints for axis 1:
		directions = _mm_load_ps(cornerRays.direction[i1]);
		directionsigns = _mm_cmpge_ps(directions, _mm_setzero_ps());
		origins = _mm_load_ps(cornerRays.origin[i1]);

		hitpoints1 = _mm_add_ps(_mm_mul_ps(t1, directions), origins);
		hitpoints2 = _mm_add_ps(_mm_mul_ps(t2, directions), origins);

		// compare the hitpoints to the bounding box:
		// when all values are either smaller or larger
		// than the bounding box intervals, then the 
		// 
		
		minvals = _mm_set1_ps(bb[0].e[i1]);
		maxvals = _mm_set1_ps(bb[1].e[i1]);

		compare_min = _mm_cmplt_ps(hitpoints1, minvals);
		compare_max = _mm_cmpgt_ps(hitpoints1, maxvals);

		// if ((!cornerRays.rayChildOffsets[i1] && compares_max) || (cornerRays.rayChildOffsets[i1] && compares_min))	
		//hit1 = _mm_movemask_ps(_mm_or_ps(_mm_or_ps(_mm_andnot_ps(directionsigns, compare_max), 
		//	                                       _mm_and_ps(directionsigns, compare_min)),
		//								 _mm_cmpgt_ps(t1, _mm_setzero_ps()))) != ALL_RAYS;
		hit1 = _mm_movemask_ps(_mm_or_ps(_mm_andnot_ps(directionsigns, compare_max), 
			                             _mm_and_ps(directionsigns, compare_min))) != ALL_RAYS;

		compare_min = _mm_cmplt_ps(hitpoints2, minvals);
		compare_max = _mm_cmpgt_ps(hitpoints2, maxvals);
		
		//if ((cornerRays.rayChildOffsets[i1] && compares_max) || (!cornerRays.rayChildOffsets[i1] && compares_min))		
		hit2 = _mm_movemask_ps(_mm_or_ps(_mm_and_ps(directionsigns, compare_max), 
			                             _mm_andnot_ps(directionsigns, compare_min))) != ALL_RAYS;

		// no hit?
		if (!hit1 && !hit2)
			return;	

		// calculate hitpoints for axis 2:
		directions = _mm_load_ps(cornerRays.direction[i2]);
		directionsigns = _mm_cmpge_ps(directions, _mm_setzero_ps());
		origins = _mm_load_ps(cornerRays.origin[i2]);
	

		hitpoints1 = _mm_add_ps(_mm_mul_ps(t1, directions), origins);		
		hitpoints2 = _mm_add_ps(_mm_mul_ps(t2, directions), origins);

		minvals = _mm_set1_ps(bb[0].e[i2]);
		maxvals = _mm_set1_ps(bb[1].e[i2]);

		compare_min = _mm_cmplt_ps(hitpoints1, minvals);
		compare_max = _mm_cmpgt_ps(hitpoints1, maxvals);

		// if ((!cornerRays.rayChildOffsets[i1] && compares_max) || (cornerRays.rayChildOffsets[i1] && compares_min))	
		if (hit1)
			hit1 = _mm_movemask_ps(_mm_or_ps(_mm_andnot_ps(directionsigns, compare_max), _mm_and_ps(directionsigns, compare_min))) != ALL_RAYS;

		compare_min = _mm_cmplt_ps(hitpoints2, minvals);
		compare_max = _mm_cmpgt_ps(hitpoints2, maxvals);
		
		//if ((cornerRays.rayChildOffsets[i1] && compares_max) || (!cornerRays.rayChildOffsets[i1] && compares_min))		
		if (numRaysHit == nRays && _mm_movemask_ps(_mm_cmpgt_ps(t2, maxIntersectT)) == 15)
			hit2 = false;
		else if (hit2)
			hit2 = _mm_movemask_ps(_mm_or_ps(_mm_and_ps(directionsigns, compare_max), _mm_andnot_ps(directionsigns, compare_min))) != ALL_RAYS;

		
	}

	int hasMatchingDirections() {
		return cornerRays.directionsMatch();
	}

	// Checks how many rays in the packet hit and which is the first one.
	// This is *not* efficient and should only be called after the traversal
	// (which keeps those stats separately) to find out which rays to shade.
	void calculateHitStats() {				
		int back = nRays-1;
		numRaysHit = 0;
		for (int r = 0; r < nRays; r++) {
			if (rayHasHit[r] == 0)
				sortedRayIDs[back--] = r;
			else
				sortedRayIDs[numRaysHit++] = r;
		}

		firstHitRay = sortedRayIDs[0];
		lastHitRay = sortedRayIDs[numRaysHit - 1];
	}
		
	//
	// ray and hit point representation of packet
	//

	// real rays contained in this RayPacket
	SIMDRay rays[nRays];

	// hit points for rays:
	SIMDHitpoint hitpoints[nRays];

	// 0 if respective ray has not hit (i.e. is active)
	char rayHasHit[nRays];
	int sortedRayIDs[nRays];
	SIMDRay cornerRays;
	__m128 directionMask;
	
	//
	// frustum representation
	//

	// all plane normals, stored in struct of arrays order [posneg][axis]
	SIMDVec4 plane_normals[2][3];

	// common origin
	__declspec(align(16)) _Vector4 origin;

	__m128 maxIntersectT;

	//
	// stats
	// 

	// number of rays that hit (not updated while in traversal! see calculateHitStats())
	int numRaysHit;
	// first ray that hit something (not updated while in traversal! see calculateHitStats())
	int firstHitRay, lastHitRay;

protected:
private:
};

RayPacketTemplate
inline std::ostream &operator<<(std::ostream &os, RayPacketT &r) {
	os << "RayPacket<" << nRays << ">: (" << r.rays[0].getOrigin(0) << ")" << endl;
	return os;
}

};