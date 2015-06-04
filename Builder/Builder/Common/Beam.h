#ifndef COMMON_BEAM_H
#define COMMON_BEAM_H

#include "Hitpoint.h"
#define BEAM_MAXRAYNUM (BEAM_REALMAXRAYNUM / 4)
#define BEAM_REALMAXRAYNUM (BEAM_PRIMARY_STARTSIZE*BEAM_PRIMARY_STARTSIZE)
class Beam
{
public:
	Beam(Vector3 origin, Vector3 cornerLU, Vector3 cornerRU, Vector3 cornerLB, Vector3 cornerRB, int raysHorizontal, int raysVertical) {
		setup( origin, cornerLU, cornerRU, cornerLB, cornerRB, raysHorizontal, raysVertical);		
		numRaysHit = 0;		
	}

	Beam() {
		numRaysHit = 0;
	}

	~Beam() {

	}

	void setup(Vector3 origin, Vector3 cornerLU, Vector3 cornerRU, Vector3 cornerLB, Vector3 cornerRB, int raysHorizontal, int raysVertical, bool verbose = false) {
		int i;
		Vector3 normals[4];
		Vector3 dirs[4];	

		dirs[0] = cornerLU - origin;
		dirs[1] = cornerRU - origin;
		dirs[2] = cornerLB - origin;
		dirs[3] = cornerRB - origin;

		// init SIMDRay origins
		__m128 xcoords = _mm_load1_ps(&origin[0]);
		__m128 ycoords = _mm_load1_ps(&origin[1]);
		__m128 zcoords = _mm_load1_ps(&origin[2]);
		_mm_store_ps(cornerRays.origin[0], xcoords);
		_mm_store_ps(cornerRays.origin[1], ycoords);
		_mm_store_ps(cornerRays.origin[2], zcoords);

		for (i = 0; i < 4; i++) {
			dirs[i].makeUnitVector();
			cornerRays.setDirection(dirs[i], i);
		}		

		// top
		normals[0] = cross(dirs[0], dirs[1]);
		// bottom
		normals[1] = cross(dirs[3], dirs[2]);
		// left
		normals[2] = cross(dirs[2], dirs[0]);
		// right
		normals[3] = cross(dirs[1], dirs[3]);

		for (i = 0; i < 4; i++) {
			normals[i].makeUnitVector();
			tempnormals[i] = normals[i];
		}

		for (int j = 0; j < 3; j++) {
			for (i = 0; i < 4; i++) {
				plane_normals[0][j][i] = max(normals[i][j], 0.0f);				
				plane_normals[1][j][i] = min(normals[i][j], 0.0f);
			}			
		}

		if (verbose) {
			cout << "normals[0] = " << normals[0] << endl;
			cout << "normals[1] = " << normals[1] << endl;
			cout << "normals[2] = " << normals[2] << endl;
			cout << "normals[3] = " << normals[3] << endl;
		}

		numRaysHorizontal = raysHorizontal;
		numRaysVertical = raysVertical;
		numRealRays = 0;
		numRaysHit = 0;		
	}

	/**
	* Split the beam into subdivisionRatio*subdivisionRatio sub-beams
	* contained in the array newBeams. 
	**/
	void split(int subdivisionRatio, Beam *newBeams) {

	}

	/**
	 * Derive frustrum normals and parameters from real rays in this 
	 * beam (for non-primary rays where the frustrum is not known)
	 **/
	void makeFrustrum() {
		/// TODO for secondary rays?
	}

	/**
	* Init internal structures assuming that the real rays were copied
	* in from the caller.
	**/
	void initFromRays(int numRealRays) {
		// set number of rays and init hitcounters:
		this->numRealRays = numRealRays;
		memset(rayHasHit, 0, sizeof(unsigned int)*numRealRays);
		memset(rayHitpoints, 0, sizeof(SIMDHitpoint)*numRealRays);
	}

	/**
	* Intersects the beam with an axis-aligned box defined by the two
	* vectors. Returns true if there is an intersection.
	**/
	FORCEINLINE bool intersectWithBox(Vector3 bb_min, Vector3 bb_max) {		
		// need to transform AABB to beam coordinate space because
		// we assume that d=0 for all normals below
		bb_min -= cornerRays.getOrigin(0);
		bb_max -= cornerRays.getOrigin(0);

		// intersect the axis-aligned box with all four frustrum faces
		// in parallel, see explanation in multi-level ray tracing paper
		__m128 nplane = _mm_setzero_ps(); 
		for (int i = 0; i < 3; i++) {
			__m128 bmin = _mm_load1_ps(&bb_min[i]);
			__m128 bmax = _mm_load1_ps(&bb_max[i]);
			__m128 normalsPositive = _mm_load_ps(plane_normals[0][i].e);
			__m128 normalsNegative = _mm_load_ps(plane_normals[1][i].e);
			bmin = _mm_mul_ps(bmin, normalsPositive);
			bmax = _mm_mul_ps(bmax, normalsNegative);
			bmin = _mm_add_ps(bmin, bmax);
			nplane = _mm_add_ps(nplane, bmin);
		}		
		int signs = _mm_movemask_ps( nplane );

		// if signs == 15, then the box intersects with this frustrum/beam
		// because we could not find an dividing plane
		return signs == 15;		
	}

	/**
	* Intersects the beam with an axis-aligned plane defined by pos[axis]=coord
	* and returns true when the bounding rectangle on the plane is outside the node
	* bounding box.
	*/
	FORCEINLINE bool intersectWithkdNode(int axis, float coord, const Vector3 &bb_min, const Vector3 &bb_max, int &childSelect) {
		static int axes[3][2] = { {1,2}, {2,0}, {0,1} }; // LUT to other axes by primary axis
		int i1 = axes[axis][0];
		int i2 = axes[axis][1];
		bool compares = false, compares_min, compares_max;

		// calculate ray distances t to the plane:
		__m128 coords = _mm_load1_ps(&coord);
		__m128 origins = _mm_load_ps(cornerRays.origin[axis]);
		__m128 directions = _mm_load_ps(cornerRays.invdirection[axis]);
		__m128 t = _mm_sub_ps(coords, origins);
		t = _mm_mul_ps(t, directions);

		__m128 hitpoints, minvals, maxvals;

		// calculate hitpoints for axis 1:
		directions = _mm_load_ps(cornerRays.direction[i1]);
		origins = _mm_load_ps(cornerRays.origin[i1]);
		hitpoints = _mm_mul_ps(t, directions);
		hitpoints = _mm_add_ps(hitpoints, origins);

		// compare the hitpoints to the bounding box:
		// when all values are either smaller or larger
		// than the bounding box intervals, then the 
		// 
		minvals = _mm_load1_ps(&bb_min.e[i1]);
		maxvals = _mm_load1_ps(&bb_max.e[i1]);
		minvals = _mm_cmplt_ps(hitpoints, minvals);
		maxvals = _mm_cmpgt_ps(hitpoints, maxvals);
		compares_min = _mm_movemask_ps(minvals) == 15;		
		compares_max = _mm_movemask_ps(maxvals) == 15;
		compares = compares_min | compares_max;

		// if compares == true, then this value will be needed,
		// otherwise we just assign it to avoid an if-test
		childSelect = ( (!cornerRays.rayChildOffsets[i1] && compares_max) || (cornerRays.rayChildOffsets[i1] && compares_min) )?1:0;			

		// outside the BB for first axis, early termination:
		if (compares)			
			return false;

		// dito for axis 2:
		directions = _mm_load_ps(cornerRays.direction[i2]);
		origins = _mm_load_ps(cornerRays.origin[i2]);
		hitpoints = _mm_mul_ps(t, directions);
		hitpoints = _mm_add_ps(hitpoints, origins);

		minvals = _mm_load1_ps(&bb_min.e[i2]);
		maxvals = _mm_load1_ps(&bb_max.e[i2]);
		minvals = _mm_cmplt_ps(hitpoints, minvals);
		maxvals = _mm_cmpgt_ps(hitpoints, maxvals);
		compares_min = _mm_movemask_ps(minvals) == 15;		
		compares_max = _mm_movemask_ps(maxvals) == 15;
		compares = compares_min | compares_max;

		// if compares == true, then this value will be needed,
		// otherwise we just assign it to avoid an if-test
		childSelect = ( (!cornerRays.rayChildOffsets[i2] && compares_max) || (cornerRays.rayChildOffsets[i2] && compares_min) )?1:0;			

		// if any of the compares succeeded, the intersection rectangle with
		// the plane is outside the given bounding box		
		return !compares;
	}

	/**
	 * Is the beam a valid one (i.e. direction signs match?)
	 **/
	FORCEINLINE bool directionsMatch() {
		int sign_x = _mm_movemask_ps(_mm_load_ps(cornerRays.direction[0]));
		int sign_y = _mm_movemask_ps(_mm_load_ps(cornerRays.direction[1]));
		int sign_z = _mm_movemask_ps(_mm_load_ps(cornerRays.direction[2]));
		return (((sign_x == 0) || (sign_x == 15)) && ((sign_y == 0) || (sign_y == 15)) && ((sign_z == 0) || (sign_z == 15)));
	}

	/*
	FORCEINLINE bool intersectWithBox(Vector3 bb_min, Vector3 bb_max) {		
		for (int i = 0; i < 3; i++) {			
			if (!intersectWithkdNode(i, bb_min[i], bb_min, bb_max))
				return true;
		}
		for (i = 0; i < 3; i++) {
			if (!intersectWithkdNode(i, bb_max[i], bb_min, bb_max))
				return true;
		}
		return false;
	}*/

	int numRaysHorizontal, numRaysVertical;
	
	SIMDRay cornerRays;
	int numRealRays;

	// real rays contained in this beam
	SIMDRay realRays[BEAM_MAXRAYNUM];

	// number of rays that hit something
	unsigned int numRaysHit;
	// 0 if respective ray has not hit
	unsigned int rayHasHit[BEAM_MAXRAYNUM];
	// hitpoint for ray:
	SIMDHitpoint rayHitpoints[BEAM_MAXRAYNUM];

	// all plane normals, stored in struct of arrays order
	__declspec(align(16)) Vector4 plane_normals[2][3];
	Vector3 tempnormals[4];

	// directions for beams, used for selecting near/far nodes
	int posneg[3];

protected:	


private:
};

inline ostream &operator<<(ostream &os, const Beam &r) {
	os << "(" << r.cornerRays.getOrigin(0) << ") W:" << r.numRaysHorizontal << " H:" << r.numRaysVertical << endl;
	for (int i = 0; i < 4; i++)
		os << "N[" << i << "] = (" << r.tempnormals[i] << ")" << endl;

	for (int j = 0; j < 3; j++) {
		os << "N: (" << r.plane_normals[0][j] << ") + (" << r.plane_normals[1][j] << ")" << endl;
	}

	return os;
}

#endif