#ifndef COMMON_PHOTONMAP_H
#define COMMON_PHOTONMAP_H

/********************************************************************
	file base:	PhotonMap
	file ext:	h
	author:		Christian Lauterbach (lauterb@informatik.uni-bremen.de)

	purpose:	Photon mapping class. Code is based on Henrik Wann Jensen's
		        code from "Realisting Image Synthesis Using Photon Mapping"
*********************************************************************/

#include "BVH.h"

/**
 * Photon structure.
 */
typedef struct Photon_t {
	Vector3 pos;		// position
	unsigned char phi, theta;	// incident angle (spherical coords)
	short plane;		// plane flag for tree
	rgb power;			// Light power	
} Photon, *PhotonPtr;

typedef std::vector<Photon> PhotonList;	
typedef PhotonList::iterator PhotonListIterator;

/* This structure is used only to locate the
* nearest photons
*/
//******************************
typedef struct NearestPhotons {
	//******************************
	int max;
	int found;
	int got_heap;
	float pos[3];
	std::vector<float> dist2;
	std::vector<Photon *> index;
} NearestPhotons;

class PhotonMap
{
public:
	PhotonMap(BVH &sceneGraph, LightList &lights, EmitterList &emitters, MaterialList &materials);

	~PhotonMap() {
		m_Photons.clear();
	}

	/**
	 * Build a photon map with the specified number of
	 * photons in total.
	 */
	void buildPhotonMap(unsigned int numPhotons);

	/**
	 * Visualize the current photon map using OpenGL.
	 */
	void GLdrawPhotonMap(Ray &viewer);

	/**
	 * Empty the photon map.
	 */
	void clearMap();

	/**
	 * Print information about the current photon map to the 
	 * specified logger (or default logger if none specified)
	 */
	void printPhotonMap(const char *LoggerName = NULL);

	FORCEINLINE void PhotonMap::photon_dir( float *dir, const Photon *p ) const;
	inline void locate_photons(NearestPhotons *np, unsigned int index );

	void irradiance_estimate(
		rgb &irrad,                // returned irradiance
		const float pos[3],            // surface position
		const float normal[3],         // surface normal at pos
		const float max_dist,          // max distance to look for photons
		const int nphotons, bool verbose = false );

protected:

	void median_split(
		Photon **p,
		const int start,               // start of photon block in array
		const int end,                 // end of photon block in array
		const int median,              // desired median number
		const int axis );

	void balance_segment(Photon **pbal,
		Photon **porg,
		const int index,
		const int start,
		const int end );

	unsigned int m_nPhotons;			 // Number of photons in map
	unsigned int m_nPhotonsMissed;	 // Number of sent photons that missed the scene
	unsigned int m_nHalfStoredPhotons; 
	
	// Time stats:
	TimerValue timeBuildStart, timeBuildEnd;
	float timeBuild;

	PhotonList m_Photons;		// array of photons, filled during generation	
	BVH *m_pSceneGraph;	// Pointer to scene graph
	LightList *m_pLightList;	// Pointer to list of lights
	EmitterList *m_pEmitterList;// Pointer to list of area lights
	MaterialList *m_pMaterialList; // Pointer to list of materials

	Vector3	min, max;			// bounding box of the scene/tree

	// Precalculated tables for packed direction conversion:
	float costheta[256];
	float sintheta[256];
	float cosphi[256];
	float sinphi[256];
	
private:
};



#endif