#ifndef COMMON_PHOTONGRID_H
#define COMMON_PHOTONGRID_H

#include "BVH.h"

////////////////////////////////////////////////////
/// Grid Settings:
//

#define _GRID_HASH

// Padding of the grid on the sides
#define PHOTONGRID_PADDING 0

// Extends of a block in the grid (^3) as a power of two
#define PHOTONGRID_BLOCKSIZEPOWER2 4
// Extends of a block in the grid (^3) in grid points
#define PHOTONGRID_BLOCKSIZE (1 << PHOTONGRID_BLOCKSIZEPOWER2)
#define ROUND_BLOCKSIZE(x) (x + (PHOTONGRID_BLOCKSIZE - (x % PHOTONGRID_BLOCKSIZE))%PHOTONGRID_BLOCKSIZE)
#define PHOTONGRID_BLOCKOFFSET (PHOTONGRID_BLOCKSIZE*PHOTONGRID_BLOCKSIZE*PHOTONGRID_BLOCKSIZE)

#include "HashGrid.h"

//
////////////////////////////////////////////////////

/**
 * Photon structure.
 */
//typedef struct GridPhoton_t {
//	rgb power;	
//} GridPhoton, *GridPhotonPtr;
typedef rgb GridPhoton, *GridPhotonPtr;
typedef GridPhoton GridBlock[PHOTONGRID_BLOCKSIZE*PHOTONGRID_BLOCKSIZE*PHOTONGRID_BLOCKSIZE];

// For sorting:
typedef struct GridPointSort_t {
	unsigned int offset;
	GridPhoton val;
} GridPointSort;
typedef std::vector<GridPointSort> GridPointVector;

inline bool operator<(GridPointSort c1, GridPointSort c2) {
	return (c1.val.sumabs() < c2.val.sumabs()); }
inline bool operator>(GridPointSort c1, GridPointSort c2) {
	return (c1.val.sumabs() > c2.val.sumabs()); }
inline bool operator<=(GridPointSort c1, GridPointSort c2) {
	return (c1.val.sumabs() <= c2.val.sumabs()); }
inline bool operator>=(GridPointSort c1, GridPointSort c2) {
	return (c1.val.sumabs() >= c2.val.sumabs()); }

class PhotonGrid
{
public:
	PhotonGrid(BVH &sceneGraph, LightList &lights, EmitterList &emitters, MaterialList &materials);

	~PhotonGrid() {	
		clearMap();
	}

	/**
	* Build a photon grid with the specified number of
	* photons in total and with the specified grid dimensions.
	*/
	void buildPhotonMap(unsigned int numPhotons, unsigned int gridPoints = 0);

	/**
	* Visualize the current photon grid using OpenGL.
	*/
	void GLdrawPhotonMap(Ray &viewer);

	/**
	* Empty the photon grid.
	*/
	void clearMap();

	/**
	* Print information about the current photon map to the 
	* specified logger (or default logger if none specified)
	*/
	void printPhotonMap(const char *LoggerName = NULL);

	void irradiance_estimate(
		rgb &irrad,                // returned irradiance
		const Vector3 pos,         // surface position
		const Vector3 normal,      // surface normal at pos
		const float max_dist,      // max distance to look for photons
		const int nphotons, 
		bool verbose = false );

protected:	
	/**
	 * "High-Level" grid access methods:
	 * Settings and adding grid values.
	 */	

	FORCEINLINE void addGrid(Vector3 &pos, Vector3 &normal, GridPhoton &value);

	/**
	 * Settings and reading the mask for saving the usage of grid points
	 */
	FORCEINLINE void setGridMask(unsigned int i, unsigned int j, unsigned int k);
	FORCEINLINE bool getGridMask(unsigned int i, unsigned int j, unsigned int k);

	/**
	 * Calculation of grid addresses (low-level grid access).
	 */
	FORCEINLINE unsigned int getBlock(unsigned int i, unsigned int j, unsigned int k);	
	FORCEINLINE GridPhoton& getGridPoint(unsigned int i, unsigned int j, unsigned int k);

	/**
	 * Wavelet transform and thresholding of all blocks in the grid
	 */
	void transformGrid();

	/**
	 * Does the wavelet transform on a block, returns the number of non-zero coefficients
	 */
	unsigned int transformBlock(unsigned int x, unsigned int y, unsigned int z);

	/**
	 * Takes a wavelet-transformed block and does thresholding according to ratio (0.0-1.0)
	 */
	void applyThresholding(unsigned int x, unsigned int y, unsigned int z, float ratio);

	/**
	 * Gets the reconstructed value on a certain grid position by using the inverse
	 * wavelet transform.
	 */
	rgb reconstructValue(unsigned int i, unsigned int j, unsigned int k);

	// number of grid points in each dimensions
	unsigned int gridDimension[3];	
				
	// distances between grid points in each dimension
	float gridDelta[3];	

	unsigned int m_nPhotons;			 // Number of photons in map
	unsigned int m_nPhotonsMissed;		 // Number of sent photons that missed the scene	
	
	unsigned int offsetZ, offsetY;				// precalculated offsets for finding the address of grid points	
	unsigned int blockoffsetZ, blockoffsetY;	// ...and blocks

	// compression and grid point estimation settings:
	float m_fCompressionRate;
	float m_fEstimationRadius, m_fEstimationRadiusSquared;
	float m_fEstimationRadius2, m_fEstimationRadiusSquared2;
	float m_fEstimationVolume;

	// Time stats:
	TimerValue timeBuildStart, timeBuildEnd;	// timestamps for beginning/end
	float timeBuild;							// time taken to build, in seconds
	
	BVH *m_pSceneGraph;	 // Pointer to scene graph
	LightList *m_pLightList;	 // Pointer to list of lights
	EmitterList *m_pEmitterList; // Pointer to list of area lights
	MaterialList *m_pMaterialList; // Pointer to list of materials

#ifndef _GRID_HASH
	// the grid:
	GridPhoton *m_pGrid;
#else
	HashGrid<rgb> m_pGrid;
#endif

	// grid usage bitmask:
	unsigned char *m_pGridMask;
	// statistics on the blocks that make up the grid
	unsigned int m_nGridBlocksUsed,
		         m_nGridBlocks;

	// compression parameters
	unsigned int m_nDecompositionSteps;	// number of wavelet decomposition steps

	Vector3	min, max;			// bounding box of the scene/tree
	Vector3 gridExtendsInv;		// 1 / (max - min)

private:
};



#endif