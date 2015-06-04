#ifndef COMMON_SCENE_H
#define COMMON_SCENE_H

/********************************************************************
	created:	2004/10/04
	created:	4.10.2004   14:54
	filename: 	c:\MSDev\MyProjects\Renderer\Common\Scene.h
	file base:	Scene
	file ext:	h
	author:		Christian Lauterbach (lauterb@informatik.uni-bremen.de)
	
	purpose:	Scene Management
*********************************************************************/
#include "HighLevelTree.h"
#include "common.h"
#include "PhotonMap.h"
#include "PhotonGrid.h"

#if HIERARCHY_TYPE == TYPE_BVH
#include "BVH.h"
#include "AreaLight.h"
//#include "clusterdefine.h"
#endif

// VRML Scene Loader
#include <cv97/CyberVRML97.h>

// For special visualizations (using OpenGL),
// allows to select the current visualization
enum SceneVisualizationType {
	SHOW_BSPTREE,
	SHOW_PHOTONMAP	
};

class Scene
{
public:
	Scene() {
		photonMap = NULL;
		background = NULL;
		objectList = NULL;		
		highLevelTree = NULL;
		nSubObjects = 1;
		hasMultipleObjects = false;

#ifdef _USE_AREA_LIGHT
		areaLight = NULL;
#endif

		resetScene();	
	}

	~Scene() {
		LogManager *log = LogManager::getSingletonPtr();
		log->logMessage("Destroying Scene...");

		#ifdef _SIMD_SHOW_STATISTICS
		if (sceneGraph)
			sceneGraph->dumpCounts();
		#endif
	
		resetScene();
	}
	
	/**
	 * Loads the scene file given by the parameter.
	 *
	 * The file type (VRML, PLY) will be determined by the 
	 * extension of fileName.
	 */
	int loadScene(const char *fileName, unsigned int subObjectId = 0);

	/**
	 * Returns the initial viewer position for this scene
	 * (e.g. camera position). This may be set from the loaded
	 * scene.
	 *
	 * The position is returned as a Ray with the origin as 
	 * the position and the ray's direction as the gaze direction.
	 * All other parameters (up direction) are assumed to be default.
	 */
	Ray &getInitialViewer() {
		assert(viewlist.size() > 0);
		return viewlist[0].view;
	}

	Vector3 getLightPos(int num) {
		return lightlist[num].pos;
	}

	void setLightPos(int num, Vector3 pos) {
		lightlist[num].pos = pos;;
	}

	/**
	 * Get a list of cameras for this scene.
	 */
	ViewList &getPresetViews() {
		return viewlist;
	}
	
	/**
	 * Writes the axis-aligned scene bounding box into the
	 * two vectors given as arguments
	 **/
	void getAABoundingBox(Vector3 &min, Vector3 &max, unsigned int subObjectID = 0) {
		min = bb_min;
		max = bb_max;
	}

	/**
	 * Dumps a textual representation of the scene contents
	 * into the specified Logger (or the default logger if 
	 * LoggerName is empty)
	 */
	void printScene(const char *LoggerName = NULL);

	/**
	* Dumps a few scene statistics 
	* into the specified Logger (or the default logger if 
	* LoggerName is empty)
	*/
	void printSceneStats(const char *LoggerName = NULL);

	void printRayStats(const char *LoggerName = NULL);

	/**
	 * Visualization of the scene using OpenGL. The visualization
	 * type is selected by mode. Each of the called functions then
	 * clears the current GL buffer and then renders into it. No
	 * buffer swap is performed !
	 */
	void GLdrawScene(Ray &viewer, SceneVisualizationType mode) {
		switch (mode) {
			case SHOW_BSPTREE:
				for (int i = 0; i < nSubObjects; ++i)
					objectList[i].bvh->GLdrawTree(viewer);
				break;

			case SHOW_PHOTONMAP:
				if (photonMap)
					photonMap->GLdrawPhotonMap(viewer);
				break;
		}		
	}

	/**
	 *	Moves the objects in the scene according to their
	 *  animation. Parameter is the time difference to the 
	 *  last update in seconds.
	 */
	//void animateObjects(float timeDifference) {
	//	for (int i = 0; i < nSubObjects; i++)
	//		objectList[i].animate(timeDifference);
	//	calculateAABoundingBox();
	//}


	//
	// we just rebulid high level tree if the object has transformation matrix list
	//
	int animateObjects() {
		// we check that we have more animation frame
		bool needAnimate = false;
		for (int i = 0; i < nSubObjects; ++i) {
			if (objectList[i].indexCurrentTransform < objectList[i].sizeTransformMatList - 1) {
				needAnimate = true;
				break;
			}
		}
		if (!needAnimate)
			return -1;

		if (highLevelTree) {
			delete highLevelTree;
			highLevelTree = new HighLevelTree(objectList, nSubObjects);
			highLevelTree->buildTree();	
			return 1;
		}

		return -1;
	}

	/**
	 * Trace a ray into the scene.
	 * 
	 * Returns the effective color for the ray in color, by shading
	 * and - if necessary - recursive evaluation (e.g. on reflective
	 * surfaces).
	 *
	 * The first two functions use "normal" raytracing for evaluation,
	 * the first for just one ray, the second for four coherent rays
	 * in parallel.
	 * 
	 * tracePath() does path-tracing, i.e. a whole path is traced into
	 * the scene until the maximal path length is reached or the ray
	 * misses the scene (in the latter case, false is returned).
	 */

	void rayCast(SIMDRay &ray, rgb *color);
	void trace(Ray &ray, rgb &color, int depth = 0, float TraveledDist = 0.f);
	void trace(SIMDRay &ray, rgb *color, int depth = 0, float *distanceOffset = NULL);
	bool tracePath(Ray &ray, rgb &outColor, int depth = 0);
	void traceBeam(Beam &beam, rgb *color);

	/**
	 * Returns the error code from the last error
	 * (or 0 if no error)
	 */
	int getLastError();

	/**
	 * Returns a textual representation of the last error
	 * (empty string if no error)
	 */
	const char *getLastErrorString();

	/**
	 * Clears all scene data.
	 */
	void resetScene();

	/**
	 * Error Codes:
	 */
	const static int STATUS_OK			=  1;
	const static int ERROR_FILENOTFOUND = -1;
	const static int ERROR_CORRUPTFILE  = -2;
	const static int ERROR_OUTOFMEM     = -3;
	const static int ERROR_UNKNOWNEXT   = -4;
	const static int ERROR_UNKNOWN      = -10;

	// Public stats
	int nPrimaryRays;
	int nShadowRays;
	int nSingleShadowRays;
	int nSIMDShadowRays;
	int nSIMDReflectionRays;
	int nSingleReflectionRays;
	int nReflectionRays;
	int nRefractionRays;

	// Special stats:
	CycleCounter statsTimerPrimary;
	CycleCounter statsTimerShadow;
	CycleCounter statsTimerReflection;
	CycleCounter statsTimerRefraction;

protected:

	#if !defined(_USE_OOC) || defined(_USE_OOC_DIRECTMM)
	/**
	 * Loads a VRML file as the current scene.
	 *
	 * All previous scene contents will be erased !
	 */
	int loadVRML(const char *fileName, unsigned int subObjectId = 0);
	
	/**
	 * Loads a PLY file as the current scene.
	 *
	 * All previous scene contents will be erased !
	 */
	int loadPLY(const char *fileName, unsigned int subObjectId = 0);

	/**
	 * Helper function for VRML loader, will recursively traverse
	 * the VRML scene graph and load all supported types.
	 */
	void traverseVRMLSceneGraph(SceneGraph *sceneGraph, Node *firstNode, unsigned int level = 0, unsigned int subObjectId = 0);
	#endif

	/**
	* Loads a group of files defining an out-of-core model
	* from the directory given by the file name.
	**/
	int loadOOC(const char *dirName, unsigned int subObjectId = 0);

	/**
	* Loads a scene file using instanced objects in other file
	* formats.
	**/
	int loadMultipleObjects(const char *dirName);

	/**
	 * Calculates the bounding-box (axis-aligned) of the scene and 
	 * saves it into members bb_min and bb_max.
	 */
	void calculateAABoundingBox(unsigned int subObjectId = 0);

	/**
	 * Make a default camera and other necessary elements for a scene
	 * that does not have them (i.e. PLY)
	 */
	void makeGenericSceneSettings();	

	/**
	 *	Initializes internal structures for the number of given 
	 *  sub-objects in the scene
	 */
	void initForNumSubObjects(unsigned int numSubObjects = 1) {
		nSubObjects = numSubObjects;
		objectList = new ModelInstance[numSubObjects];		

		for (unsigned int i = 0; i < numSubObjects; i++) {
			// Clear material list and fill up with one dummy material
			// for objects without material info
			objectList[i].materiallist.clear();

			//#ifdef _USE_FAKE_BASEPLANE
			////rgb tempCol = rgb(0.7, 0.7f, 0.7f);
			////MaterialDiffuse *base = new MaterialDiffuse(tempCol);
			//MaterialSpecular *base = new MaterialSpecular(rgb(0.7, 0.7f, 0.7f));
			//base->setReflectance(0.7f);
			//objectList[i].materiallist.push_back(base);
			//#endif

			#ifdef MARK_UPDATED_NODES
			MaterialDiffuse *plyMat2 = new MaterialDiffuse(rgb(1, 0, 0));	
			objectList[i].materiallist.push_back(plyMat2);
			markUpdateMaterial = objectList[i].materiallist.size() - 1;
			#endif
		}

	}
	
	// File name of the scene file that is currently loaded
	char sceneFileName[MAX_PATH];

	// list of models/objects in the scene, contains geometry
	// information
	ModelInstance		*objectList;
	
	// object count 
	unsigned int nSubObjects;
	bool hasMultipleObjects;

#ifdef _USE_AREA_LIGHT
	AreaLight* areaLight;
#endif

	// Lights	
	LightList lightlist;
	
	// Views
	ViewList viewlist;

	// Triangles emitting light
	EmitterList emitterlist;

	// High Level Tree for the multiple object
	HighLevelTree* highLevelTree;

#if HIERARCHY_TYPE == TYPE_KD_TREE
	// BSP tree of the scene
	SIMDBSPTree *sceneGraph;
#endif
//#if HIERARCHY_TYPE == TYPE_BVH
//	// BSP tree of the scene
//	BVH *sceneGraph;
//
//#endif

	// Photon map
	#ifdef PHOTON_MAP_USE_GRID
	PhotonGrid *photonMap;
	#else
	PhotonMap *photonMap;
	#endif
	int m_nPhotonsInEstimate;
	float m_PhotonDensityRadius;

	// Bounding-Box information on the loaded scene
	Vector3 bb_min, bb_max;

	// the background (e.g. environment map)
	Background *background;
	rgb ambientColor;

	// Option: how far to recurse while tracing rays
	// (set in options.xml)
	int maxRecursionDepth;

	// Option: how many shadow ray per area light source
	// (set in options.xml)
	unsigned int m_nShadowRays;
	float m_nShadowRayFactor; // 1.0f / m_nShadowRays

	// last errors (or empty)
	int lastErrorCode;	
	char lastError[2000];
	char outputBuffer[2000];
	
	/************************************************************************/
	/* ´ö¼ö                                                                 */
	/* Since : 2008/5/13													*/
	/************************************************************************/

	// detect collision from current frame ( ´ö¼ö )
public :
	void collisionDetection() ;
};

#endif