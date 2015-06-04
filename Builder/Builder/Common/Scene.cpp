/********************************************************************
	filename: 	c:\MSDev\MyProjects\Renderer\Common\Scene.cpp
	file path:	c:\MSDev\MyProjects\Renderer\Common
	file base:	Scene
	file ext:	cpp
	author:		Christian Lauterbach (lauterb@informatik.uni-bremen.de)
	
	purpose:	Scene class, provides functions for ray tracing into
	            scene. Shading and (global) illumination calculation
				will be done according to the settings in common.h
*********************************************************************/

#include "stdafx.h"
#include "common.h"
#include "Scene.h"
#include "Sample.h"
#include "ply.h"
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>

#ifdef _USE_RACM
#include "compressed_mesh.h"
#endif

// macros for accessing single triangles and vertices, depending on
// whether we're using OOC or normal mode
#undef GETTRI 
#undef GETVERTEX
#if defined(_USE_OOC) && !defined(_USE_OOC_DIRECTMM)
#define GETTRI(object,idx) (*(objectList[object].trilist))[idx]
//#define GETVERTEX(object,idx) (*(objectList[object].vertices))[idx*sizeof(_Vector4)]
#define GETVERTEX(object,idx) (*(objectList[object].vertices))[idx]
#else
#define GETTRI(object,idx) objectList[object].trilist[idx]
#define GETVERTEX(object,idx) objectList[object].vertices[idx]
#endif

#define GETMATERIAL(object,materialid) ((object)->materiallist[materialid])
#define SHADE(lightcolor, localcolor, cos_angle) (((lightcolor) * (localcolor)) * (cos_angle))

// num of random light sources
#define PLY_NUM_RANDOM_LIGHTS 1

float g_AvgNumTraversedPath = 0;
extern int g_NumIntersected;
int WorkingSetSizeKB = 0;

extern bool g_Verbose;

void Scene::traceBeam(Beam &beam, rgb *color) {
}

extern int g_NumIntersected;

#ifdef WORKING_SET_COMPUTATION
#include "gpu_cache_sim.h"
CGPUCacheSim g_kdTreeSim (1, 4*1024/sizeof (BSPArrayTreeNode), 
						  1024/4 * 1024, true); // 1GB RAM
CGPUCacheSim g_kdIdxSim (1, 4*1024/sizeof (unsigned int), 
						 1024/4 * 128, true); // 128MB RAM
CGPUCacheSim g_TriSim (1, 4*1024/sizeof (Triangle), 
					   1024/4 * 128, true); // 128MB RAM
CGPUCacheSim g_VerSim (1, 4*1024/sizeof (_Vector4), 
					   1024/4 * 64, true); // 64MB RAM
CGPUCacheSim g_LODSim (1, 4*1024/sizeof (LODNode), 
					   1024/4 * 512, true); // 512MB RAM

CGPUCacheSim g_kdTreeSimL2 (2048, 64/sizeof (BSPArrayTreeNode), 
						  8, true); // 1MB L2 cache 8 associativity
#endif


void Scene::rayCast(SIMDRay &ray, rgb *color) {
}



void Scene::trace(SIMDRay &ray, rgb *color, int depth, float *distanceOffset) {
}

// For DOE coloring
#define COLORS_IN_MAP 4
float maxDistForDOE = 1;
float zMidForDOE = 0;
const float scaleForDOE = 1.0f/(COLORS_IN_MAP-1);

unsigned char bluemap(float dist)
{
	static int blues[COLORS_IN_MAP] = {14, 127, 227, 211};

    // rainbow-ish
    // blues[0]=42;
    // blues[1]=46;
    // blues[2]=124;
    // blues[3]=187;

    // softer
    //blues[0]=14;
    //blues[1]=127;
    //blues[2]=227;
    //blues[3]=211;

    int index = 0;
    float perc=dist/maxDistForDOE;
    if (perc > 1)
        perc = 1;
    while (perc>scaleForDOE)
    {
        perc-=scaleForDOE;
        index++;
    }

    float val = perc/scaleForDOE*blues[index+1]+(1-perc/scaleForDOE)*blues[index];
    return (unsigned char) floor(val);
}

unsigned char greenmap(float dist)
{
	static int greens[COLORS_IN_MAP] = {198, 227, 149, 41};

	// rainbow-ish
    // greens[0]=42;
    // greens[1]=169;
    // greens[2]=187;
    // greens[3]=178;

    // softer colors
    //greens[0]=198;
    //greens[1]=227;
    //greens[2]=149;
    //greens[3]=41;

    int index = 0;
    float perc=dist/maxDistForDOE;
    if (perc > 1)
        perc = 1;
    while (perc>scaleForDOE)
    {
        perc-=scaleForDOE;
        index++;
    }

    float val = perc/scaleForDOE*greens[index+1]+(1-perc/scaleForDOE)*greens[index];
    return (unsigned char) floor(val);
}

unsigned char redmap(float dist)
{
	static int reds[COLORS_IN_MAP] = {211, 39, 39, 49};

	// rainbow-ish
    // reds[0]=211;
    // reds[1]=230;
    // reds[2]=35;
    // reds[3]=46;

    // softer colors
    //reds[0]=211;
    //reds[1]=39;
    //reds[2]=39;
    //reds[3]=49;

    int index = 0;
    float perc=dist/maxDistForDOE;
    if (perc > 1)
        perc = 1;
    while (perc>scaleForDOE)
    {
        perc-=scaleForDOE;
        index++;
    }

    float val = perc/scaleForDOE*reds[index+1]+(1-perc/scaleForDOE)*reds[index];
    return (unsigned char) floor(val);
}

void getDOEColor(float z, rgb &color)
{
	float dist = fabs(z-zMidForDOE);

	color.data[0] = redmap(dist)/255.0f;
	color.data[1] = greenmap(dist)/255.0f;
	color.data[2] = bluemap(dist)/255.0f;
}


void Scene::trace(Ray &ray, rgb &color, int depth, float TraveledDist) 
{	
	static Sampler sampler;
	Hitpoint hitpoint;
	rgb specularColor, shaded_color;
	float specularWeight = 0.0f;
	bool hasHit = false;
	OptionManager *opt = OptionManager::getSingletonPtr();	
	bool isEnvironmentMap = ((opt->getOptionAsInt("raytracing", "environment", 1)) != 1);
	bool isBaseHit = false;

	hitpoint.t = FLT_MAX;

	highLevelTree->intersectTest(ray, &hitpoint, highLevelTree->root, TraveledDist, &hasHit);

	if (!hasHit) {

#ifdef _USE_FAKE_BASEPLANE		
		static float m_FakePlaneHeight = highLevelTree->root->bbox.pp[0].y();
		if (ray.direction().y() < 0.0f && ray.origin().y() >= m_FakePlaneHeight) {
		
			if (isEnvironmentMap)
				background->shade(ray.direction(), shaded_color);

			isBaseHit = true;

			hitpoint.m = 1;
			hitpoint.n[0] = 0;
			hitpoint.n[1] = 1.0f;
			hitpoint.n[2] = 0;
			hitpoint.t = (m_FakePlaneHeight - ray.origin().y()) * ray.invDirection().y();
			hitpoint.x = ray.pointAtParameter(hitpoint.t);	

 			hitpoint.triIdx = INT_MAX;
			hitpoint.objectPtr = &objectList[0];
		}
		else // ray did miss the scene
#endif
		{
			// return background color from background class:			
			background->shade(ray.direction(), color);
			return;
		}
	}

	Material *mat = NULL;

	if (hasHit) {
		mat = GETMATERIAL(hitpoint.objectPtr, hitpoint.m);

		// shade hit point
		mat->shade(ray, hitpoint, shaded_color);
		
		// Only for DOE
		//getDOEColor(ray.pointAtParameter(hitpoint.t).e[2], shaded_color);
	}


#ifdef _USE_REFLECTIONS
	// do we need to calculate reflexivity ?
	bool needReflectionRay = mat &&
							 mat->hasReflection() &&       // object has reflective material
							 depth < maxRecursionDepth &&  // max. recursion depth not exceeded
							 hitpoint.t > 0.0f;
	if (needReflectionRay) {
		Vector3 reflectionDirection = reflect(hitpoint.x - ray.origin() , hitpoint.n);
		reflectionDirection.makeUnitVector();	

		Ray reflectionRay(hitpoint.x, reflectionDirection);
		trace(reflectionRay, specularColor, depth + 1, hitpoint.m_TraveledDist + TraveledDist);

		// blend with local color
		specularWeight = mat->getReflectance();
	}
#endif

//#ifdef _USE_REFRACTIONS
//	bool needRefractionRay = mat->hasRefraction() &&      // object has reflective material
//							 depth < maxRecursionDepth && // max. recursion depth not exceeded
//							 hitpoint.t > 0.0f;
//	// do we need to calculate refraction ?
//	if (needRefractionRay) {	
//		Vector3 incoming = hitpoint.x - ray.origin();
//		incoming.makeUnitVector();
//		float dn = dot(incoming, hitpoint.n);	
//		specularColor = rgb(0,0,0);
//		specularWeight = 1.0f - mat->getOpacity();
//
//		// TODO: all this need to be stored in the material
//		float nt = 1.0f;
//		float R0 = (nt - 1.0f) / (nt + 1.0f);
//		R0 *= R0;
//
//		// Entering material...
//		if (dn < 0.0f) {
//			float temp1 = 1.0f / nt;
//			dn = -dn;
//			float root = 1.0f - (temp1*temp1) * (1.0f - dn*dn);
//			Vector3 refractionDirection = incoming * temp1 + hitpoint.n * (temp1*dn - sqrt(root));
//			Ray refractionRay(hitpoint.x, refractionDirection);
//
//			trace(refractionRay, specularColor, depth + 1, hitpoint.m_TraveledDist + TraveledDist);
//		}
//		//else { // Leaving material...
//		//	float root = 1.0f - (nt*nt) * (1.0f - dn*dn);
//
//		//	// if root < 0.0f, then we have total internal reflection 
//		//	// and therefore do not need to trace the refraction..
//		//	if (root >= 0.0f) {					
//		//		Vector3 refractionDirection = incoming*nt - hitpoint.n*(nt * dn - sqrt(root));
//		//		Ray refractionRay(hitpoint.x, refractionDirection);
//
//		//		trace(refractionRay, specularColor, depth+1, hitpoint.m_TraveledDist + TraveledDist);
//		//	}
//		//}
//	}
//#endif

	// add constant ambient term
	color = ambientColor * shaded_color;
	
	Hitpoint tempHitpoint;
	rgb totalColor(0.0f, 0.0f, 0.0f);
	rgb tokenColor;
	bool isShadow = false;

	// for each light source:
	for (int l = 0; l < lightlist.size(); l++) {
		Light &i = lightlist[l];		
		Vector3 v = i.pos - hitpoint.x;
		float len = v.length();
		v /= len;			

		// Is light behind surface ? then don't even trace a ray towards it
		// Note: v and hitpoint.n are normalized, so the dot product is the cosine
		float fac = dot(v, hitpoint.n);	
		if (fac < 0.0f)
			continue;

		hasHit = false;

#ifdef _USE_SHADOW_RAYS

		tempHitpoint = hitpoint;

		tokenColor = SHADE(i.color, shaded_color, fac);

		totalColor += tokenColor;
		// use light's cut-off distance
		// is light source visible from current point ?
		if (i.cutoff_distance < len)
			color += tokenColor;
		else if (highLevelTree->isVisible(highLevelTree->root, i.pos, &tempHitpoint, &hasHit))
#endif
		
		{			
			// yes, add lighting contribution
			color += tokenColor;
		}
#ifdef _USE_SHADOW_RAYS
		else 
#endif
			isShadow = true;

	}

	// NOTE!!
	// EnvironmetMap code is incomplete!
	if (isEnvironmentMap && isBaseHit) 	{
		if (!isShadow)
			color = shaded_color;
		else {
			color.data[0] = shaded_color.r() * (color.r() / totalColor.r());
			color.data[1] = shaded_color.g() * (color.g() / totalColor.g());
			color.data[2] = shaded_color.b() * (color.b() / totalColor.b());
		}
	}
	
	if (specularWeight > 0.0f)
		color = specularWeight*specularColor + (1.0f - specularWeight)*color;
}


bool Scene::tracePath(Ray &ray, rgb &outColor, int depth) {		
	return false;
}



int Scene::loadScene(const char *fileName, unsigned int subObjectId) {
	char *extension;
	const char *sceneBasePath;
	int retCode = STATUS_OK;
	OptionManager *opt = OptionManager::getSingletonPtr();
	LogManager *log = LogManager::getSingletonPtr();

	//float ambientColorR = opt->getOptionAsFloat("raytracing", "ambientColorR", 0.15f);
	float ambientColorR = opt->getOptionAsFloat("raytracing", "ambientColorR", 0.1f);
	float ambientColorG = opt->getOptionAsFloat("raytracing", "ambientColorG", 0.1f);
	float ambientColorB = opt->getOptionAsFloat("raytracing", "ambientColorB", 0.1f);
	ambientColor = rgb(ambientColorR, ambientColorG, ambientColorB);
		

	// Test whether fileName exists (use base path from options
	// if necessary)
	// 
	strcpy(sceneFileName, fileName);	
	struct stat fileInfo;	

	if (stat(sceneFileName, &fileInfo) != 0) {		
		// Get default base path and prepend it
		sceneBasePath = opt->getOption("global", "scenePath");

		// append '/' if needed
		if (sceneBasePath[strlen(sceneBasePath)-1] != '/')
			sprintf(sceneFileName, "%s/%s", sceneBasePath, fileName);
		else
			sprintf(sceneFileName, "%s%s", sceneBasePath, fileName);

		if (stat(sceneFileName, &fileInfo) != 0 ) {
			// File does not exist -> error
			lastErrorCode = ERROR_FILENOTFOUND;
			sprintf(lastError, "Could not find file \"%s\" or \"%s\" !", fileName, sceneFileName);
			log->logMessage(LOG_ERROR, lastError);
			return lastErrorCode;
		}		
	}
	
	// Test if file size > 0 Bytes	
	if ((fileInfo.st_mode & _S_IFDIR) != _S_IFDIR && fileInfo.st_size == 0) {
		// File does not exist -> error
		lastErrorCode = ERROR_FILENOTFOUND;
		sprintf(lastError, "File is empty: \"%s\" or \"%s\" !", fileName, sceneFileName);
		log->logMessage(LOG_ERROR, lastError);
		return lastErrorCode;
	}

	// Get file's extension and call respective loader function
	//
	extension = _strdup(strrchr(sceneFileName, '.'));
	strtolower(extension);

	// Clear material list and fill up with one dummy material
	// for objects without material info
	//materiallist[subObjectId].clear();
	//MaterialDiffuse *base = new MaterialDiffuse(rgb(1.0f, 1.0f, 1.0f));
	//materiallist[subObjectId].push_back(base);

	#if !defined(_USE_OOC) || defined(_USE_OOC_DIRECTMM)
	if (strcmp(extension, ".wrl") == 0) { // VRML
		if (!hasMultipleObjects)
			initForNumSubObjects();
		retCode = loadVRML(sceneFileName, subObjectId);
		calculateAABoundingBox();
		log->logMessage(LOG_DEBUG, "VRML file loaded.");
	}
	else if (strcmp(extension, ".ply") == 0) { // PLY
		if (!hasMultipleObjects)
			initForNumSubObjects();
		retCode = loadPLY(sceneFileName, subObjectId);				
	}
	else 
	#endif // !_USE_OOC
	if (strcmp(extension, ".ooc") == 0) { // Out-of-Core (own format)
		if (!hasMultipleObjects)
			initForNumSubObjects();

		// model's material index = m = 0
		retCode = loadOOC(sceneFileName, subObjectId);	

	}
	else if (strcmp(extension, ".scene") == 0) { // Scene format for instancing models (own format)
		hasMultipleObjects = true;
		retCode = loadMultipleObjects(sceneFileName);	

	}
	else { // Error: Unknown Format
		lastErrorCode = ERROR_UNKNOWNEXT;
		sprintf(lastError, "Unknown Extension '%s' (File: \"%s\")", extension, sceneFileName);
		log->logMessage(LOG_ERROR, lastError);		
		retCode = lastErrorCode;
	}

	if (!hasMultipleObjects || strcmp(extension, ".scene") == 0) {
		// Create background class:

		int type = opt->getOptionAsInt("raytracing", "environment", 1);

		switch (type) {
			case 3:
				background = new BackgroundEMCubeMap(opt->getOption("raytracing", "environmentTextureName", "cubemap"));
				break;

			case 2:
				background = new BackgroundEMSpherical(opt->getOption("raytracing", "environmentTextureName", "cubemap"));
				break;

			case 1:
			default:
				float r = opt->getOptionAsFloat("raytracing", "environmentColorR", 0.0f);
				float g = opt->getOptionAsFloat("raytracing", "environmentColorG", 0.0f);
				float b = opt->getOptionAsFloat("raytracing", "environmentColorB", 0.0f);
				background = new BackgroundConstant(rgb(r, g, b));

		}
	}

	// Build/load scene graph (BSP tree)
#if HIERARCHY_TYPE == TYPE_KD_TREE
	if (sceneGraph == NULL)
		sceneGraph = new SIMDBSPTree(nSubObjects, objectList, GETMATERIAL(&objectList[0],0));		
#endif
#if HIERARCHY_TYPE == TYPE_BVH
	objectList[subObjectId].bvh = new BVH(nSubObjects, objectList, GETMATERIAL(&objectList[subObjectId],0));
#endif
	
	if (strcmp(extension, ".scene") != 0) {	
		// test whether we've got a saved BSP tree for this:
		//
		char bspFileName[MAX_PATH];
#if HIERARCHY_TYPE == TYPE_KD_TREE
		sprintf(bspFileName, "%s/kdtree", sceneFileName);
#endif
#if HIERARCHY_TYPE == TYPE_BVH
		sprintf(bspFileName, "%s/BVH", sceneFileName);
#endif

		if (strcmp(extension, ".ooc") != 0 || !objectList[subObjectId].loadTreeFromFiles(bspFileName)) {
			#if !defined(_USE_OOC) || defined(_USE_OOC_DIRECTMM)
			// test whether we've got a saved BSP tree for this:
			//		
			sprintf(bspFileName, "%s.bsp", sceneFileName);		
				
			if (!objectList[subObjectId].loadTreeFromFile(bspFileName)) {
				// no saved BSP tree, build one from geometry:
				sceneGraph->buildTree(subObjectId);
				log->logMessage(LOG_DEBUG, "BSP tree built.");		
				objectList[subObjectId].saveTreeToFile(bspFileName);
			}
			#else

			// in OOC mode, we don't allow generating the tree on the fly:
			log->logMessage(LOG_ERROR, "Could not find kD-tree file for scene.");		
			log->logMessage(LOG_ERROR, "On-the-fly construction of kD-tree not supported in OoC mode.");		
			return ERROR_FILENOTFOUND;

			#endif
		}		
	}

	//if (strcmp(extension, ".ooc") == 0 && !hasMultipleObjects) {
		// scene bounding box from model bb:
	//	calculateAABoundingBox();		
	//	makeGenericSceneSettings();
	//}

	// resize vectors to optimal size:	
	/*LightList(lightlist).swap(lightlist);
	MaterialList(objectList[subObjectId].materiallist).swap(objectList[subObjectId].materiallist);
	EmitterList(emitterlist).swap(emitterlist);*/

	/*
	// enable this for a dump of the BSP tree to BSPTree.log
	// (use for simple scenes only or the file will be *huge*!)
	log->createLogger("BSPTree", LOG_FILETEXT);
	sceneGraph->printTree(true, "BSPTree");
	log->closeLogger("BSPTree");
	*/	

	if (!hasMultipleObjects) {

		if (highLevelTree)
			delete highLevelTree;

		// scene bounding box from model bb:
		calculateAABoundingBox();	

		highLevelTree = new HighLevelTree(objectList, nSubObjects);
		highLevelTree->buildTree();	
	
		// Check: if this is the first view, remove the default view
		// from the list:
		if (viewlist.size() == 1 && strcmp("Default", viewlist[0].name) == 0) {
			viewlist.clear();	
			makeGenericSceneSettings();
		}

		// baseplane's material index = m = 1
		#ifdef _USE_FAKE_BASEPLANE
		//rgb tempCol = rgb(0.7, 0.7f, 0.7f);
		//MaterialDiffuse *base = new MaterialDiffuse(tempCol);
		MaterialSpecular *base = new MaterialSpecular(rgb(1.0f, 0.0f, 0.0f)); 
		base->setReflectance(0.5f);
		base->setOpacity(1.0f);
		objectList[0].materiallist.push_back(base);
		#endif
	}

	if (subObjectId == 0) {	
		// Build photon map
		#ifdef _USE_PHOTON_MAP
		#ifdef PHOTON_MAP_USE_GRID
		photonMap = new PhotonGrid(*sceneGraph, lightlist, emitterlist, materiallist);
		unsigned int numPhotons = opt->getOptionAsInt("photongrid", "numPhotons", 10000);
		#else
		photonMap = new PhotonMap(*sceneGraph, lightlist, emitterlist);
		unsigned int numPhotons = opt->getOptionAsInt("photonmap", "numPhotons", 10000);
		#endif
		m_nPhotonsInEstimate = opt->getOptionAsInt("photonmap", "numPhotonsInEstimate", 50);
		m_PhotonDensityRadius = opt->getOptionAsFloat("photonmap", "estimationRadius", 2.0f);
		photonMap->buildPhotonMap(numPhotons);
		photonMap->printPhotonMap();
		#endif
	}

	free(extension);

	objectList[subObjectId].bvh->initialize(sceneFileName);

	return retCode;
}

/**
 *	Load .scene file including instances of other models
 */
int Scene::loadMultipleObjects(const char *fileName) {
	typedef stdext::hash_map<const char *, ModelInstance *, stringhasher > ModelMap;
	typedef ModelMap::iterator ModelMapIterator;

	LogManager *log = LogManager::getSingletonPtr();	
	ModelMap fileToModelMap;
	int retCode = STATUS_OK;
	nSubObjects = 0;
	hasMultipleObjects = true;
	
	// open a stream of the file given to us
	FILE *sceneFileStream = fopen(fileName, "rb");
	if (!sceneFileStream) {
		sprintf(lastError, "Could not find file \"%s\"!", fileName);
		lastErrorCode = ERROR_FILENOTFOUND;
		log->logMessage(LOG_ERROR, lastError);
		return lastErrorCode;
	}

	char currentLine[500];

	if (fgets(currentLine, 499, sceneFileStream)) {
		sscanf(currentLine, "#model %d", &nSubObjects);
	}
	else {
		cerr << "Error to load the number of model\n";
		return -1;
	}

	// allocate memory for structures
	initForNumSubObjects(nSubObjects);

	char modelFileName[100];	
	int numFrame;
	Matrix tempMat;
	for (int curObject = 0; curObject < nSubObjects; ++curObject) {
		if (!fgets(currentLine, 499, sceneFileStream)) {
			cerr << "Error to load the file name\n";
			return -1;
		}

		sscanf(currentLine, "%s", modelFileName);
		char *tempModelFileName = _strdup(modelFileName);
		ModelMapIterator it = fileToModelMap.find(tempModelFileName);

		// first time model is loaded?
		if (it == fileToModelMap.end()) {
			cout << "loading new " << tempModelFileName <<  endl;

			// copy model name
			objectList[curObject].modelFileName = tempModelFileName;

			// load scene:
			loadScene(objectList[curObject].modelFileName, curObject);

			// insert into file to model map:
			fileToModelMap.insert(std::pair<const char *, ModelInstance *>(tempModelFileName,&objectList[curObject]));;
		}
		else { // model was loaded before, make this one an instance:
			cout << "loading instance " << tempModelFileName << endl;
			objectList[curObject].instanceFrom(it->second);
		}

		// load current model's transformation info
		fgets(currentLine, 499, sceneFileStream);
		if (!sscanf(currentLine, "#frame %d", &numFrame)) {
			cerr << "Error to load the number of the frame\n";
			return -1;
		}
		
		objectList[curObject].ID = curObject;
		objectList[curObject].sizeTransformMatList = numFrame;
		objectList[curObject].indexCurrentTransform = -1;
		objectList[curObject].transformMatList = new Matrix[numFrame];
		objectList[curObject].invTransformMatList = new Matrix[numFrame];

		// load current model's animation info
		for (int i = 0; i < numFrame; ++i) {
			fgets(currentLine, 499, sceneFileStream);
			if (!sscanf(currentLine, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f", \
				 &tempMat.x[0][0], &tempMat.x[1][0], &tempMat.x[2][0], &tempMat.x[3][0], \
				 &tempMat.x[0][1], &tempMat.x[1][1], &tempMat.x[2][1], &tempMat.x[3][1], \
				 &tempMat.x[0][2], &tempMat.x[1][2], &tempMat.x[2][2], &tempMat.x[3][2], \
				 &tempMat.x[0][3], &tempMat.x[1][3], &tempMat.x[2][3], &tempMat.x[3][3])) {

				 //&tempMat.x[0][0], &tempMat.x[0][1], &tempMat.x[0][2], &tempMat.x[0][3], \
				 //&tempMat.x[1][0], &tempMat.x[1][1], &tempMat.x[1][2], &tempMat.x[1][3], \
				 //&tempMat.x[2][0], &tempMat.x[2][1], &tempMat.x[2][2], &tempMat.x[2][3], \
				 //&tempMat.x[3][0], &tempMat.x[3][1], &tempMat.x[3][2], &tempMat.x[3][3])) {
				cerr << "Error to load the number of the frame\n";
				return -1;
			}						
			objectList[curObject].transformMatList[i] = tempMat;
			objectList[curObject].invTransformMatList[i] = tempMat.getInverse();
		}

		objectList[curObject].printModel();
	}

	if (highLevelTree)
		delete highLevelTree;

	// scene bounding box from model bb:
	calculateAABoundingBox();	

	highLevelTree = new HighLevelTree(objectList, nSubObjects);
	highLevelTree->buildTree();	
	
	// Check: if this is the first view, remove the default view
	// from the list:
	if (viewlist.size() == 1 && strcmp("Default", viewlist[0].name) == 0) {
		viewlist.clear();	
		makeGenericSceneSettings();
	}

	// baseplane's material index = m = 1
	#ifdef _USE_FAKE_BASEPLANE
	rgb tempCol = rgb(0.0f, 0.0f, 0.0f);
	MaterialDiffuse *base = new MaterialDiffuse(tempCol);
	//base->setReflectance(0.0f);
	//base->setOpacity(0.5f);
	objectList[0].materiallist.push_back(base);
	#endif

	/*
	 *	Lucy & Blade Scene Setting
	 */

	// Material color Test
	//if (nSubObjects == 2) {		
	//	MaterialDiffuse *testMat = (MaterialDiffuse*)objectList[0].materiallist[0];

	//	testMat->setColor(rgb(1.0f,0.0f,0.0f));
	//	
	//	objectList[1].materiallist.clear();

	//	MaterialDiffuse *testMat2 = new MaterialDiffuse(rgb(0.0f, 0.0f, 1.0f));

	//	objectList[1].materiallist.push_back(testMat2);
	//}
	 //Material color Test end

	/*
	 *   St.Matthew & Bulind Scene Setting
	 */
	// Material color Test
	if (nSubObjects == 4) {		
		// st.matthew
		MaterialDiffuse *testMat = (MaterialDiffuse*)objectList[0].materiallist[0];
		testMat->setColor(rgb(0.3f, 0.3f, 0.3f));
		
		// part1 - body
		objectList[1].materiallist.clear();
		MaterialDiffuse *testMat2 = new MaterialDiffuse(rgb(0.93f, 0.9f, 0.79f));
		//testMat2->setReflectance(0.5f);
		//testMat2->setOpacity(1.0f);
		objectList[1].materiallist.push_back(testMat2);
		
		// part2 - body - color
		objectList[2].materiallist.clear();
		MaterialSpecular *testMat3 = new MaterialSpecular(rgb(0.8f, 0.6f, 0.4f));
		testMat3->setReflectance(0.2f);
		testMat3->setOpacity(1.0f);
		objectList[2].materiallist.push_back(testMat3);
		
		// part3 - bottom
		objectList[3].materiallist.clear();
		MaterialSpecular *testMat4 = new MaterialSpecular(rgb(0.6f, 0.6f, 0.6f));
		testMat4->setReflectance(0.5f);
		testMat4->setOpacity(1.0f);
		objectList[3].materiallist.push_back(testMat4);
	}



	///////////////////////////////////////////////////
	/*
	 *  Lucy & Blade Scene Setting
	 */
	// Material color Test
	//if (nSubObjects == 2) {		
	//	MaterialDiffuse *testMat = (MaterialDiffuse*)objectList[0].materiallist[0];

	//	testMat->setColor(rgb(0.7f, 0.5f, 0.3f));
	//	
	//	objectList[1].materiallist.clear();

	//	MaterialDiffuse *testMat2 = new MaterialDiffuse(rgb(0.3f, 0.5f, 0.7f));

	//	//testMat2->setReflectance(0.5f);
	//	//testMat2->setOpacity(1.0f);
	//	objectList[1].materiallist.push_back(testMat2);
	//}
	///////////////////////////////////////////////////
	return retCode;
}

#if !defined(_USE_OOC) || defined(_USE_OOC_DIRECTMM)
int Scene::loadVRML(const char *fileName, unsigned int subObjectId) {
	SceneGraph *sceneGraph; 
	int retCode = STATUS_OK;
	LogManager *log = LogManager::getSingletonPtr();
	
	sprintf(outputBuffer, "Loading VRML file \"%s\" ...", fileName);
	log->logMessage(LOG_INFO, outputBuffer);

	// load VRML file
	char *vrmlFileName = _strdup(fileName);
	sceneGraph = new SceneGraph();
	sceneGraph->load(vrmlFileName);
	free(vrmlFileName);

	// loaded without errors ?
	if (sceneGraph->isOK()) { // yes, get information from scene graph:	
		log->logMessage(LOG_DEBUG, "parsing VRML scene graph...");
		traverseVRMLSceneGraph(sceneGraph, sceneGraph->getNodes(), 0, subObjectId);			
	}
	else { // no, error reading VRML file !
		sprintf(lastError, "Error reading VRML file (line %d) : %s", sceneGraph->getErrorLineNumber(), sceneGraph->getErrorLineString());
		lastErrorCode = ERROR_CORRUPTFILE;
		log->logMessage(LOG_ERROR, lastError);
		return lastErrorCode;
	}

	return STATUS_OK;
}

void Scene::traverseVRMLSceneGraph(SceneGraph *sceneGraph, Node *firstNode, unsigned int level, unsigned int subObjectId) {
	static float specularReflectionThreshold = OptionManager::getSingletonPtr()->getOptionAsFloat("raytracing", "specularReflectionThreshold", 0.0f);
	static LogManager *log = LogManager::getSingletonPtr();
	static TriangleList *tempTriList = NULL;	
	static Vector3List *tempVectorList = NULL;	

	if (!firstNode)
		return;
	
	if (tempTriList == NULL)
		tempTriList = new TriangleList;
	if (tempVectorList == NULL)
		tempVectorList = new Vector3List;
	
	Node *node;

	// Traverse all nodes:
	//
	for (node = firstNode; node; node=node->next()) {

		// Which kind of node is this ?		
		if (node->isLightNode()) { // Light
			LightNode *pVRMLLightNode = (LightNode *)node;
			Light newLight;	
			float lightColor[4];
			float lightPos[4];
			float lightAttenuation[3];
			float lightDirection[3];			

			// Light active ?
			if (!pVRMLLightNode->isOn())
				continue;
			
			// Which light type is this ?
			//

			// Point light (all directions)
			if (pVRMLLightNode->isPointLightNode()) {
				PointLightNode *pVRMLLight = (PointLightNode *)pVRMLLightNode;
				
				// fetch light information
				pVRMLLight->getLocation(lightPos);
				pVRMLLight->getDiffuseColor(lightColor);
				pVRMLLight->getAttenuation(lightAttenuation);

				// initialize new Light
				newLight.type = LIGHT_POINT;
				newLight.pos = Vector3(lightPos);
				newLight.direction = Vector3(0,0,0);				
				newLight.intensity = pVRMLLight->getIntensity();
				newLight.color = rgb(lightColor[0]*newLight.intensity, lightColor[1]*newLight.intensity, lightColor[2]*newLight.intensity);
				newLight.cutoff_distance = pVRMLLight->getRadius();
				
				// insert into light list
				lightlist.push_back(newLight);				
			}
			// non-local directional light, only in one direction
			else if (pVRMLLightNode->isDirectionalLightNode()) {
				DirectionalLightNode *pVRMLLight = (DirectionalLightNode *)pVRMLLightNode;
				
				// fetch light information
				pVRMLLight->getDirection(lightDirection);
				pVRMLLight->getDiffuseColor(lightColor);

				// initialize new Light
				newLight.type = LIGHT_DIRECTIONAL;
				newLight.pos = Vector3(0,0,0);
				newLight.direction = Vector3(lightDirection);
				newLight.intensity = pVRMLLight->getIntensity();
				newLight.color = rgb(lightColor[0]*newLight.intensity, lightColor[1]*newLight.intensity, lightColor[2]*newLight.intensity);
				newLight.cutoff_distance = 9999999;

				// insert into light list
				lightlist.push_back(newLight);				
			}
			// Spot light, local directional light with light cone
			else if (pVRMLLightNode->isSpotLightNode()) {
				SpotLightNode *pVRMLLight = (SpotLightNode *)pVRMLLightNode;
				
				// fetch light information
				pVRMLLight->getLocation(lightPos);
				pVRMLLight->getLocation(lightDirection);
				pVRMLLight->getDiffuseColor(lightColor);
				pVRMLLight->getAttenuation(lightAttenuation);

				// initialize new Light
				newLight.type = LIGHT_SPOT;
				newLight.pos = Vector3(lightPos);
				newLight.direction = Vector3(lightDirection);
				newLight.intensity = pVRMLLight->getIntensity();
				newLight.color = rgb(lightColor[0]*newLight.intensity, lightColor[1]*newLight.intensity, lightColor[2]*newLight.intensity);
				newLight.cutoff_distance = pVRMLLight->getRadius();

				// insert into light list
				lightlist.push_back(newLight);	
			}			
		}
		else if (node->isViewpointNode()) { // Viewpoint Node, change initial view of camera
			float temp[4];
			View newView;

			// Name
			strncpy(newView.name, ((ViewpointNode *)node)->getName(), sizeof(newView.name) - 1);

			// Position
			((ViewpointNode *)node)->getPosition(temp);
			newView.view.setOrigin(Vector3(temp));

			// Orientation is in Axis-Angle-Format
			((ViewpointNode *)node)->getOrientation(temp);
			Vector3 orient(0.0, 0.0, -1.0);
			Matrix m;
			m.fromAngleAxis(Vector3(temp[0], temp[1], temp[2]), temp[3]);
			orient = m*orient;			
			newView.view.setDirection(orient);	

			// Check: if this is the first view, remove the default view
			// from the list:
			if (viewlist.size() == 1 && strcmp("Default", viewlist[0].name) == 0)
				viewlist.clear();

			viewlist.push_back(newView);
		}
		else if (node->isShapeNode()) { // geometry node, read triangle information

			ShapeNode *shape = (ShapeNode *)node;
			GeometryNode *gnode = shape->getGeometry();
			AppearanceNode	*appearance = shape->getAppearanceNodes();
			float point[3];
			Triangle tri;			
			Vector3 n;
			Vector3 normals[3];
			Vector2 uv[3];
			Matrix m, mRot;
			Emitter newEmitter;
			bool isEmitter = false;

			// Transformation matrix of the following shape.
			// All coordinates (vertices and normals) need to
			// be transformed with this so we get the proper
			// values
			shape->getTransformMatrix(m.x);
			m.transpose();

			// Generate copy of matrix without translation
			// for transforming the normal vectors
			mRot = m;
			mRot.x[0][3] = 0;
			mRot.x[1][3] = 0;
			mRot.x[2][3] = 0;

			#ifdef _USE_TRI_MATERIALS

			// Node has material:
			// Insert new material and set for all triangles in this object
			if (appearance) {
				Material *newmat;
				float diffuseColor[3];
				float opacity;
				float specularReflexivity;
				float emittedColor[3];

				// General material parameters (colors, ..)
				MaterialNode *vrmlMaterial = appearance->getMaterialNodes();
				if (vrmlMaterial) {
					opacity = 1.0f - vrmlMaterial->getTransparency();
					if (opacity < 0.0f)
						opacity = 0.0f;

					vrmlMaterial->getDiffuseColor(diffuseColor);

					// this is actually a hack since VRML knows no parameter for reflexivity,
					// but AmbientIntensity should not be used often anyways..
					specularReflexivity = 1.0f - vrmlMaterial->getAmbientIntensity();
					if (specularReflexivity < specularReflectionThreshold)
						specularReflexivity = 0.0f;					

					vrmlMaterial->getEmissiveColor(emittedColor);					
				}
				else {
					diffuseColor[0] = 0;
					diffuseColor[1] = 0;
					diffuseColor[2] = 0;
					opacity = 1.0f;
					specularReflexivity = 0.0f;
				}

				// Has texture ?
				TextureNode *textureNode = appearance->getTextureNode();

				if ((emittedColor[0] + emittedColor[1] + emittedColor[2]) > 0.0f) { // Emitter !
					newmat = new MaterialEmitter();
					newEmitter.emissionColor = rgb(emittedColor[0], emittedColor[1], emittedColor[2]);
					emittedColor[0] *= 5.0f;
					emittedColor[1] *= 5.0f;
					emittedColor[2] *= 5.0f;					
					newEmitter.emissionIntensity = rgb(emittedColor[0], emittedColor[1], emittedColor[2]);
					newEmitter.summedIntensity = emittedColor[0] + emittedColor[1] + emittedColor[2];					
					isEmitter = true;
					((MaterialEmitter *)newmat)->setColor(newEmitter.emissionColor);					
				}
				//#ifdef _USE_TEXTURING
				else if ( textureNode && textureNode->isImageTextureNode()) {

					//printf ("Un-supported error\n");
					if (specularReflexivity > 0.0f || opacity < 1.0f) {
						newmat = new MaterialSpecularAndBitmap(((ImageTextureNode *)textureNode)->getUrl(0));
						((MaterialSpecularAndBitmap *)newmat)->setReflectance(specularReflexivity); 
						((MaterialSpecularAndBitmap *)newmat)->setOpacity(opacity); 

					}
					else
						newmat = new MaterialBitmapTexture(((ImageTextureNode *)textureNode)->getUrl(0));

					// TODO: error handling for texture
				}
				//#endif
				else if (specularReflexivity > 0.0f || opacity < 1.0f) { // specular material
					newmat = new MaterialSpecular();
					((MaterialSpecular *)newmat)->setColor(rgb(diffuseColor[0], diffuseColor[1], diffuseColor[2]));
					((MaterialSpecular *)newmat)->setReflectance(specularReflexivity); 
					((MaterialSpecular *)newmat)->setOpacity(opacity); 
				}				
				else { // normal material
					newmat = new MaterialDiffuse();
					((MaterialDiffuse *)newmat)->setColor(rgb(diffuseColor[0], diffuseColor[1], diffuseColor[2]));
				}

				if (newmat) { // insert					
					objectList[subObjectId].materiallist.push_back(newmat);
					tri.material = objectList[subObjectId].materiallist.size()-1;
				}
				else tri.material = 0; // some error must have occured - set to base material

			}
			else tri.material = 0; // no material - set to base material

			#endif // _USE_TRI_MATERIALS
			
			// Handle geometry types:			
			if (gnode->isIndexedFaceSetNode()) {
				IndexedFaceSetNode *idxFaceSet = (IndexedFaceSetNode *)gnode;
				int pointCount = 0;
				int coordIndex;
				_Vector4 triPoints[3];

				// Triangle coordinates
				CoordinateNode *coordinateNode = idxFaceSet->getCoordinateNodes();
				if (!coordinateNode)
					continue;
				
				// Texture coordinates, if existing
				TextureCoordinateNode *texCoordNode	= idxFaceSet->getTextureCoordinateNodes();

				// Vertex normals, if existing
				NormalNode	*normalNode	= idxFaceSet->getNormalNodes();

				int nNormalIndexes = idxFaceSet->getNNormalIndexes();
				int nCoordIndexes = idxFaceSet->getNCoordIndexes();	
				int nTexCoordIndexes = idxFaceSet->getNTexCoordIndexes();

				for (int nCoordIndex=0; nCoordIndex<nCoordIndexes; nCoordIndex++) {
					coordIndex = idxFaceSet->getCoordIndex(nCoordIndex);
					
					// If index set ended *and* we have 3 points, finish off triangle
					if (coordIndex == -1) {
						if (pointCount == 3) {
							// Finished with coordinates.							

							// Calculate unit normal vector (check winding so it will
							// point the right way !)
							if (idxFaceSet->getCCW())
								n = cross(triPoints[1] - triPoints[0], triPoints[2] - triPoints[0]);
							else
								n = cross(triPoints[2] - triPoints[0], triPoints[1] - triPoints[0]);								

							// Degenerate triangle, points must be collinear/identical:
							// skip this triangle
							if (n.squaredLength() == 0.0) {
								log->logMessage(LOG_WARNING, "Degenerate triangle (collinear points?) detected!");								
								pointCount = 0;
								continue;
							}

							n.makeUnitVector();

							//////////////////////////////////
							// fill intersection triangle

							// normal vector
							tri.n = n;
							
							// distance d of plane equation (really is -d, saves us one op in intersection, the
							// only place this is needed..)
							tri.d = dot(triPoints[0], n);	

							// find best projection plane (YZ, XZ, XY)
							if (fabs(tri.n[0]) > fabs(tri.n[1]) && fabs(tri.n[0]) > fabs(tri.n[2])) {								
								tri.i1 = 1;
								tri.i2 = 2;
							}
							else if (fabs(tri.n[1]) > fabs(tri.n[2])) {								
								tri.i1 = 0;
								tri.i2 = 2;
							}
							else {								
								tri.i1 = 0;
								tri.i2 = 1;
							}

							int firstIdx;
							float u1list[3];
							u1list[0] = fabs(triPoints[1][tri.i1] - triPoints[0][tri.i1]);
							u1list[1] = fabs(triPoints[2][tri.i1] - triPoints[1][tri.i1]);
							u1list[2] = fabs(triPoints[0][tri.i1] - triPoints[2][tri.i1]);

							if (u1list[0] >= u1list[1] && u1list[0] >= u1list[2])
								firstIdx = 0;
							else if (u1list[1] >= u1list[2])
								firstIdx = 1;
							else
								firstIdx = 2;

							int secondIdx = (firstIdx + 1) % 3;
							int thirdIdx = (firstIdx + 2) % 3;
							
							// normalize the normal vectors (just to make sure..)
							normals[firstIdx].makeUnitVector();
							normals[secondIdx].makeUnitVector();
							normals[thirdIdx].makeUnitVector();

							#ifdef _USE_VERTEX_NORMALS
							// if we don't have vertex normals in VRML, save surface normal
							// to each of the three vertex normals..
							if (!normalNode) {
								tri.normals[0] = n;
								tri.normals[1] = Vector3(0,0,0); // last two normals are encoded as difference to first one
								tri.normals[2] = Vector3(0,0,0);
							}
							else {			
								tri.normals[0] = normals[firstIdx];
								// last two normals are encoded as difference to first one for faster
								// calculation of effective normal when interpolating
								tri.normals[1] = normals[secondIdx] - normals[firstIdx];
								tri.normals[2] = normals[thirdIdx] - normals[firstIdx];
							}
							#endif


							#ifdef _USE_TEXTURING
							if (texCoordNode) {
								tri.uv[0] = uv[firstIdx];
								// if we have texture coordinates, encode last two as differences for
								// faster interpolation with barycentric coords
								tri.uv[1] = uv[secondIdx] - uv[firstIdx];
								tri.uv[2] = uv[thirdIdx] - uv[firstIdx];
							}
							#endif
								
							// get triangle indices from best index combination:							
							tri.p[0] = objectList[subObjectId].nVerts + firstIdx - 3; 
							tri.p[1] = objectList[subObjectId].nVerts + secondIdx - 3; 
							tri.p[2] = objectList[subObjectId].nVerts + thirdIdx - 3; 
														
							// insert triangle into list								
							tempTriList->push_back(tri);		

							// if this is an emitter, also insert an new emitter
							// into the list for lighting purposes
							if (isEmitter) {
								newEmitter.p[0] = triPoints[0];
								newEmitter.p[1] = triPoints[1];
								newEmitter.p[2] = triPoints[2];
								newEmitter.n = tri.n;
								newEmitter.area = triangleArea(newEmitter.p[0], newEmitter.p[1], newEmitter.p[2]);
								emitterlist.push_back(newEmitter);									
							}
						}
						
						pointCount = 0;
					}
					else {
						coordinateNode->getPoint(coordIndex, point);

						if (pointCount >= 3) { // if we've got more than 3 points, ignore this
							log->logMessage(LOG_WARNING, "Surface with more than 3 vertices detected, ignoring...\n");
							pointCount = 0;
						}

						// save point in triangle (transformed by matrix)
						triPoints[pointCount] = _Vector4(m*Vector3(point));
						tempVectorList->push_back(triPoints[pointCount]);
						objectList[subObjectId].nVerts++;
						
						#ifdef _USE_TEXTURING
						// if we've got texture coordinates, save them, too
						if (texCoordNode) {
							if (0 < nTexCoordIndexes)
								texCoordNode->getPoint(idxFaceSet->getTexCoordIndex(nCoordIndex), uv[pointCount].e);
							else
								texCoordNode->getPoint(coordIndex, uv[pointCount].e);
							uv[pointCount].e[1] = 1.0f - uv[pointCount].e[1];
						}
						#endif

						// if we have vertex normals, save them 
						if (normalNode) {
							if (0 < nNormalIndexes)
								normalNode->getVector(idxFaceSet->getNormalIndex(nCoordIndex), normals[pointCount].e);
							else
								normalNode->getVector(coordIndex, normals[pointCount].e);

							// rotate normal with matrix (if shape has a transformation, it must
							// be applied to the normal as well)
							normals[pointCount] = mRot*normals[pointCount];
						}

						pointCount++;
					}
				}
			}
			

		}
		else
			traverseVRMLSceneGraph(sceneGraph, node->getChildNodes(), level + 1);
	}

	// Traversal ended if this is the first level.
	// Now copy dynamically allocated lists to static arrays:
	//	
	if (level == 0) {	
		unsigned int i;

		// Allocate memory for tris:
		cout << tempVectorList->size() << " Vertices," << endl;
		cout << "Allocating " << (tempVectorList->size() * sizeof(Vector3)) << " Bytes." << endl;
		objectList[subObjectId].vertices = new _Vector4[tempVectorList->size()];

		// Allocate memory for tris:
		cout << tempTriList->size() << " Triangles," << endl;
		cout << "Allocating " << (tempTriList->size() * sizeof(Triangle)) << " Bytes." << endl;
		objectList[subObjectId].trilist = new Triangle[tempTriList->size()];		

		//
		// Copy vertices from list to static array:
		//
		
		for (i = 0; i < tempVectorList->size(); i++)
			objectList[subObjectId].vertices[i] = tempVectorList->at(i);

		//
		// Copy triangles from list to static array:
		//

		for (i = 0; i < tempTriList->size(); i++) {
			objectList[subObjectId].trilist[objectList[subObjectId].nTris] = tempTriList->at(i);			

			


			// 
			// Convert indices from old array to pointers for 
			// the vertex references stored in the triangle:
			//

			//trilist[nTris].p[0] = trilist[nTris].p[0];
			//trilist[nTris].p[1] = trilist[nTris].p[1];
			//trilist[nTris].p[2] = trilist[nTris].p[2];

			objectList[subObjectId].nTris++;
		}		

		
		delete tempTriList;
		delete tempVectorList;		

		tempTriList = 0;
		tempVectorList = 0;
	}
}
#endif // !_USE_OOC

#if !defined(_USE_OOC) || defined(_USE_OOC_DIRECTMM)
/**
 * Loads and parses a PLY file. This uses the PLY library by Greg Turk, some
 * of the parsing code was also shamelessly ripped from an example by the same
 * author
 **/
int Scene::loadPLY(const char *fileName, unsigned int subObjectId) {
	LogManager *log = LogManager::getSingletonPtr();

	// PLY object:
	PlyFile *ply;

	// PLY properties:
	char **elist;
	int nelems, num_elems;
	int file_type;
	float version;
	float plyScale = 1.0; // for prescaled models
	
	// open a stream of the file given to us
	FILE *plyFileStream = fopen(fileName, "rb");

	if (!plyFileStream) 
		return ERROR_FILENOTFOUND;

	// hand over the stream to the ply functions:
	ply = ply_read(plyFileStream, &nelems, &elist);

	if (ply == NULL) {
		log->logMessage("Error while parsing the PLY file!");
		return ERROR_CORRUPTFILE;
	}

	ply_get_info(ply, &version, &file_type);

	char outputBuffer[2000];
	sprintf(outputBuffer, "PLY file '%s' loaded, version %f, %d elements", fileName, version, nelems);
	log->logMessage(outputBuffer);

	// buffer to store vertices in while parsing:
	unsigned int curVert = 0;

	typedef struct PLYVertex{
		float coords[3];
		unsigned char color[3];
		void *other_props;		
	} PLYVertex;

	typedef struct PLYFace{
		unsigned char nverts;
		int *verts;		
		void *other_props;
	} PLYFace;

	PlyProperty vert_props[] = { /* list of property information for a vertex */
			{"x", PLY_FLOAT, PLY_FLOAT, 0, 0, 0, 0, 0},
			{"y", PLY_FLOAT, PLY_FLOAT, 4, 0, 0, 0, 0},
			{"z", PLY_FLOAT, PLY_FLOAT, 8, 0, 0, 0, 0},
			{"red", PLY_UCHAR, PLY_UCHAR, offsetof(PLYVertex,color[0]), 0, 0, 0, 0},
			{"green", PLY_UCHAR, PLY_UCHAR, offsetof(PLYVertex,color[1]), 0, 0, 0, 0},
			{"blue", PLY_UCHAR, PLY_UCHAR, offsetof(PLYVertex,color[2]), 0, 0, 0, 0},
	};

	PlyProperty face_props[] = { /* list of property information for a vertex */
		{"vertex_indices", PLY_INT, PLY_INT, offsetof(PLYFace,verts), 1, PLY_UCHAR, PLY_UCHAR, offsetof(PLYFace,nverts)},
	};

	// default PLY material:
	/*MaterialSpecular *plyMat = new MaterialSpecular();
	((MaterialSpecular *)plyMat)->setColor(rgb(0.9f, 0.9f, 0.9f));
	((MaterialSpecular *)plyMat)->setReflectance(0.9f); 
	((MaterialSpecular *)plyMat)->setOpacity(1.0f); */

	MaterialDiffuse *plyMat = new MaterialDiffuse(rgb(0.7f,0.7f,0.7f));	

	//if (objectList[subObjectId].materiallist.size() == 0)
	objectList[subObjectId].materiallist.push_back(plyMat);

	// needed for parsing elements:
	char *elem_name;
	PlyProperty **plist;
	PlyOtherProp *vert_other = NULL, *face_other = NULL;
	PlyOtherElems *other_elements = NULL;
	bool has_vertex_indices = false;
	bool has_vertex_x = false, has_vertex_y = false, has_vertex_z = false, has_colors = false;

	unsigned char color_components = 0;
	unsigned char *colorBuffer;

	typedef stdext::hash_map<unsigned int, unsigned int> ColorTable;
	typedef ColorTable::iterator ColorTableIterator;
	ColorTable usedColors;
	ColorTableIterator colorIter;


	int nprops;
	for (int i = 0; i < nelems; i++) {		

		// get the description of the first element (i.e. type)
		elem_name = elist[i];

		cout << "Element: " << i << " : " << elem_name << endl;

		plist = ply_get_element_description(ply, elem_name, &num_elems, &nprops);		

		// this is a vertex:
		if (equal_strings ("vertex", elem_name)) {
			int j;
			for (j=0; j<nprops; j++)
			{
				if (equal_strings("x", plist[j]->name))
				{
					ply_get_property (ply, elem_name, &vert_props[0]);  /* x */
					has_vertex_x = TRUE;
				}
				else if (equal_strings("y", plist[j]->name))
				{
					ply_get_property (ply, elem_name, &vert_props[1]);  /* y */
					has_vertex_y = TRUE;
				}
				else if (equal_strings("z", plist[j]->name))
				{
					ply_get_property (ply, elem_name, &vert_props[2]);  /* z */
					has_vertex_z = TRUE;
				}
				else if (equal_strings("red", plist[j]->name))
				{
					ply_get_property (ply, elem_name, &vert_props[3]);  /* z */
					color_components++;
				}
				else if (equal_strings("green", plist[j]->name))
				{
					ply_get_property (ply, elem_name, &vert_props[4]);  /* z */
					color_components++;
				}
				else if (equal_strings("blue", plist[j]->name))
				{
					ply_get_property (ply, elem_name, &vert_props[5]);  /* z */
					color_components++;
				}
			}

			has_colors = color_components == 3;

			//vert_other = ply_get_other_properties(ply, elem_name,offsetof(PLYVertex, other_props));

			// test for necessary properties
			if ((!has_vertex_x) || (!has_vertex_y) || (!has_vertex_z))
			{
				log->logMessage(LOG_WARNING, "Vertex with less than 3 coordinated detected. Output will most likely be corrupt!");
				continue;
			}

			// grab all the vertex elements
			PLYVertex plyNewVertex;		
			cout << "Allocating " << (num_elems*sizeof(_Vector4)) << " bytes of storage." << endl;
			objectList[subObjectId].vertices = new _Vector4[num_elems + 1];	

			// do we have vertex colors?
			if (has_colors) {
				colorBuffer = new unsigned char[num_elems*3];
				cout << "Allocating " << (num_elems*3) << " bytes of storage for vertex colors." << endl;
			}
			
			for (j = 0; j < num_elems; j++) {								
				ply_get_element(ply, (void *)&plyNewVertex);								
				
				if (has_colors) {
					colorBuffer[curVert*3] = plyNewVertex.color[0];
					colorBuffer[curVert*3 + 1] = plyNewVertex.color[1];
					colorBuffer[curVert*3 + 2] = plyNewVertex.color[2];
				}

				objectList[subObjectId].vertices[curVert++] = (plyScale * Vector3(plyNewVertex.coords));
				
				if (j != 0 && j%1000000 == 0) {
					cout << " - " << j << " of " << num_elems << " loaded." << endl;					
				}				
			}			
		}
		// this is a face (and, hopefully, a triangle):
		else if (equal_strings ("face", elem_name)) {
			int j;
			for (j=0; j<nprops; j++)
			{
				if (equal_strings("vertex_indices", plist[j]->name))
				{
					ply_get_property (ply, elem_name, &face_props[0]);  /* vertex_indices */
					has_vertex_indices = TRUE;
				}
			}

			//face_other = ply_get_other_properties (ply, elem_name, offsetof(PLYFace, other_props));

			/* test for necessary properties */
			if (!has_vertex_indices)
			{
				log->logMessage(LOG_WARNING, "Face without vertex indices detected in PLY file. Output will most likely be corrupt!");
				continue;
			}

			/* grab all the face elements */
			Triangle tri;
			PLYFace plyFace;	
			plyFace.other_props = NULL;			

			cout << "Allocating " << (num_elems*sizeof(Triangle)) << " bytes of storage." << endl;			
			objectList[subObjectId].trilist = new Triangle[num_elems];	

			for (j = 0; j < num_elems; j++) {				
				ply_get_element(ply, (void *)&plyFace);
				if (plyFace.nverts != 3) {
					log->logMessage(LOG_WARNING, "Face with more than 3 vertices detected. The importer will only read triangles, skipping...");
					continue;
				}

				//
				// make a triangle in our format from PLY face + vertices
				//						
				
				
				// copy vertex indices
				tri.p[0] = plyFace.verts[0];
				tri.p[1] = plyFace.verts[1];
				tri.p[2] = plyFace.verts[2];

				Vector3 n;
				Vector3 normals[3];
				Vector2 uv[3];

				// Calculate unit normal vector (check winding so it will
				// point the right way !)				
				//n = cross(tri.p[2] - tri.p[0], tri.p[1] - tri.p[0]);								
				n = cross(objectList[subObjectId].vertices[tri.p[1]] - objectList[subObjectId].vertices[tri.p[0]],
					      objectList[subObjectId].vertices[tri.p[2]] - objectList[subObjectId].vertices[tri.p[0]]);

				if (n.squaredLength() == 0.0) {
					n = cross(objectList[subObjectId].vertices[tri.p[2]] - objectList[subObjectId].vertices[tri.p[1]], 
						      objectList[subObjectId].vertices[tri.p[0]] - objectList[subObjectId].vertices[tri.p[1]]);

					if (n.squaredLength() == 0.0)
						n = cross(objectList[subObjectId].vertices[tri.p[0]] - objectList[subObjectId].vertices[tri.p[2]], 
								  objectList[subObjectId].vertices[tri.p[1]] - objectList[subObjectId].vertices[tri.p[2]]);
				}

				// Degenerate triangle, points must be collinear/identical:
				// skip this triangle
				if (n.squaredLength() == 0.0) {
					log->logMessage(LOG_WARNING, "Degenerate triangle (collinear points?) detected!");								
				}

				n.makeUnitVector();
				tri.n = n;

				// distance d of plane equation (really is -d, saves us one op in intersection, the
				// only place this is needed..)
				tri.d = dot(objectList[subObjectId].vertices[tri.p[0]], n);	

				// find best projection plane (YZ, XZ, XY)
				if (fabs(tri.n[0]) > fabs(tri.n[1]) && fabs(tri.n[0]) > fabs(tri.n[2])) {								
					tri.i1 = 1;
					tri.i2 = 2;
				}
				else if (fabs(tri.n[1]) > fabs(tri.n[2])) {								
					tri.i1 = 0;
					tri.i2 = 2;
				}
				else {								
					tri.i1 = 0;
					tri.i2 = 1;
				}

				int firstIdx;
				float u1list[3];
				u1list[0] = fabs(objectList[subObjectId].vertices[tri.p[1]].e[tri.i1] - objectList[subObjectId].vertices[tri.p[0]].e[tri.i1]);
				u1list[1] = fabs(objectList[subObjectId].vertices[tri.p[2]].e[tri.i1] - objectList[subObjectId].vertices[tri.p[1]].e[tri.i1]);
				u1list[2] = fabs(objectList[subObjectId].vertices[tri.p[0]].e[tri.i1] - objectList[subObjectId].vertices[tri.p[2]].e[tri.i1]);

				if (u1list[0] >= u1list[1] && u1list[0] >= u1list[2])
					firstIdx = 0;
				else if (u1list[1] >= u1list[2])
					firstIdx = 1;
				else
					firstIdx = 2;

				int secondIdx = (firstIdx + 1) % 3;
				int thirdIdx = (firstIdx + 2) % 3;

				// apply coordinate order to tri structure:
				tri.p[0] = plyFace.verts[firstIdx];
				tri.p[1] = plyFace.verts[secondIdx];
				tri.p[2] = plyFace.verts[thirdIdx];

				// vertex colors:
				#ifdef _USE_TRI_MATERIALS
				if (has_colors) {
					rgb color = rgb(colorBuffer[plyFace.verts[firstIdx]*3] / 255.0f,
									colorBuffer[plyFace.verts[firstIdx]*3 + 1] / 255.0f,
									colorBuffer[plyFace.verts[firstIdx]*3 + 2] / 255.0f);
					unsigned int hash = (unsigned int)(color.r() + 256*color.g() + 256*256*color.b());

					if ((colorIter = usedColors.find(hash)) != usedColors.end()) {
						tri.material = colorIter->second;
					}
					else {
						MaterialDiffuse *newMat = new MaterialDiffuse(color);						
						objectList[subObjectId].materiallist.push_back(newMat);
						tri.material = objectList[subObjectId].materiallist.size()-1;
						usedColors[hash] = tri.material;
					}
				}
				else // make default material 					
					tri.material = objectList[subObjectId].materiallist.size()-1;
				#endif



				free(plyFace.verts);

#ifdef _USE_VERTEX_NORMALS
				tri.normals[0] = n;
				tri.normals[1] = Vector3(0,0,0); // last two normals are encoded as difference to first one
				tri.normals[2] = Vector3(0,0,0);
#endif
				
				// insert triangle into list							
				objectList[subObjectId].trilist[objectList[subObjectId].nTris++] = tri;

				if (j != 0 && j%500000 == 0) {
					cout << " - " << j << " of " << num_elems << " loaded." << endl;					
				}
			}

			if (has_colors)
				delete [] colorBuffer;

		}
		else // otherwise: skip all further
			other_elements = ply_get_other_element (ply, elem_name, num_elems);

	}

	// PLY parsing ended, clean up vertex buffer and close the file		
	ply_close(ply);
	fclose(plyFileStream);

	// We neither have lights nor camera definition in the PLY file, so 
	// include generic ones:
	calculateAABoundingBox();

	if (subObjectId == 0)
		makeGenericSceneSettings();

	return STATUS_OK;
}
#endif // !_USE_OOC

int Scene::loadOOC(const char *dirName, unsigned int subObjectId) {
	LogManager *log = LogManager::getSingletonPtr();
	OptionManager *opt = OptionManager::getSingletonPtr();
	char fileNameTri[MAX_PATH], fileNameVertex[MAX_PATH], fileNameMaterial[MAX_PATH], output[1000];

	#ifdef USE_LOD
	char fileNameLOD[MAX_PATH];
	#endif

	FILE *vertexFile = NULL;
	FILE *triFile = NULL;
	FILE *lodFile = NULL;
	FILE *materialFile = NULL;
	

	#if COMMON_PLATFORM == PLATFORM_WIN32
	struct stat fileInfo;
	#else
	stat fileInfo;
	#endif

	//#ifdef _USE_TRI_MATERIALS
	//objectList[subObjectId].materiallist.clear();
	//#endif


	#ifdef _USE_RACM
	char fileNameRACM[MAX_PATH];
	char fileNameRACMMeta[MAX_PATH];
	sprintf(fileNameRACM, "%s/RACM", dirName);
	sprintf(fileNameRACMMeta, "%s/meta.h", fileNameRACM);
	FILE *fpTemp = fopen(fileNameRACMMeta, "r");
	objectList[subObjectId].useRACM = fpTemp != NULL;
	if(fpTemp!=NULL)fclose(fpTemp);
	if(objectList[subObjectId].useRACM)
	{
		objectList[subObjectId].pRACM = new CCompressedMesh;
		objectList[subObjectId].pRACM->PrepareData(fileNameRACM ,true);
		objectList[subObjectId].nVerts = objectList[subObjectId].pRACM->GetNumVertices();
		objectList[subObjectId].nTris = objectList[subObjectId].pRACM->GetNumFaces();
	}
	#endif

	if(!objectList[subObjectId].useRACM)
	{
		sprintf(fileNameVertex, "%s/vertex.ooc", dirName);
		if ((vertexFile = fopen(fileNameVertex, "rb")) == NULL) {
			sprintf(output, "loadOOC(): could not find %s in directory!", fileNameVertex);
			log->logMessage(LOG_ERROR, output);
			return ERROR_FILENOTFOUND;
		}

		// get file information
		stat(fileNameVertex, &fileInfo);
		objectList[subObjectId].nVerts = fileInfo.st_size / sizeof(_Vector4);

		sprintf(fileNameTri, "%s/tris.ooc", dirName);
		if ((triFile = fopen(fileNameTri, "rb")) == NULL) {
			sprintf(output, "loadOOC(): could not find %s in directory!", fileNameTri);
			log->logMessage(LOG_ERROR, output);
			fclose(vertexFile); 
			return ERROR_FILENOTFOUND;
		}	

		stat(fileNameTri, &fileInfo);
		objectList[subObjectId].nTris = fileInfo.st_size / sizeof(Triangle);
	}
	
#ifdef USE_LOD
	sprintf(fileNameLOD, "%s/kdtree.lod", dirName);
	if ((lodFile = fopen(fileNameLOD, "rb")) == NULL) {
		sprintf(output, "loadOOC(): could not find %s in directory!", fileNameLOD);
		log->logMessage(LOG_ERROR, output);
		fclose(vertexFile);
		fclose(triFile);
		return ERROR_FILENOTFOUND;
	}

	// get file information
	stat(fileNameLOD, &fileInfo);
	objectList[subObjectId].nLODs = fileInfo.st_size / sizeof(LODNode);
#endif


	sprintf(fileNameMaterial, "%s/materials.ooc", dirName);
	if ((materialFile = fopen(fileNameMaterial, "rb")) == NULL) {
		sprintf(output, "loadOOC(): could not find %s in directory!", fileNameMaterial);
		log->logMessage(LOG_ERROR, output);
		fclose(vertexFile);
		fclose(triFile);
		return ERROR_FILENOTFOUND;
	}

	stat(fileNameMaterial, &fileInfo);
	unsigned int nMaterials = fileInfo.st_size / sizeof(MaterialDiffuse);

	// Normal load, just dumps the entire files into two arrays,
	// just works for smaller scenes for obvious reasons:
	#ifndef _USE_OOC
	
	cout << "Allocating " << (objectList[subObjectId].nVerts*sizeof(Vector3)) << " bytes of storage for " << objectList[subObjectId].nVerts << " vertices." << endl;
	objectList[subObjectId].vertices = new _Vector4[objectList[subObjectId].nVerts + 1];	

	cout << "Allocating " << (objectList[subObjectId].nTris*sizeof(Triangle)) << " bytes of storage for " << objectList[subObjectId].nTris << " triangles." << endl;			
	objectList[subObjectId].trilist = new Triangle[objectList[subObjectId].nTris + 1];

	// vertices:
	fread(objectList[subObjectId].vertices, sizeof(_Vector4), objectList[subObjectId].nVerts, vertexFile);
	fclose(vertexFile);

	// triangles:
	fread(objectList[subObjectId].trilist, sizeof(Triangle), objectList[subObjectId].nTris, triFile);
	fclose(triFile);
	
	#ifdef USE_LOD
	cout << "Allocating " << (objectList[subObjectId].nLODs*sizeof(LODNode)) << " bytes of storage for " << objectList[subObjectId].nLODs << " LODs." << endl;			
	objectList[subObjectId].lodlist = new LODNode[objectList[subObjectId].nLODs + 1];
	fread(objectList[subObjectId].lodlist, sizeof(LODNode), objectList[subObjectId].nLODs, lodFile);
	fclose(lodFile);
	#endif

	#else 
	// OOC load, open the vertex and triangle list via memory-mapped files
	if(!objectList[subObjectId].useRACM)
	{
		fclose(triFile);
		fclose(vertexFile);
	}

	#ifdef _USE_OOC_DIRECTMM
	objectList[subObjectId].trilist = (SceneTriangleList)allocateFullMemoryMap(fileNameTri);
	objectList[subObjectId].vertices = (SceneVertexList)allocateFullMemoryMap(fileNameVertex);
	#ifdef USE_LOD
	objectList[subObjectId].lodlist = (SceneLodList)allocateFullMemoryMap(fileNameLOD);		
	#endif 

	#else // !_USE_OOC_DIRECTMM

  /*
	objectList[subObjectId].trilist = new OOC_TRI_FILECLASS<Triangle>(fileNameTri, 
						   			  1024*1024*opt->getOptionAsInt("ooc", "maxCacheMemTrisMB", 512),									   
									  1024*opt->getOptionAsInt("ooc", "cacheEntrySizeTrisKB", 1024*4));	
	objectList[subObjectId].vertices = new OOC_VERTEX_FILECLASS<_Vector4>(fileNameVertex, 
									1024*1024*opt->getOptionAsInt("ooc", "maxCacheMemVerticesMB", 256),
		    						1024*opt->getOptionAsInt("ooc", "cacheEntrySizeVerticesKB", 1024*2));		
  */

	/*
<<<<<<< .mine

=======
*/
	if(!objectList[subObjectId].useRACM)
	{
	  char MappingMode [255];
	  strcpy (MappingMode, "r");
	  
	  int NumElement = TRI_PAGE_SIZE;
	  int NumPages = opt->getOptionAsInt("ooc", "maxCacheMemTrisMB", 512) * 1024 * 1024 / (NumElement * sizeof (Triangle));
	  objectList[subObjectId].trilist = new CMemManager <Triangle> ();
	  objectList[subObjectId].trilist->Init (fileNameTri, objectList[subObjectId].nTris, NumPages, NumElement, true, MappingMode);

	  NumElement = VER_PAGE_SIZE*2;   // to align with 64K
	  NumPages = opt->getOptionAsInt("ooc", "maxCacheMemVerticesMB", 256) * 1024 * 1024 / (NumElement * sizeof (_Vector4));
	  objectList[subObjectId].vertices = new CMemManager <_Vector4> ();
	  objectList[subObjectId].vertices->Init (fileNameVertex, objectList[subObjectId].nVerts, NumPages, NumElement, true, MappingMode);
	}


/*
>>>>>>> .r2405
*/
	#ifdef USE_LOD
	printf ("LOD OOC prepare\n");
	objectList[subObjectId].lodlist = new OOC_LOD_FILECLASS<LODNode>(fileNameLOD, 
									1024*1024*opt->getOptionAsInt("ooc", "maxCacheMemLODsMB", 256),
									1024*opt->getOptionAsInt("ooc", "cacheEntrySizeLODsKB", 1024*2));									
		    						//1024*opt->getOptionAsInt("ooc", "cacheEntrySizeVerticesKB", 88*8*64));									
	#endif // USE_LOD

	#endif // !_USE_OOC_DIRECTMM
	#endif // _USE_OOC

	// materials:
	MaterialDiffuse *tempMaterialList = new MaterialDiffuse[nMaterials];
	fread(tempMaterialList, sizeof(MaterialDiffuse), nMaterials, materialFile);
	fclose(materialFile);	

	objectList[subObjectId].materiallist.clear();
	for (unsigned int i = 0; i < nMaterials; i++) {
		Material *newMat = new MaterialDiffuse(tempMaterialList[i]);
		//((MaterialDiffuse *) newMat)->setColor(rgb(0.1f, 0.1f, 0.1f));
		

		/*Material *newMat = new MaterialSpecular;

		((MaterialSpecular *) newMat)->c = tempMaterialList[i].getColor ();
		((MaterialSpecular *) newMat)->opacity = 0.8f;
		((MaterialSpecular *) newMat)->specularReflectance = 0.2f;*/

		objectList[subObjectId].materiallist.push_back(newMat);
	}

	//delete tempMaterialList;
	// note: we don't deallocate the material's memory since the main material list
	// is just a vector of pointers, and we need the array to still be in memory. The 

	return STATUS_OK;
}


void Scene::printScene(const char *LoggerName) {
	LogManager *log = LogManager::getSingletonPtr();

	printSceneStats();

	// Output camera positions:
	log->logMessage("Cameras:", LoggerName);
	for (ViewListIterator i = viewlist.begin(); i != viewlist.end(); i++) {
		sprintf(outputBuffer, "Camera \"%s\" at (%.2f,%.2f,%.2f), Dir (%.2f,%.2f,%.2f)", (*i).name,
			(*i).view.origin().x(), (*i).view.origin().y(), (*i).view.origin().z(),
			(*i).view.direction().x(), (*i).view.direction().y(), (*i).view.direction().z());			
		log->logMessage(outputBuffer, LoggerName);
	}

	log->logMessage("-------------------------------------------", LoggerName);
	log->logMessage("Lights:", LoggerName);

	for (LightListIterator j = lightlist.begin(); j != lightlist.end(); j++) {
		sprintf(outputBuffer, "Light at (%.4f,%.4f,%.4f), Intensity %.2f, Color (%.2f,%.2f,%.2f)", 
			    (*j).pos.x(), (*j).pos.y(), (*j).pos.z(),
				(*j).intensity, (*j).color.r(), (*j).color.g(), (*j).color.b());
		log->logMessage(outputBuffer, LoggerName);
	}	
		
	log->logMessage("Emitters:", LoggerName);

	for (EmitterListIterator k = emitterlist.begin(); k != emitterlist.end(); k++) {
		sprintf(outputBuffer, "Emitter at (%.2f,%.2f,%.2f), Intensity %.2f, Area %.2f", 
			(*k).p[0].x(), (*k).p[0].y(), (*k).p[0].z(),
			(*k).summedIntensity, (*k).area);
		log->logMessage(outputBuffer, LoggerName);
	}

	log->logMessage("-------------------------------------------", LoggerName);
}

void Scene::printSceneStats(const char *LoggerName) {	
	LogManager *log = LogManager::getSingletonPtr();
	char outputBuffer[2000];
	log->logMessage("-------------------------------------------", LoggerName);
	log->logMessage("Scene Statistics", LoggerName);
	log->logMessage("-------------------------------------------", LoggerName);
	sprintf(outputBuffer, "Name:\t\t\"%s\"", sceneFileName);
	log->logMessage(outputBuffer, LoggerName);

	unsigned int realTriangles = 0;
	unsigned int instancedTriangles = 0;
	unsigned int nMaterials = 0;

	for (unsigned int i = 0; i < nSubObjects; i++) {		
		if (!objectList[i].isInstanced)
			realTriangles += objectList[i].nTris;
		instancedTriangles += objectList[i].nTris;
		nMaterials += objectList[i].materiallist.size();
	}

	if (hasMultipleObjects) {
		sprintf(outputBuffer, "Objects:\t\t%d", nSubObjects);
		log->logMessage(outputBuffer, LoggerName);

		if (instancedTriangles != realTriangles) {
			sprintf(outputBuffer, "Triangles (w/ instanced):\t%d", instancedTriangles);
			log->logMessage(outputBuffer, LoggerName);	
			sprintf(outputBuffer, "Triangles (actual):\t%d", realTriangles);
			log->logMessage(outputBuffer, LoggerName);	
		}	
		else {
			sprintf(outputBuffer, "Triangles:\t%d", realTriangles);
			log->logMessage(outputBuffer, LoggerName);		
		}
	}
	else {
		sprintf(outputBuffer, "Triangles:\t%d", realTriangles);
		log->logMessage(outputBuffer, LoggerName);		
	}

	
	sprintf(outputBuffer, "Materials:\t%d",nMaterials);
	log->logMessage(outputBuffer, LoggerName);

	sprintf(outputBuffer, "Lights:\t\t%d", lightlist.size());
	log->logMessage(outputBuffer, LoggerName);
	sprintf(outputBuffer, "Viewpoints:\t%d", viewlist.size());
	log->logMessage(outputBuffer, LoggerName);
	sprintf(outputBuffer, "Bounding Box:\t(%.3f,%.3f,%.3f) - (%.3f,%.3f,%.3f)", \
		highLevelTree->root->bbox.pp[0].x(), highLevelTree->root->bbox.pp[0].y(), highLevelTree->root->bbox.pp[0].z(), \
		highLevelTree->root->bbox.pp[1].x(), highLevelTree->root->bbox.pp[1].y(), highLevelTree->root->bbox.pp[1].z());
	log->logMessage(outputBuffer, LoggerName);
	log->logMessage("-------------------------------------------", LoggerName);	
}

void Scene::printRayStats(const char *LoggerName) {
	LogManager *log = LogManager::getSingletonPtr();
	char outputBuffer[2000];
	log->logMessage("-------------------------------------------", LoggerName);
	log->logMessage("Ray Time Statistics", LoggerName);
	log->logMessage("-------------------------------------------", LoggerName);
	
	sprintf(outputBuffer, "Rays: %d primary, %d shadow, %d reflection, %d refraction", nPrimaryRays, nShadowRays, nReflectionRays, nRefractionRays);
	log->logMessage(outputBuffer, LoggerName);

	sprintf(outputBuffer, "Primary rays: %I64u cycles total (%I64u cycles / ray)", statsTimerPrimary.getElapsed(), statsTimerPrimary.getElapsed() / nPrimaryRays);
	log->logMessage(outputBuffer, LoggerName);

	if (nShadowRays + nReflectionRays + nRefractionRays) {	
		sprintf(outputBuffer, "Secondary rays: %I64u cycles total (%I64u cycles / ray)", statsTimerShadow.getElapsed() + statsTimerReflection.getElapsed() + statsTimerRefraction.getElapsed(), 
																				(statsTimerShadow.getElapsed() + statsTimerReflection.getElapsed() + statsTimerRefraction.getElapsed()) / (nShadowRays + nReflectionRays + nRefractionRays));
		log->logMessage(outputBuffer, LoggerName);
	}

	if (nShadowRays) {
		sprintf(outputBuffer, " ->     shadow: %I64u cycles total (%I64u cycles / ray)", statsTimerShadow.getElapsed(), statsTimerShadow.getElapsed() / nShadowRays);
		log->logMessage(outputBuffer, LoggerName);
	}
	if (nReflectionRays) {
		sprintf(outputBuffer, " -> reflection: %I64u cycles total (%I64u cycles / ray)", statsTimerReflection.getElapsed(), statsTimerReflection.getElapsed() / nReflectionRays);
		log->logMessage(outputBuffer, LoggerName);
	}
	if (nRefractionRays) {
		sprintf(outputBuffer, " -> refraction: %I64u cycles total (%I64u cycles / ray)", statsTimerRefraction.getElapsed(), statsTimerRefraction.getElapsed() / nRefractionRays);	
		log->logMessage(outputBuffer, LoggerName);
	}
}


const char *Scene::getLastErrorString() {
	return lastError;
}
int Scene::getLastError() {
	int ret = lastErrorCode;
	lastErrorCode = 0;
	return ret;
}

void Scene::resetScene() {
	OptionManager *opt = OptionManager::getSingletonPtr();

	maxRecursionDepth = opt->getOptionAsInt("raytracing", "maxRecursionDepth", 5);
	m_nShadowRays = opt->getOptionAsInt("raytracing", "numShadowRays", 1);
	m_nShadowRayFactor = 1.0f / (float)m_nShadowRays;

	if (photonMap)
		delete photonMap;
	
	for (int i = 0; i < nSubObjects; ++i) {
		if (objectList) {
			objectList[i].bvh->finalize();
			delete objectList[i].bvh;
		}
	}

	if (background)
		delete background;	

	if (highLevelTree)
		delete highLevelTree;

	nSubObjects = 1;

	lightlist.clear();
	emitterlist.clear();
	
	lastErrorCode;
	lastError[0] = 0;

	sceneFileName[0] = 0;

	nPrimaryRays = 0;
	nShadowRays = 0;
	nReflectionRays = 0;
	nRefractionRays = 0;
	nSingleShadowRays = 0;
	nSIMDShadowRays = 0;
	nSIMDReflectionRays = 0;
	nSingleReflectionRays = 0;

	// insert one default viewer
	View defView;
	viewlist.clear();
	sprintf(defView.name, "Default");
	defView.view.setOrigin(Vector3(0,0,0));
	defView.view.setDirection(Vector3(0,0,-1));
	viewlist.push_back(defView);

#ifdef _USE_AREA_LIGHT
	if (areaLight)
		delete areaLight;
#endif
}

void Scene::calculateAABoundingBox(unsigned int subObjectId) {
	bb_min = Vector3(FLT_MAX,FLT_MAX,FLT_MAX);
	bb_max = Vector3(-FLT_MAX,-FLT_MAX,-FLT_MAX);

	for (unsigned int i = 0; i < nSubObjects; i++) {
		if (objectList[i].bb[0][0] == -FLT_MAX)
			objectList[i].calculateAABoundingBox();
		
		for (int k = 0; k<3; k++) {	
			bb_min[k] = min(bb_min[k], objectList[i].bb[0][k]);
			bb_max[k] = max(bb_max[k], objectList[i].bb[1][k]);
		}
	}

	objectList[subObjectId].bvh->objectBB[0] = bb_min;
	objectList[subObjectId].bvh->objectBB[1] = bb_max;
}

void Scene::makeGenericSceneSettings() {

#ifdef PLY_RANDOM_LIGHTS
	// insert number of random lights into PLY scene:

	// centroid point & starting point
	Vector3 bb_center = (bb_max + bb_min)*0.5f;
	bb_center[1] = bb_max[1];
	Vector3 startPos = bb_max + 0.1f * (bb_max - bb_min);	

	// RNG
	RandomLinear rotationSampler;

	for (int l=0; l < PLY_NUM_RANDOM_LIGHTS; l++) {
		Light genericLight;
		Vector3 curPos = startPos - bb_center;
		
		genericLight.intensity = 2.0f / (float)PLY_NUM_RANDOM_LIGHTS;

		genericLight.color = rgb(genericLight.intensity,genericLight.intensity,genericLight.intensity);
		genericLight.type = LIGHT_POINT;

		// generate random rotation angle around center
		float phi = rotationSampler.sample() * 2.0f * PI;
		
		curPos[0] = curPos[0] * sin(phi);
		curPos[1] = curPos[1];
		curPos[2] = curPos[2] * cos(phi);

		genericLight.pos = curPos + bb_center;
		lightlist.push_back(genericLight);
	}

	#else  // simple light, put outside the scene bounding box:

#ifndef _USE_AREA_LIGHT
	//
	// point light
	//
	Light genericLight;
	genericLight.color = rgb(1,1,1);
	genericLight.intensity = 2.0;
	genericLight.type = LIGHT_POINT;

	//genericLight.type = LIGHT_DIRECTIONAL;
	//genericLight.pos = bb_max + 0.1f * (bb_max - bb_min);	

	genericLight.direction = Vector3 (1, 1, 1);
	genericLight.pos = bb_max + 10.1f * (bb_max - bb_min);	

	//genericLight.pos = (bb_max + bb_min)/2. + Vector3 (10000, 10000, 0);
	//	genericLight.pos = (bb_max + bb_min)/2. + Vector3 (0, 10000, 10000);

	//genericLight.pos = Vector3(1.0f, 4.0f, -2.0f);

	lightlist.push_back(genericLight);
#else
	//
	// Area light
	// 

	//float paramAreaLightPos = (bb_max - bb_min).maxAbsComponent() * 3.0f;
	//float paramAreaLightSize = (bb_max - bb_min).maxAbsComponent() * 0.1f;
	//Vector3 areaLightPos(paramAreaLightPos, paramAreaLightPos, -paramAreaLightPos);
	//areaLight = new AreaLight(areaLightPos, paramAreaLightSize, paramAreaLightSize, m_nShadowRays);
	//areaLight->generateAreaLight(&lightlist);		

	//LightList lightlist2;
	//Vector3 areaLightPos2(-2.0f, 2.0f, 2.0f);	
	//AreaLight* areaLight2 = new AreaLight(areaLightPos2, paramAreaLightSize, paramAreaLightSize, m_nShadowRays);
	//areaLight2->generateAreaLight(&lightlist2);

	//for (int ii=0; ii<lightlist2.size(); ++ii)
	//	lightlist.push_back(lightlist2[ii]);

	/*
	 * DOE 128M, BB Info <539, 539, 500> ~ <1459, 1459, 1269>
	 */
	/////////////////////////////////////////////////////////////
	// inside light
	//float paramAreaLightSize = (bb_max - bb_min).maxAbsComponent() * 0.02f;

	//// outside light
	//// right
	//LightList lightlist2;
	//Vector3 areaLightPos2(400.f, 1000.f, 800.0f);	
	//AreaLight* areaLight2 = new AreaLight(areaLightPos2, paramAreaLightSize, paramAreaLightSize, m_nShadowRays, OUTSIDE_LIGHT);
	//areaLight2->setColor(rgb(1.f, 1.f, 1.f));
	//areaLight2->generateAreaLight(&lightlist2);

	//for (int ii=0; ii<lightlist2.size(); ++ii)
	//	lightlist.push_back(lightlist2[ii]);

	//// left
	//areaLightPos2 = Vector3(1800.f, 1000.f, 800.f);	
	//areaLight2 = new AreaLight(areaLightPos2, paramAreaLightSize, paramAreaLightSize, m_nShadowRays, OUTSIDE_LIGHT);
	//areaLight2->setColor(rgb(1.f, 1.f, 1.f));
	//areaLight2->generateAreaLight(&lightlist2);

	//for (int ii=0; ii<lightlist2.size(); ++ii)
	//	lightlist.push_back(lightlist2[ii]);

	//// key Light
	//areaLightPos2 = Vector3(1000.f, 1000.0f, 1500.f);	
	//areaLight2 = new AreaLight(areaLightPos2, paramAreaLightSize, paramAreaLightSize, m_nShadowRays, OUTSIDE_LIGHT);
	//areaLight2->setColor(rgb(1.f, 1.f, 1.f));
	//areaLight2->generateAreaLight(&lightlist2);

	//for (int ii=0; ii<lightlist2.size(); ++ii)
	//	lightlist.push_back(lightlist2[ii]);
	/////////////////////////////////////////////////////////////

	/*
	 * Lucy & Blade
	 */
	/////////////////////////////////////////////////////////////
	// inside light
	//float paramAreaLightPos = (bb_max - bb_min).maxAbsComponent() * 3.0f;
	//float paramAreaLightSize = (bb_max - bb_min).maxAbsComponent() * 0.02f;
	//Vector3 areaLightPos(100.0f, 1200.0f, 1200.0f);
	//areaLight = new AreaLight(areaLightPos, paramAreaLightSize, paramAreaLightSize, m_nShadowRays, INSIDE_LIGHT);
	//areaLight->setColor(rgb(1.f, 0.8f, 0.6f));
	////areaLight->generateAreaLight(&lightlist);		

	//// outside light
	//// right
	//LightList lightlist2;
	//Vector3 areaLightPos2(paramAreaLightPos, 10.0f, 0.0f);	
	//AreaLight* areaLight2 = new AreaLight(areaLightPos2, paramAreaLightSize, paramAreaLightSize, m_nShadowRays, OUTSIDE_LIGHT);
	//areaLight2->setColor(rgb(1.f, 0.8f, 0.6f));
	//areaLight2->generateAreaLight(&lightlist2);

	//for (int ii=0; ii<lightlist2.size(); ++ii)
	//	lightlist.push_back(lightlist2[ii]);

	//// left
	//areaLightPos2 = Vector3(-paramAreaLightPos, 10.0f, 0.0f);	
	//areaLight2 = new AreaLight(areaLightPos2, paramAreaLightSize, paramAreaLightSize, m_nShadowRays, OUTSIDE_LIGHT);
	//areaLight2->setColor(rgb(1.f, 0.8f, 0.6f));
	//areaLight2->generateAreaLight(&lightlist2);

	//for (int ii=0; ii<lightlist2.size(); ++ii)
	//	lightlist.push_back(lightlist2[ii]);

	//// key Light
	//areaLightPos2 = Vector3(0.1f, 10.0f, 10.f);	
	//areaLight2 = new AreaLight(areaLightPos2, paramAreaLightSize, paramAreaLightSize, m_nShadowRays, OUTSIDE_LIGHT);
	//areaLight2->setColor(rgb(1.f, 0.8f, 0.6f));
	//areaLight2->generateAreaLight(&lightlist2);

	//for (int ii=0; ii<lightlist2.size(); ++ii)
	//	lightlist.push_back(lightlist2[ii]);
	/////////////////////////////////////////////////////////////


	/*
	 * ST.matthew & builnd Scene (Light)
	 */
	/////////////////////////////////////////////////////////////
	// inside light
	float paramAreaLightPos = (bb_max - bb_min).maxAbsComponent() * 3.0f;
	float paramAreaLightSize = (bb_max - bb_min).maxAbsComponent() * 0.02f;
	Vector3 areaLightPos(1000.0f, 9000.0f, 6000.0f);
	areaLight = new AreaLight(areaLightPos, paramAreaLightSize, paramAreaLightSize, m_nShadowRays, INSIDE_LIGHT);
	areaLight->setColor(rgb(1.f, 0.8f, 0.6f));
	areaLight->generateAreaLight(&lightlist);		

	// outside light
	// right
	LightList lightlist2;
	Vector3 areaLightPos2(20000.f, 40000.0f, 5000.0f);	
	AreaLight* areaLight2 = new AreaLight(areaLightPos2, paramAreaLightSize, paramAreaLightSize, 1, OUTSIDE_LIGHT);
	areaLight2->setColor(rgb(0.6f, 0.6f, 0.6f));
	areaLight2->generateAreaLight(&lightlist2);

	for (int ii=0; ii<lightlist2.size(); ++ii)
		lightlist.push_back(lightlist2[ii]);

	// left
	areaLightPos2 = Vector3(-20000.f, 40000.0f, 5000.0f);	
	areaLight2 = new AreaLight(areaLightPos2, paramAreaLightSize, paramAreaLightSize, 1, OUTSIDE_LIGHT);
	areaLight2->setColor(rgb(0.6f, 0.6f, 0.6f));
	areaLight2->generateAreaLight(&lightlist2);

	for (int ii=0; ii<lightlist2.size(); ++ii)
		lightlist.push_back(lightlist2[ii]);

	// key Light
	areaLightPos2 = Vector3(10.0f, 20000.0f, 30000.f);	
	areaLight2 = new AreaLight(areaLightPos2, paramAreaLightSize, paramAreaLightSize, 1, OUTSIDE_LIGHT);
	areaLight2->setColor(rgb(0.6f, 0.6f, 0.6f));
	areaLight2->generateAreaLight(&lightlist2);

	for (int ii=0; ii<lightlist2.size(); ++ii)
		lightlist.push_back(lightlist2[ii]);
	/////////////////////////////////////////////////////////////


#endif

	#endif

	View newView;
	sprintf(newView.name, "Default View");	

	// Position		
	float exVScale = 0.1f;
	Vector3 exVecScale =  exVScale*(bb_max - bb_min);
	Vector3 viewPos = Vector3(bb_max[0] - exVecScale[0], bb_max[1], bb_max[2] + exVecScale[2]);
	Vector3 viewDir = ((bb_max + bb_min)/2.0f + Vector3(0, exVecScale[1], 0)) - viewPos;		
	//	Vector3 viewDir = ((bb_max + bb_min)/2.0f) - viewPos;	
	//float paramViewPos = (bb_max - bb_min).maxAbsComponent() * 1.0f;
	//Vector3 viewPos(paramViewPos, paramViewPos, paramViewPos);
	//Vector3 viewDir = Vector3(0.0f,0.0f,0.0f) - viewPos;


	/*
	 * ST.matthew & builnd Scene (Start View)
	 */
	/////////////////////////////////////////////////////////////
//	Vector3 viewPos(0.0f, 5000.0f, 15000.0f);
//	Vector3 viewDir = Vector3(0.0f, 5000.0f, 0.0f) - viewPos;
	/////////////////////////////////////////////////////////////

	/*
	 * DOE 128M, BB Info <539, 539, 500> ~ <1459, 1459, 1269>
	 */
	/////////////////////////////////////////////////////////////
	//Vector3 viewPos(1000.0f, 1000.0f, 2000.0f);
	//Vector3 viewDir = Vector3(1000.0f, 1000.0f, 1500.0f) - viewPos;
	/////////////////////////////////////////////////////////////

	newView.view.setOrigin(viewPos);	
	viewDir.makeUnitVector();
	newView.view.setDirection(viewDir);

	// Check: if this is the first view, remove the default view
	// from the list:
	if (viewlist.size() == 1 && strcmp("Default", viewlist[0].name) == 0)
		viewlist.clear();

	viewlist.push_back(newView);
}

/************************************************************************/
/* ´ö¼ö, Collision Detection                                            */
/************************************************************************/
void Scene::collisionDetection(){
	highLevelTree->collisionDetection() ;
}