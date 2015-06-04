#ifndef COMMON_CAMERA_H
#define COMMON_CAMERA_H

#include "Geometry.h"
#include "Image.h"
#include "Scene.h"

/********************************************************************
	created:	2004/06/19
	created:	19.6.2004   16:35
	filename: 	c:\MSDev\MyProjects\Renderer\Common\Camera.h
	file base:	Camera
	file ext:	h
	author:		Christian Lauterbach (lauterb@informatik.uni-bremen.de)
	
	purpose:	Camera class, generates rays to shoot into the scene
*********************************************************************/


class Camera  
{
public:

	Camera() {}

	// Copy constructor
	Camera(const Camera& orig)
	{
		center = orig.center;
		corner = orig.corner;
		across = orig.across;
		up     = orig.up;
		uvw    = orig.uvw;
		left      = orig.left;
		right      = orig.right;
		bottom      = orig.bottom;
		top      = orig.top;
		distance      = orig.distance;
		lens_radius = orig.lens_radius;
		scene  = scene;
		recalculateProjection();

		timeLastFrame = 0.0f;
		timeAllFrames = 0.0f;
		bSaveImage = orig.bSaveImage;
		outImage = NULL;
	} 

	Camera(Scene &scene,			// Scene Object
		   int width,				// Image Dimensions
		   int height,				//
		   Vector3 position  = Vector3(0,0,0),		// Viewer position
		   Vector3 direction = Vector3(0,0,-1),		// Viewing direction
		   Vector3 vup       = Vector3(0,1,0)			// Up vector
		  )	
		: center(position), direction(direction), vup(vup), scene(&scene)
	{
		assert(width > 0);
		assert(height > 0);

		// require image width and height to be a multiple of 2:
		width += (width % 2);
		height += (height % 2);

		this->width = width;
		this->height = height;

		tile_width = 16;
		tile_height = 16;

		left = -1.0;	// dimensions of projection plane
		right = 1.0f;	
		bottom = -1.0f; 
		top = 1.0f;		
		
		distance = 2.0f;	// distance to projection plane		
		distance = 1.4142f; // (GLVU compatible)

		lens_radius = 0.1f/2.0f; // the camera's aperture (radius of lens)
		setViewToPreset(0);
		recalculateProjection();

		timeLastFrame = 0.0f;
		timeAllFrames = 0.0f;
		bSaveImage = true;
		outImage = NULL;
	}

	~Camera() {
		LogManager *log = LogManager::getSingletonPtr();
		log->logMessage("Destroying Camera...");

		if (outImage)
			delete outImage;
		if (scene)
			delete scene;
	}

	/**
	 * Render a frame of the specified scene. 
	 */
	bool renderFrame(int frameNum);

	/**
	 * Set view position manually
	 */
	void setViewer(Ray ray) {
		direction = ray.direction();
		center = ray.origin();
		recalculateProjection();
	}

	Ray getViewer() {
		return Ray(center, direction);		
	}

	void viewFromLight(int num = 0) {
		Vector3 bb_min, bb_max, bb_dim, bb_center;
		scene->getAABoundingBox(bb_min, bb_max);
		bb_dim = bb_max - bb_min;
		bb_center = (bb_max + bb_min) / 2.0f;
		
		float exVScale = 0.2f;
		Vector3 exVecScale =  exVScale*bb_dim;

		// look at scene center
		Vector3 lookAt = (bb_center + Vector3(0, exVecScale[1], 0));	

		// viewer pos = light pos
		center = scene->getLightPos(0);

		direction = lookAt - center;
		direction.makeUnitVector();
		recalculateProjection();
	}

	void setLightViewer(Vector3 pos) {
		scene->setLightPos(0, pos);
	}

	//void animate(float timeDifference) {
	//	if (scene)
	//		scene->animateObjects(timeDifference);
	//}

	int animate() {
		if (scene)
			return scene->animateObjects();
		else return -1;
	}

	ViewList &getPresetViews();
	View &getPresetView(unsigned int preset_number);
	void setViewToPreset(unsigned int preset_number);
	void setViewToPresetByName(const char *name);

	/**
	* Get a viewpoint at point to look at for a rotation
	* animation around the scene. The viewer application will
	* then rotate around the center point.
	**/
	void getAnimationView(Vector3 &viewPos, Vector3 &lookAt) {
		Vector3 bb_min, bb_max, bb_dim, bb_center;
		scene->getAABoundingBox(bb_min, bb_max);
		bb_dim = bb_max - bb_min;
		bb_center = (bb_max + bb_min) / 2.0f;

		int axis = (bb_dim[0] >= bb_dim[2])?0:2;
		int axis2 = 2 - axis;
		int axis3 = (bb_dim[axis] >= bb_dim[1])?axis:1;
		 		
		float exVScale = 0.2f;
		Vector3 exVecScale =  exVScale*bb_dim;

		// Position		
		viewPos[axis]  = bb_center[axis];
		viewPos[1]     = bb_center[1] + exVecScale[1];
		viewPos[axis2] = bb_min[axis2] - bb_dim[axis3];
		
		viewPos = Vector3(bb_min[0] - exVecScale[0], bb_max[1] + exVecScale[1], bb_max[2] + exVecScale[2]);

		lookAt = (bb_center + Vector3(0, exVecScale[1], 0));					
	}

	void printFrameTimes() {
		LogManager *log = LogManager::getSingletonPtr();
		sprintf(timeStringBuffer, "Elapsed time for this frame: %d seconds, %d milliseconds", (int)timeLastFrame, (int)((timeLastFrame - floor(timeLastFrame)) * 1000));
		log->logMessage(LOG_INFO, timeStringBuffer);
		sprintf(timeStringBuffer, "Rays: %d primary, %d shadow, %d reflection, %d refraction", scene->nPrimaryRays, scene->nShadowRays, scene->nReflectionRays, scene->nRefractionRays);
		log->logMessage(LOG_INFO, timeStringBuffer);
		sprintf(timeStringBuffer, "Shadow rays: %d SIMD, %d single", scene->nSIMDShadowRays, scene->nSingleShadowRays);
		log->logMessage(LOG_INFO, timeStringBuffer);
		sprintf(timeStringBuffer, "Reflection rays: %d SIMD, %d single", scene->nSIMDReflectionRays, scene->nSingleReflectionRays);
		log->logMessage(LOG_INFO, timeStringBuffer);

		scene->printRayStats();
		sprintf(timeStringBuffer, "Elapsed time overall: %d seconds, %d milliseconds", (int)timeAllFrames, (int)((timeAllFrames - floor(timeAllFrames)) * 1000));
		log->logMessage(LOG_INFO, timeStringBuffer);
	}

	/**
	 * Time (in milliseconds) the last frame took to render.
	 */
	int getTimeLastFrame() {
		return (int)(timeLastFrame * 1000.0f);
	}

	/**
	 * Time (in milliseconds) the all frames took to render.
	 */
	int getTimeAllFrames() {
		return (int)(timeAllFrames * 1000.0f);
	}

	/**
	 * Get a pointer to the rendered image data in RGB format.
	 */
	void *getImageDataPtr() {
		if (outImage)
			return (void *)outImage->data;
		else
			return NULL;
	}

	/**
	 * Save the current image to file.
	 * The file name of the saved image will be filename_<framenr>.jpg.
	 */
	void saveCurrentImageToFile(const char *filename);

	/**
	* Save the current image to file.
	* The file name of the saved image will be exactly as specified.
	*/

	void saveCurrentImageToSpecificFile(char *filename);	
	/**
	 * Sets whether the renderered image should be written to file
	 * after each frame.
	 */
	void setSaveImage(bool state) {
		bSaveImage = state;
	}

	/**
	 * Returns whether the renderered image is written to file
	 * after each frame.
	 */
	bool getSaveImage() const {
		return bSaveImage;
	}

	void GLdrawView(SceneVisualizationType mode) {
		Ray viewer;
		getRayWithOrigin(viewer, 0.5f, 0.5f);
		scene->GLdrawScene(viewer, mode);
	}

	bool setFromGLVUPath(FILE *pathFile) {
		static int count = 0;
		static int interpolateFrames = 1;
		static float interpolateFactor = 1.0f / (float)interpolateFrames;

		char buffer[2000];

		static Vector3 X, Y, Z;					 // NORMALIZED CAMERA COORDINATE-AXIS VECTORS
		static Vector3 oldX, oldY, oldZ;            // NORMALIZED CAMERA COORDINATE-AXIS VECTORS
		static Vector3 Orig, oldOrig;               // LOCATION OF ORIGIN OF CAMERA SYSTEM IN WORLD COORDS

		/*
		// first invocation, set both to same values:
		if (count == 0) {
			if (fgets(buffer, 2000, pathFile) <= 0)
				return false;

			// read one line from file:
			int Cond = sscanf(buffer, "%f %f %f %f %f %f %f %f %f %f %f %f\n",
				  			 &X.e[0],&X.e[1],&X.e[2], &Y.e[0],&Y.e[1],&Y.e[2], 
							 &Z.e[0],&Z.e[1],&Z.e[2], &Orig.e[0],&Orig.e[1],&Orig.e[2]);					
		}
			
		if ((count % interpolateFrames) == 0) { // get new line from file:			
			if (fgets(buffer, 2000, pathFile) <= 0)
				return false;

			oldX = X;
			oldY = Y;
			oldZ = Z;
			oldOrig = Orig;

			// read one line from file:
			int Cond = sscanf(buffer, "%f %f %f %f %f %f %f %f %f %f %f %f\n",
				  			 &X.e[0],&X.e[1],&X.e[2], &Y.e[0],&Y.e[1],&Y.e[2], 
							 &Z.e[0],&Z.e[1],&Z.e[2], &Orig.e[0],&Orig.e[1],&Orig.e[2]);					
		}		

		float interpolate = float(count % interpolateFrames) * interpolateFactor;
		
		center = LERP(interpolate,oldOrig,Orig);		
		uvw.set(LERP(interpolate,oldX,X), LERP(interpolate,oldY,Y), LERP(interpolate,oldZ,Z));
		corner = center + left*uvw.u() - bottom*uvw.v() - distance*uvw.w();
		across = (right-left)*uvw.u();
		up = (-(top-bottom))*uvw.v();
		direction = -LERP(interpolate,oldZ,Z);

		count++;*/

		if (fgets(buffer, 2000, pathFile) <= 0)
			return false;

		int Skip = 1;

		for (int i = 0;i < Skip;i++)
			// read one line from file:
			int Cond = sscanf(buffer, "%f %f %f %f %f %f %f %f %f %f %f %f\n",
				&X.e[0],&X.e[1],&X.e[2], &Y.e[0],&Y.e[1],&Y.e[2], 
				&Z.e[0],&Z.e[1],&Z.e[2], &Orig.e[0],&Orig.e[1],&Orig.e[2]);	

		float ScaleFactor = 1;		// we scale the model; 
									// so we also need to scale position of the path to use it

		//ScaleFactor = 1/1000.;		// full_PP

		Orig = Orig * ScaleFactor;

		center = Orig ;		
		uvw.set(X,Y,Z);
		corner = center + left*uvw.u() - bottom*uvw.v() - distance*uvw.w();
		across = (right-left)*uvw.u();
		up = (-(top-bottom))*uvw.v();
		direction = -Z;


		return true;
	}

	bool saveToGLVUPath(FILE *pathFile) {		
		// read one line from file:
		fprintf(pathFile, "%f %f %f %f %f %f %f %f %f %f %f %f\n",
			uvw.u().x(), uvw.u().y(), uvw.u().z(), 
			uvw.v().x(), uvw.v().y(), uvw.v().z(), 
			uvw.w().x(), uvw.w().y(), uvw.w().z(), 
			center.x(), center.y(), center.z());		
		return true;
	}

	bool setLightFromPath(FILE *pathFile) {
		char buffer[2000];
		Vector3 X, Y, Z;            // NORMALIZED CAMERA COORDINATE-AXIS VECTORS
		Vector3 Orig;               // LOCATION OF ORIGIN OF LIGHT SYSTEM IN WORLD COORDS

		if (fgets(buffer, 2000, pathFile) <= 0)
			return false;

		// read one line from file:
		int Cond = sscanf(buffer, "%f %f %f %f %f %f %f %f %f %f %f %f\n",
						  &X.e[0],&X.e[1],&X.e[2], 
						  &Y.e[0],&Y.e[1],&Y.e[2], 
						  &Z.e[0],&Z.e[1],&Z.e[2], 
						  &Orig.e[0],&Orig.e[1],&Orig.e[2]);		
		
		Vector3 bb_min, bb_max, bb_dim, bb_center;
		scene->getAABoundingBox(bb_min, bb_max);
		bb_dim = bb_max - bb_min;
		bb_center = (bb_max + bb_min) / 2.0f;
		
		Orig = bb_center + 1000.f * Z;

		scene->setLightPos(0, Orig);
		cout << "Setting light position to " << Orig << endl;

		return true;
	}

	bool saveLightToPath(FILE *pathFile) {		
		Vector3 lightPos = scene->getLightPos(0);
		// read one line from file:
		fprintf(pathFile, "%f %f %f %f %f %f %f %f %f %f %f %f\n",
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			lightPos.x(), lightPos.y(), lightPos.z());		
		return true;
	}

	float getDistance() { return distance; }
	void setDistance(float newDistance) { 
		distance = newDistance;
		recalculateProjection();
	}
	Vector3 & getAcross (void)  {return across;}
	Vector3 & getUp (void)  {return up;}



	// Camera intrinsics:
	//

	int width, height;
	int tile_width, tile_height;

	float lens_radius;
	float left, right, bottom, top;	
	float distance;			// distance of projection plane

	ONB uvw;  // orthonormal basis of viewer

protected:
	/**
	 * Set internal projection settings to current view
	 * (executed after each change to view parameters)
	 */
	void recalculateProjection() {
		uvw.initFromWV( -direction, vup );
		corner = center + left*uvw.u() - bottom*uvw.v() - distance*uvw.w();
		across = (right-left)*uvw.u();
		up = (-(top-bottom))*uvw.v();
	}
	
	/**
	 * Get an eye-ray for the specified width (a) and height (b)
	 * parameters. Parameters need to be in the [0..1] range.
	 */
	inline void getRay(Ray &ray, float a, float b) 
	{
		Vector3 target = corner + across*a + up*b;		
		ray.setDirection(unitVector(target-center));	
	}

	inline void getRayWithOrigin(Ray &ray, float a, float b) 
	{
		Vector3 target = corner + across*a + up*b;
		ray.setOrigin(center);
		ray.setDirection(unitVector(target-center));	
	}

	/**
	 * Sets ray directions according to arrays a and b
	 * (see getRay() above, but as an float[4]).
	 * IMPORTANT: assumes a and b are 16-byte aligned !
	 */
	inline void getRay(SIMDRay &rays, float *a, float *b) 
	{

		__m128 a4 = _mm_load_ps(a);
		__m128 b4 = _mm_load_ps(b);

		__m128 corner4 = _mm_load1_ps(&corner.e[0]);
		__m128 across4 = _mm_load1_ps(&across.e[0]);
		__m128 up4 = _mm_load1_ps(&up.e[0]);
		__m128 center4 = _mm_load1_ps(&center.e[0]);

		// target = corner + across*a[i] + up*b[i];
		// direction = target - center
		__m128 directionx = _mm_sub_ps(_mm_add_ps(corner4, _mm_add_ps(_mm_mul_ps(a4,across4), _mm_mul_ps(b4,up4))),center4);
		
		corner4 = _mm_load1_ps(&corner.e[1]);
		across4 = _mm_load1_ps(&across.e[1]);
		up4 = _mm_load1_ps(&up.e[1]);
		center4 = _mm_load1_ps(&center.e[1]);

		// target = corner + across*a[i] + up*b[i];
		// direction = target - center
		__m128 directiony = _mm_sub_ps(_mm_add_ps(corner4, _mm_add_ps(_mm_mul_ps(a4,across4), _mm_mul_ps(b4,up4))),center4);
			
		corner4 = _mm_load1_ps(&corner.e[2]);
		across4 = _mm_load1_ps(&across.e[2]);
		up4 = _mm_load1_ps(&up.e[2]);
		center4 = _mm_load1_ps(&center.e[2]);

		// target = corner + across*a[i] + up*b[i];
		// direction = target - center
		__m128 directionz = _mm_sub_ps(_mm_add_ps(corner4, _mm_add_ps(_mm_mul_ps(a4,across4), _mm_mul_ps(b4,up4))),center4);
			
		// make unit vector:
		// get length of direction vector
		__m128 dir_len = _mm_sqrt_ps( _mm_add_ps(_mm_add_ps(_mm_mul_ps(directionx, directionx), 
															_mm_mul_ps(directiony, directiony)), 
															_mm_mul_ps(directionz, directionz)));
		// calculate reciprocal of direction length
		__m128 dir_len_inv = _mm_rcp_ps(dir_len);

		// multiply with reciprocal to normalize:
		directionx = _mm_mul_ps(directionx, dir_len_inv);
		directiony = _mm_mul_ps(directiony, dir_len_inv);
		directionz = _mm_mul_ps(directionz, dir_len_inv);

		// store directions:
		_mm_store_ps(rays.direction[0], directionx);
		_mm_store_ps(rays.direction[1], directiony);
		_mm_store_ps(rays.direction[2], directionz);

		// and reciprocals of directions:
		_mm_store_ps(rays.invdirection[0], _mm_rcp_ps(directionx));
		_mm_store_ps(rays.invdirection[1], _mm_rcp_ps(directiony));
		_mm_store_ps(rays.invdirection[2], _mm_rcp_ps(directionz));


		rays.rayChildOffsets[0] = (rays.direction[0][0] >= 0.0f)?1:0;
		rays.rayChildOffsets[1] = (rays.direction[1][0] >= 0.0f)?1:0;
		rays.rayChildOffsets[2] = (rays.direction[2][0] >= 0.0f)?1:0;
	}

	inline void getRayWithOrigin(SIMDRay &rays, float *a, float *b) 
	{

		_mm_store_ps(rays.origin[0], _mm_load1_ps(&center.e[0]));
		_mm_store_ps(rays.origin[1], _mm_load1_ps(&center.e[1]));
		_mm_store_ps(rays.origin[2], _mm_load1_ps(&center.e[2]));

		__m128 a4 = _mm_load_ps(a);
		__m128 b4 = _mm_load_ps(b);

		__m128 corner4 = _mm_load1_ps(&corner.e[0]);
		__m128 across4 = _mm_load1_ps(&across.e[0]);
		__m128 up4 = _mm_load1_ps(&up.e[0]);
		__m128 center4 = _mm_load1_ps(&center.e[0]);

		// target = corner + across*a[i] + up*b[i];
		// direction = target - center
		__m128 directionx = _mm_sub_ps(_mm_add_ps(corner4, _mm_add_ps(_mm_mul_ps(a4,across4), _mm_mul_ps(b4,up4))),center4);

		corner4 = _mm_load1_ps(&corner.e[1]);
		across4 = _mm_load1_ps(&across.e[1]);
		up4 = _mm_load1_ps(&up.e[1]);
		center4 = _mm_load1_ps(&center.e[1]);

		// target = corner + across*a[i] + up*b[i];
		// direction = target - center
		__m128 directiony = _mm_sub_ps(_mm_add_ps(corner4, _mm_add_ps(_mm_mul_ps(a4,across4), _mm_mul_ps(b4,up4))),center4);

		corner4 = _mm_load1_ps(&corner.e[2]);
		across4 = _mm_load1_ps(&across.e[2]);
		up4 = _mm_load1_ps(&up.e[2]);
		center4 = _mm_load1_ps(&center.e[2]);

		// target = corner + across*a[i] + up*b[i];
		// direction = target - center
		__m128 directionz = _mm_sub_ps(_mm_add_ps(corner4, _mm_add_ps(_mm_mul_ps(a4,across4), _mm_mul_ps(b4,up4))),center4);

		// make unit vector:
		// get length of direction vector
		__m128 dir_len = _mm_sqrt_ps( _mm_add_ps(_mm_add_ps(_mm_mul_ps(directionx, directionx), 
			_mm_mul_ps(directiony, directiony)), 
			_mm_mul_ps(directionz, directionz)));
		// calculate reciprocal of direction length
		__m128 dir_len_inv = _mm_rcp_ps(dir_len);

		// multiply with reciprocal to normalize:
		directionx = _mm_mul_ps(directionx, dir_len_inv);
		directiony = _mm_mul_ps(directiony, dir_len_inv);
		directionz = _mm_mul_ps(directionz, dir_len_inv);

		// store directions:
		_mm_store_ps(rays.direction[0], directionx);
		_mm_store_ps(rays.direction[1], directiony);
		_mm_store_ps(rays.direction[2], directionz);

		// and reciprocals of directions:
		_mm_store_ps(rays.invdirection[0], _mm_rcp_ps(directionx));
		_mm_store_ps(rays.invdirection[1], _mm_rcp_ps(directiony));
		_mm_store_ps(rays.invdirection[2], _mm_rcp_ps(directionz));

		rays.rayChildOffsets[0] = (rays.direction[0][0] >= 0.0f)?1:0;
		rays.rayChildOffsets[1] = (rays.direction[1][0] >= 0.0f)?1:0;
		rays.rayChildOffsets[2] = (rays.direction[2][0] >= 0.0f)?1:0;

		/*
		for (int i = 0; i < 4; i++) {
			target = corner + across*a[i] + up*b[i];
			rays.setDirection(unitVector(target-center), i);	
		}*/
	}

	
	// Image to render to:
	Image *outImage;
	bool bSaveImage;		// save image to file after each frame	

	// Scene we're rendering
	Scene *scene;

	int currentFrameNum;

	__declspec(align(16)) Vector3 center, direction, vup, // Viewer position, viewing angle, up vector
								  corner, across, up;	  // Vector of top left corner, vector to right/up side of viewing rectangle


	// Stats:
	TimerValue timeFrameStart, timeFrameEnd;
	float timeLastFrame;
	float timeAllFrames;
	char timeStringBuffer[200];
};

#endif