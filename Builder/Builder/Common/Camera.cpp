#include "stdafx.h"
#include "common.h"
#include "Camera.h"
#include "Sample.h"

#include "LOD_header.h"
#include "stopwatch.hpp"

#include "Zcurve.h"

extern CLODMetric LODMetric;
//extern float g_PoE;
extern float g_MaxAllowModifier;
extern int g_numCrackedTri;
Stopwatch **TCluster;
Stopwatch **TTriangle;
Stopwatch **TComplete;
Stopwatch TBoxTest("BB intersection test");

#include "gpu_cache_sim.h"
#ifdef WORKING_SET_COMPUTATION
extern int WorkingSetSizeKB;
extern CGPUCacheSim g_kdTreeSim;
#endif

#define _TRACE_ZCURVE   1
#define _TRACE_ROWBYROW 2
#define _TRACE_TILE_ROW 3

#define _TRACE_TYPE	   _TRACE_TILE_ROW

bool g_Verbose;

bool Camera::renderFrame(int frameNum) {		
	LogManager *log = LogManager::getSingletonPtr();
	OptionManager *opt = OptionManager::getSingletonPtr();

	// sungeui start --------------------
	// update LOD metric
	LODMetric.m_NearPlane = getDistance ();
	LODMetric.m_PixelSize = 1.0f / (float)width;	// assume symmetric image resolution
	g_MaxAllowModifier = LODMetric.ComputeMaxAllowanceModifier ();

	#ifdef WORKING_SET_COMPUTATION
	g_kdTreeSim.InitCache ();
	g_kdTreeSimL2.InitCache ();
	g_kdIdxSim.InitCache ();
	g_TriSim.InitCache ();
	g_VerSim.InitCache ();
	g_LODSim.InitCache ();
	#endif
	// sungeui end -----------------------

	// output image:
	if (outImage == NULL)
		outImage = new Image(width, height);

	// start counter for time measurment
	currentFrameNum = frameNum;	
	timeFrameStart.set();

	int i;
	TCluster = new Stopwatch*[NUM_THREADS];
	TTriangle = new Stopwatch*[NUM_THREADS];
	TComplete = new Stopwatch*[NUM_THREADS];
	for(i=0;i<NUM_THREADS;i++)
	{
		TCluster[i] = new Stopwatch("Cluster load");
		TTriangle[i] = new Stopwatch("Triangle");
		TComplete[i] = new Stopwatch("Complete BB");
	}

	// If the supersample parameter
	// is given, the image will be supersampled with supersample^2 rays.
	// get the size of the supersample from "options.xml"
	int supersample = opt->getOptionAsInt("raytracing", "numSuperSampling", 1);

#ifdef _USE_BEAM_TRACING

	// set up beam:	
	unsigned int beams_x = width / BEAM_PRIMARY_STARTSIZE;
	unsigned int beams_y = height / BEAM_PRIMARY_STARTSIZE;
	unsigned int numBeams = beams_x * beams_y;	

	cout << "renderFrame(): " <<  numBeams << " beams " << endl;
	//
	// for all tiles in the system:
	// 
	#ifdef _USE_OPENMP
	#pragma omp parallel for shared(beams_x, beams_y, numBeams) schedule(guided,1)
	#endif
	for (int curBeam = 0; curBeam < numBeams; curBeam++) {		
		Beam beam;				
		rgb colors[BEAM_REALMAXRAYNUM];

		unsigned int start_x = (curBeam % beams_x) * BEAM_PRIMARY_STARTSIZE;
		unsigned int start_y = (unsigned int)(curBeam / beams_x) * BEAM_PRIMARY_STARTSIZE;

		int rayNum = 0;
		float deltax = 1.0f / (float)width;
		float deltay = 1.0f / (float)height;
		__declspec(align(16)) float xList[4], yList[4];
		float deltax2 = 2.0f * deltax;
		float deltay2 = 2.0f * deltay;
		float ypos = deltay/2.0f + start_y*deltay,
		      xpos = deltax/2.0f + start_x*deltax;

		Vector3 cornerLU = corner + across*(start_x*deltax) + up*(start_y*deltay);
		Vector3 cornerRU = cornerLU + across*(BEAM_PRIMARY_STARTSIZE*deltax);
		Vector3 cornerLB = cornerLU + up*(BEAM_PRIMARY_STARTSIZE*deltay);
		Vector3 cornerRB = cornerLU + across*(BEAM_PRIMARY_STARTSIZE*deltax) + up*(BEAM_PRIMARY_STARTSIZE*deltay);

		// initialize beam:
		beam.setup(center, cornerLU, cornerRU, cornerLB, cornerRB, BEAM_PRIMARY_STARTSIZE, BEAM_PRIMARY_STARTSIZE, false);		

		// make real rays for beam:		

		for (int y = 0; y < BEAM_PRIMARY_STARTSIZE; y+=2) {				
			xpos = deltax/2.0f + start_x*deltax;

			yList[0] = ypos; 
			yList[1] = ypos;          
			yList[2] = ypos + deltay; 
			yList[3] = ypos + deltay;

			xList[0] = xpos;
			xList[1] = xpos + deltax;
			xList[2] = xpos;
			xList[3] = xpos + deltax;

			for (int x = 0; x < BEAM_PRIMARY_STARTSIZE; x+=2) {

				// 4 ray offsets					
				xList[0] += deltax2;
				xList[1] += deltax2;
				xList[2] += deltax2;
				xList[3] += deltax2;	

				getRayWithOrigin(beam.realRays[rayNum++], xList, yList);
				xpos += deltax2;
			}
			ypos += deltay2;
		}

		beam.initFromRays(rayNum);

		scene->traceBeam(beam, colors);

		int pixelNum = 0;
		for (int y = start_y; y < start_y+BEAM_PRIMARY_STARTSIZE; y+=2) {							
			for (int x = start_x; x < start_x+BEAM_PRIMARY_STARTSIZE; x+=2) {
				colors[pixelNum].clamp();
				outImage->setPixel(x, y, colors[pixelNum]);
				pixelNum++;
				colors[pixelNum].clamp();
				outImage->setPixel(x+1, y, colors[pixelNum]);
				pixelNum++;
				colors[pixelNum].clamp();
				outImage->setPixel(x, y+1, colors[pixelNum]);
				pixelNum++;
				colors[pixelNum].clamp();
				outImage->setPixel(x+1, y+1, colors[pixelNum]);
				pixelNum++;				
			}
		}
	}

#else
#if _TRACE_TYPE == _TRACE_ZCURVE
	Ray ray;
	rgb outColor;
	float deltax = 1.0f / (float)width;
	float deltay = 1.0f / (float)height;
	float xpos, ypos;

	Zcurve zcurve(0,0,width,height);
	for (int ii=0; ii<zcurve.seq.size(); ++ii) {
		ypos = deltay/2.0f + zcurve.seq[ii].second*deltay;
		xpos = deltax/2.0f + zcurve.seq[ii].first*deltax;

		getRayWithOrigin(ray, xpos, ypos);
		scene->trace(ray, outColor);
		outColor.clamp();
		outImage->setPixel(zcurve.seq[ii].first, zcurve.seq[ii].second, outColor);
	}

#elif _TRACE_TYPE == _TRACE_ROWBYROW
	Ray ray;
	rgb outColor;
	float deltax = 1.0f / (float)width;
	float deltay = 1.0f / (float)height;
	float ypos = deltay/2.0f,
		  xpos = deltax/2.0f;

	for (int y=0; y<height; ++y) {
		xpos = deltax/2.0f;
		for (int x=0; x<width; ++x) {
			getRayWithOrigin(ray, xpos, ypos);

			scene->trace(ray, outColor);
			outColor.clamp();
			outImage->setPixel(x, y, outColor);
			xpos += deltax;
		}
		ypos += deltay;
	}
#else
	//
	// set up tiling:
	//

	unsigned int tiles_x = width / tile_width;
	unsigned int tiles_y = height / tile_height;
	unsigned int numTiles = tiles_x * tiles_y;	

	//
	// for all tiles in the system:
	// 
	#ifdef _USE_OPENMP
	omp_set_num_threads(NUM_THREADS);
	#pragma omp parallel for shared(tiles_x, tiles_y, numTiles) schedule(guided,2)
	#endif
	for (int curTile = 0; curTile < numTiles; curTile++) {
		//cout << "Thread Num : " << omp_get_thread_num() << endl;
		unsigned int start_x = (curTile % tiles_x) * tile_width;
		unsigned int start_y = (unsigned int)(curTile / tiles_x) * tile_height;		

#ifdef _USE_SIMD_RAYTRACING
	__declspec(align(16)) float xList[4], yList[4];
	SIMDRay ray;
	Ray singleRay;
	rgb outColor[4];
	float deltax = 1.0f / (float)width;
	float deltay = 1.0f / (float)height;
	float deltax2 = 2.0f * deltax;
	float deltay2 = 2.0f * deltay;
	float ypos = deltay/2.0f + start_y*deltay,
		  xpos = deltax/2.0f + start_x*deltax;

	if (supersample > 1) { // Supersampling requested
		__declspec(align(16)) float xSampleList[4], ySampleList[4];
		rgb accumulatorColor[4];

		int supersample2 = supersample*supersample;
		Vector2 *samples = new Vector2[supersample2];
		Sampler *sampler = new Sampler;
		
		for (int y = 0; y < tile_height; y+=2) {	
			xpos = deltax/2.0f + start_x*deltax;

			yList[0] = ypos; 
			yList[1] = ypos;          
			yList[2] = ypos + deltay; 
			yList[3] = ypos + deltay;

			xList[0] = xpos;
			xList[1] = xpos + deltax;
			xList[2] = xpos;
			xList[3] = xpos + deltax;

			for (int x = 0; x < tile_width; x+=2) {
				
				outColor[0] = rgb(0,0,0);
				outColor[1] = rgb(0,0,0);
				outColor[2] = rgb(0,0,0);
				outColor[3] = rgb(0,0,0);

				// generate samples
				sampler->jitter2(samples, supersample);				

				// 4 ray offsets					
				xList[0] += deltax2;
				xList[1] += deltax2;
				xList[2] += deltax2;
				xList[3] += deltax2;

				for (int k = 0; k < supersample2; k++) {



					xSampleList[0] = xList[0] + (samples[k].e[0] * deltax);
					xSampleList[1] = xList[1] + (samples[k].e[0] * deltax);
					xSampleList[2] = xList[2] + (samples[k].e[0] * deltax);
					xSampleList[3] = xList[3] + (samples[k].e[0] * deltax);

					ySampleList[0] = yList[0] + (samples[k].e[1] * deltay);
					ySampleList[1] = yList[1] + (samples[k].e[1] * deltay);
					ySampleList[2] = yList[2] + (samples[k].e[1] * deltay);
					ySampleList[3] = yList[3] + (samples[k].e[1] * deltay);
									
					// get new ray, just change direction, not origin
					getRayWithOrigin(ray, xSampleList, ySampleList);

					// if all direction signs match (otherwise, the BSP traverse
					// order is different), then trace all 4 rays now
					if (directionsMatch(ray)) {				
						scene->trace(ray, accumulatorColor);
					}
					else { // fallback to 4 single rays..
						getRayWithOrigin(singleRay, xSampleList[0], ySampleList[0]);
						scene->trace(singleRay, accumulatorColor[0]);
						getRayWithOrigin(singleRay, xSampleList[1], ySampleList[1]);
						scene->trace(singleRay, accumulatorColor[1]);
						getRayWithOrigin(singleRay, xSampleList[2], ySampleList[2]);
						scene->trace(singleRay, accumulatorColor[2]);
						getRayWithOrigin(singleRay, xSampleList[3], ySampleList[3]);
						scene->trace(singleRay, accumulatorColor[3]);						
					}

					outColor[0] += accumulatorColor[0];
					outColor[1] += accumulatorColor[1];
					outColor[2] += accumulatorColor[2];
					outColor[3] += accumulatorColor[3];
				}

				outColor[0] /= (float)supersample2;
				outColor[1] /= (float)supersample2;
				outColor[2] /= (float)supersample2;
				outColor[3] /= (float)supersample2;
				outColor[0].clamp();
				outColor[1].clamp();
				outColor[2].clamp();
				outColor[3].clamp();
				outImage->setPixel(start_x+x,   start_y+y,   outColor[0]);
				outImage->setPixel(start_x+x+1, start_y+y,   outColor[1]);
				outImage->setPixel(start_x+x,   start_y+y+1, outColor[2]);
				outImage->setPixel(start_x+x+1, start_y+y+1, outColor[3]);
				xpos += deltax2;			
			}

			ypos += deltay2;
		}

		delete sampler;
	}
	else {
		for (int y = 0; y < tile_height; y+=2) {	
			xpos = deltax/2.0f + start_x*deltax;

			yList[0] = ypos; 
			yList[1] = ypos;          
			yList[2] = ypos + deltay; 
			yList[3] = ypos + deltay;

			xList[0] = xpos;
			xList[1] = xpos + deltax;
			xList[2] = xpos;
			xList[3] = xpos + deltax;

			for (int x = 0; x < tile_width; x+=2) {			
				//g_Verbose = (start_x + x) == 296 && (start_y + y) == 266;
				g_Verbose = false;

				// 4 ray offsets					

				xList[0] += deltax2;
				xList[1] += deltax2;
				xList[2] += deltax2;
				xList[3] += deltax2;			
				
				// get new ray, just change direction, not origin
				getRayWithOrigin(ray, xList, yList);

				// if all direction signs match (otherwise, the BSP traverse
				// order is different), then trace all 4 rays now
				if (directionsMatch(ray)) {							
					scene->trace(ray, outColor);
					//scene->rayCast(ray, outColor);
				}
				else { // fallback to 4 single rays..
					getRayWithOrigin(singleRay, xList[0], yList[0]);
					scene->trace(singleRay, outColor[0]);
					getRayWithOrigin(singleRay, xList[1], yList[1]);
					scene->trace(singleRay, outColor[1]);
					getRayWithOrigin(singleRay, xList[2], yList[2]);
					scene->trace(singleRay, outColor[2]);
					getRayWithOrigin(singleRay, xList[3], yList[3]);
					scene->trace(singleRay, outColor[3]);
				}

				outColor[0].clamp();
				outColor[1].clamp();
				outColor[2].clamp();
				outColor[3].clamp();
				outImage->setPixel(start_x+x,   start_y+y,   outColor[0]);
				outImage->setPixel(start_x+x+1, start_y+y,   outColor[1]);
				outImage->setPixel(start_x+x,   start_y+y+1, outColor[2]);
				outImage->setPixel(start_x+x+1, start_y+y+1, outColor[3]);
				xpos += deltax2;			
			}

			ypos += deltay2;
		}
	}

#else // ifdef _USE_SIMD_RAYTRACING
#ifdef _USE_PATH_TRACING

	Sampler sampler;
	Vector2 jitter;
	float deltax = 1.0f / (float)width;
	float deltay = 1.0f / (float)height;
	Ray ray;
	int k;
	rgb outColor, tempColor;
	float ypos = deltay/2.0f,
		xpos = deltax/2.0f;

	//getRayWithOrigin(ray, 0, 0);

	for (int y = 0; y < height; y++) {	
		xpos = deltax/2.0f;		
		for (int x = 0; x < width; x++) {
			outColor = rgb(0,0,0);			

			// sample this path supersample times:
			for (k = 0; k < supersample; k++) {								
				sampler.random2(&jitter, 1);
				getRayWithOrigin(ray, xpos + (jitter[0] * deltax), ypos + (jitter[1] * deltay));
				// trace one path:
				if (scene->tracePath(ray, tempColor) == false) {
					// if tracePath returned false, we only hit the background
					// and may as well stop tracing paths..
					outColor = tempColor;
					k++;
					break;
				}
				outColor += tempColor;
			}

			outColor /= (float)k;
			outColor.clamp();
			outImage->setPixel(x, y, outColor);
			xpos += deltax;			
		}
		ypos += deltay;
	}

#else 
    // if neither SIMD nor path tracing:
    // normale single ray tracing:
	float deltax = 1.0f / (float)width;
	float deltay = 1.0f / (float)height;

	extern float maxDistForDOE;
	extern float zMidForDOE;
	Vector3 minForDOE, maxForDOE;
	scene->getAABoundingBox(minForDOE, maxForDOE);
	zMidForDOE = minForDOE.e[2];
	maxDistForDOE = maxForDOE.e[2]-minForDOE.e[2];

	if (supersample > 1) { // Supersampling requested
		Ray ray;
		rgb outColor;		
 		float ypos = deltay/2.0f + start_y*deltay,
   			  xpos = deltax/2.0f + start_x*deltax;
 		rgb tempcolor;
		int supersample2 = supersample*supersample;
		Vector2 *samples = new Vector2[supersample2];
		Sampler *sampler = new Sampler();

		//getRayWithOrigin(ray, 0, 0);
	
		for (int y = 0; y < tile_height; y++) {			
			xpos = deltax/2.0f + start_x*deltax;
			for (int x = 0; x < tile_width; x++) {
				outColor = rgb(0,0,0);

				// generate samples
				sampler->jitter2(samples, supersample);

				// supersample this pixel:
				for (int k = 0; k < supersample2; k++) {
					getRayWithOrigin(ray, xpos + (samples[k].e[0] * deltax) , ypos  + (samples[k].e[1] * deltay));
					scene->trace(ray, tempcolor);
					outColor += tempcolor;
				}
				
				outColor /= (float)supersample2;
				outColor.clamp();
				outImage->setPixel(start_x + x, start_y + y, outColor);
				xpos += deltax;			
			}
			ypos += deltay;
		}

		delete[] samples;
		delete sampler;
	}
	else { // Just normal single-sampling
		Ray ray;
		rgb outColor;
		float ypos = deltay/2.0f + start_y*deltay,
			  xpos = deltax/2.0f + start_x*deltax;

		for (int y = 0; y < tile_height; y++) {				
			xpos = deltax/2.0f + start_x*deltax;
			for (int x = 0; x < tile_width; x++) {									

				getRayWithOrigin(ray, xpos, ypos);
				
				scene->trace(ray, outColor);

				outColor.clamp();
				outImage->setPixel(start_x + x, start_y + y,   outColor);				
				xpos += deltax;
			}
			ypos += deltay;
		}

	}
#endif
#endif

	} // for all tiles

#endif
#endif
	cout << "g_numCrackedTri = " << g_numCrackedTri << endl;

	for(i=0;i<NUM_THREADS;i++)
	{
		cout << *TCluster[i] << "[" << i << "](" << TCluster[i]->GetTime() << " seconds)" << endl;
	}
	for(i=0;i<NUM_THREADS;i++)
	{
		cout << *TTriangle[i] << "[" << i << "](" << TTriangle[i]->GetTime() << " seconds)" << endl;
	}
	for(i=0;i<NUM_THREADS;i++)
	{
		cout << *TComplete[i] << "[" << i << "](" << TComplete[i]->GetTime() << " seconds)" << endl;
	}

	for(i=0;i<NUM_THREADS;i++)
	{
		delete TCluster[i];
		delete TTriangle[i];
		delete TComplete[i];
	}
	delete[] TCluster;
	delete[] TTriangle;
	delete[] TComplete;
	cout << TBoxTest << "(" << TBoxTest.GetTime() << " seconds)" << endl;
	TBoxTest.Reset();

	// calculate frame times
	//
	timeFrameEnd.set();
	timeLastFrame = timeFrameEnd - timeFrameStart;
	timeAllFrames += timeLastFrame;

	if (bSaveImage) {
		// write output image
		//
		saveCurrentImageToFile("frame");		
	}



	#ifdef WORKING_SET_COMPUTATION
	unsigned int NumkdTree = g_kdTreeSim.m_WS.size ();
	unsigned int NumkdIdx = g_kdIdxSim.m_WS.size ();
	unsigned int NumTri = g_TriSim.m_WS.size ();
	unsigned int NumVer = g_VerSim.m_WS.size ();
	unsigned int NumLOD = g_LODSim.m_WS.size ();

	unsigned int SizekdTree = NumkdTree * 4;		// KB granuality
	unsigned int SizekdIdx = NumkdIdx * 4;
	unsigned int SizeTri = NumTri * 4;
	unsigned int SizeVer = NumVer * 4;
	unsigned int SizeLOD = NumLOD * 4;

	printf ("LOD:    %dK, %dMB\n", NumLOD/ 1024, SizeLOD /1024);
	printf ("Tri:    %dK, %dMB\n", NumTri/1024, SizeTri /1024);
	printf ("Ver:    %dK, %dMB\n", NumVer/1024, SizeVer /1024);
	printf ("kdNode: %dK, %dMB\n", NumkdTree/1024, SizekdTree/1024);
	printf ("kdIdx:  %dK, %dMB\n", NumkdIdx/1024, SizekdIdx/1024);

	WorkingSetSizeKB = SizekdTree + SizekdIdx + SizeTri + SizeVer + SizeLOD;
	printf ("Total Used Mem = %dKB\n", WorkingSetSizeKB);
	printf ("%f L2 cache hit ratio\n", g_kdTreeSimL2.GetCacheHit ());
	#endif

	/************************************************************************/
	/* ´ö¼ö. Collision Detection                                            */
	/************************************************************************/
	// call start collision detection

#ifdef DO_COLLISION_DETECTION
	scene->collisionDetection();
#endif

	return true;
}

ViewList &Camera::getPresetViews() {
	return scene->getPresetViews();
}

View &Camera::getPresetView(unsigned int preset_number) {
	return scene->getPresetViews().at(preset_number);
}

void Camera::setViewToPreset(unsigned int preset_number) {
	ViewList &vl = scene->getPresetViews();
	if (preset_number < vl.size())
		this->setViewer(vl.at(preset_number).view);
}

void Camera::setViewToPresetByName(const char *name) {
	ViewList &vl = scene->getPresetViews();

	for (ViewListIterator i=vl.begin(); i != vl.end(); i++) 
		if (strcmp((*i).name, name) == 0) {
			setViewer((*i).view);
		}
}

void Camera::saveCurrentImageToFile(const char *filename) {
	OptionManager *opt = OptionManager::getSingletonPtr();
	LogManager *log = LogManager::getSingletonPtr();
	char imageName[MAX_PATH];

	// write output image
	//
	const char *imageOutputPath = opt->getOption("global", "imageOutputPath", ".");
	if (imageOutputPath[strlen(imageOutputPath)-1] != '/')
		sprintf(imageName, "%s/%s_%04d.tga", imageOutputPath, filename, currentFrameNum);
	else
		sprintf(imageName, "%s%s_%04d.tga", imageOutputPath, filename, currentFrameNum);

	if (!outImage->writeToFile(imageName))
		log->logMessage(LOG_ERROR, "Saving image: %s\n", outImage->getLastError());
}

void Camera::saveCurrentImageToSpecificFile(char *filename) {
	OptionManager *opt = OptionManager::getSingletonPtr();
	LogManager *log = LogManager::getSingletonPtr();

	// write output image
	//

	if (!outImage->writeToFile(filename))
		log->logMessage(LOG_ERROR, "Saving image: %s\n", outImage->getLastError());
}