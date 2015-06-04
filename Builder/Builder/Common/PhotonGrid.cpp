#include "stdafx.h"
#include <GL/GL.h>
#include <GL/glut.h>
#include <math.h>

#include <iostream>
#include <algorithm>
#include <functional>
#include <vector>

#include "common.h"
#include "PhotonGrid.h"
#include "Sample.h"
#include "ZeroGrid.h"

PhotonGrid::PhotonGrid(BVH &sceneGraph, LightList &lights, EmitterList &emitters, MaterialList &materials) {
	m_pSceneGraph = &sceneGraph;
	m_pLightList = &lights;
	m_pEmitterList = &emitters;
	m_pMaterialList = &materials;

#ifndef _GRID_HASH
	m_pGrid = NULL;
#endif
	m_pGridMask = NULL;

	clearMap();
}


FORCEINLINE void PhotonGrid::addGrid(Vector3 &pos, Vector3 &normal, GridPhoton &value) {	

	//static Vector3 temp;
	//static Vector3 radius = Vector3(m_fEstimationRadius, m_fEstimationRadius, m_fEstimationRadius);	

	//Vector3 bbMin = pos - radius;
	//Vector3 bbMax = pos + radius;

	//temp = (bbMin - min);
	//temp[0] *= gridExtendsInv[0];
	//temp[1] *= gridExtendsInv[1];
	//temp[2] *= gridExtendsInv[2];

	//// start indices
	//unsigned int i1 = max(ceilf(temp[0]), 0);
	//unsigned int j1 = max(ceilf(temp[1]), 0);
	//unsigned int k1 = max(ceilf(temp[2]), 0);

	//temp = (bbMax - min);
	//temp[0] *= gridExtendsInv[0];
	//temp[1] *= gridExtendsInv[1];
	//temp[2] *= gridExtendsInv[2];

	//// end indices
	//unsigned int i2 = min(floorf(temp[0]), gridDimension[0]-1);
	//unsigned int j2 = min(floorf(temp[1]), gridDimension[1]-1);
	//unsigned int k2 = min(floorf(temp[2]), gridDimension[2]-1);	

	//for (unsigned int k = k1; k <= k2; k++)
	//	for (unsigned int j = j1; j <= j2; j++)
	//		for	(unsigned int i = i1; i <= i2; i++) {
	//			Vector3 pointPos = min + Vector3(i*gridDelta[0], j*gridDelta[1], k*gridDelta[2]);
	//			Vector3 relPointPos = pointPos - pos;
	//			
	//			
	//			float r2 = dot(normal, relPointPos);
	//			float r1 = ((r2 * normal) - relPointPos).squaredLength();

	//			//if (r1 <= m_fEstimationRadiusSquared &&  fabs(r2) <= (m_fEstimationRadius2 * cos((r1*PI) / m_fEstimationRadiusSquared))) {
	//			if (r1 <= m_fEstimationRadiusSquared &&  fabs(r2) <= m_fEstimationRadius2) {
	//				getGridPoint(i,j,k) += m_fEstimationVolume * value;
	//				setGridMask(i,j,k);

	//			}
	//		}
}




void PhotonGrid::buildPhotonMap(unsigned int numPhotons, unsigned int gridPoints) {
//	OptionManager *opt = OptionManager::getSingletonPtr();
//	LogManager *log = LogManager::getSingletonPtr();
//
//#ifndef _GRID_HASH
//	if (m_pGrid != NULL)
//		clearMap();
//#endif
//
//	if (gridPoints <= 0) {
//		gridPoints = opt->getOptionAsInt("photongrid", "gridPoints", 20);		
//	}
//
//	// wavelet compression rate (for thresholding)
//	m_fCompressionRate = opt->getOptionAsFloat("photongrid", "compressionRate", 1.0f);
//
//	// number of wavelet decomposition steps
//	m_nDecompositionSteps = opt->getOptionAsInt("photongrid", "decompositionSteps", 1);
//
//	
//	
//	// grid extends as determined by bounding box
//	Vector3 gridExtends = max - min;
//
//	// grid points: add to next multiple of blocksize
//	gridPoints = ROUND_BLOCKSIZE(gridPoints);
//	
//	// specified number of grid points applies to largest axis directly
//	int largestAxis = gridExtends.indexOfMaxComponent();
//	gridDimension[largestAxis] = gridPoints;
//
//	// set dimensions for other two axes proportional to axis extend
//	gridDimension[(largestAxis + 1) % 3] = ROUND_BLOCKSIZE((unsigned int)ceil( (gridExtends[(largestAxis + 1) % 3] / gridExtends[largestAxis]) * gridPoints ));
//	gridDimension[(largestAxis + 2) % 3] = ROUND_BLOCKSIZE((unsigned int)ceil( (gridExtends[(largestAxis + 2) % 3] / gridExtends[largestAxis]) * gridPoints ));
//	
//	// calculate delta values (should be equal, but for testing)
//	gridDelta[0] = gridExtends[0] / (float)(gridDimension[0] - 1 - 2*PHOTONGRID_PADDING);
//	gridDelta[1] = gridExtends[1] / (float)(gridDimension[1] - 1 - 2*PHOTONGRID_PADDING);
//	gridDelta[2] = gridExtends[2] / (float)(gridDimension[2] - 1 - 2*PHOTONGRID_PADDING);
//
//	// precalc for translating world to grid pos (1 / delta)
//	gridExtendsInv[0] = (float)(gridDimension[0] - 1 - 2*PHOTONGRID_PADDING) / gridExtends[0];
//	gridExtendsInv[1] = (float)(gridDimension[1] - 1 - 2*PHOTONGRID_PADDING) / gridExtends[1];
//	gridExtendsInv[2] = (float)(gridDimension[2] - 1 - 2*PHOTONGRID_PADDING) / gridExtends[2];
//
//	#ifndef _GRID_HASH
//	// create the grid !
//	m_pGrid = new GridPhoton[gridDimension[0] * gridDimension[1] * gridDimension[2]];
//	if (!m_pGrid)
//		return;
//	#else
//	m_pGrid.setDimensions(gridDimension[0], gridDimension[1], gridDimension[2]);
//	m_pGrid.setDefaultValue(rgb(0,0,0));
//	#endif
//	
//	// rescale dimensions
//	gridDimension[0] -= 2*PHOTONGRID_PADDING;
//	gridDimension[1] -= 2*PHOTONGRID_PADDING;
//	gridDimension[2] -= 2*PHOTONGRID_PADDING;
//	
//	#ifndef _GRID_HASH
//	// initialize to zero
//	memset(m_pGrid, 0, gridDimension[0] * gridDimension[1] * gridDimension[2] * sizeof(GridPhoton));
//	#endif
//
//	// precalculate offset for components:
//	offsetZ = gridDimension[0] * gridDimension[1];
//	offsetY = gridDimension[0];
//
//	// initialize bitmask for block usage:
//	m_nGridBlocks = (gridDimension[0] * gridDimension[1] * gridDimension[2]) / PHOTONGRID_BLOCKOFFSET;
//	m_nGridBlocksUsed = 0;
//	m_pGridMask = new unsigned char[(unsigned int)ceil(m_nGridBlocks / 8.0f)];
//
//	// initialize to zero
//	memset(m_pGridMask, 0, ceil(m_nGridBlocks / 8.0f));
//
//	// precalculate offset for components (number of blocks for a step in Z or Y direction):
//	blockoffsetZ = (gridDimension[0] * gridDimension[1]) / (PHOTONGRID_BLOCKSIZE * PHOTONGRID_BLOCKSIZE);
//	blockoffsetY = gridDimension[0] / PHOTONGRID_BLOCKSIZE;
//
//
//	float avgDelta = (gridDelta[0] + gridDelta[1] + gridDelta[2]) / 3.0f;
//
//	// radius for finding grid points around the hit point (scaled by grid delta)
//	m_fEstimationRadius = opt->getOptionAsFloat("photongrid", "estimationRadius", 1.0f) * avgDelta;
//	m_fEstimationRadiusSquared = m_fEstimationRadius*m_fEstimationRadius;
//	m_fEstimationRadius2 = opt->getOptionAsFloat("photongrid", "estimationRadius2", 1.0f) * avgDelta;
//	m_fEstimationRadiusSquared2 = m_fEstimationRadius2*m_fEstimationRadius2;
//
//	// approximate volume for one estimation, used for intensity normalization (inverse)
//	
//	m_fEstimationVolume = (4.0f * opt->getOptionAsFloat("photongrid", "photonPowerScale", 1.0f)) / (m_fEstimationRadiusSquared * PI * m_fEstimationRadius2);
//
//	/**
//	 * Trace Photons:
//	 */
//	 
//	Ray photonRay;
//	Hitpoint hitpoint;
//	Sampler sampler;
//	RandomLinear generator;
//	Vector2 sampleDirection;
//	Vector3 pointOnEmitter;
//	Vector3 photonRayDirection;	
//	rgb color, reflectance;
//	int photonHit, depth = 0;
//	GridPhoton currentPhotonPower;
//
//	// photon tracing parameters
//	int maxReflectionDepth = opt->getOptionAsInt("photongrid", "maxParticleTraceDepth", 5);	
//
//	// for progress output:
//	unsigned int currPhoton = 0;
//	unsigned int photonPercent = (numPhotons / 20);
//	char progressString[100];
//
//	timeBuildStart.set();	
//
//	log->logMessage("Beginning photon tracing...");
//
//	if (m_pEmitterList->size() > 0) { // do we have area light sources ?
//
//		// calc overall power of emitters
//		float sumIntensity = 0.0f;
//		for(EmitterListIterator e1 = m_pEmitterList->begin(); e1 != m_pEmitterList->end(); e1++)
//			sumIntensity += (*e1).summedIntensity;
//
//		// power per photon
//		float dPowerFactor = sumIntensity * (1000.0f / numPhotons);
//
//		// emit photons for every emitter
//		for(EmitterListIterator e = m_pEmitterList->begin(); e != m_pEmitterList->end(); e++) {
//			// number of photons for this emitter is related to its
//			// intensity in relation to the total intensity;
//			unsigned int numPhotonsForEmitter = ceil(((*e).summedIntensity / sumIntensity) * numPhotons);
//			rgb dPower = (*e).emissionIntensity * (1000.0f / (float)numPhotonsForEmitter);
//			float avgdPower = (dPower[0] + dPower[1] + dPower[2]) / 3.0f;
//
//			for (unsigned int i = 0; i < numPhotonsForEmitter; i++) {
//				currPhoton++;
//
//				if ((currPhoton % photonPercent) == 0) {
//					sprintf(progressString, "   ... %d%% complete\t(%d of %d)", 5*(currPhoton / photonPercent), currPhoton, numPhotons);
//					log->logMessage(progressString);
//				}
//				currentPhotonPower = dPower;
//
//				// Sample point on emitter
//				sampler.random2(&sampleDirection, 1);
//				(*e).sample(sampleDirection[0], sampleDirection[1], pointOnEmitter);
//
//				// Sample direction from emitter 
//				sampler.random2(&sampleDirection, 1);
//				(*e).sampleDirection(sampleDirection[0], sampleDirection[1], photonRayDirection);
//
//				// Generate ray
//				photonRay.setOrigin(pointOnEmitter);
//				photonRay.setDirection(photonRayDirection);
//
//				// Trace photon into scene:
//				if (photonHit = m_pSceneGraph->RayTreeIntersect(photonRay, &hitpoint)) {
//					// Photon hit something:
//					float photonPowerFraction = 1.0f;
//					depth = 0;
//
//					// reflection/refraction direction
//					Vector3 incomingDirection = unitVector(hitpoint.x - photonRay.origin());
//					m_pMaterialList->at(hitpoint.m)->sampleDirection(hitpoint, incomingDirection, photonRayDirection, reflectance);
//
//					float avgReflectance = (reflectance[0] + reflectance[1] + reflectance[2]) / 3.0f;
//
//					// is this particle reflected or absorbed ?
//					// (russian roulette instead of reflecting a particle
//					//  with less power)					
//					while (generator.sample() <= avgReflectance && depth < maxReflectionDepth) {
//						depth++;
//
//						// reflected, so generate new ray:
//						photonRay.setOrigin(hitpoint.x);
//						photonRay.setDirection(photonRayDirection);
//
//						// modify photon power by BRDF for each wavelength:						
//						//hitpoint.m->brdf(hitpoint, incomingDirection, photonRayDirection, reflectance);
//						currentPhotonPower[0] *= reflectance[0]/avgReflectance;
//						currentPhotonPower[1] *= reflectance[1]/avgReflectance;
//						currentPhotonPower[2] *= reflectance[2]/avgReflectance;
//						photonPowerFraction = (currentPhotonPower[0] + currentPhotonPower[1] + currentPhotonPower[2]) / (3.0f * avgdPower);
//
//						// trace photon into scene again:
//						if (m_pSceneGraph->RayTreeIntersect(photonRay, &hitpoint)) {
//							incomingDirection = unitVector(hitpoint.x - photonRay.origin());
//							m_pMaterialList->at(hitpoint.m)->sampleDirection(hitpoint, incomingDirection, photonRayDirection, reflectance);
//							avgReflectance = (reflectance[0] + reflectance[1] + reflectance[2]) / 3.0f;
//							photonHit = 1;							
//						}
//						else {
//							photonHit = 0;
//							break;
//						}
//					}
//
//					if (photonHit) {
//						m_nPhotons++;	
//						addGrid(hitpoint.x, hitpoint.n, currentPhotonPower);
//					}
//					else 
//						m_nPhotonsMissed++;
//					
//				}		
//				else { 
//					m_nPhotonsMissed++;
//				}
//			}
//		}
//	}
//
//	timeBuildEnd.set();
//	timeBuild = timeBuildEnd - timeBuildStart;
//
//	char outputBuffer[2000];
//	sprintf(outputBuffer, "Photon Tracing complete:\t%d seconds, %d milliseconds", (int)timeBuild, (int)((timeBuild - floor(timeBuild)) * 1000));
//	log->logMessage(outputBuffer);	
//	sprintf(outputBuffer, "(%d photons/sec. average)", (int)((float)numPhotons/timeBuild));
//	log->logMessage(outputBuffer);	
//
//	timeBuildStart.set();
//	transformGrid();
//	timeBuildEnd.set();
//	timeBuild = timeBuildEnd - timeBuildStart;
//
//	sprintf(outputBuffer, "Grid Transform complete:\t%d seconds, %d milliseconds", (int)timeBuild, (int)((timeBuild - floor(timeBuild)) * 1000));
//	log->logMessage(outputBuffer);	
//	
//	/*
//	// DEBUG: 
//	GridPhoton val;
//	for (unsigned int j = 0; j < gridDimension[1]; j++) {				
//		for (unsigned int k = 0; k < gridDimension[2]; k++)
//			for (unsigned int i = 0; i < gridDimension[0]; i++) {
//				val[0] = ((float)j / (float)gridDimension[1]);
//				val[1] = ((float)j / (float)gridDimension[1]);
//				val[2] = ((float)j / (float)gridDimension[1]);
//				if (getGridMask(i,j,k))
//					setGrid(i,j,k, val);
//			}
//	}*/
//	
}

void PhotonGrid::GLdrawPhotonMap(Ray &viewer) {
	glMatrixMode( GL_PROJECTION );
	glLoadIdentity();
	glFrustum(-1,1, -1,1, 2, 4000 );
	glMatrixMode( GL_MODELVIEW );

#define GET_GRIDPOINT(i,j,k) reconstructValue(i,j,k)
//#define GET_GRIDPOINT(i,j,k) m_pGrid.at(i,j,k)

	glLoadIdentity();
	Vector3 lookAt = viewer.origin() + viewer.direction();
	gluLookAt(viewer.origin().x(), viewer.origin().y(), viewer.origin().z(),  
		lookAt.x(), lookAt.y(), lookAt.z(), 
		0, 1, 0);

	// clear image
	glClearColor(0,0,0,1);
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

	// set OpenGL state
	glEnable(GL_DEPTH_TEST);		
	glDisable(GL_TEXTURE_2D);	
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_BLEND);

	// wire color
	glColor4f(0.9, 0.1, 0.1, 0.7);

	// draw grid points
	Vector3 startPos, endPos;
	rgb temp;
	glBegin(GL_LINES);
	for (unsigned int k = PHOTONGRID_PADDING; k < gridDimension[2] - PHOTONGRID_PADDING; k++) {
		startPos.setZ(min.z() + (float)(k-PHOTONGRID_PADDING) * gridDelta[2]);
		for (unsigned int j = PHOTONGRID_PADDING; j < gridDimension[1] - PHOTONGRID_PADDING; j++) {
			startPos.setY(min.y() + (float)(j-PHOTONGRID_PADDING) * gridDelta[1]);
			for (unsigned int i = PHOTONGRID_PADDING; i < gridDimension[0] - PHOTONGRID_PADDING; i++) {		
				startPos.setX(min.x() + (float)(i-PHOTONGRID_PADDING) * gridDelta[0]);
				GridPhoton &currPhoton = GET_GRIDPOINT(i,j,k);
				if (fabs(currPhoton[0] 
				       + currPhoton[1] 
				       + currPhoton[2]) > 0.01f) {				

					glColor3fv(currPhoton.data);

					
					if (i < (gridDimension[0] - PHOTONGRID_PADDING - 1)&& GET_GRIDPOINT(i+1,j,k).sumabs() > 0.01f) {
						glVertex3f(startPos.x(), startPos.y(), startPos.z());
						glVertex3f(startPos.x() + gridDelta[0], startPos.y(), startPos.z());
					}

					if (j < (gridDimension[1] - PHOTONGRID_PADDING - 1)&& GET_GRIDPOINT(i,j+1,k).sumabs() > 0.01f) {
						glVertex3f(startPos.x(), startPos.y(), startPos.z());
						glVertex3f(startPos.x(), startPos.y() + gridDelta[1], startPos.z());
					}

					if (k < (gridDimension[2] - PHOTONGRID_PADDING - 1) && GET_GRIDPOINT(i,j,k+1).sumabs() > 0.01f) {
						glVertex3f(startPos.x(), startPos.y(), startPos.z());
						glVertex3f(startPos.x(), startPos.y(), startPos.z() + gridDelta[2]);
					}

					//glVertex3f(startPos.x(), startPos.y(), startPos.z());
				}
			}
		}
	}	
	glEnd();

	// restore state
	glDisable(GL_BLEND);
	glDisable(GL_DEPTH_TEST);	
	glEnable(GL_TEXTURE_2D);
}

void PhotonGrid::clearMap() {
	m_nPhotons = 0;
	m_nPhotonsMissed = 0;	
	m_nGridBlocksUsed = 0;
	m_nGridBlocks = 0;

#ifndef _GRID_HASH	
	if (m_pGrid)
		delete m_pGrid;
	m_pGrid = NULL;
#endif

	if (m_pGridMask)
		delete m_pGridMask;	
	m_pGridMask = NULL;

	min = m_pSceneGraph->objectBB[0];
	max = m_pSceneGraph->objectBB[1];
}

void PhotonGrid::printPhotonMap(const char *LoggerName) {
	LogManager *log = LogManager::getSingletonPtr();
	char outputBuffer[2000];
	log->logMessage("-------------------------------------------", LoggerName);
	log->logMessage("Photon Grid Statistics", LoggerName);
	log->logMessage("-------------------------------------------", LoggerName);
	//sprintf(outputBuffer, "Time to build:\t%d seconds, %d milliseconds", (int)timeBuild, (int)((timeBuild - floor(timeBuild)) * 1000));
	//log->logMessage(outputBuffer, LoggerName);
	sprintf(outputBuffer, "Photons requested:\t%d", m_nPhotons + m_nPhotonsMissed);
	log->logMessage(outputBuffer, LoggerName);
	sprintf(outputBuffer, "Photons in map:\t%d", m_nPhotons );
	log->logMessage(outputBuffer, LoggerName);
	sprintf(outputBuffer, "Photons missed:\t%d", m_nPhotonsMissed);
	log->logMessage(outputBuffer, LoggerName);
	sprintf(outputBuffer, "Grid dimensions:\t%d x %d x %d Voxels, %.3f x %.3f x %.3f per Voxel", 
			gridDimension[0], gridDimension[1], gridDimension[2],
			gridDelta[0], gridDelta[1], gridDelta[2]);
	log->logMessage(outputBuffer, LoggerName);
#ifndef _GRID_HASH
	sprintf(outputBuffer, "Memory usage:\tGrid: %d KB\tMask: %d KB", 
		    int(gridDimension[0] * gridDimension[1] * gridDimension[2] * sizeof(GridPhoton)) / 1024,
			m_nGridBlocks / (8 * 1024));
	log->logMessage(outputBuffer, LoggerName);
#else
	m_pGrid.printStats(LoggerName);
#endif

	sprintf(outputBuffer, "Blocks:\t%d / %d blocks used", m_nGridBlocksUsed, m_nGridBlocks);	
	log->logMessage(outputBuffer, LoggerName);
}

void PhotonGrid::irradiance_estimate(
						 rgb &irrad,				// returned irradiance
						 const Vector3 pos,         // surface position
						 const Vector3 normal,      // surface normal at pos
						 const float max_dist,      // max distance to look for photons
						 const int nphotons, bool verbose ) 
{	

	
	//
	/// Linear interpolation:
	//

	Vector3 temp = (pos - min);
	temp[0] *= gridExtendsInv[0];
	temp[1] *= gridExtendsInv[1];
	temp[2] *= gridExtendsInv[2];

	// calculate indices
	unsigned int i = (unsigned int)(temp[0]);
	unsigned int j = (unsigned int)(temp[1]);
	unsigned int k = (unsigned int)(temp[2]);

	float alpha = temp[0] - i;
	float beta  = temp[1] - j;
	float gamma = temp[2] - k;

	// indices start at 1, allowing for padding
	i+= PHOTONGRID_PADDING;
	j+= PHOTONGRID_PADDING;
	k+= PHOTONGRID_PADDING;
		
	// Bilinear Interpolation 1
	rgb temp1 = (1.0f - alpha)*reconstructValue(i,j,k)   + alpha*reconstructValue(i+1,j,k);
	rgb temp2 = (1.0f - alpha)*reconstructValue(i,j+1,k) + alpha*reconstructValue(i+1,j+1,k);
	temp1 = (1.0f - beta)*temp1 + beta*temp2;

	// Bilinear Interpolation 2
	rgb temp3 = (1.0f - alpha)*reconstructValue(i,j,k+1)   + alpha*reconstructValue(i+1,j,k+1);
	rgb temp4 = (1.0f - alpha)*reconstructValue(i,j+1,k+1) + alpha*reconstructValue(i+1,j+1,k+1);
	temp3 = (1.0f - beta)*temp3 + beta*temp4;
	
	// interpolate values (=> trilinear)
	irrad = (1.0f - gamma)*temp1 + gamma*temp3;
	
	
	/*
	//
	/// Nearest neighbor:
	//

	Vector3 temp = (pos - min);
	temp[0] *= gridExtendsInv[0];
	temp[1] *= gridExtendsInv[1];
	temp[2] *= gridExtendsInv[2];

	unsigned int i = (unsigned int)(temp[0] + 0.5f) + PHOTONGRID_PADDING;
	unsigned int j = (unsigned int)(temp[1] + 0.5f) + PHOTONGRID_PADDING;
	unsigned int k = (unsigned int)(temp[2] + 0.5f) + PHOTONGRID_PADDING;	

	irrad = reconstructValue(i,j,k);
	*/
	
}

FORCEINLINE void PhotonGrid::setGridMask(unsigned int i, unsigned int j, unsigned int k) {
	assert(m_pGridMask != NULL);	
	i >>= PHOTONGRID_BLOCKSIZEPOWER2;
	j >>= PHOTONGRID_BLOCKSIZEPOWER2;
	k >>= PHOTONGRID_BLOCKSIZEPOWER2;
	unsigned int offset = blockoffsetZ*k + blockoffsetY*j + i;	// index
	unsigned int bitnum = offset % 8;							// bit in the respective byte
	offset >>= 3;												// byte address in array (offset / 8)
	unsigned char curVal = m_pGridMask[offset];	
	m_nGridBlocksUsed += 1 - ((curVal >> bitnum) & 1);
	curVal |= 1 << bitnum;
	m_pGridMask[offset] = curVal;
}

FORCEINLINE bool PhotonGrid::getGridMask(unsigned int i, unsigned int j, unsigned int k) {
	assert(m_pGridMask != NULL);	
	i >>= PHOTONGRID_BLOCKSIZEPOWER2;
	j >>= PHOTONGRID_BLOCKSIZEPOWER2;
	k >>= PHOTONGRID_BLOCKSIZEPOWER2;
	unsigned int offset = blockoffsetZ*k + blockoffsetY*j + i;	// index
	unsigned int bitnum = offset % 8;							// bit in the respective byte
	offset >>= 3;												// byte address in array (offset / 8)	
	return ((m_pGridMask[offset] >> bitnum) & 1);	
}

FORCEINLINE unsigned int PhotonGrid::getBlock(unsigned int i, unsigned int j, unsigned int k) {
	i >>= PHOTONGRID_BLOCKSIZEPOWER2;
	j >>= PHOTONGRID_BLOCKSIZEPOWER2;
	k >>= PHOTONGRID_BLOCKSIZEPOWER2;

	return k*PHOTONGRID_BLOCKOFFSET*blockoffsetZ
		   + j*PHOTONGRID_BLOCKOFFSET*blockoffsetY
		   + i*PHOTONGRID_BLOCKOFFSET;
}

FORCEINLINE GridPhoton& PhotonGrid::getGridPoint(unsigned int i, unsigned int j, unsigned int k) {		
	// address of grid point = 
	//  block address 
	//  + z offset in block
	//  + y offset in block
	//  + x offset in block

	return m_pGrid[getBlock(i,j,k)
					+ ((k & (PHOTONGRID_BLOCKSIZE-1)) << (PHOTONGRID_BLOCKSIZEPOWER2+PHOTONGRID_BLOCKSIZEPOWER2))
					+ ((j & (PHOTONGRID_BLOCKSIZE-1)) << PHOTONGRID_BLOCKSIZEPOWER2)
					+ (i & (PHOTONGRID_BLOCKSIZE-1))];
}

void PhotonGrid::transformGrid() {

	// information on used coefficients in the blocks:
	float *coeffUsed;
	// sum of all used coefficients ratios
	double coeffUsedTotalRatio = 0.0;

	// detail coefficients per block:
	float numDetailCoeffsPerBlock = (PHOTONGRID_BLOCKSIZE*PHOTONGRID_BLOCKSIZE*PHOTONGRID_BLOCKSIZE) 
		                          - ((PHOTONGRID_BLOCKSIZE*PHOTONGRID_BLOCKSIZE*PHOTONGRID_BLOCKSIZE) >> (3*m_nDecompositionSteps));
	
	unsigned int numDetailCoeffsTotal = numDetailCoeffsPerBlock * m_nGridBlocks;

	// allocate and initialize array
	coeffUsed = new float[m_nGridBlocks];
	if (!coeffUsed) {
		LogManager *log = LogManager::getSingletonPtr();
		log->logMessage(LOG_CRITICAL, "Could not allocate memory for coefficients during wavelet transform !");
		return;
	}

	memset(coeffUsed, 0, m_nGridBlocks * sizeof(float));


	m_pGrid.printStats();

	//
	// STEP 1:
	//
	// transform all blocks:
	//
	unsigned int k;
	for (k = 0; k < gridDimension[2]/PHOTONGRID_BLOCKSIZE; k++) {		
		for (unsigned int j = 0; j < gridDimension[1]/PHOTONGRID_BLOCKSIZE; j++) {			
			for (unsigned int i = 0; i < gridDimension[0]/PHOTONGRID_BLOCKSIZE; i++) {	

				// if there's no photon in the whole block, we can skip it:
				// TODO: fix grid mask !
				if (!getGridMask(i<<PHOTONGRID_BLOCKSIZEPOWER2,j<<PHOTONGRID_BLOCKSIZEPOWER2,k<<PHOTONGRID_BLOCKSIZEPOWER2))
					continue;

				// wavelet transform the block
				float used = transformBlock(i<<PHOTONGRID_BLOCKSIZEPOWER2,
					                        j<<PHOTONGRID_BLOCKSIZEPOWER2,
											k<<PHOTONGRID_BLOCKSIZEPOWER2) / numDetailCoeffsPerBlock;

				// add the number of used coefficients
				coeffUsed[k*blockoffsetZ + j*blockoffsetY + i] = used;
				coeffUsedTotalRatio += used;
			}
		}
	}

	//
	// STEP 2:
	//
	// thresholding with relative block compression ratio
	//

	// number of detail coefficients to retain, all others should be thresholded:
	float coeffToUseTotal = numDetailCoeffsTotal * m_fCompressionRate;

	// divide by the sum of all the ratios calculated while transforming the blocks, 
	// this is our base ratio. The effective block ratio now is determined by the
	// local ratio in the next loop
	float baseRatio = coeffToUseTotal / coeffUsedTotalRatio;

	for (k = 0; k < gridDimension[2]/PHOTONGRID_BLOCKSIZE; k++) {		
		for (unsigned int j = 0; j < gridDimension[1]/PHOTONGRID_BLOCKSIZE; j++) {			
			for (unsigned int i = 0; i < gridDimension[0]/PHOTONGRID_BLOCKSIZE; i++) {
				if (!getGridMask(i<<PHOTONGRID_BLOCKSIZEPOWER2,j<<PHOTONGRID_BLOCKSIZEPOWER2,k<<PHOTONGRID_BLOCKSIZEPOWER2))
					continue;

				// calculate compression rate
				float compressionRate = (coeffUsed[k*blockoffsetZ + j*blockoffsetY + i] * baseRatio) / numDetailCoeffsPerBlock;
				assert(compressionRate >= 0.0f && compressionRate <= 1.0f);
				
				applyThresholding(i<<PHOTONGRID_BLOCKSIZEPOWER2,j<<PHOTONGRID_BLOCKSIZEPOWER2,k<<PHOTONGRID_BLOCKSIZEPOWER2, compressionRate);
			}
		}
	}

	delete coeffUsed;
}

unsigned int PhotonGrid::transformBlock(unsigned int x, unsigned int y, unsigned int z) {
	static GridBlock originalBlock;
	unsigned int coeffPointers[8];
	GridPhoton *origPointer, coeffCache[8];
	unsigned int nonZeroCoefficients = 0;

	// initialize pointers to places where we will put the transformed
	// coefficients in the resulting block (8 subblocks)
	unsigned int transformedBlock = getBlock(x,y,z);

#ifndef _GRID_HASH
	// Copy original block to puffer:
	memcpy(originalBlock, &m_pGrid[transformedBlock], sizeof(GridBlock));
#else
	// Copy original block to puffer:
	for (unsigned int i=0; i < PHOTONGRID_BLOCKOFFSET; i++) {
		originalBlock[i] = m_pGrid[transformedBlock + i];
		//m_pGrid.erase(transformedBlock + i);
	}
#endif

	unsigned int s, blocksizepower2;
	for (s = 0, blocksizepower2 = PHOTONGRID_BLOCKSIZEPOWER2; s < m_nDecompositionSteps; s++, blocksizepower2--) {
		unsigned int blocksize = 1 << blocksizepower2;
		unsigned int subblock_size = 1 << (blocksizepower2 + blocksizepower2 + blocksizepower2 - 3);
			
		// indices for transformed coefficients
		coeffPointers[0] = transformedBlock;					// approximation coeff. (l l l)
		coeffPointers[1] = coeffPointers[0] + subblock_size;	// l l h
		coeffPointers[2] = coeffPointers[1] + subblock_size;    // l h l
		coeffPointers[3] = coeffPointers[2] + subblock_size;	// l h h
		coeffPointers[4] = coeffPointers[3] + subblock_size;	// h l l
		coeffPointers[5] = coeffPointers[4] + subblock_size;	// h l h
		coeffPointers[6] = coeffPointers[5] + subblock_size;	// h h l
		coeffPointers[7] = coeffPointers[6] + subblock_size;	// h h h
	
		// do haar transform on this level:
		for (unsigned int k = 0; k < blocksize; k+=2) {
			for (unsigned int j = 0; j < blocksize; j+=2) {
				for (unsigned int i = 0; i < blocksize; i+=2) {	
					origPointer = &originalBlock[(k << (blocksizepower2+blocksizepower2))+ (j << blocksizepower2) + i ];

					// approximation signal
					coeffCache[0] = ( *(origPointer) 
												+ *(origPointer + 1) 
												+ *(origPointer + blocksize) 
												+ *(origPointer + blocksize + 1) 
												+ *(origPointer + blocksize*blocksize) 
												+ *(origPointer + blocksize*blocksize + 1) 
												+ *(origPointer + blocksize*blocksize + blocksize) 
												+ *(origPointer + blocksize*blocksize + blocksize + 1) 
											) / 8.0f;	

					/*
					coeffCache[0] = rgb(1.0f, 1.0f, 1.0f);

					for (int t = 1; t < 8; t++)
						coeffCache[t] = rgb(0.0f, 0.0f, 0.0f);
						*/

					// detail signal 1..7:

					coeffCache[1] = ( *(origPointer) 
												+ *(origPointer + 1) 
												+ *(origPointer + blocksize) 
												+ *(origPointer + blocksize + 1) 
												- *(origPointer + blocksize*blocksize) 
												- *(origPointer + blocksize*blocksize + 1) 
												- *(origPointer + blocksize*blocksize + blocksize) 
												- *(origPointer + blocksize*blocksize + blocksize + 1) 
											) / 8.0f;
					coeffCache[2] = ( *(origPointer) 
												+ *(origPointer + 1) 
												- *(origPointer + blocksize) 
												- *(origPointer + blocksize + 1) 
												+ *(origPointer + blocksize*blocksize) 
												+ *(origPointer + blocksize*blocksize + 1) 
												- *(origPointer + blocksize*blocksize + blocksize) 
												- *(origPointer + blocksize*blocksize + blocksize + 1) 
											) / 8.0f;
					coeffCache[3] = ( *(origPointer) 
												+ *(origPointer + 1) 
												- *(origPointer + blocksize) 
												- *(origPointer + blocksize + 1) 
												- *(origPointer + blocksize*blocksize) 
												- *(origPointer + blocksize*blocksize + 1) 
												+ *(origPointer + blocksize*blocksize + blocksize) 
												+ *(origPointer + blocksize*blocksize + blocksize + 1) 
											) / 8.0f;
					coeffCache[4] = ( *(origPointer) 
												- *(origPointer + 1) 
												+ *(origPointer + blocksize) 
												- *(origPointer + blocksize + 1) 
												+ *(origPointer + blocksize*blocksize) 
												- *(origPointer + blocksize*blocksize + 1) 
												+ *(origPointer + blocksize*blocksize + blocksize) 
												- *(origPointer + blocksize*blocksize + blocksize + 1) 
												) / 8.0f;
					coeffCache[5] = ( *(origPointer) 
												- *(origPointer + 1) 
												+ *(origPointer + blocksize) 
												- *(origPointer + blocksize + 1) 
												- *(origPointer + blocksize*blocksize) 
												+ *(origPointer + blocksize*blocksize + 1) 
												- *(origPointer + blocksize*blocksize + blocksize) 
												+ *(origPointer + blocksize*blocksize + blocksize + 1) 
												) / 8.0f;
					coeffCache[6] = ( *(origPointer) 
												- *(origPointer + 1) 
												- *(origPointer + blocksize) 
												+ *(origPointer + blocksize + 1) 
												+ *(origPointer + blocksize*blocksize) 
												- *(origPointer + blocksize*blocksize + 1) 
												- *(origPointer + blocksize*blocksize + blocksize) 
												+ *(origPointer + blocksize*blocksize + blocksize + 1) 
												) / 8.0f;
					coeffCache[7] = ( *(origPointer) 
												- *(origPointer + 1) 
												- *(origPointer + blocksize) 
												+ *(origPointer + blocksize + 1) 
												- *(origPointer + blocksize*blocksize) 
												+ *(origPointer + blocksize*blocksize + 1) 
												+ *(origPointer + blocksize*blocksize + blocksize) 
												- *(origPointer + blocksize*blocksize + blocksize + 1) 
												) / 8.0f;

					// write approximation coefficients to original block for next decomposition step:
					originalBlock[coeffPointers[0]-transformedBlock] = coeffCache[0];
					
					
					// if coefficient larger than threshold, then write
					if (coeffCache[0].sumabs() > EPSILON) {	
						m_pGrid[coeffPointers[0]] = coeffCache[0];						
					}
					else
						m_pGrid.erase(coeffPointers[0]);

					coeffPointers[0]++;	

					// write detail coefficients
					for (int t = 1; t < 8; t++) {
						assert(coeffPointers[t] < gridDimension[0]*gridDimension[1]*gridDimension[2]);

						// if coefficient larger than threshold, then write
						if (coeffCache[t].sumabs() > EPSILON) {	
							m_pGrid[coeffPointers[t]] = coeffCache[t];
							nonZeroCoefficients++;
						}
						else
							m_pGrid.erase(coeffPointers[t]);

						// move to next coefficient 
						coeffPointers[t]++;
					}

				}
			}
		}
	}

	return nonZeroCoefficients;	
}

void PhotonGrid::applyThresholding(unsigned int x, unsigned int y, unsigned int z, float ratio) {
	static GridBlock originalBlock;
	GridPointVector pointHeap(PHOTONGRID_BLOCKSIZE*PHOTONGRID_BLOCKSIZE*PHOTONGRID_BLOCKSIZE);
	unsigned int nonZeroCoefficients = 0;
	pointHeap.clear();

	// initialize pointers to places where we will put the transformed
	// coefficients in the resulting block (8 subblocks)
	unsigned int transformedBlock = getBlock(x,y,z);

#ifndef _GRID_HASH
	// Copy original block to puffer:
	memcpy(originalBlock, &m_pGrid[transformedBlock], sizeof(GridBlock));
#else
	// Copy original block to puffer:
	for (unsigned int i=0; i < (PHOTONGRID_BLOCKSIZE*PHOTONGRID_BLOCKSIZE*PHOTONGRID_BLOCKSIZE); i++)
		originalBlock[i] = m_pGrid[transformedBlock + i];
#endif

	// size of the sub-block containing the approximation coefficients
	unsigned int approximationBlockSize = PHOTONGRID_BLOCKSIZE >> m_nDecompositionSteps;

	for (unsigned int k = 0; k < PHOTONGRID_BLOCKSIZE; k++) {
		for (unsigned int j = 0; j < PHOTONGRID_BLOCKSIZE; j++) {
			for (unsigned int i = 0; i < PHOTONGRID_BLOCKSIZE; i++) {	
				// ignore approximation coefficients:
				if (k < approximationBlockSize && j < approximationBlockSize && k < approximationBlockSize)
					continue;

				GridPointSort temp;

				// relative offset in block:
				temp.offset = k * PHOTONGRID_BLOCKSIZE * PHOTONGRID_BLOCKSIZE 
					        + j * PHOTONGRID_BLOCKSIZE 
							+ i;

				// get value
				temp.val = originalBlock[temp.offset];

				// make absolute offset
				temp.offset += transformedBlock;

				// insert into heap
				pointHeap.push_back(temp);

				// delete in original grid
				m_pGrid.erase(temp.offset);
			}
		}
	}

	//
	// We got all detail coefficients, now sort them and cut off
	// below the threshold (the ratio*num'th coefficient)
	//

	// create heap from vector of coefficients
	std::make_heap(pointHeap.begin(), pointHeap.end());

	unsigned int numVals = (unsigned int)((float)(PHOTONGRID_BLOCKSIZE*PHOTONGRID_BLOCKSIZE*PHOTONGRID_BLOCKSIZE) * ratio);
	numVals = min(numVals, pointHeap.size());

	//for (GridPointVector::iterator i = pointHeap.begin(); i != pointHeap.end(); i++)
	//	m_pGrid[i->offset] = i->val;

	for (unsigned int l=0; l < numVals; l++) {
		if (pointHeap.begin()->val.sumabs() <= 0.001f)
			break;
		m_pGrid.set(pointHeap.begin()->offset, pointHeap.begin()->val);
		std::pop_heap(pointHeap.begin(),pointHeap.end());	
		pointHeap.pop_back();
	}
	
}

rgb PhotonGrid::reconstructValue(unsigned int i, unsigned int j, unsigned int k) {	
	unsigned int ib, jb, kb;				// indices in the current block	
	unsigned int coeffPointers[8];	
	unsigned int transformedBlock = getBlock(i,j,k);
	unsigned int coeff;
	rgb target;

	// transform coordinates to block coords
	i &= (PHOTONGRID_BLOCKSIZE-1);
	j &= (PHOTONGRID_BLOCKSIZE-1);
	k &= (PHOTONGRID_BLOCKSIZE-1);

	unsigned int blocksizepower2 = PHOTONGRID_BLOCKSIZEPOWER2 - m_nDecompositionSteps + 1;
	unsigned int subblock_size = 1 << (blocksizepower2 + blocksizepower2 + blocksizepower2 - 3);
	unsigned int blocksize = 1 << blocksizepower2;			
	
	ib = i >> (m_nDecompositionSteps-1);
	jb = j >> (m_nDecompositionSteps-1);
	kb = k >> (m_nDecompositionSteps-1);

	// selection of one of the 8 coefficients
	coeff = (ib & 1) | ((jb & 1)<<1) | ((kb & 1)<<2);

	ib &= ~0x01;
	jb &= ~0x01;
	kb &= ~0x01;

	// initialize first approximation coefficient
	target = m_pGrid[(transformedBlock + (kb << (blocksizepower2+blocksizepower2-3))
							   + (jb << (blocksizepower2-2)) 
							   + (ib >> 1))];
	
	for (unsigned int s = m_nDecompositionSteps; s > 0;) {
		
		// coefficient pointers:
		coeffPointers[0] = (transformedBlock + (kb << (blocksizepower2+blocksizepower2-3))
							                 + (jb << (blocksizepower2-2)) 
							                 + (ib >> 1));
		coeffPointers[1] = coeffPointers[0] + subblock_size;	// l l h
		coeffPointers[2] = coeffPointers[1] + subblock_size;    // l h l
		coeffPointers[3] = coeffPointers[2] + subblock_size;	// l h h
		coeffPointers[4] = coeffPointers[3] + subblock_size;	// h l l
		coeffPointers[5] = coeffPointers[4] + subblock_size;	// h l h
		coeffPointers[6] = coeffPointers[5] + subblock_size;	// h h l
		coeffPointers[7] = coeffPointers[6] + subblock_size;	// h h h

		// formula for reconstruction varies depending on which coefficient
		// of the 2x2x2 block we want:
		switch (coeff) {
		case 0: // c_000
			target = target + m_pGrid[coeffPointers[1]] + m_pGrid[coeffPointers[2]] + m_pGrid[coeffPointers[3]]
							+ m_pGrid[coeffPointers[4]] + m_pGrid[coeffPointers[5]] + m_pGrid[coeffPointers[6]] + m_pGrid[coeffPointers[7]];
			break;
		case 1: // c_001
			target = target + m_pGrid[coeffPointers[1]] + m_pGrid[coeffPointers[2]] + m_pGrid[coeffPointers[3]]
							- m_pGrid[coeffPointers[4]] - m_pGrid[coeffPointers[5]] - m_pGrid[coeffPointers[6]] - m_pGrid[coeffPointers[7]];
			break;
		case 2: // c_010
			target = target + m_pGrid[coeffPointers[1]] - m_pGrid[coeffPointers[2]] - m_pGrid[coeffPointers[3]]
							+ m_pGrid[coeffPointers[4]] + m_pGrid[coeffPointers[5]] - m_pGrid[coeffPointers[6]] - m_pGrid[coeffPointers[7]];
			break;
		case 3: // c_011
			target = target + m_pGrid[coeffPointers[1]] - m_pGrid[coeffPointers[2]] - m_pGrid[coeffPointers[3]]
							- m_pGrid[coeffPointers[4]] - m_pGrid[coeffPointers[5]] + m_pGrid[coeffPointers[6]] + m_pGrid[coeffPointers[7]];
			break;
		case 4: // c_100
			target = target - m_pGrid[coeffPointers[1]] + m_pGrid[coeffPointers[2]] - m_pGrid[coeffPointers[3]]
							+ m_pGrid[coeffPointers[4]] - m_pGrid[coeffPointers[5]] + m_pGrid[coeffPointers[6]] - m_pGrid[coeffPointers[7]];
			break;
		case 5: // c_101
			target = target - m_pGrid[coeffPointers[1]] + m_pGrid[coeffPointers[2]] - m_pGrid[coeffPointers[3]]
							- m_pGrid[coeffPointers[4]] + m_pGrid[coeffPointers[5]] - m_pGrid[coeffPointers[6]] + m_pGrid[coeffPointers[7]];
			break;
		case 6: // c_110
			target = target - m_pGrid[coeffPointers[1]] - m_pGrid[coeffPointers[2]] + m_pGrid[coeffPointers[3]]
							+ m_pGrid[coeffPointers[4]] - m_pGrid[coeffPointers[5]] - m_pGrid[coeffPointers[6]] + m_pGrid[coeffPointers[7]];
			break;
		case 7: // c_111
			target = target - m_pGrid[coeffPointers[1]] - m_pGrid[coeffPointers[2]] + m_pGrid[coeffPointers[3]]
							- m_pGrid[coeffPointers[4]] + m_pGrid[coeffPointers[5]] + m_pGrid[coeffPointers[6]] - m_pGrid[coeffPointers[7]];
			break;
		default: // error: this shouldn't occur
			target = rgb(0.0f, 0.0f, 0.0f);
		}
		
		// increase block size
		s--;
		blocksizepower2++;
		subblock_size <<= 3;
		blocksize <<= 1;

		if (s > 1) {
			ib = i >> (s-1);
			jb = j >> (s-1);
			kb = k >> (s-1);
		}
		else {
			ib = i;
			jb = j;
			kb = k;
		}
		
		// selection of one of the 8 coefficients
		coeff = (ib & 1) | ((jb & 1)<<1) | ((kb & 1)<<2);

		ib &= ~0x01;
		jb &= ~0x01;
		kb &= ~0x01;
	}

	return target;	
}