/********************************************************************
	file base:	PhotonMap
	file ext:	cpp
	author:		Christian Lauterbach (lauterb@informatik.uni-bremen.de)
	
	purpose:	Photon mapping class. Code is based on Henrik Wann Jensen's
	            code from "Realisting Image Synthesis Using Photon Mapping"
*********************************************************************/

#include "stdafx.h"
#include <GL/GL.h>
#include <GL/glut.h>
#include "common.h"
#include "PhotonMap.h"
#include "Sample.h"


PhotonMap::PhotonMap(BVH &sceneGraph, LightList &lights, EmitterList &emitters, MaterialList &materials) {
	m_pSceneGraph = &sceneGraph;
	m_pLightList = &lights;
	m_pEmitterList = &emitters;
	m_pMaterialList = &materials;

	// init precalculated conversion tables
	for (int i = 0; i < 256; i++) {
		float angle = float(i) * (1.0f / 256.0f) * PI;
		costheta[i] = cosf(angle);
		sintheta[i] = sinf(angle);
		cosphi[i] = cosf(2.0f*angle);
		sinphi[i] = sinf(2.0f*angle);
	}

	clearMap();
}


void PhotonMap::buildPhotonMap(unsigned int numPhotons) {
	//Ray photonRay;
	//Hitpoint hitpoint;
	//Sampler sampler;
	//RandomLinear generator;
	//Vector2 sampleDirection;
	//Vector3 pointOnEmitter;
	//Vector3 photonRayDirection;
	//Photon newPhoton;
	//rgb color, reflectance;
	//int photonHit;
	//rgb currentPhotonPower;

	//// clear the previous map (just to be certain)
	//clearMap();

	//timeBuildStart.set();	

	//if (m_pEmitterList->size() > 0) { // do we have area light sources ?
	//	EmitterListIterator e;

	//	// calc overall power of emitters
	//	float sumIntensity = 0.0f;
	//	for(e = m_pEmitterList->begin(); e != m_pEmitterList->end(); e++)
	//		sumIntensity += (*e).summedIntensity;

	//	// power per photon
	//	float dPowerFactor = sumIntensity * (3000.0f / numPhotons);

	//	// emit photons for every emitter
	//	for(e = m_pEmitterList->begin(); e != m_pEmitterList->end(); e++) {
	//		// number of photons for this emitter is related to its
	//		// intensity in relation to the total intensity;
	//		unsigned int numPhotonsForEmitter = ceil(((*e).summedIntensity / sumIntensity) * numPhotons);
	//		rgb dPower = (*e).emissionIntensity * (3000.0f / (float)numPhotonsForEmitter);

	//		for (unsigned int i = 0; i < numPhotonsForEmitter; i++) {
	//			currentPhotonPower = dPower;
	//			
	//			// Sample point on emitter
	//			sampler.random2(&sampleDirection, 1);
	//			(*e).sample(sampleDirection[0], sampleDirection[1], pointOnEmitter);

	//			// Sample direction from emitter 
	//			sampler.random2(&sampleDirection, 1);
	//			(*e).sampleDirection(sampleDirection[0], sampleDirection[1], photonRayDirection);
	//			
	//			// Generate ray
	//			photonRay.setOrigin(pointOnEmitter);
	//			photonRay.setDirection(photonRayDirection);

	//			// Trace photon into scene:
	//			if (photonHit = m_pSceneGraph->RayTreeIntersect(photonRay, &hitpoint)) {
	//				// Photon hit something:
	//				Vector3 incomingDirection = unitVector(hitpoint.x - photonRay.origin());
	//				m_pMaterialList->at(hitpoint.m)->sampleDirection(hitpoint, incomingDirection, photonRayDirection, reflectance);

	//				float avgReflectance = (reflectance[0] + reflectance[1] + reflectance[2]) / 3.0f;

	//				// is this particle reflected or absorbed ?
	//				// (russian roulette instead of reflecting a particle
	//				//  with less power)
	//				while (generator.sample() <= avgReflectance) {
	//					// reflected, so generate new ray:
	//					photonRay.setOrigin(hitpoint.x);
	//					photonRay.setDirection(photonRayDirection);

	//					// modify photon power by wavelength (for color bleeding and similar effects)
	//					currentPhotonPower[0] *= reflectance[0] / avgReflectance;
	//					currentPhotonPower[1] *= reflectance[1] / avgReflectance;
	//					currentPhotonPower[2] *= reflectance[2] / avgReflectance;

	//					// trace photon into scene again:
	//					if (m_pSceneGraph->RayTreeIntersect(photonRay, &hitpoint)) {
	//						incomingDirection = unitVector(hitpoint.x - photonRay.origin());
	//						m_pMaterialList->at(hitpoint.m)->sampleDirection(hitpoint, incomingDirection, photonRayDirection, reflectance);
	//						avgReflectance = (reflectance[0] + reflectance[1] + reflectance[2]) / 3.0f;							
	//					}
	//					else {
	//						photonHit = 0;
	//						break;
	//					}
	//				}
	//				
	//				if (photonHit) {
	//					// TODO: fix this angles!
	//					newPhoton.theta = acosf(-dot(hitpoint.n, photonRayDirection));

	//					newPhoton.phi   = sampleDirection[1];
	//					newPhoton.pos = hitpoint.x;
	//					newPhoton.power = currentPhotonPower;
	//					newPhoton.plane = 0;

	//					for (int j = 0; j < 3; j++) {
	//						if (newPhoton.pos[j] < min[j])
	//							min[j] = newPhoton.pos[j];
	//						if (newPhoton.pos[j] > max[j])
	//							max[j] = newPhoton.pos[j];
	//					}

	//					// store new photon in array
	//					m_Photons.push_back(newPhoton);
	//					m_nPhotons++;
	//				}
	//				else { 
	//					m_nPhotonsMissed++;
	//				}
	//			}		
	//			else { 
	//				m_nPhotonsMissed++;
	//			}
	//		}
	//	}
	//}
	//else { // no area light source, legacy case for point lights
	//	unsigned char phi, theta;
	//	LightListIterator l = m_pLightList->begin();		

	//	// power per photon
	//	rgb dPower = (*l).color * (100000.0f / numPhotons);

	//	for (unsigned int i = 0; i < numPhotons; i++) {
	//		currentPhotonPower = dPower;

	//		// Generate phi, theta for light direction
	//		sampler.random2(&sampleDirection, 1);
	//		phi = (unsigned char)(sampleDirection[0] * 255.0f);
	//		theta = (unsigned char)(sampleDirection[1] * 255.0f);		

	//		// Initialize new ray from light into direction specified by theta and phi
	//		photonRay.setOrigin((*l).pos);
	//		photonRayDirection[0] = sinf(sampleDirection[1] * PI) * cosf(sampleDirection[0] * 2.0f * PI);
	//		photonRayDirection[1] = sinf(sampleDirection[1] * PI) * sinf(sampleDirection[0] * 2.0f * PI);
	//		photonRayDirection[2] = cosf(sampleDirection[1] * PI);
	//		photonRay.setDirection(photonRayDirection);
	//		
	//		// Trace photon into scene:
	//		if (m_pSceneGraph->RayTreeIntersect(photonRay, &hitpoint)) {
	//			// Photon hit something:

	//			newPhoton.theta = sampleDirection[0];
	//			newPhoton.phi = sampleDirection[1];
	//			newPhoton.pos = hitpoint.x;
	//			newPhoton.power = currentPhotonPower;
	//			newPhoton.plane = 0;

	//			for (int j = 0; j < 3; j++) {
	//				if (newPhoton.pos[j] < min[j])
	//					min[j] = newPhoton.pos[j];
	//				if (newPhoton.pos[j] > max[j])
	//					max[j] = newPhoton.pos[j];
	//			}
	//			
	//			// store new photon in array
	//			m_Photons.push_back(newPhoton);
	//			m_nPhotons++;
	//		}		
	//		else { 
	//			m_nPhotonsMissed++;
	//		}
	//	}
	//}

	//// resize vector to optimal size:
	//PhotonList(m_Photons).swap(m_Photons);


	//// If we have stored photons, now create tree.
	//// (NB: code is heavily borrowed from H.W.Jenssen: "Realistic
	////  image synthesis using photon mapping")
	//if (m_nPhotons > 1) {
	//	// allocate two temporary arrays for the balancing procedure
	//	Photon **pa1 = (Photon**)malloc(sizeof(Photon*)*(m_nPhotons+1));
	//	Photon **pa2 = (Photon**)malloc(sizeof(Photon*)*(m_nPhotons+1));

	//	unsigned int i;
	//	for ( i=0; i<=m_nPhotons; i++)
	//		pa2[i] = &m_Photons[i];

	//	balance_segment( pa1, pa2, 1, 1, m_nPhotons );
	//	free(pa2);

	//	// reorganize balanced kd-tree (make a heap)
	//	int d, j=1;
	//	unsigned int foo=1;
	//	Photon foo_photon = m_Photons[j];

	//	for (i=1; i<=m_nPhotons; i++) {
	//		d = pa1[j] - &m_Photons[0];
	//		pa1[j] = NULL;
	//		if (d != foo)
	//			m_Photons[j] = m_Photons[d];
	//		else {
	//			m_Photons[j] = foo_photon;

	//			if (i<m_nPhotons) {
	//				for (;foo<=m_nPhotons; foo++)
	//					if (pa1[foo] != NULL)
	//						break;
	//				foo_photon = m_Photons[foo];
	//				j = foo;
	//			}
	//			continue;
	//		}
	//		j = d;
	//	}
	//	free(pa1);
	//}

	//m_nHalfStoredPhotons = m_nPhotons/2 - 1;

	//timeBuildEnd.set();
	//timeBuild = timeBuildEnd - timeBuildStart;
}

#define swap(ph,a,b) { Photon *ph2=ph[a]; ph[a]=ph[b]; ph[b]=ph2; }

// median_split splits the photon array into two separate
// pieces around the median with all photons below the
// the median in the lower half and all photons above
// than the median in the upper half. The comparison
// criteria is the axis (indicated by the axis parameter)
// (inspired by routine in "Algorithms in C++" by Sedgewick)
//*****************************************************************
void PhotonMap::median_split(
								Photon **p,
								const int start,               // start of photon block in array
								const int end,                 // end of photon block in array
								const int median,              // desired median number
								const int axis )               // axis to split along
								//*****************************************************************
{
	int left = start;
	int right = end;

	while ( right > left ) {
		const float v = p[right]->pos[axis];
		int i=left-1;
		int j=right;
		for (;;) {
			while ( p[++i]->pos[axis] < v )
				;
			while ( p[--j]->pos[axis] > v && j>left )
				;
			if ( i >= j )
				break;
			swap(p,i,j);
		}

		swap(p,i,right);
		if ( i >= median )
			right=i-1;
		if ( i <= median )
			left=i+1;
	}
}


// See "Realistic image synthesis using Photon Mapping" chapter 6
// for an explanation of this function
//****************************
void PhotonMap::balance_segment(Photon **pbal,
								Photon **porg,
								const int index,
								const int start,
								const int end )								
{
	//--------------------
	// compute new median
	//--------------------

	int median=1;
	while ((4*median) <= (end-start+1))
		median += median;

	if ((3*median) <= (end-start+1)) {
		median += median;
		median += start-1;
	} else	
		median = end-median+1;

	//--------------------------
	// find axis to split along
	//--------------------------

	int axis=2;
	if ((max[0]-min[0])>(max[1]-min[1]) &&
		(max[0]-min[0])>(max[2]-min[2]))
		axis=0;
	else if ((max[1]-min[1])>(max[2]-min[2]))
		axis=1;

	//------------------------------------------
	// partition photon block around the median
	//------------------------------------------

	median_split( porg, start, end, median, axis );

	pbal[ index ] = porg[ median ];
	pbal[ index ]->plane = axis;

	//----------------------------------------------
	// recursively balance the left and right block
	//----------------------------------------------

	if ( median > start ) {
		// balance left segment
		if ( start < median-1 ) {
			const float tmp=max[axis];
			max[axis] = pbal[index]->pos[axis];
			balance_segment( pbal, porg, 2*index, start, median-1 );
			max[axis] = tmp;
		} else {
			pbal[ 2*index ] = porg[start];
		}
	}

	if ( median < end ) {
		// balance right segment
		if ( median+1 < end ) {
			const float tmp = min[axis];		
			min[axis] = pbal[index]->pos[axis];
			balance_segment( pbal, porg, 2*index+1, median+1, end );
			min[axis] = tmp;
		} else {
			pbal[ 2*index+1 ] = porg[end];
		}
	}	
}

void PhotonMap::GLdrawPhotonMap(Ray &viewer) {
	glMatrixMode( GL_PROJECTION );
	glLoadIdentity();
	glFrustum(-1,1, -1,1, 2, 4000 );
	glMatrixMode( GL_MODELVIEW );

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
	glColor4f(0.9, 0.9, 0.9, 0.7);

	
	// draw all photons as points
	if (m_nPhotons > 0) {
		glBegin(GL_POINTS);
		for (PhotonListIterator i = m_Photons.begin(); i != m_Photons.end(); i++) {
			//glColor3fv((*i).power.data);
			glVertex3f((*i).pos.x(), (*i).pos.y(), (*i).pos.z());
		}
		glEnd();
	}	

	// restore state
	glDisable(GL_BLEND);
	glDisable(GL_DEPTH_TEST);	
	glEnable(GL_TEXTURE_2D);
	
}

void PhotonMap::clearMap() {
	m_Photons.clear();
	m_nPhotons = 0;
	m_nPhotonsMissed = 0;
	m_nHalfStoredPhotons = 0;

	min[0] = -FLT_MAX;
	min[1] = -FLT_MAX;
	min[2] = -FLT_MAX;
	max[0] =  FLT_MAX;
	max[1] =  FLT_MAX;
	max[2] =  FLT_MAX;
}

void PhotonMap::printPhotonMap(const char *LoggerName) {
	LogManager *log = LogManager::getSingletonPtr();
	char outputBuffer[2000];
	log->logMessage("-------------------------------------------", LoggerName);
	log->logMessage("Photon Map Statistics", LoggerName);
	log->logMessage("-------------------------------------------", LoggerName);
	sprintf(outputBuffer, "Time to build:\t%d seconds, %d milliseconds", (int)timeBuild, (int)((timeBuild - floor(timeBuild)) * 1000));
	log->logMessage(outputBuffer, LoggerName);
	sprintf(outputBuffer, "Photons requested:\t%d", m_nPhotons + m_nPhotonsMissed);
	log->logMessage(outputBuffer, LoggerName);
	sprintf(outputBuffer, "Photons in map:\t%d", m_nPhotons );
	log->logMessage(outputBuffer, LoggerName);
	sprintf(outputBuffer, "Photons missed:\t%d", m_nPhotonsMissed);
	log->logMessage(outputBuffer, LoggerName);
}

/* photon_dir returns the direction of a photon
*/
//*****************************************************************
FORCEINLINE void PhotonMap::photon_dir( float *dir, const Photon *p ) const
//*****************************************************************
{
	dir[0] = sintheta[p->theta]*cosphi[p->phi];
	dir[1] = sintheta[p->theta]*sinphi[p->phi];
	dir[2] = costheta[p->theta];
}

/* irradiance_estimate computes an irradiance estimate
* at a given surface position
*/
//**********************************************
void PhotonMap::irradiance_estimate(rgb &irrad,                 // returned irradiance
									const float pos[3],         // surface position
									const float normal[3],      // surface normal at pos
									const float max_dist,       // max distance to look for photons
									const int nphotons,			// number of photons to use
									bool verbose)		 
{
	static NearestPhotons np;
	irrad = rgb(0,0,0);

	np.dist2.clear();
	np.index.clear();

	np.dist2.resize(nphotons + 1);
	np.index.resize(nphotons + 1);

	np.pos[0] = pos[0]; np.pos[1] = pos[1]; np.pos[2] = pos[2];
	np.max = nphotons;
	np.found = 0;
	np.got_heap = 0;
	np.dist2[0] = max_dist*max_dist;

	// locate the nearest photons
	locate_photons( &np, 1 );

	// if less than certain number of photons return
	if (np.found < 3)
		return;

	// sum irradiance from all photons
	//float pdir[3];
	for (int i=1; i<=np.found; i++) {
		const Photon *p = np.index[i];
		/*
		// the photon_dir call and following if can be omitted (for speed)
		// if the scene does not have any thin surfaces
		photon_dir( pdir, p );
		if ( (pdir[0]*normal[0]+pdir[1]*normal[1]+pdir[2]*normal[2]) < 0.0f ) {*/

		irrad += p->power;			

		//}
	}
	
		
	//*irrad = np.index[2]->power * np.found;

	const float tmp= (1.0f / PI) / np.dist2[0];	// estimate of density

	irrad *= tmp;

}


/* locate_photons finds the nearest photons in the
* photon map given the parameters in np
*/
//******************************************
void PhotonMap::locate_photons(NearestPhotons *np, unsigned int index )
{
	Photon *p = &m_Photons[index];
	float dist1;

	if (index<m_nHalfStoredPhotons) {
		dist1 = np->pos[ p->plane ] - p->pos[ p->plane ];

		if (dist1 > 0.0) { // if dist1 is positive search right plane
			locate_photons( np, 2*index+1 );
			if ( dist1*dist1 < np->dist2[0] )
				locate_photons( np, 2*index );
		} 
		else { // dist1 is negative search left first
			locate_photons( np, 2*index );
			if ( dist1*dist1 < np->dist2[0] )
				locate_photons( np, 2*index+1 );
		}
	}

	// compute squared distance between current photon and np->pos

	dist1 = p->pos[0] - np->pos[0];
	float dist2 = dist1*dist1;
	dist1 = p->pos[1] - np->pos[1];
	dist2 += dist1*dist1;
	dist1 = p->pos[2] - np->pos[2];
	dist2 += dist1*dist1;

	if ( dist2 < np->dist2[0] ) {
		// we found a photon  [:)] Insert it in the candidate list

		if ( np->found < np->max ) {
			// heap is not full; use array
			np->found++;
			np->dist2[np->found] = dist2;
			np->index[np->found] = p;
		} else {
			int j,parent;

			if (np->got_heap==0) { // Do we need to build the heap?
				// Build heap
				float dst2;
				Photon *phot;
				int half_found = np->found>>1;
				for ( int k=half_found; k>=1; k--) {
					parent=k;
					phot = np->index[k];
					dst2 = np->dist2[k];
					while ( parent <= half_found ) {
						j = parent+parent;
						if (j<np->found && np->dist2[j]<np->dist2[j+1])
							j++;
						if (dst2>=np->dist2[j])
							break;
						np->dist2[parent] = np->dist2[j];
						np->index[parent] = np->index[j];
						parent=j;
					}
					np->dist2[parent] = dst2;
					np->index[parent] = phot;
				}
				np->got_heap = 1;
			}

			// insert new photon into max heap
			// delete largest element, insert new and reorder the heap

			parent=1;
			j = 2;
			while ( j <= np->found ) {
				if ( j < np->found && np->dist2[j] < np->dist2[j+1] )
					j++;
				if ( dist2 > np->dist2[j] )
					break;
				np->dist2[parent] = np->dist2[j];
				np->index[parent] = np->index[j];
				parent = j;
				j += j;
			}
			np->index[parent] = p;
			np->dist2[parent] = dist2;

			np->dist2[0] = np->dist2[1];
		}
	}
}