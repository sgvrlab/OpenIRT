#include "CUDAPhotonMapping.h"
#include <stopwatch.h>

#ifndef fminf
#define fminf(a,b) (((a) < (b)) ? (a) : (b))
#endif

#ifndef fmaxf
#define fmaxf(a,b) (((a) > (b)) ? (a) : (b))
#endif

#include "CUDA/CUDADataStructures.cuh"

extern "C" int tracePhotons(int size, void *outPhotons);
extern "C" int buildPhotonKDTree(int size, void *kdtree);
extern "C" void loadSceneCUDAPhotonMapping(CUDA::Scene *scene);
extern "C" void renderCUDAPhotonMapping(CUDA::Camera *camera, CUDA::Image *image, CUDA::Controller *controller, int frame, int seed);
extern "C" void unloadSceneCUDAPhotonMapping();
extern "C" void clearResultCUDAPhotonMapping(int &frame);

using namespace irt;

CUDAPhotonMapping::CUDAPhotonMapping(void)
{
	m_dstScene = 0;
	m_photons = 0;
}

CUDAPhotonMapping::~CUDAPhotonMapping(void)
{
	done();

	if(m_photons) delete[] m_photons;

	unloadSceneCUDAPhotonMapping();
}

void CUDAPhotonMapping::init(Scene *scene)
{
	Renderer::init(scene);

	if(!scene) return;

	done();

	m_intersectionStream = scene->getIntersectionStream();

	adaptScene(scene, m_dstScene);
	loadSceneCUDAPhotonMapping((CUDA::Scene*)m_dstScene);

	sceneChanged();
}

void CUDAPhotonMapping::done()
{
}

void CUDAPhotonMapping::resized(int width, int height)
{
	Renderer::resized(width, height);
}

void CUDAPhotonMapping::sceneChanged()
{
	extern int g_timerPhotonTracing;
	if(!g_timerPhotonTracing)
		g_timerPhotonTracing = StopWatch::create();
	StopWatch::get(g_timerPhotonTracing).start();

	int numMaxPhotons = m_scene->getNumMaxPhotons();
	if(m_photons)
		delete[] m_photons;
	m_photons = new Photon[numMaxPhotons];
	int numValidPhotons = tracePhotons(numMaxPhotons, m_photons);
	/*
	FILE *fp = fopen("photons", "w");
	for(int i=0;i<numValidPhotons;i++)
	{
		fprintf(fp, "%d %f %f %f\n", m_photons[i].axis, m_photons[i].pos.e[0], m_photons[i].pos.e[1], m_photons[i].pos.e[2]);
	}
	fclose(fp);
	*/

	StopWatch::get(g_timerPhotonTracing).stop();
	printf("Tracing photons : %f ms\n", StopWatch::get(g_timerPhotonTracing).getTime());

	// build kd-tree on CPU
	AABB bb;
	int sizeKDTree = m_scene->buildPhotonKDTree(numValidPhotons, &m_photons, bb);
	printf("numValidPhotons = %d, sizeKDTree = %d\n", numValidPhotons, sizeKDTree);
	if(sizeKDTree > 0) buildPhotonKDTree(sizeKDTree, m_photons);

	/*
	FILE *fp = fopen("photons2", "w");
	for(int i=0;i<sizeKDTree;i++)
	{
		fprintf(fp, "%d %f %f %f\n", m_photons[i].axis, m_photons[i].pos.e[0], m_photons[i].pos.e[1], m_photons[i].pos.e[2]);
	}
	fclose(fp);
	*/
}

void CUDAPhotonMapping::render(Camera *camera, Image *image, unsigned int seed)
{
	CUDA::Camera cameraCUDA;

	memset(image->data, 0, image->width*image->height*image->bpp);
	//adaptDataStructures(m_scene, camera, &sceneCUDA, &cameraCUDA);

	extern int g_frame;
	renderCUDAPhotonMapping((CUDA::Camera*)camera, (CUDA::Image*)&image->width, (CUDA::Controller *)&m_controller, g_frame++, seed);
}