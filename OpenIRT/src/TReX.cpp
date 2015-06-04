#include "CommonOptions.h"
#include "defines.h"
#include "TReX.h"
#include <stopwatch.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <process.h>
#include <deque>
#include <algorithm>

#ifndef fminf
#define fminf(a,b) (((a) < (b)) ? (a) : (b))
#endif

#ifndef fmaxf
#define fmaxf(a,b) (((a) > (b)) ? (a) : (b))
#endif

#include "CUDA/CUDADataStructures.cuh"
//#include "GBufferFilter.h"

//#define DISABLE_GPU
//#define PRINT_LOG

using namespace irt;

extern int g_statTryCount;
int g_numTimeSample = g_statTryCount*8;

int g_timerProgressivePhotonTracing = -1;
int g_frameStartPhoton;

extern "C" int tracePhotonsAPI();
extern "C" int traceSubPhotonsToBackBufferAPI(int pos, int numPhotonsPerEmitter, int frame3);
extern "C" int swapPhotonBufferAPI();
extern "C" int buildPhotonKDTreeAPI(int size, void *kdtree);
extern "C" void lightChangedAPI(CUDA::Scene *scene);
extern "C" void materialChangedAPI(CUDA::Scene *scene);
extern "C" void loadSceneAPI(CUDA::Scene *scene, CUDA::OctreeHeader *octreeHeader, int numVoxels, CUDA::Voxel *octree, int numVoxelsForOOC);
extern "C" void updateControllerAPI(CUDA::Controller *controller);
extern "C" void renderBeginAPI(CUDA::Camera *camera, CUDA::Image *image, CUDA::Controller *controller, int frame);
extern "C" void renderPartAPI(int frame, int frame2, float &time, CUDA::HitPoint *hitCache, CUDA::ExtraRayInfo *extRayCache, int numRays, int offset);
extern "C" void renderEndAPI(CUDA::Image *image, int frame);
extern "C" void resetAPI(int frame2);
extern "C" void initWithImageAPI(CUDA::Image *image);
extern "C" void loadOOCVoxelInfoAPI(int count, const CUDA::OOCVoxel *oocVoxels);
extern "C" void beginGatheringRequestCountAPI();
extern "C" void endGatheringRequestCountAPI(float *requestCountList);
extern "C" void updateControllerAPI(CUDA::Controller *controller);
extern "C" void unloadSceneAPI();

TReX::TReX(void)
{
	m_exit = false;

	m_dstScene = 0;
	m_photons = 0;

	m_rayCacheBlockSize = 256*128;

	m_oocVoxelMgr = 0;

	m_cameraUpdated = true;
	m_materialUpdated = false;
	m_lightUpdated = false;

	m_tileOrder = 0;

	m_curTracedPhotonsInBackground = 0;
	m_numPhotonsInBackground = 0;

	m_hRenderThread = 0;

	m_rayCache = 0;
	m_hitCache = 0;
	m_extRayCache = 0;

	m_mode = MODE_DEFAULT;
}

TReX::~TReX(void)
{
	done();

	//clearScene();

	if(m_photons) delete[] m_photons;

	if(m_oocVoxelMgr) delete m_oocVoxelMgr;

	if(m_rayCache) delete[] m_rayCache;
	if(m_hitCache) delete[] m_hitCache;
	if(m_extRayCache) delete[] m_extRayCache;
	if(m_tileOrder) delete[] m_tileOrder;

	unloadSceneAPI();
}

void TReX::init(Scene *scene)
{
	Renderer::init(scene);

	if(!scene) return;

	done();

	m_intersectionStream = scene->getIntersectionStream();
	m_renderingThumbNailStream = scene->getIntersectionStream();
	m_controller.numGatheringRays = 1;
	m_controller.numShadowRays = 1;
	m_controller.pathLength = 1;
	m_controller.useShadowRays = true;
	m_controller.timeLimit = 30.0f;
	m_controller.sizeMBForOOCVoxel = 256;

	sceneChanged();

	adaptScene(scene, m_dstScene);

	loadSceneAPI((CUDA::Scene*)m_dstScene, (CUDA::OctreeHeader*)&(m_octree.getHeader()), m_octree.getNumVoxels(), (CUDA::Voxel*)m_octree.getOctreePtr(), m_controller.sizeMBForOOCVoxel*1024*1024/sizeof(Voxel));

	if(m_controller.numGatheringRays > 0)
	{
		extern int g_timerPhotonTracing;
		if(!g_timerPhotonTracing)
			g_timerPhotonTracing = StopWatch::create();
		StopWatch::get(g_timerPhotonTracing).start();
		tracePhotonsAPI();
		StopWatch::get(g_timerPhotonTracing).stop();
	}

	restart();

	// setup thread
	int threadBlockSize = m_rayCacheBlockSize;
	m_controller.threadBlockSize = threadBlockSize;
	m_threadData.renderer = this;
}

void TReX::done()
{
	m_exit = true;

	if(m_hGPULaunchingThreadList.size())
	{
		for(size_t i=0;i<m_hGPULaunchingThreadList.size();i++)
		{
			if(m_hGPULaunchingThreadList[i])
			{
				WaitForSingleObject(m_hGPULaunchingThreadList[i], INFINITE);
				CloseHandle(m_hGPULaunchingThreadList[i]);
			}
		}
		m_hGPULaunchingThreadList.clear();
	}

	if(m_hRenderThread)
	{
		WaitForSingleObject(m_hRenderThread, INFINITE);
		CloseHandle(m_hRenderThread);
		m_hRenderThread = NULL;
	}

	m_exit = false;
}

void TReX::restart(Camera *camera, Image *image)
{
	extern int g_frame2;
	resetAPI(g_frame2);
}

void TReX::resized(int width, int height)
{
	m_controller.useZCurveOrdering = true;
	Renderer::resized(width, height);

	if(m_tileOrder) delete[] m_tileOrder;
	m_tileOrder = new int[width*height/(TILE_SIZE*TILE_SIZE)];
	for(int i=0, j=0;i<width*height;i++)
	{
		if(m_rayOrder[i].x % TILE_SIZE == 0 && m_rayOrder[i].y % TILE_SIZE == 0)
		{
			int offset = (m_rayOrder[i].x / TILE_SIZE) + (m_rayOrder[i].y / TILE_SIZE)*(width / TILE_SIZE);
			m_tileOrder[offset] = j++;
		}
	}

	// setup tile list
	m_tileList.clear();

	int tileX = (width + TILE_SIZE - 1) / TILE_SIZE;
	int tileY = (height + TILE_SIZE - 1) / TILE_SIZE;

	int numTiles = tileX*tileY;
	int startY = (tileY - 1) * TILE_SIZE;
	for(int y=0;y<tileY;y++)
	{
		int startX = 0;
		for(int x=0;x<tileX;x++)
		{
			int pos = (y*tileX + x)*TILE_SIZE;
			m_tileList.push_back(TileElem(TILE_SIZE, startX, startY));
			startX += TILE_SIZE;
		}
		startY -= TILE_SIZE;
	}

	if(m_rayCache) delete[] m_rayCache;
	if(m_hitCache) delete[] m_hitCache;
	if(m_extRayCache) delete[] m_extRayCache;
	m_rayCache = new Ray[width*height];
	m_hitCache = new CUDAHit[width*height];
	m_extRayCache = new ExtraRayInfo[width*height];

	m_sampler.reset(m_width, m_height, 16);
	m_sampler.resample(0);
}

void TReX::sceneChanged()
{
	/*
	extern int g_timerPhotonTracing;
	if(!g_timerPhotonTracing)
		g_timerPhotonTracing = StopWatch::create();
	StopWatch::get(g_timerPhotonTracing).start();

	int numMaxPhotons = m_scene->getNumMaxPhotons();
	if(m_photons)
		delete[] m_photons;
	m_photons = new Photon[numMaxPhotons];
	int numValidPhotons = tracePhotonsAPI(numMaxPhotons, m_photons);

	StopWatch::get(g_timerPhotonTracing).stop();
	printf("Tracing photons : %f ms\n", StopWatch::get(g_timerPhotonTracing).getTime());

	// build kd-tree on CPU
	AABB bb;
	int sizeKDTree = m_scene->buildPhotonKDTree(numValidPhotons, &m_photons, bb);
	printf("numValidPhotons = %d, sizeKDTree = %d\n", numValidPhotons, sizeKDTree);
	buildPhotonKDTreeAPI(sizeKDTree, m_photons);
	*/
	
	// load precomputed octree
	int numMaxPhotons = m_scene->getNumMaxPhotons();

	char fileNameVoxel[MAX_PATH];
	sprintf_s(fileNameVoxel, MAX_PATH, "%s_voxel.ooc", m_scene->getASVOFileBase());
	m_octree.load(fileNameVoxel, false);
	//Voxel::initD(AABB(m_octree.getHeader().min, m_octree.getHeader().max));

#	if defined(TRACE_PHOTONS) && defined(USE_OOCVOXEL) && !defined(USE_FULL_DETAIL_GI)	// trace photons with full voxel
	static int timer;
	if(!timer)
		timer = StopWatch::create();
	StopWatch::get(timer).start();
	PhotonOctree photonTracer;
	photonTracer.tracePhotonsWithFullDetailedVoxels(m_scene->getASVOFileBase());
	StopWatch::get(timer).stop();
	printf("Tracing photons to full voxels: %f ms\n", StopWatch::get(timer).getTime());
#	endif

	if(!m_oocVoxelMgr)
	{
		m_oocVoxelMgr = new OOCVoxelManager(&m_octree, m_scene->getASVOFileBase(), m_octree.getNumVoxels(), Vector3(m_octree.getLeafVoxelSize()),  m_controller.sizeMBForOOCVoxel);

		int count = m_oocVoxelMgr->getNumOOCVoxels();
		CUDA::OOCVoxel *oocVoxels = new CUDA::OOCVoxel[count];
		for(int i=0;i<count;i++)
		{
			const OOCVoxelManager::OOCVoxel &temp = m_oocVoxelMgr->getOOCVoxel(i);
			oocVoxels[i].rootChildIndex = temp.rootChildIndex;
			oocVoxels[i].startDepth = temp.startDepth;
			oocVoxels[i].offset = temp.offset;
			oocVoxels[i].numVoxels = temp.numVoxels;
			oocVoxels[i].rootBB.min.e[0] = temp.rootBB.min.e[0];
			oocVoxels[i].rootBB.min.e[1] = temp.rootBB.min.e[1];
			oocVoxels[i].rootBB.min.e[2] = temp.rootBB.min.e[2];
			oocVoxels[i].rootBB.max.e[0] = temp.rootBB.max.e[0];
			oocVoxels[i].rootBB.max.e[1] = temp.rootBB.max.e[1];
			oocVoxels[i].rootBB.max.e[2] = temp.rootBB.max.e[2];
		}
		loadOOCVoxelInfoAPI(count, (CUDA::OOCVoxel*)oocVoxels);
		delete[] oocVoxels;
	}
}

void TReX::materialChanged()
{
	m_materialUpdated = true;
}

void TReX::lightChanged(bool soft)
{
	m_lightUpdated = true;
}

#include "HCCMesh.h"
#include "HCCMesh2.h"
float g_CPUTime, g_GPUTime;

int TReX::previewRendering(Camera *camera, Image *image)
{
	static int multiple = 2;

	static const int nRaysPerSide = TILE_SIZE/2;
	static const int nRealRaysPerSide = TILE_SIZE;
	static const int nRays = nRaysPerSide*nRaysPerSide;
	static const int nRealRays = nRealRaysPerSide*nRealRaysPerSide;
	int numSubPixelsX = image->width / multiple;
	int numSubPixelsY = image->height / multiple;
	int numPacketsX = numSubPixelsX / nRealRaysPerSide;
	int numPacketsY = numSubPixelsY / nRealRaysPerSide;
	int numPackets = numPacketsX * numPacketsY;

	static Image *bakImage = NULL;
	if(!bakImage)
	{
		if(bakImage) delete bakImage;
		bakImage = new Image(numSubPixelsX, numSubPixelsY);
	}

	Camera curCame = *(camera);
	Vector3 corner2 = curCame.getCorner2();

	float deltax = 1.0f / (float)numSubPixelsX;
	float deltay = 1.0f / (float)numSubPixelsY;

	//omp_set_num_threads(1);
#	pragma omp parallel for schedule(dynamic)
	for (int curPacket = 0; curPacket < numPackets; curPacket++) 
	{		
		RayPacket<nRays, true, true, true> rayPacket;
		RayPacket<nRays, false, true, true> *rayPacketNonCoherent = (RayPacket<nRays, false, true, true> *)((void *)&rayPacket);
		__declspec(align(16)) RGB4f colors[nRealRays];
		//__declspec(align(16)) RGB4f colors2[nRealRays];

		unsigned int start_x = (curPacket % numPacketsX)*nRealRaysPerSide;
		unsigned int start_y = (curPacket / numPacketsX)*nRealRaysPerSide;

		float ypos = start_y*deltay,
			xpos = start_x*deltax;

		__declspec(align(16)) float jitter[nRealRays][2] = {0, };

		// initialize beam with real eye-rays:
		rayPacket.setupForPrimaryRays(curCame.getEye(), corner2, curCame.getScaledRight(), curCame.getScaledUp(), nRaysPerSide, nRaysPerSide, xpos, ypos, deltax, deltay, jitter);

		if(rayPacket.hasMatchingDirections())
			m_scene->trace(rayPacket, colors, 0, m_renderingThumbNailStream);
		else
			m_scene->trace(*rayPacketNonCoherent, colors, 0, m_renderingThumbNailStream);

		// fill ray & hit information from packet
		for(int i=0;i<nRealRays;i++)
		{
			unsigned int sX = i % nRealRaysPerSide;
			unsigned int sY = i / nRealRaysPerSide;
			unsigned int offset = (sY/2)*nRaysPerSide+(sX/2);
			unsigned int offset2 = (sY%2)*2+(sX%2);

			bakImage->setPixel(start_x + sX, start_y + sY, colors[offset*4 + offset2]);
		}
	}
	int sampleSize = multiple;

	// filter
#	pragma omp parallel for schedule(guided, 2)
	for(int y=0;y<image->height;y++)
	{
		for(int x=0;x<image->width;x++)
		{
			int x0 = x/sampleSize;
			int y0 = y/sampleSize;
			int x1 = x0 + 1;
			int y1 = y0 + 1;

			x1 = min(bakImage->width-1, x1);
			y1 = min(bakImage->height-1, y1);
			RGBf c00, c01, c10, c11;
			bakImage->getPixel(x0, y0, c00);
			bakImage->getPixel(x1, y0, c10);
			bakImage->getPixel(x0, y1, c01);
			bakImage->getPixel(x1, y1, c11);

			float w00 = 1.0f;
			float w01 = 1.0f;
			float w10 = 1.0f;
			float w11 = 1.0f;

			// Bilateral weights:
			// Bilinear interpolation (filter distance)
			float subPixelPosX = (x % sampleSize)/((float)(sampleSize));
			float subPixelPosY = (y % sampleSize)/((float)(sampleSize));
			w00 = (1.0f - subPixelPosX) * (1.0f - subPixelPosY);
			w01 = (1.0f - subPixelPosX) * subPixelPosY;
			w10 = subPixelPosX * (1.0f - subPixelPosY);
			w11 = subPixelPosX * subPixelPosY;

			float norm = 1.0f / (w00 + w01 + w10 + w11);

			RGBf col = (c00 * w00 + c01 * w01 + c10 * w10 + c11 * w11)*norm;
			image->setPixel(x, y, col);
		}
	}

	initWithImageAPI((CUDA::Image*)&image->width);

	return 0;
}

int TReX::resetImages(Image *image)
{
	memset(image->data, 0, image->width*image->height*image->bpp);

	initWithImageAPI((CUDA::Image*)&image->width);

	return 0;
}

int TReX::tileOrdering(Image *image)
{
	int tileX = (image->width + TILE_SIZE - 1) / TILE_SIZE;
	int tileY = (image->height + TILE_SIZE - 1) / TILE_SIZE;

	switch(m_controller.tileOrderingType)
	{
	case Controller::RANDOM : 
		{
			static unsigned int seed = 0;
			for(int i=0;i<(int)m_tileList.size();i++)
			{
				//m_tileList[i].custom = rand() / (float)RAND_MAX;
				m_tileList[i].custom = rnd(seed);
			}
			std::sort(m_tileList.begin(), m_tileList.end(), TileElem::custumized);
		}
		break;
	case Controller::ROW_BY_ROW : std::sort(m_tileList.begin(), m_tileList.end(), TileElem::rowByRow); break;
	case Controller::Z_CURVE : 
		{
			for(int i=0;i<(int)m_tileList.size();i++)
			{
				m_tileList[i].custom = (float)m_tileOrder[m_tileList[i].tileStartX/TILE_SIZE + (m_tileList[i].tileStartY/TILE_SIZE)*tileX];
			}
			std::sort(m_tileList.begin(), m_tileList.end(), TileElem::custumized);
		}
		break;
	case Controller::HIGHEST_SALIENT_TILE_FIRST : 
		{
			static HSaliency saliency;
			if(saliency.m_width != tileX || saliency.m_height != tileY)
				saliency.reset(tileX, tileY);
			saliency.clear();
			saliency.set(image);
			saliency.calculate();
			for(int i=0;i<(int)m_tileList.size();i++)
			{
				m_tileList[i].custom = saliency.getSaliency(m_tileList[i].tileStartX/TILE_SIZE, m_tileList[i].tileStartY/TILE_SIZE);	// /4.0f
			}
			std::sort(m_tileList.begin(), m_tileList.end(), TileElem::custumized);

		}
		break;
	case Controller::CURSOR_GUIDED : std::sort(m_tileList.begin(), m_tileList.end(), TileElem::cursorGuided); break;
	}
	return 0;
}

int TReX::CRayTracing(Camera *camera, Image *image, int numTiles, int offset, int frame2)
{
	static const int nRaysPerSide = TILE_SIZE/2;
	static const int nRealRaysPerSide = TILE_SIZE;
	static const int nRays = nRaysPerSide*nRaysPerSide;
	static const int nRealRays = nRealRaysPerSide*nRealRaysPerSide;
	float deltax = 1.0f / (float)image->width;
	float deltay = 1.0f / (float)image->height;

	int numRays = numTiles * nRealRays;
	int rayOffset = offset * nRealRays;

	memset(&m_rayCache[rayOffset], 0, sizeof(Ray)*numRays);
	memset(&m_hitCache[rayOffset], 0, sizeof(CUDAHit)*numRays);
	memset(&m_extRayCache[rayOffset], 0, sizeof(ExtraRayInfo)*numRays);

	//
	// for all ray packets in the system:
	// 
	Vector3 corner2 = camera->getCorner2();

	int startTile = offset;
	int endTile = offset + numTiles;
	//const TileElem *tiles = &m_tileList[0];

	static unsigned int numIter = 0;
	numIter++;
	unsigned int tileSeed = tea<16>(0, numIter);

	static int tCPU = StopWatch::create();
	StopWatch::get(tCPU).reset();
	StopWatch::get(tCPU).start();
#	pragma omp parallel for shared(numTiles) schedule(dynamic)
	for (int curTile=startTile;curTile<endTile;curTile++) 
	{
		extern bool g_cameraMoving;
		extern int g_frame2;
		if(!g_cameraMoving && frame2 == g_frame2)
		{

#		ifdef USE_PACKET
		RayPacket<nRays, true, true, true> rayPacket;
		RayPacket<nRays, false, true, true> *rayPacketNonCoherent = (RayPacket<nRays, false, true, true> *)((void *)&rayPacket);
		__declspec(align(16)) RGB4f colors[nRealRays];
#		else
		Ray rays[nRealRays];
		HitPointInfo hits[nRealRays];
#		endif

		unsigned int start_x = m_tileList[curTile].tileStartX;
		unsigned int start_y = m_tileList[curTile].tileStartY;

		float ypos = start_y*deltay,
			xpos = start_x*deltax;

		extern int g_frame;
		__declspec(align(16)) float jitter[nRealRays][2];
		//unsigned int seed = tea<16>(g_frame/m_sampler.getNumSubSample(), curTile);
		unsigned int seed = tea<16>(tileSeed, curTile);
		int sampleStartX = (((int)(rnd(seed)*image->width)) / TILE_SIZE) * TILE_SIZE;
		int sampleStartY = (((int)(rnd(seed)*image->height)) / TILE_SIZE) * TILE_SIZE;
		for(int i=0;i<nRealRays;i++)
		{
			const Vector2 &sample = m_sampler.getSample(sampleStartX + (i % nRealRaysPerSide), sampleStartY + (i / nRealRaysPerSide), g_frame);
			jitter[i][0] = sample.e[0]*1.0f;
			jitter[i][1] = sample.e[1]*1.0f;
		}

		// initialize beam with real eye-rays:
		rayPacket.setupForPrimaryRays(camera->getEye(), corner2, camera->getScaledRight(), camera->getScaledUp(), nRaysPerSide, nRaysPerSide, xpos, ypos, deltax, deltay, jitter);

		if(rayPacket.hasMatchingDirections())
			m_scene->getIntersection(rayPacket, m_intersectionStream);
		else
			m_scene->getIntersection(*rayPacketNonCoherent, m_intersectionStream);

#		ifdef USE_CPU_GLOSSY
		for(int i=0;i<nRealRays;i++)
		{
			unsigned int sX = i % nRealRaysPerSide;
			unsigned int sY = i / nRealRaysPerSide;
			unsigned int offset = (sY/2)*nRaysPerSide+(sX/2);
			unsigned int offset2 = (sY%2)*2+(sX%2);

			unsigned int offset3 = start_x + sX + (start_y + sY) * p->image->width;
			//unsigned int raySeed = tea<16>(offset3, p->frame);
			unsigned int raySeed = tea<16>(offset3, g_frame);

			bool hasHit = (rayPacket.rayHasHit[offset] & (1u << offset2)) != 0;
			if(!hasHit) continue;

			unsigned int model = r->m_scene->getModelListMap()[rayPacket.hitpoints[offset].modelPtr[offset2]];
			unsigned int m = rayPacket.hitpoints[offset].m[offset2];
			Material &mat = r->m_scene->getModelList()[model]->getMaterial(m);

			ExtraRayInfo *extRayInfo = ((ExtraRayInfo *)tiles[curTile].extRayCache) + i;
			extRayInfo->setIsSpecular(!mat.isDiffuse(raySeed));
				
			//if(mat.mat_Ns > 256.0f && extRayInfo->isSpecular)
			if(mat.isPerfectSpecular() && extRayInfo->isSpecular())
			{
				extRayInfo->setWasBounced(true);

				Vector3 d, n, x, nd;

				// read ray & hit info
				d = rayPacket.rays[offset].getDirection(offset2);

				n.e[0] = rayPacket.hitpoints[offset].n[0].e[offset2];
				n.e[1] = rayPacket.hitpoints[offset].n[1].e[offset2];
				n.e[2] = rayPacket.hitpoints[offset].n[2].e[offset2];
				x.e[0] = rayPacket.hitpoints[offset].x[0].e[offset2];
				x.e[1] = rayPacket.hitpoints[offset].x[1].e[offset2];
				x.e[2] = rayPacket.hitpoints[offset].x[2].e[offset2];

				nd = mat.sampleDirection(n, d, raySeed);

				x = x + nd * 1.f;

				// overwrite ray info
				rayPacket.rays[offset].setOrigin(x, offset2);
				rayPacket.rays[offset].setDirection(nd, offset2);

				Ray ray;
				HitPointInfo hit;
				ray.set(x, nd);
				hit.t = FLT_MAX;

				if(r->m_scene->getIntersection(ray, hit, 0.0f, r->m_intersectionStream))
				{
					rayPacket.hitpoints[offset].t.e[offset2] = hit.t;
					rayPacket.hitpoints[offset].modelPtr[offset2] = hit.modelPtr;
					rayPacket.hitpoints[offset].m[offset2] = hit.m;
					rayPacket.hitpoints[offset].n[0].e[offset2] = hit.n.e[0];
					rayPacket.hitpoints[offset].n[1].e[offset2] = hit.n.e[1];
					rayPacket.hitpoints[offset].n[2].e[offset2] = hit.n.e[2];
					rayPacket.hitpoints[offset].u.e[offset2] = hit.uv.e[0];
					rayPacket.hitpoints[offset].v.e[offset2] = hit.uv.e[1];
					rayPacket.hitpoints[offset].x[0].e[offset2] = hit.x.e[0];
					rayPacket.hitpoints[offset].x[1].e[offset2] = hit.x.e[1];
					rayPacket.hitpoints[offset].x[2].e[offset2] = hit.x.e[2];
				}
				else
				{
					rayPacket.rayHasHit[offset] &= ~(1u << offset2);
				}
			}
		}

#		endif

		// fill ray & hit information from packet
		for(int i=0;i<nRealRays;i++)
		{
			unsigned int sX = i % nRealRaysPerSide;
			unsigned int sY = i / nRealRaysPerSide;
			unsigned int offset = (sY/2)*nRaysPerSide+(sX/2);
			unsigned int offset2 = (sY%2)*2+(sX%2);
			ExtraRayInfo *extRayInfo = &m_extRayCache[curTile*nRealRays+i];

			extRayInfo->pixelX = (float)(start_x + sX)+jitter[i][0];
			extRayInfo->pixelY = (float)(start_y + sY)+jitter[i][1];//g_scrHeight-1-(start_y + sY);

#			ifdef USE_PACKET
			extRayInfo->setHasHit((rayPacket.rayHasHit[offset] & (1u << offset2)) != 0);
#			else
			extRayInfo->setHasHit(hits[i].t < FLT_MAX);
#			endif

			unsigned int offset3 = start_x + sX + (start_y + sY) * image->width;
			//unsigned int raySeed = tea<16>(offset3, g_frame);
			unsigned int raySeed = tea<16>(offset3, tileSeed);
			extRayInfo->seed = raySeed;
			//extern int g_frame2;
			//extRayInfo->frame2 = g_frame2;

			CUDAHit *cudaHit = &m_hitCache[curTile*nRealRays+i];
			cudaHit->t = FLT_MAX;

			if(extRayInfo->hasHit())
			{
#				ifdef USE_PACKET
				cudaHit->t = rayPacket.hitpoints[offset].t.e[offset2];
				cudaHit->model = m_scene->getModelListMap()[rayPacket.hitpoints[offset].modelPtr[offset2]];
				cudaHit->material = rayPacket.hitpoints[offset].m[offset2];
				cudaHit->n.e[0] = rayPacket.hitpoints[offset].n[0].e[offset2];
				cudaHit->n.e[1] = rayPacket.hitpoints[offset].n[1].e[offset2];
				cudaHit->n.e[2] = rayPacket.hitpoints[offset].n[2].e[offset2];
				cudaHit->uv.e[0] = rayPacket.hitpoints[offset].u.e[offset2];
				cudaHit->uv.e[1] = rayPacket.hitpoints[offset].v.e[offset2];
				cudaHit->x.e[0] = rayPacket.hitpoints[offset].x[0].e[offset2];
				cudaHit->x.e[1] = rayPacket.hitpoints[offset].x[1].e[offset2];
				cudaHit->x.e[2] = rayPacket.hitpoints[offset].x[2].e[offset2];
#				else
				cudaHit->t = hits[i].t;
				cudaHit->model = m_scene->getModelListMap()[hits[i].modelPtr];
				cudaHit->material = hits[i].m;
				cudaHit->n = *((CUDA::Vector3*)&(hits[i].n));
				cudaHit->uv = *((CUDA::Vector2*)&(hits[i].uv));
				cudaHit->x = *((CUDA::Vector3*)&(hits[i].x));

#				endif

			}
			else
			{
				cudaHit->t = FLT_MAX;
				Vector3 ori(rayPacket.rays[offset].origin[0][offset2], rayPacket.rays[offset].origin[1][offset2], rayPacket.rays[offset].origin[2][offset2]);
				Vector3 dir(rayPacket.rays[offset].direction[0][offset2], rayPacket.rays[offset].direction[1][offset2], rayPacket.rays[offset].direction[2][offset2]);
				cudaHit->x = ori + dir * 1000.0f;
			}
		}
		}
	}

	StopWatch::get(tCPU).stop();

	return 0;
}

int TReX::launchGRayTracing(int numTiles, int offset)
{
	int rayOffset = offset * TILE_SIZE * TILE_SIZE;
	Ray *rayCache = &m_rayCache[rayOffset];
	CUDAHit *hitCache = &m_hitCache[rayOffset];
	ExtraRayInfo *extRayCache = &m_extRayCache[rayOffset];

	float tGPU;
	renderPartAPI(0, 0, tGPU, (CUDA::HitPoint*)hitCache, (CUDA::ExtraRayInfo*)extRayCache, numTiles * TILE_SIZE * TILE_SIZE, rayOffset);
	return 0;
}

typedef struct
{
	TReX *renderer;
	int numTiles;
	int offset;
	HANDLE hEvent1;
	HANDLE hEvent2;
	Camera *camera;
	Image *image;
	int frame2;
	int block;
} GPULaunchingThreadArg;

unsigned __stdcall TReX::GPULaunchingThread(void* arg)
{
	GPULaunchingThreadArg *p = (GPULaunchingThreadArg*)arg;
	TReX *r = p->renderer;

	while(!r->m_exit)
	{
		r->m_oocVoxelMgr->update();

		if(p->block == 0)
		{
			if(r->m_controllerUpdated)
			{
				updateControllerAPI((CUDA::Controller *)&r->m_controller);
				r->m_controllerUpdated = false;
			}

			if(r->m_cameraUpdated)
			{
				renderBeginAPI((CUDA::Camera*)p->camera, (CUDA::Image*)&p->image->width, (CUDA::Controller *)&r->m_controller, 0);
				r->m_cameraUpdated = false;
			}

			if(r->m_materialUpdated)
			{
				r->applyChangedMaterial();
				r->m_materialUpdated = false;
			}

			if(r->m_lightUpdated)
			{
				r->applyChangedLight();
				r->m_lightUpdated = false;
				if(g_timerProgressivePhotonTracing == -1)
					g_timerProgressivePhotonTracing = StopWatch::create();
				StopWatch::get(g_timerProgressivePhotonTracing).start();
				extern int g_frame3;
				g_frameStartPhoton = g_frame3;
			}
		}

		if(r->m_curTracedPhotonsInBackground < r->m_numPhotonsInBackground)
		{
			traceSubPhotonsToBackBufferAPI(r->m_curTracedPhotonsInBackground, r->m_numSubPhotonsInBackground, 0);
			r->m_curTracedPhotonsInBackground += r->m_numSubPhotonsInBackground;
			if(r->m_curTracedPhotonsInBackground >= r->m_numPhotonsInBackground)
			{
				swapPhotonBufferAPI();
				StopWatch::get(g_timerProgressivePhotonTracing).stop();
				extern int g_frame, g_frame3;
				r->m_cameraUpdated = true;
				StopWatch::destroy(g_timerProgressivePhotonTracing);
				g_timerProgressivePhotonTracing = -1;
			}
		}

		WaitForSingleObject(p->hEvent2, INFINITE);

		float tGPU = 0.0f;
		static float tGPUSum = 0.0f;
		extern int g_frame2;
		int numRays = p->numTiles * TILE_SIZE * TILE_SIZE;

		if(p->frame2 == g_frame2)
			renderPartAPI(0, p->frame2, tGPU, (CUDA::HitPoint*)r->m_hitCache, (CUDA::ExtraRayInfo*)r->m_extRayCache, numRays, p->offset * TILE_SIZE * TILE_SIZE);
		tGPUSum += tGPU;
		r->m_frameTimeGPUCur = tGPU;

		SetEvent(p->hEvent1);
	}
	_endthreadex(0);
	return 0;
}

unsigned __stdcall TReX::renderThread(void* arg)
{
	SharedThreadData *p = (SharedThreadData *)arg;
	TReX *r = p->renderer;

	Image *image = p->image;
	Camera *camera = p->camera;

	extern int g_frame;
	extern int g_frame2;
	extern int g_frame3;

	int timerFrame = StopWatch::create();
	int timerAvgFrame = StopWatch::create();
	int timerFrame3 = StopWatch::create();

	const int numGPUThreads = 1;
	GPULaunchingThreadArg gArg[numGPUThreads];
	for(int i=0;i<numGPUThreads;i++)
	{
		gArg[i].renderer = r;
		gArg[i].hEvent1 = CreateEvent(NULL, FALSE, TRUE, NULL);
		gArg[i].hEvent2 = CreateEvent(NULL, FALSE, FALSE, NULL);
		gArg[i].camera = camera;
		gArg[i].image = image;
		HANDLE handle = (HANDLE)_beginthreadex(NULL, 0, GPULaunchingThread, &gArg[i], 0, NULL);
		r->m_hGPULaunchingThreadList.push_back(handle);
	}

	//r->previewRendering(camera, image);

	// main rendering loop
	while(!r->m_exit)
	{
		p->frame2 = g_frame2;
		StopWatch::get(timerFrame).start();
		StopWatch::get(timerAvgFrame).start();
		r->tileOrdering(image);

		int tileX = (image->width + TILE_SIZE - 1) / TILE_SIZE;
		int tileY = (image->height + TILE_SIZE - 1) / TILE_SIZE;

		int numTilesPerBlock = r->m_rayCacheBlockSize / (TILE_SIZE*TILE_SIZE);
		int numBlocks = tileX*tileY / numTilesPerBlock;

		for(int block=0;block<numBlocks;block++)
		{
			extern bool g_cameraMoving;
			if(g_cameraMoving || p->frame2 != g_frame2) break;

			r->CRayTracing(camera, image, numTilesPerBlock, block*numTilesPerBlock, p->frame2);

			int idx = block % numGPUThreads;
			WaitForSingleObject(gArg[idx].hEvent1, INFINITE);

			gArg[idx].frame2 = p->frame2;
			gArg[idx].block = block;
			gArg[idx].numTiles = numTilesPerBlock;
			gArg[idx].offset = block*numTilesPerBlock;
			SetEvent(gArg[idx].hEvent2);

			if(r->m_exit) return 0;

			g_frame3++;

		}

		StopWatch::get(timerFrame).stop();
		StopWatch::get(timerAvgFrame).stop();

		extern float g_frameTime;
		extern float g_frameSumTime;
		extern int g_frameNumSum;

		g_frameTime = StopWatch::get(timerFrame).getTime();
		g_frameSumTime += g_frameTime;
		g_frameNumSum++;
		StopWatch::get(timerFrame).reset();

		static int s_bFrame = g_frame;
		if(s_bFrame > g_frame)
			StopWatch::get(timerAvgFrame).reset();
		s_bFrame = g_frame;

		g_frame3 = g_frame*numBlocks;
		g_frame++;
		extern int g_frame4;
		g_frame4++;
	}
	_endthreadex(0);
	return 0;
}

void TReX::flushImage(Image *image)
{
	extern int g_frame;

#	ifdef USE_PREVIEW
	extern bool g_cameraMoving;
	if(g_cameraMoving)
	{
		return;
	}
#	endif

	renderEndAPI((CUDA::Image*)&image->width, g_frame);

	int width = image->width;
	int height = image->height;
	int bpp = image->bpp;
	unsigned char *data = image->data;

	for(int i=0;i<height/2;i++)
	{
		for(int j=0;j<width;j++)
		{
			int offset = (j+i*width)*bpp;
			int offset2 = (j+(height - 1 - i)*width)*bpp;
			unsigned char temp;
			temp = data[offset];
			data[offset] = data[offset2];
			data[offset2] = temp;
			temp = data[offset+1];
			data[offset+1] = data[offset2+1];
			data[offset2+1] = temp;
			temp = data[offset+2];
			data[offset+2] = data[offset2+2];
			data[offset2+2] = temp;
		}
	}

}

void TReX::render(Camera *camera, Image *image, unsigned int seed)
{
	DWORD tic = GetTickCount();
	CUDA::Camera cameraCUDA;

	extern int g_frame;
	int frame = g_frame;
	g_frame = 0;

	extern int g_numForDebug;

	AABB sceneBB = m_scene->getSceneBB();

	m_threadData.camera = camera;
	m_threadData.image = image;
	renderBeginAPI((CUDA::Camera*)camera, (CUDA::Image*)&image->width, (CUDA::Controller *)&m_controller, frame);

	if(frame <= 1)
	{
		extern int g_frame2;
		extern int g_frame4;
		g_frame2++;
		g_frame4 = 1;
		restart(camera, image);
		if(m_oocVoxelMgr)
		{
#			if VOXEL_PRIORITY_POLICY == 1
			m_oocVoxelMgr->moveCamera(*camera);
#			endif
#			if VOXEL_PRIORITY_POLICY == 2
			beginGatheringRequestCountAPI();
#			endif
		}
		m_cameraUpdated = true;
	}
	else
	{
		flushImage(image);
	}

	if(!m_hRenderThread)
		m_hRenderThread = (HANDLE)_beginthreadex(NULL, 0, renderThread, &m_threadData, 0, NULL);

	if(g_frame == 0) g_frame = frame+1;
}

void TReX::applyChangedMaterial()
{
	CUDA::Scene *sceneCUDA = (CUDA::Scene *)m_dstScene;
	Scene *sceneCPU = m_scene;

	for(int i=0;i<sceneCUDA->numModels;i++)
	{
		Model *modelPtr = sceneCPU->getModelList()[i];
		Model &modelCPU = *(modelPtr);
		CUDA::Model &modelCUDA = sceneCUDA->models[i];

		for(int j=0;j<modelCUDA.numMats;j++)
		{
			Material &matCPU = modelCPU.getMaterial(j);
			memcpy(&modelCUDA.mats[j], &matCPU, sizeof(Material) - sizeof(BitmapTexture *)*3);
		}
	}

	materialChangedAPI(sceneCUDA);
}

void TReX::applyChangedLight()
{
	CUDA::Scene *sceneCUDA = (CUDA::Scene *)m_dstScene;
	Scene *sceneCPU = m_scene;

	sceneCUDA->numEmitters = 0;

	int numEmitters = sceneCPU->getNumEmitters();
	int maxPhotons = 0;
	for(int i=0;i<numEmitters;i++)
	{
		if(sceneCPU->getEmitter(i).type != Emitter::ENVIRONMENT_LIGHT)
			sceneCUDA->numEmitters++;

		maxPhotons = max(maxPhotons, sceneCPU->getEmitter(i).numScatteringPhotons);
	}

	if(sceneCUDA->emitters) delete[] sceneCUDA->emitters;
	sceneCUDA->emitters = new CUDA::Emitter[sceneCUDA->numEmitters];

	for(int i=0, j=0;i<numEmitters;i++)
	{
		Emitter &emitter = sceneCPU->getEmitter(i);
		if(emitter.type == Emitter::ENVIRONMENT_LIGHT)
		{
			if(emitter.environmentTexName[0] == 0)
			{
				memcpy(&sceneCUDA->envColor, &emitter.color_Kd, sizeof(CUDA::Vector3));
			}
		}
		else
		{
			memcpy(&sceneCUDA->emitters[j++], &emitter, sizeof(CUDA::Emitter));
		}
	}

	lightChangedAPI(sceneCUDA);
	m_numPhotonsInBackground = maxPhotons;
	//m_numSubPhotonsInBackground = maxPhotons / 100;
	//m_numSubPhotonsInBackground = numEmitters > 0 ? 4096*100/(g_numTimeSample/STAT_TRY_COUNT) / numEmitters : 0;
	m_numSubPhotonsInBackground = numEmitters > 0 ? 4096*100/(8) / numEmitters : 0;
	m_curTracedPhotonsInBackground = 0;
}