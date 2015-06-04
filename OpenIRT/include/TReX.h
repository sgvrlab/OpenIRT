/********************************************************************
	created:	2011/09/28
	file path:	d:\Projects\Redering\OpenIRT\include
	file base:	TReX
	file ext:	h
	author:		Tae-Joon Kim (tjkim.kaist@gmail.com)
	
	comment:	Hybrid renderer using both CPU and GPU
*********************************************************************/

#pragma once

#include "CUDARayTracer.h"
#include "Photon.h"
#include "PhotonOctree.h"
#include "WinLock.h"
#include "HSaliency.h"
#include "OOCVoxelManager.h"
#include "Sampler.h"

namespace irt
{

class TReX :
	public CUDARayTracer
{
public:
	typedef struct SharedThreadData_t
	{
		TReX *renderer;
		Camera *camera;
		Image *image;
		int frame2;
	} SharedThreadData;

	typedef struct TileElem_t
	{
		int tileSize;
		int tileStartX;
		int tileStartY;
		float custom;
		TileElem_t(int tileSize = -1, int tileStartX = -1, int tileStartY = -1, float custom = 0.0f)
			: tileSize(tileSize), tileStartX(tileStartX), tileStartY(tileStartY), custom(custom) {}

		~TileElem_t()
		{
		}

		bool isValid() {return tileStartX >= 0;}

		static bool rowByRow(const TileElem_t &a, const TileElem_t &b)
		{
			if(a.tileStartY == b.tileStartY)
				return a.tileStartX < b.tileStartX;
			return a.tileStartY < b.tileStartY;
		}

		static bool zCurve(const TileElem_t &a, const TileElem_t &b)
		{
			if(a.tileStartY == b.tileStartY)
				return a.tileStartX > b.tileStartX;
			return a.tileStartY > b.tileStartY;
		}

		static bool cursorGuided(const TileElem_t &a, const TileElem_t &b)
		{
			extern int g_mouseX;
			extern int g_mouseY;
			return 
				(g_mouseX - a.tileStartX)*(g_mouseX - a.tileStartX) + (g_mouseY - a.tileStartY)*(g_mouseY - a.tileStartY) <
				(g_mouseX - b.tileStartX)*(g_mouseX - b.tileStartX) + (g_mouseY - b.tileStartY)*(g_mouseY - b.tileStartY);
		}

		static bool custumized(const TileElem_t &a, const TileElem_t &b)
		{
			return a.custom > b.custom;
		}

	} TileElem;

	typedef struct CUDAHit_t 
	{
		float t;				// Param of ray
		unsigned int model;
		unsigned int material;	// Rerefence to material
		Vector3 n;
		Vector2 uv;				// tex coords
		Vector3 x;
		float dummy;
	} CUDAHit;

	typedef struct ExtraRayInfo_t
	{
		float pixelX;
		float pixelY;
		//unsigned int frame2;
		unsigned int seed;
		unsigned char mask;

		bool hasHit() {return (mask & 0x1) != 0;}
		bool isSpecular() {return (mask & 0x2) != 0;}
		bool wasBounced() {return (mask & 0x4) != 0;}

		void setHasHit(bool val) {mask = val ? (mask | 0x1) : (mask & ~0x1);}
		void setIsSpecular(bool val) {mask = val ? (mask | 0x2) : (mask & ~0x2);}
		void setWasBounced(bool val) {mask = val ? (mask | 0x4) : (mask & ~0x4);}

		ExtraRayInfo_t() {mask = 0;}
		//__device__ __host__ void setIsSpecular(bool val) {mask |= 0x2;}
		//__device__ __host__ void setWasBounced(bool val) {mask |= 0x4;}

		//__device__ __host__ void resetHasHit() {mask &= ~0x1;}
		//__device__ __host__ void resetIsSpecular() {mask &= ~0x2;}
		//__device__ __host__ void resetWasBounced() {mask &= ~0x4;}
		//int hasHit;
		//bool isSpecular;
		//bool wasBounced;
	} ExtraRayInfo;

	typedef std::vector<TileElem> TileList;

	enum Mode
	{
		MODE_DEFAULT,
		MODE_QUALITY_ESTIMATION,
		MODE_TIMING,
		MODE_TIME_TO_CONVERGE,
		MODE_PERFORMANCE,
		MODE_TRACE
	};
protected:
	Photon *m_photons;
	int m_numPhotons;
	AABB m_bbPhotons;
	PhotonOctree m_octree;

	int m_rayCacheBlockSize;

	bool m_controllerUpdated;
	bool m_cameraUpdated;
	bool m_materialUpdated;
	bool m_lightUpdated;

	bool m_exit;

	HANDLE m_hRenderThread;
	std::vector<HANDLE> m_hGPULaunchingThreadList;
	SharedThreadData m_threadData;
	TileList m_tileList;
	Ray *m_rayCache;
	CUDAHit *m_hitCache;
	ExtraRayInfo *m_extRayCache;

	int previewRendering(Camera *camera, Image *image);
	int resetImages(Image *image);
	int tileOrdering(Image *image);
	int CRayTracing(Camera *camera, Image *image, int numTiles, int offset, int frame2);
	int launchGRayTracing(int numTiles, int offset);

	static unsigned __stdcall GPULaunchingThread(void* arg);

	static unsigned __stdcall renderThread(void* arg);

	int m_renderingThumbNailStream;

	OOCVoxelManager *m_oocVoxelMgr;

	Sampler m_sampler;

	int *m_tileOrder;

	int m_numPhotonsInBackground;
	int m_numSubPhotonsInBackground;
	int m_curTracedPhotonsInBackground;

	float m_frameTime;
	float m_frameTimeCPU;
	float m_frameTimeGPU;
	float m_frameTimeCur;
	float m_frameTimeCPUCur;
	float m_frameTimeGPUCur;
	float m_MRaysPerSec;

	Mode m_mode;
public:
	TReX(void);
	virtual ~TReX(void);

	virtual void init(Scene *scene);
	virtual void done();
	virtual void restart(Camera *camera = NULL, Image *image = NULL);

	virtual void resized(int width, int height);

	virtual void sceneChanged();
	virtual void materialChanged();
	virtual void lightChanged(bool soft = false);

	// renderer
	virtual void flushImage(Image *image);
	virtual void render(Camera *camera, Image *image, unsigned int seed);

	bool rayOctreeIntersect(const Ray &ray, HitPointInfo &hit);
	bool rayOctreeIntersect(const Ray &ray, HitPointInfo &hit, const Voxel &voxel, AABB bb, int flag);

	bool rayOctreeIntersectIter(const Ray &ray, HitPointInfo &hit);

	void applyChangedMaterial();
	void applyChangedLight();
};

};