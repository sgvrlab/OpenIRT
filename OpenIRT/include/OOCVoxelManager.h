#pragma once

#include "CommonOptions.h"
#include <deque>
#include <set>
#include <map>
#include <Windows.h>
#include "WinLock.h"
#include "Camera.h"
#include "Voxel.h"
#include "BV.h"
#include "Photon.h"
#include "Octree.h"

#if VOXEL_PRIORITY_POLICY == 1
#define VOXEL_COMPARE VoxelDistLess
#endif

#if VOXEL_PRIORITY_POLICY == 2
#define VOXEL_COMPARE VoxelRequestedGreater
#endif

namespace irt
{

class OOCVoxelManager
{
public:
	typedef struct OOCVoxel_t
	{
		int rootChildIndex;
		int startDepth;
		int offset;
		int numVoxels;
		AABB rootBB;
	} OOCVoxel;

protected:
	typedef struct FreeMem_t
	{
		int offset;
		int count;

		FreeMem_t(int offset, int count) : offset(offset), count(count) {}

	} FreeMem;

	struct FreeMemCountLess
	{
		bool operator() (const FreeMem &a, const FreeMem &b)
		{
			if(a.count == b.count) return a.offset < b.offset;
			return a.count < b.count;
		}
	};

	struct FreeMemOffsetLess
	{
		bool operator() (const FreeMem &a, const FreeMem &b)
		{
			return a.offset < b.offset;
		}
	};

	struct VoxelDistLess
	{
		bool operator() (int a, int b)
		{
			extern Vector3 g_camPos;
			extern std::vector<OOCVoxel> *g_oocVoxelList;

			const OOCVoxel &voxelA = g_oocVoxelList->at(a);
			const OOCVoxel &voxelB = g_oocVoxelList->at(b);

			float distA = (0.5f*(voxelA.rootBB.max + voxelA.rootBB.min) - g_camPos).squaredLength();
			float distB = (0.5f*(voxelB.rootBB.max + voxelB.rootBB.min) - g_camPos).squaredLength();

			return distA < distB;

			/*
			return 
				(0.5f*(voxelA.rootBB.max + voxelA.rootBB.min) - g_camPos).squaredLength() < 
				(0.5f*(voxelB.rootBB.max + voxelB.rootBB.min) - g_camPos).squaredLength();
			*/
		}
	};

	struct VoxelRequestedGreater
	{
		bool operator() (int a, int b)
		{
			extern float *g_requestCountList;
			return g_requestCountList[a] > g_requestCountList[b];
		}
	};

	typedef struct VoxelLoadedElem_t
	{
		int rootIndex;
		int gpuOffset;

		VoxelLoadedElem_t(int rootIndex, int gpuOffset) : rootIndex(rootIndex), gpuOffset(gpuOffset) {}
	} VoxelLoadedElem;

	typedef struct VoxelOutElem_t
	{
		int voxel;
		int rootIndex;
		int gpuOffset;
		int numVoxels;

		VoxelOutElem_t(int voxel, int rootIndex, int gpuOffset, int numVoxels) : voxel(voxel), rootIndex(rootIndex), gpuOffset(gpuOffset), numVoxels(numVoxels) {}
	} VoxelOutElem;

	typedef std::map<int, int, VOXEL_COMPARE> ActiveVoxelMap;

protected:
	bool m_enabled;

	Vector3 m_thresholdSize;

	std::vector<OOCVoxel> m_oocVoxelList;
	ActiveVoxelMap m_activeVoxels;
	std::deque<int> m_voxelNeeded;
	std::deque<VoxelLoadedElem> m_voxelLoaded;
	std::deque<VoxelOutElem> m_voxelOut;
	std::set<FreeMem, FreeMemCountLess> m_freeMem;

	WinLock m_lockSet;
	WinLock m_lockQNeeded;
	WinLock m_lockQLoaded;
	WinLock m_lockQOut;
	WinLock m_lockMem;

	HANDLE m_hVoxelFile, m_hVoxelMap;
	Voxel *m_voxelFile;

	HANDLE m_hPhotonVoxelFile, m_hPhotonVoxelMap;
	PhotonVoxel *m_photonVoxelFile;

	int m_allowedMem;
	int m_oriNumVoxels;

	int *m_in, *m_out;

	Octree *m_highOctree;
	int m_requestCountListSize;
	float *m_requestCountList;

	int allocMem(int count);
	void freeMem(int offset, int count);
	bool packMem();

	// for thread
	HANDLE m_hLoadingThread;
	HANDLE m_hQNotEmpty;
	HANDLE m_hEnoughMem;
	bool m_exit;
	static unsigned __stdcall loadingThread(void* arg);

public:
	OOCVoxelManager(Octree *highOctree, const char *fileBase, int oriNumVoxels, const Vector3 &m_thresholdSize, int allowedMemMB = 512);
	~OOCVoxelManager();

	void moveCamera(Camera &camera);
	void updateVoxelList();
	void update();

	int getNumOOCVoxels() {return (int)m_oocVoxelList.size();}
	OOCVoxel &getOOCVoxel(int pos) {return m_oocVoxelList[pos];}
	float *getRequestCountListPtr() {return m_requestCountList;}
};

};