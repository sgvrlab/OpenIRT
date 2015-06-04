#include "OOCVoxelManager.h"
#include "CommonOptions.h"
#include <io.h>
#include <process.h>
#include <algorithm>
#include "Photon.h"
#include "PhotonOctree.h"
#include <map>
#include <stopwatch.h>
#include "OpenIRT.h"
#include "Renderer.h"

#ifndef fminf
#define fminf(a,b) (((a) < (b)) ? (a) : (b))
#endif

#ifndef fmaxf
#define fmaxf(a,b) (((a) > (b)) ? (a) : (b))
#endif

#include "CUDA/CUDADataStructures.cuh"

using namespace irt;

OOCVoxelManager::OOCVoxelManager(Octree *highOctree, const char *fileBase, int oriNumVoxels, const Vector3 &thresholdSize, int allowedMemMB)
{
#	ifndef USE_OOCVOXEL
	return;
#	endif
	m_highOctree = highOctree;

	m_oriNumVoxels = oriNumVoxels;
	m_allowedMem = allowedMemMB * 1024 * 1024;
	m_freeMem.insert(FreeMem(0, m_allowedMem/sizeof(Voxel)));
	m_thresholdSize = thresholdSize;

	char fileName[256];
	FILE *fp;

	m_exit = true;

	// read header
	sprintf_s(fileName, 255, "%s_OOCVoxel.hdr", fileBase);
	fopen_s(&fp, fileName, "rb");
	if(!fp)
	{
		printf("Cannot find %s\n", fileName);
		return;
	}

	int numOOCVoxels = (int)_filelength(_fileno(fp)) / sizeof(OOCVoxel);
	for(int i=0;i<numOOCVoxels;i++)
	{
		OOCVoxel buf;
		fread(&buf, sizeof(OOCVoxel), 1, fp);
		m_oocVoxelList.push_back(buf);

		(*m_highOctree)[buf.rootChildIndex].setLink2Low(i);
	}
	fclose(fp);

	m_requestCountListSize = numOOCVoxels;
	m_requestCountList = new float[m_requestCountListSize];

	BY_HANDLE_FILE_INFORMATION fileInfo;
	// map voxel file
	sprintf_s(fileName, 255, "%s_OOCVoxel.ooc", fileBase);
	m_hVoxelFile = CreateFile(fileName, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, NULL);
	if(!m_hVoxelFile)
	{
		printf("Cannot find %s\n", fileName);
		return;
	}
	GetFileInformationByHandle(m_hVoxelFile, &fileInfo);
	m_hVoxelMap = CreateFileMapping(m_hVoxelFile, NULL, PAGE_READONLY, fileInfo.nFileSizeHigh, fileInfo.nFileSizeLow, NULL);
	m_voxelFile = (Voxel*)MapViewOfFile(m_hVoxelMap, FILE_MAP_READ, 0, 0, 0);

	// map photon voxel file

	sprintf_s(fileName, 255, "%s_photonVoxel.ooc", fileBase);
	m_hPhotonVoxelFile = CreateFile(fileName, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, NULL);
	if(!m_hPhotonVoxelFile)
	{
		printf("Cannot find %s\n", fileName);
		return;
	}
	GetFileInformationByHandle(m_hPhotonVoxelFile, &fileInfo);
	m_hPhotonVoxelMap = CreateFileMapping(m_hPhotonVoxelFile, NULL, PAGE_READONLY, fileInfo.nFileSizeHigh, fileInfo.nFileSizeLow, NULL);
	m_photonVoxelFile = (PhotonVoxel*)MapViewOfFile(m_hPhotonVoxelMap, FILE_MAP_READ, 0, 0, 0);

	m_in = new int[numOOCVoxels*2];
	m_out = new int[numOOCVoxels*4];

	m_hQNotEmpty = CreateEvent(NULL, TRUE, FALSE, NULL);
	m_hEnoughMem = CreateEvent(NULL, TRUE, TRUE, NULL);
	m_exit = false;
	m_enabled = true;
	m_hLoadingThread = (HANDLE)_beginthreadex(NULL, 0, loadingThread, this, 0, NULL);
}

OOCVoxelManager::~OOCVoxelManager()
{
	m_exit = true;

	SetEvent(m_hQNotEmpty);
	SetEvent(m_hEnoughMem);

	if(m_hLoadingThread)
	{
		WaitForSingleObject(m_hLoadingThread, INFINITE);
		CloseHandle(m_hLoadingThread);
		m_hLoadingThread = NULL;
	}

	UnmapViewOfFile(m_voxelFile);
	CloseHandle(m_hVoxelMap);
	CloseHandle(m_hVoxelFile);

	UnmapViewOfFile(m_photonVoxelFile);
	CloseHandle(m_hPhotonVoxelMap);
	CloseHandle(m_hPhotonVoxelFile);
	delete[] m_in;
	delete[] m_out;
	delete[] m_requestCountList;

	CloseHandle(m_hQNotEmpty);
	CloseHandle(m_hEnoughMem);
}

int OOCVoxelManager::allocMem(int count)
{
	int offset = -1;
	m_lockMem.lock();

	/*
	printf("before alloc: ");
	for(std::set<FreeMem, FreeMemCountLess>::iterator it=m_freeMem.begin();it!=m_freeMem.end();it++)
	{
		printf("(%d, %d) ", it->offset, it->count);
	}
	printf("\n");
	*/


	for(std::set<FreeMem, FreeMemCountLess>::iterator it=m_freeMem.begin();it!=m_freeMem.end();it++)
	{
		if(it->count >= count)
		{
			offset = it->offset;
			if(it->count > count)
				m_freeMem.insert(FreeMem(it->offset+count, it->count - count));
			m_freeMem.erase(it);
			break;
		}
	}

	/*
	if(offset >= 0)
	{
		printf("alloc (%d, %d)\n", offset, count);
		printf("after alloc: ");
		for(std::set<FreeMem, FreeMemCountLess>::iterator it=m_freeMem.begin();it!=m_freeMem.end();it++)
		{
			printf("(%d, %d) ", it->offset, it->count);
		}
		printf("\n");
	}
	else
		printf("not enough memory\n");
	*/

	m_lockMem.unlock();

	if(offset < 0 && packMem())
		offset = allocMem(count);

	return offset;
}

void OOCVoxelManager::freeMem(int offset, int count)
{
	m_lockMem.lock();
	m_freeMem.insert(FreeMem(offset, count));
	m_lockMem.unlock();
}

bool OOCVoxelManager::packMem()
{
	bool packed = false;
	std::vector<FreeMem> list;

	m_lockMem.lock();
	for(std::set<FreeMem, FreeMemCountLess>::iterator it=m_freeMem.begin();it!=m_freeMem.end();it++)
		list.push_back(*it);
	std::sort(list.begin(), list.end(), FreeMemOffsetLess());
	for(int i=0;i<(int)list.size()-1;i++)
	{
		if(list[i].offset + list[i].count == list[i+1].offset)
		{
			list[i+1].offset = list[i].offset;
			list[i+1].count += list[i].count;
			list[i].count = 0;
			packed = true;
		}
	}
	m_freeMem.clear();
	for(int i=0;i<(int)list.size();i++)
	{
		if(list[i].count > 0)
			m_freeMem.insert(list[i]);
	}

	m_lockMem.unlock();
	return packed;
}

Vector3 g_camPos;
std::vector<OOCVoxelManager::OOCVoxel> *g_oocVoxelList = NULL;
float *g_requestCountList = NULL;
extern "C" void loadVoxelsAPI(int count, const CUDA::Voxel *voxels, int oocVoxelIdx, int offset, int ttt);
extern "C" void loadPhotonVoxelsAPI(int count, const CUDA::PhotonVoxel *voxels, int oocVoxelIdx, int offset);
extern "C" int tracePhotonsAPI();
extern "C" int tracePhotonsToOOCVoxelAPI(int oocVoxelIdx);
unsigned __stdcall OOCVoxelManager::loadingThread(void* arg)
{
#	ifndef USE_OOCVOXEL
	return 0;
#	endif
	OOCVoxelManager *m = (OOCVoxelManager*)arg;
	while(!m->m_exit)
	{
		WaitForSingleObject(m->m_hQNotEmpty, INFINITE);
		if(m->m_exit) break;

#		if VOXEL_PRIORITY_POLICY == 1
		if(!g_oocVoxelList) 
#		endif
#		if VOXEL_PRIORITY_POLICY == 2
		if(!g_requestCountList)
#		endif
		{
			Sleep(10);
			continue;
		}
		

		int voxel;
		m->m_lockQNeeded.lock();
		if(m->m_voxelNeeded.empty()) 
		{
			m->m_lockQNeeded.unlock();
			continue;
		}
		voxel = m->m_voxelNeeded.front();
		m->m_lockQNeeded.unlock();

		//printf("Loading... %d\n", voxel);

		int gpuOffset = -1;
		const OOCVoxel &oocVoxel = m->m_oocVoxelList[voxel];
		int fileOffset = oocVoxel.offset;
		int numVoxels = oocVoxel.numVoxels;
		m->m_lockSet.lock();
		bool isIn = m->m_activeVoxels.find(voxel) != m->m_activeVoxels.end();
		m->m_lockSet.unlock();
		if(!isIn)
		{
			//while((gpuOffset = m->allocMem(numVoxels)) < 0)
			while(!m->m_exit)
			{
				m->m_lockSet.lock();
				//printf("alloc for voxel[%d] with %d\n", voxel, numVoxels);
				gpuOffset = m->allocMem(numVoxels);
				if(gpuOffset >= 0)
				{
					m->m_activeVoxels[voxel] = gpuOffset;
					//printf("<- %d\n", voxel);
					m->m_lockSet.unlock();
					break;
				}
				//printf("failed wait...removeing...");

				ResetEvent(m->m_hEnoughMem);
				// if requested voxel has lower priority than all active voxels then ignore the request
				if(VOXEL_COMPARE()(m->m_activeVoxels.rbegin()->first, voxel))
				{
					m->m_lockQNeeded.lock();
					m->m_voxelNeeded.clear();
					ResetEvent(m->m_hQNotEmpty);
					m->m_lockQNeeded.unlock();
					m->m_lockSet.unlock();

					//tracePhotonsAPI();
					goto END_OF_LOOP;
				}
				m->m_lockQOut.lock();
				int targetSpace = m->m_allowedMem / 10;
				for(ActiveVoxelMap::reverse_iterator it=m->m_activeVoxels.rbegin();it!=m->m_activeVoxels.rend();it++)
				{
					const OOCVoxel &voxel = m->m_oocVoxelList[it->first];

					bool isIn = false;
					for(int i=0;i<m->m_voxelOut.size();i++)
					{
						//printf("[%d, %d] ", it->first, m->m_voxelOut[i].voxel);
						if(m->m_voxelOut[i].voxel == it->first)
						{
							isIn = true;
							break;
						}
					}
					if(!isIn) m->m_voxelOut.push_back(VoxelOutElem(it->first, voxel.rootChildIndex, it->second, voxel.numVoxels));

					targetSpace -= voxel.numVoxels * sizeof(Voxel);

					//printf("(%d) ", it->first);

					if(targetSpace < 0) break;
				}
				//printf("\n");
				m->m_lockQOut.unlock();
				m->m_lockSet.unlock();
				WaitForSingleObject(m->m_hEnoughMem, INFINITE);
				if(m->m_exit) break;
			}

			//m->m_lockSet.lock();
			//m->m_activeVoxels[voxel] = gpuOffset;
			//m->m_lockSet.unlock();
		}

		m->m_lockQNeeded.lock();
		if(!m->m_voxelNeeded.empty()) 
			m->m_voxelNeeded.pop_front();
		m->m_lockQNeeded.unlock();

		if(gpuOffset >= 0)
		{
			Voxel *tempVoxels = new Voxel[numVoxels];
			//memcpy(tempVoxels, &m->m_voxelFile[fileOffset], sizeof(Voxel)*numVoxels);
			for(int i=0;i<numVoxels;i++)
			{
				Voxel &voxel = tempVoxels[i];
				voxel = m->m_voxelFile[fileOffset+i];
				if(voxel.hasChild())
					voxel.setChildIndex(voxel.getChildIndex() + (m->m_oriNumVoxels + gpuOffset)/8);
			}
			//loadVoxelsAPI(numVoxels, (CUDA::Voxel*)&m->m_voxelFile[fileOffset], gpuOffset);
			loadVoxelsAPI(numVoxels, (CUDA::Voxel*)tempVoxels, voxel, gpuOffset, oocVoxel.rootChildIndex);
			loadPhotonVoxelsAPI(numVoxels, (CUDA::PhotonVoxel*)&(m->m_photonVoxelFile[fileOffset]), voxel, gpuOffset);
			//tracePhotonsToOOCVoxelAPI(voxel);
			//printf("<- %d\n", voxel);
			delete[] tempVoxels;
			m->m_lockQLoaded.lock();
			m->m_voxelLoaded.push_back(VoxelLoadedElem(oocVoxel.rootChildIndex, gpuOffset));
			m->m_lockQLoaded.unlock();
		}

END_OF_LOOP:
		if(m->m_voxelNeeded.empty())
			if(!m->m_exit) ResetEvent(m->m_hQNotEmpty);

	}
	_endthreadex(0);
	return 0;
}

void OOCVoxelManager::moveCamera(Camera &camera)
{
	static bool s_moving = false;
	if(!m_enabled)
	{
		//printf("disabled\n");
		return;
	}
	if(s_moving)
	{
		//printf("moving\n");
		return;
	}

	static Vector3 prevPos;
	const Vector3 &pos = camera.getEye();
	if(m_exit || pos == prevPos)
	{
		//if(m_exit) printf("exit\n");
		//if(pos == prevPos) printf("same pos\n");
		return;
	}

	s_moving = true;
	prevPos = pos;

	m_lockQNeeded.lock();
	m_voxelNeeded.clear();
	ResetEvent(m_hQNotEmpty);
	m_lockQNeeded.unlock();

	int numOOCVoxels = (int)m_oocVoxelList.size();

	std::vector<int> voxelDistList;
	for(int i=0;i<numOOCVoxels;i++)
		voxelDistList.push_back(i);

	// sort voxels w.r.t distance from camera position
	g_camPos = pos;
	g_oocVoxelList = &m_oocVoxelList;
	std::sort(voxelDistList.begin(), voxelDistList.end(), VoxelDistLess());

	m_lockSet.lock();

	// rearrange sorted map
	std::vector<std::pair<int, int> > tempList;
	for(ActiveVoxelMap::iterator it=m_activeVoxels.begin();it!=m_activeVoxels.end();it++)
		tempList.push_back(*it);

	m_activeVoxels.clear();
	for(int i=0;i<(int)tempList.size();i++)
		m_activeVoxels[tempList[i].first] = tempList[i].second;

#	if 0
	printf("ActiveVoxels = ");
	int usedMem = 0;
	for(ActiveVoxelMap::iterator it=m_activeVoxels.begin();it!=m_activeVoxels.end();it++)
	{
		printf("%d ", it->first);
		usedMem += m_oocVoxelList[it->first].numVoxels * 32;
	}
	printf("\n");
	printf("%d bytes used\n", usedMem);
	int freeMem = 0;
	for(std::set<FreeMem, FreeMemCountLess>::iterator it=m_freeMem.begin();it!=m_freeMem.end();it++)
	{
		freeMem += it->count*32;
	}
	printf("%d/%d bytes available\n", freeMem, m_allowedMem);
	if(usedMem + freeMem != m_allowedMem)
	{
		printf("Memory Error! %d != %d\n", usedMem + freeMem, m_allowedMem);
		/*
		printf("free mem: ");
		for(std::set<FreeMem, FreeMemCountLess>::iterator it=m_freeMem.begin();it!=m_freeMem.end();it++)
		{
			printf("(%d, %d) ", it->offset, it->count);
		}
		printf("\n ActiveVoxels: ");
		for(ActiveVoxelMap::iterator it=m_activeVoxels.begin();it!=m_activeVoxels.end();it++)
		{
			printf("(%d, %d) ", m_oocVoxelList[it->first].offset, m_oocVoxelList[it->first].numVoxels);
		}
		printf("\n");
		*/
		//exit(1);
	}
#	endif

	//printf("NeededVoxels = ");
	float diag = m_thresholdSize.length();
	int scrWidth = OpenIRT::getSingletonPtr()->getRenderer()->getWidth();
	float ratio = (camera.getScaledRight().length() / scrWidth) / camera.getZNear();
	for(int i=0;i<numOOCVoxels;i++)
	{
		int voxelIdx = voxelDistList[i];
		const OOCVoxel &voxel = m_oocVoxelList[voxelIdx];

		float dist = (0.5f*(voxel.rootBB.min + voxel.rootBB.max) - pos).length() - (0.5f*(voxel.rootBB.max - voxel.rootBB.min)).length();
		
		if(dist > 0 && ratio > diag / dist)
			break;

		if(m_activeVoxels.find(voxelIdx) == m_activeVoxels.end())
		{
			m_lockQNeeded.lock();
			m_voxelNeeded.push_back(voxelIdx);
			//printf("%d ", voxelIdx);
			m_lockQNeeded.unlock();
			SetEvent(m_hQNotEmpty);
		}
	}
	//printf("++\n");
	m_lockSet.unlock();

	s_moving = false;
}

void OOCVoxelManager::updateVoxelList()
{
	m_lockQNeeded.lock();
	m_voxelNeeded.clear();
	ResetEvent(m_hQNotEmpty);
	m_lockQNeeded.unlock();

	int numOOCVoxels = (int)m_oocVoxelList.size();

	/*
	printf("------------------------------------------------\n");
	for(int i=0;i<numOOCVoxels;i++)
	{
		if(m_requestCountList[i] >= 1.0f)
			printf("[%d] %f, ", i, m_requestCountList[i]);
	}
	printf("\n================================================\n");
	*/

	std::vector<int> voxelRequestCountList;
	for(int i=0;i<numOOCVoxels;i++)
		voxelRequestCountList.push_back(i);

	// sort voxels w.r.t request count from GPU
	g_requestCountList = m_requestCountList;
	std::sort(voxelRequestCountList.begin(), voxelRequestCountList.end(), VoxelRequestedGreater());

	m_lockSet.lock();

	// rearrange sorted map
	std::vector<std::pair<int, int> > tempList;
	for(ActiveVoxelMap::iterator it=m_activeVoxels.begin();it!=m_activeVoxels.end();it++)
		tempList.push_back(*it);

	m_activeVoxels.clear();
	for(int i=0;i<(int)tempList.size();i++)
		m_activeVoxels[tempList[i].first] = tempList[i].second;

	/*
	printf("ActiveVoxels = ");
	int usedMem = 0;
	for(ActiveVoxelMap::iterator it=m_activeVoxels.begin();it!=m_activeVoxels.end();it++)
	{
		printf("%d ", it->first);
		usedMem += m_oocVoxelList[it->first].numVoxels * 32;
	}
	printf("\n");
	printf("%d bytes used\n", usedMem);
	int freeMem = 0;
	for(std::set<FreeMem, FreeMemCountLess>::iterator it=m_freeMem.begin();it!=m_freeMem.end();it++)
	{
		freeMem += it->count*32;
	}
	printf("%d/%d bytes available\n", freeMem, m_allowedMem);
	if(usedMem + freeMem != m_allowedMem)
	{
		printf("Memory Error! %d != %d\n", usedMem + freeMem, m_allowedMem);
		exit(1);
	}
	*/

	//printf("NeededVoxels = ");
	for(int i=0;i<numOOCVoxels;i++)
	{
		int voxelIdx = voxelRequestCountList[i];
		const OOCVoxel &voxel = m_oocVoxelList[voxelIdx];

		if(m_requestCountList[voxelIdx] < 1.0f)
			break;

		if(m_activeVoxels.find(voxelIdx) == m_activeVoxels.end())
		{
			m_lockQNeeded.lock();
			m_voxelNeeded.push_back(voxelIdx);
			//printf("%d ", voxelIdx);
			m_lockQNeeded.unlock();
			SetEvent(m_hQNotEmpty);
		}
	}
	//printf("++\n");
	m_lockSet.unlock();
}

/*
int g_ttt = 0;
float g_fff = 0;
FILE *g_fpfp = 0;
*/
extern "C" void OnVoxelChangedAPI(int numIn, int *in, int numOut, int *out);
void OOCVoxelManager::update()
{
	if(!m_enabled) return;
	if(m_exit) return;

	m_lockQLoaded.lock();
	int numIn = (int)m_voxelLoaded.size();
	for(int i=0;i<numIn;i++)
	{
		const VoxelLoadedElem &e = m_voxelLoaded.front();
		m_in[i*2+0] = e.rootIndex;
		m_in[i*2+1] = e.gpuOffset;
		m_voxelLoaded.pop_front();
	}
	m_lockQLoaded.unlock();

	m_lockQOut.lock();
	int numOut = (int)m_voxelOut.size();
	for(int i=0;i<numOut;i++)
	{
		const VoxelOutElem &e = m_voxelOut.front();
		m_out[i*2+0] = e.rootIndex;
		m_out[i*2+1] = e.gpuOffset;
		m_out[2*numOut+i*2+0] = e.voxel;
		m_out[2*numOut+i*2+1] = e.numVoxels;
		m_voxelOut.pop_front();
	}
	m_lockQOut.unlock();

	/*
	int timer = StopWatch::create();
	StopWatch::get(timer).start();
	*/

	OnVoxelChangedAPI(numIn, m_in, numOut, m_out);

	/*
	StopWatch::get(timer).stop();
	g_fff += StopWatch::get(timer).getTime();
	g_ttt += numIn;
	if(numIn > 0)
	{
		printf("numin = %d, update time = %f ms, summed[%d] = %f ms\n", numIn, StopWatch::get(timer).getTime(), g_ttt, g_fff);
		fprintf_s(g_fpfp, "%d %f\n", numIn, StopWatch::get(timer).getTime());
		fflush(g_fpfp);
	}
	StopWatch::destroy(timer);
	*/

	m_lockSet.lock();
	for(int i=0;i<numOut;i++)
	{
		int voxel = m_out[2*numOut+i*2+0];
		int gpuOffset = m_out[i*2+1];
		int numVoxels = m_out[2*numOut+i*2+1];

		ActiveVoxelMap::iterator it = m_activeVoxels.find(voxel);
		if(it != m_activeVoxels.end())
		{
			if(gpuOffset != it->second)
			{
				printf("offset error! %d != %d\n", gpuOffset, it->second);
				exit(-1);
			}
			m_activeVoxels.erase(it);
			//printf("-> %d\n", voxel);
			freeMem(gpuOffset, numVoxels);
		}
		/*
		m_activeVoxels.erase(m_out[2*numOut+i*2+0]);
		//printf("-> %d\n", m_out[2*numOut+i*2+0]);
		freeMem(m_out[i*2+1], m_out[2*numOut+i*2+1]);
		*/
	}
	m_lockSet.unlock();

	if(numOut > 0) SetEvent(m_hEnoughMem);
}