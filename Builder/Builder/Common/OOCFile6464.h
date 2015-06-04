#ifndef COMMON_OOCFILE6464_H
#define COMMON_OOCFILE6464_H

// debugging mode, tests for errors on functions calls and
// has some asserts to test correctness in debug build
//#define OOCFILE_DEBUG

// measure approximate cache hit rate
//#define OOCFILE_PROFILE

#include <Windows.h>
#include "LogManager.h"
#include "common.h"

// 64 bit addressing structure
typedef ULARGE_INTEGER OOCSize64;

template <class T>
class OOCFile6464 {

public:
	OOCFile6464() {}
	OOCFile6464(const char * pFileName, __int64 maxAllowedMem, int blockSize);
	FORCEINLINE const T &operator[](unsigned int i); 
	~OOCFile6464();

	OOCSize64 m_fileSize;

protected:

	void outputWindowsErrorMessage() {
		DWORD  ErrorCode = GetLastError();
		LPVOID lpMsgBuf;

		FormatMessage ( FORMAT_MESSAGE_ALLOCATE_BUFFER | 
			FORMAT_MESSAGE_FROM_SYSTEM | 
			FORMAT_MESSAGE_IGNORE_INSERTS, 0, ErrorCode, 0, // Default language
			(LPTSTR) &lpMsgBuf,   0,   NULL );

		cout << (char *)lpMsgBuf << endl;
	}

	void dumpCacheStats() {
		LogManager *log = LogManager::getSingletonPtr();
		char output[200];

		sprintf(output, "Cache: %u access, %u misses. Cache hit rate: %f\n", cacheAccesses, cacheMisses, 100.0f - (float)cacheMisses/(float)cacheAccesses);
		log->logMessage(LOG_INFO, output);
	}

	// handles for memory-mapped file
#ifdef WIN32
	HANDLE m_hFile, m_hMapping;
#else
	int m_FD;
#endif

	unsigned char *data;
};

#ifdef WORKING_SET_COMPUTATION
extern bool g_targetModelCacheSimualte;	
extern CGPUCacheSim g_BVHSim;
extern CGPUCacheSim g_TriSim;
extern CGPUCacheSim g_VerSim;
extern CGPUCacheSim g_IdxSim;
extern CGPUCacheSim g_PhotonSim;
extern CGPUCacheSim g_LODSim;
#endif

// constructor
template <class T>
OOCFile6464<T>::OOCFile6464(const char * pFileName, __int64 maxAllowedMem, int blockSize) {
	SYSTEM_INFO systemInfo;
	BY_HANDLE_FILE_INFORMATION fileInfo;
	LogManager *log = LogManager::getSingletonPtr();
	char output[200];

	// get windows allocation granularity:
	GetSystemInfo(&systemInfo);

	// open file:
	if (! (m_hFile = CreateFile(pFileName, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
		FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, NULL))) {

			cerr << "OOCFile6464: Cannot open file: " << pFileName << endl;			
			outputWindowsErrorMessage();
			exit (-1);
	}

	// get file size:
	GetFileInformationByHandle(m_hFile, &fileInfo);
	m_fileSize.LowPart = fileInfo.nFileSizeLow; 
	m_fileSize.HighPart = fileInfo.nFileSizeHigh;

	if (!(m_hMapping = CreateFileMapping(m_hFile, NULL, PAGE_READONLY, 
		m_fileSize.HighPart, m_fileSize.LowPart, NULL))) {
			cerr << "OOCFile6464<" << sizeof(T) << ">: CreateFileMapping() failed" << endl;
			outputWindowsErrorMessage();
			exit (-1);
	}

	data = (unsigned char *)MapViewOfFile(m_hMapping, FILE_MAP_READ, 0, 0, 0);
}

template <class T> 
OOCFile6464<T>::~OOCFile6464() {
	if(data) UnmapViewOfFile(data);

	// close file mapping and file:
	if(m_hMapping) CloseHandle (m_hMapping);
	if(m_hFile) CloseHandle (m_hFile);
	m_hMapping = 0;
	m_hFile = 0 ;
#ifdef OOCFILE_PROFILE
	dumpCacheStats();
#endif
}

#ifdef USE_SEPARATE_TYPE
FORCEINLINE const Vertex& OOCFile6464<Vertex>::operator[](unsigned int i) {
	OOCSize64 j, startOffset;

	// find pos in table for this i
	j.QuadPart = (__int64)i * sizeof(Vertex);

	startOffset.QuadPart = j.QuadPart >> m_BlockSizePowerTwo;	
	unsigned int tablePos = startOffset.LowPart & m_CacheNumMask;

	int PageID = startOffset.QuadPart;
	tablePos = m_Loaded [PageID];
	
	#ifdef WORKING_SET_COMPUTATION
	//if (g_targetModelCacheSimualte) 
	//	g_VerSim.Access (i);
	#endif

	if (m_Loaded [PageID] == UNLOADED)
	{
		if (m_CurNumCachedPage < m_NumCachedPage)
		{
			tablePos = m_CurNumCachedPage++;
			LRUEntry *newLRUEntry = new LRUEntry;
			m_CacheTable[tablePos].entryLRU = newLRUEntry;
			newLRUEntry->tablePos = tablePos;
			newLRUEntry->pageID = PageID;
			m_CacheTable[tablePos].fileStartOffset.QuadPart = startOffset.QuadPart;
			load(tablePos);
			m_LRUList.ForceAdd(newLRUEntry);
			m_Loaded [PageID] = tablePos;
		}
		else
		{
			LRUEntry *pLeastUsed = m_LRUList.m_pEnd->m_pPrev;
			tablePos = pLeastUsed->tablePos;
			unload(tablePos);
			m_Loaded [pLeastUsed->pageID] = UNLOADED;
			m_CacheTable[tablePos].fileStartOffset.QuadPart = startOffset.QuadPart;
			load(tablePos);
			pLeastUsed->pageID = PageID;
			m_Loaded [PageID] = tablePos;
			m_LRUList.ForceAdd(pLeastUsed);
		}
	}
	if (tablePos != m_LastAccessedPage) {
		// manage LRU list, already loaded. So put it front.
		m_LRUList.ForceAdd (m_CacheTable[tablePos].entryLRU);
		m_LastAccessedPage = tablePos;
	}
	return *((Vertex *)(m_CacheTable[tablePos].address + (j.LowPart & m_BlockMaskToOffset)));
}

FORCEINLINE const Photon& OOCFile6464<Photon>::operator[](unsigned int i) {
	OOCSize64 j, startOffset;

	// find pos in table for this i
	j.QuadPart = (__int64)i * sizeof(Photon);

	startOffset.QuadPart = j.QuadPart >> m_BlockSizePowerTwo;	
	unsigned int tablePos = startOffset.LowPart & m_CacheNumMask;

	int PageID = startOffset.QuadPart;
	tablePos = m_Loaded [PageID];
	
	#ifdef WORKING_SET_COMPUTATION
	//if (g_targetModelCacheSimualte) 
	//	g_PhotonSim.Access (i);
	#endif

	if (m_Loaded [PageID] == UNLOADED)
	{
		if (m_CurNumCachedPage < m_NumCachedPage)
		{
			tablePos = m_CurNumCachedPage++;
			LRUEntry *newLRUEntry = new LRUEntry;
			m_CacheTable[tablePos].entryLRU = newLRUEntry;
			newLRUEntry->tablePos = tablePos;
			newLRUEntry->pageID = PageID;
			m_CacheTable[tablePos].fileStartOffset.QuadPart = startOffset.QuadPart;
			load(tablePos);
			m_LRUList.ForceAdd(newLRUEntry);
			m_Loaded [PageID] = tablePos;
		}
		else
		{
			LRUEntry *pLeastUsed = m_LRUList.m_pEnd->m_pPrev;
			tablePos = pLeastUsed->tablePos;
			unload(tablePos);
			m_Loaded [pLeastUsed->pageID] = UNLOADED;
			m_CacheTable[tablePos].fileStartOffset.QuadPart = startOffset.QuadPart;
			load(tablePos);
			pLeastUsed->pageID = PageID;
			m_Loaded [PageID] = tablePos;
			m_LRUList.ForceAdd(pLeastUsed);
		}
	}
	if (tablePos != m_LastAccessedPage) {
		// manage LRU list, already loaded. So put it front.
		m_LRUList.ForceAdd (m_CacheTable[tablePos].entryLRU);
		m_LastAccessedPage = tablePos;
	}
	return *((Photon *)(m_CacheTable[tablePos].address + (j.LowPart & m_BlockMaskToOffset)));
}

FORCEINLINE const Triangle& OOCFile6464<Triangle>::operator[](unsigned int i) {
	OOCSize64 j, startOffset;

	// find pos in table for this i
	j.QuadPart = (__int64)i * sizeof(Triangle);

	startOffset.QuadPart = j.QuadPart >> m_BlockSizePowerTwo;	
	unsigned int tablePos = startOffset.LowPart & m_CacheNumMask;

	int PageID = startOffset.QuadPart;
	tablePos = m_Loaded [PageID];
	
	#ifdef WORKING_SET_COMPUTATION
	if (g_targetModelCacheSimualte) 
		g_TriSim.Access (i);
	#endif

	if (m_Loaded [PageID] == UNLOADED)
	{
		if (m_CurNumCachedPage < m_NumCachedPage)
		{
			tablePos = m_CurNumCachedPage++;
			LRUEntry *newLRUEntry = new LRUEntry;
			m_CacheTable[tablePos].entryLRU = newLRUEntry;
			newLRUEntry->tablePos = tablePos;
			newLRUEntry->pageID = PageID;
			m_CacheTable[tablePos].fileStartOffset.QuadPart = startOffset.QuadPart;
			load(tablePos);
			m_LRUList.ForceAdd(newLRUEntry);
			m_Loaded [PageID] = tablePos;
		}
		else
		{
			LRUEntry *pLeastUsed = m_LRUList.m_pEnd->m_pPrev;
			tablePos = pLeastUsed->tablePos;
			unload(tablePos);
			m_Loaded [pLeastUsed->pageID] = UNLOADED;
			m_CacheTable[tablePos].fileStartOffset.QuadPart = startOffset.QuadPart;
			load(tablePos);
			pLeastUsed->pageID = PageID;
			m_Loaded [PageID] = tablePos;
			m_LRUList.ForceAdd(pLeastUsed);
		}
	}
	if (tablePos != m_LastAccessedPage) {
		// manage LRU list, already loaded. So put it front.
		m_LRUList.ForceAdd (m_CacheTable[tablePos].entryLRU);
		m_LastAccessedPage = tablePos;
	}
	return *((Triangle *)(m_CacheTable[tablePos].address + (j.LowPart & m_BlockMaskToOffset)));
}

FORCEINLINE const unsigned int & OOCFile6464<unsigned int >::operator[](unsigned int i) {
	OOCSize64 j, startOffset;

	// find pos in table for this i
	j.QuadPart = (__int64)i * sizeof(unsigned int);

	startOffset.QuadPart = j.QuadPart >> m_BlockSizePowerTwo;	
	unsigned int tablePos = startOffset.LowPart & m_CacheNumMask;

	int PageID = startOffset.QuadPart;
	tablePos = m_Loaded [PageID];
	
	#ifdef WORKING_SET_COMPUTATION
	//if (g_targetModelCacheSimualte) 
		//g_IdxSim.Access (i);
	#endif

	if (m_Loaded [PageID] == UNLOADED)
	{
		if (m_CurNumCachedPage < m_NumCachedPage)
		{
			tablePos = m_CurNumCachedPage++;
			LRUEntry *newLRUEntry = new LRUEntry;
			m_CacheTable[tablePos].entryLRU = newLRUEntry;
			newLRUEntry->tablePos = tablePos;
			newLRUEntry->pageID = PageID;
			m_CacheTable[tablePos].fileStartOffset.QuadPart = startOffset.QuadPart;
			load(tablePos);
			m_LRUList.ForceAdd(newLRUEntry);
			m_Loaded [PageID] = tablePos;
		}
		else
		{
			LRUEntry *pLeastUsed = m_LRUList.m_pEnd->m_pPrev;
			tablePos = pLeastUsed->tablePos;
			unload(tablePos);
			m_Loaded [pLeastUsed->pageID] = UNLOADED;
			m_CacheTable[tablePos].fileStartOffset.QuadPart = startOffset.QuadPart;
			load(tablePos);
			pLeastUsed->pageID = PageID;
			m_Loaded [PageID] = tablePos;
			m_LRUList.ForceAdd(pLeastUsed);
		}
	}
	if (tablePos != m_LastAccessedPage) {
		// manage LRU list, already loaded. So put it front.
		m_LRUList.ForceAdd (m_CacheTable[tablePos].entryLRU);
		m_LastAccessedPage = tablePos;
	}
	return *((unsigned int *)(m_CacheTable[tablePos].address + (j.LowPart & m_BlockMaskToOffset)));
}

FORCEINLINE const BSPArrayTreeNode& OOCFile6464<BSPArrayTreeNode>::operator[](unsigned int i) {
	OOCSize64 j, startOffset;

	// find pos in table for this i
	j.QuadPart = (__int64)i * sizeof(BSPArrayTreeNode);

	startOffset.QuadPart = j.QuadPart >> m_BlockSizePowerTwo;	
	unsigned int tablePos = startOffset.LowPart & m_CacheNumMask;

	int PageID = startOffset.QuadPart;
	tablePos = m_Loaded [PageID];
	
	#ifdef WORKING_SET_COMPUTATION
	if (g_targetModelCacheSimualte) {
		g_BVHSim.Access (i);
	}
	#endif

	if (m_Loaded [PageID] == UNLOADED)
	{
		if (m_CurNumCachedPage < m_NumCachedPage)
		{
			tablePos = m_CurNumCachedPage++;
			LRUEntry *newLRUEntry = new LRUEntry;
			m_CacheTable[tablePos].entryLRU = newLRUEntry;
			newLRUEntry->tablePos = tablePos;
			newLRUEntry->pageID = PageID;
			m_CacheTable[tablePos].fileStartOffset.QuadPart = startOffset.QuadPart;
			load(tablePos);
			m_LRUList.ForceAdd(newLRUEntry);
			m_Loaded [PageID] = tablePos;
		}
		else
		{
			LRUEntry *pLeastUsed = m_LRUList.m_pEnd->m_pPrev;
			tablePos = pLeastUsed->tablePos;
			unload(tablePos);
			m_Loaded [pLeastUsed->pageID] = UNLOADED;
			m_CacheTable[tablePos].fileStartOffset.QuadPart = startOffset.QuadPart;
			load(tablePos);
			pLeastUsed->pageID = PageID;
			m_Loaded [PageID] = tablePos;
			m_LRUList.ForceAdd(pLeastUsed);
		}
	}
	if (tablePos != m_LastAccessedPage) {
		// manage LRU list, already loaded. So put it front.
		m_LRUList.ForceAdd (m_CacheTable[tablePos].entryLRU);
		m_LastAccessedPage = tablePos;
	}
	return *((BSPArrayTreeNode *)(m_CacheTable[tablePos].address + (j.LowPart & m_BlockMaskToOffset)));
}

FORCEINLINE const LODNode& OOCFile6464<LODNode>::operator[](unsigned int i) {
	OOCSize64 j, startOffset;

	// find pos in table for this i
	j.QuadPart = (__int64)i * sizeof(LODNode);

	startOffset.QuadPart = j.QuadPart >> m_BlockSizePowerTwo;	
	unsigned int tablePos = startOffset.LowPart & m_CacheNumMask;

	int PageID = startOffset.QuadPart;
	tablePos = m_Loaded [PageID];
	
	#ifdef WORKING_SET_COMPUTATION
	//if (g_targetModelCacheSimualte) 
	//	g_LODSim.Access (i);
	#endif

	if (m_Loaded [PageID] == UNLOADED)
	{
		if (m_CurNumCachedPage < m_NumCachedPage)
		{
			tablePos = m_CurNumCachedPage++;
			LRUEntry *newLRUEntry = new LRUEntry;
			m_CacheTable[tablePos].entryLRU = newLRUEntry;
			newLRUEntry->tablePos = tablePos;
			newLRUEntry->pageID = PageID;
			m_CacheTable[tablePos].fileStartOffset.QuadPart = startOffset.QuadPart;
			load(tablePos);
			m_LRUList.ForceAdd(newLRUEntry);
			m_Loaded [PageID] = tablePos;
		}
		else
		{
			LRUEntry *pLeastUsed = m_LRUList.m_pEnd->m_pPrev;
			tablePos = pLeastUsed->tablePos;
			unload(tablePos);
			m_Loaded [pLeastUsed->pageID] = UNLOADED;
			m_CacheTable[tablePos].fileStartOffset.QuadPart = startOffset.QuadPart;
			load(tablePos);
			pLeastUsed->pageID = PageID;
			m_Loaded [PageID] = tablePos;
			m_LRUList.ForceAdd(pLeastUsed);
		}
	}
	if (tablePos != m_LastAccessedPage) {
		// manage LRU list, already loaded. So put it front.
		m_LRUList.ForceAdd (m_CacheTable[tablePos].entryLRU);
		m_LastAccessedPage = tablePos;
	}
	return *((LODNode *)(m_CacheTable[tablePos].address + (j.LowPart & m_BlockMaskToOffset)));
}
#else
template <class T>
FORCEINLINE const T& OOCFile6464<T>::operator[](unsigned int i) {
	OOCSize64 j;

	// find pos in table for this i
	j.QuadPart = (__int64)i * sizeof(T);

	return *((T *)(data + j.QuadPart));
}
#endif // USE_SEPARATE_TYPE
// unload the specified cache entry
#undef OOCFILE_PROFILE
#undef OOCFILE_DEBUG

#endif