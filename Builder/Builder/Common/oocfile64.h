#ifndef COMMON_OOCFILE64_H
#define COMMON_OOCFILE64_H

// debugging mode, tests for errors on functions calls and
// has some asserts to test correctness in debug build
//#define OOCFILE_DEBUG

// measure approximate cache hit rate
//#define OOCFILE_PROFILE

#include "LogManager.h"
//#include "common.h"
#include "windows.h"

#define USE_OOC_FILE_LRU

#ifdef USE_OOC_FILE_LRU
#include "VDTActiveList.h"
#endif

#ifdef _USE_QBVH
#include "positionquantizer_new.h"
#endif

// 64 bit addressing structure
typedef ULARGE_INTEGER OOCSize64;

template <class T>
class OOCFile64 {

public:
	OOCFile64(const char * pFileName, int maxAllowedMem, int blockSize);
	FORCEINLINE const T &operator[](unsigned int i); 
	~OOCFile64();
	OOCSize64 m_fileSize;

#ifdef _USE_QBVH
	PositionQuantizerNew pq;
	void initForQBVH(int nbits, float* min, float* max);
#endif

protected:

	void FORCEINLINE unload(unsigned int tablePos);
	void FORCEINLINE load(unsigned int tablePos);

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
	// size of individual mappings:
	unsigned int m_BlockSize;
	// same, as power of two
	unsigned int m_BlockSizePowerTwo; 
	// (2^m_BlockSizePowerTwo - 1) for use as AND mask
	unsigned int m_BlockMaskToOffset;	

	// mapping granularity from OS
	unsigned int m_OSMappingGranularity;

	// num cache entries:
	unsigned int m_NumCacheEntries;
	unsigned int m_NumCacheEntriesPowerTwo;
	unsigned int m_CacheNumMask;

	// maximally used memory (in bytes):
	unsigned int m_MaxAllowedMem;

#ifdef USE_OOC_FILE_LRU
	class LRUEntry {
	public:
		unsigned int tablePos;
		unsigned int pageID;
		LRUEntry* m_pNext, *m_pPrev;

		LRUEntry()
		{
			tablePos = 0;
			m_pNext = m_pPrev = NULL;
		}
	};
#endif

	// cache entry in table:
	typedef struct CacheEntry_t {
		char *address;
		OOCSize64 fileStartOffset;
#ifdef USE_OOC_FILE_LRU
		LRUEntry *entryLRU;
#endif
	} CacheEntry;

	// main table of cache entries
	CacheEntry *m_CacheTable;

	// for profiling
	unsigned int cacheAccesses;
	unsigned int cacheMisses;

#ifdef USE_OOC_FILE_LRU
	// for LRU management
	int m_NumCachedPage;        // maximum cached Pages in the manager
	int m_CurNumCachedPage;     // current # of cached Pages
	int m_LastAccessedPage;
	CActiveList <LRUEntry *> m_LRUList;
	int * m_Loaded;             // indicate idx of cached Page if loaded
	int UNLOADED;
#endif

};

#ifdef WORKING_SET_COMPUTATION
extern CGPUCacheSim g_kdTreeSim;
extern CGPUCacheSim g_kdTreeSimL2;
extern CGPUCacheSim g_kdIdxSim;
extern CGPUCacheSim g_TriSim;
extern CGPUCacheSim g_VerSim;
extern CGPUCacheSim g_LODSim;
#endif

// constructor
template <class T>
OOCFile64<T>::OOCFile64(const char * pFileName, int maxAllowedMem, int blockSize) {
	SYSTEM_INFO systemInfo;
	BY_HANDLE_FILE_INFORMATION fileInfo;
	LogManager *log = LogManager::getSingletonPtr();
	char output[200];

	m_MaxAllowedMem = maxAllowedMem;

	// get windows allocation granularity:
	GetSystemInfo(&systemInfo);
	m_OSMappingGranularity = systemInfo.dwAllocationGranularity;

	// open file:
	if (! (m_hFile = CreateFile(pFileName, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
		FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, NULL))) {

			cerr << "OOCFile64: Cannot open file: " << pFileName << endl;			
			outputWindowsErrorMessage();
			exit (-1);
	}

	// get file size:
	GetFileInformationByHandle(m_hFile, &fileInfo);
	m_fileSize.LowPart = fileInfo.nFileSizeLow; 
	m_fileSize.HighPart = fileInfo.nFileSizeHigh;

	if (!(m_hMapping = CreateFileMapping(m_hFile, NULL, PAGE_READONLY, 
		m_fileSize.HighPart, m_fileSize.LowPart, NULL))) {
			cerr << "OOCFile64<" << sizeof(T) << ">: CreateFileMapping() failed" << endl;
			outputWindowsErrorMessage();
			exit (-1);
	}

	//
	// determine block size:	
	//	

	m_BlockSize = blockSize; 

	#ifdef _USE_QBVH
	m_BlockSize = m_fileSize.QuadPart/1024;
	#endif

	if (m_BlockSize % sizeof(T) != 0) {
		sprintf(output, "OOCFile64: cache block size need to be multiple of structure size! sizeof(T) = %d, block size = %d", sizeof(T), m_BlockSize);
		log->logMessage(LOG_ERROR, output);
	}

	m_BlockSizePowerTwo = (unsigned int)(log10((double)m_BlockSize) / log10(2.0));
	m_BlockMaskToOffset = (1<<m_BlockSizePowerTwo) - 1;	

	//
	// determine number of blocks:
	//

	// make max mem a multiple of block size:
	m_MaxAllowedMem = (m_MaxAllowedMem >> m_BlockSizePowerTwo) << m_BlockSizePowerTwo;
	m_NumCacheEntries = m_MaxAllowedMem / m_BlockSize;

	sprintf(output, "OOCFile64: total:%u bytes (%u KB) in %d entries of %d KB\n", m_MaxAllowedMem, m_MaxAllowedMem/1024, m_NumCacheEntries, m_BlockSize / 1024);
	log->logMessage(LOG_DEBUG, output);

	m_NumCacheEntriesPowerTwo = (unsigned int)(log10((double)m_NumCacheEntries) / log10(2.0));
	m_CacheNumMask = m_NumCacheEntries - 1;

	//
	// init cache table:
	//
	m_CacheTable = new CacheEntry[m_NumCacheEntries];
	memset(m_CacheTable, 0, sizeof(CacheEntry) * m_NumCacheEntries);


	cacheAccesses = 0;
	cacheMisses = 0;

#ifdef USE_OOC_FILE_LRU
	m_CurNumCachedPage = 0;
	m_NumCachedPage = m_NumCacheEntries;
	m_LastAccessedPage = -1;
	// init LRU list
	LRUEntry *pStartHead = new LRUEntry;
	LRUEntry *pEndHead = new LRUEntry;
	m_LRUList.InitList (pStartHead, pEndHead);
	int m_MaxNumPage = int (ceil (float(m_fileSize.QuadPart / m_BlockSize))) + 1;
	m_Loaded = new int [m_MaxNumPage];
	UNLOADED = -1;
	for (int i = 0;i < m_MaxNumPage;i++)
	{
		m_Loaded [i] = UNLOADED;
	}

	printf("Num Page = %d\n", m_MaxNumPage);
#endif

}

#ifdef _USE_QBVH
template <class T> 
void OOCFile64<T>::initForQBVH(int nbits, float* min, float* max)
{
	pq.SetMinMax(min, max);
	pq.SetPrecision(16);
	pq.SetupQuantizer();
}
#endif

template <class T> 
OOCFile64<T>::~OOCFile64() {

	// unmap all existing cache entries
	for (unsigned int i = 0; i < m_NumCacheEntries; i++) {
		if (m_CacheTable[i].address != NULL)
			unload(i);
	}

	// close file mapping and file:
	CloseHandle (m_hMapping);
	CloseHandle (m_hFile);
#ifdef OOCFILE_PROFILE
	dumpCacheStats();
#endif
#ifdef USE_OOC_FILE_LRU
	if (m_Loaded) {
		delete [] m_Loaded;
		m_Loaded = NULL;
	}
#endif
}

/*
// main access operator, i is array offset (i.e. depends on sizeof(T))!
template <class T>
FORCEINLINE const T& OOCFile64<T>::operator[](unsigned int i) {
	OOCSize64 j, startOffset;

	// find pos in table for this i
	j.QuadPart = (__int64)i * (__int64)sizeof(T);

	startOffset.QuadPart = (j.QuadPart >> (__int64)m_BlockSizePowerTwo);
	unsigned int tablePos = startOffset.LowPart & m_CacheNumMask;

#ifdef OOCFILE_DEBUG
	assert(j.QuadPart < m_fileSize.QuadPart);
	assert(j.QuadPart % sizeof(T) == 0);

	//cout << "access(" << i << "): tablePos=" << tablePos << ", startOffset=" << startOffset << " in Table: " << m_CacheTable[tablePos].fileStartOffset <<  endl;
#endif

#ifdef OOCFILE_PROFILE
	cacheAccesses++;
#endif

	// check if cache entry valid for this i:
	if (m_CacheTable[tablePos].address != NULL && m_CacheTable[tablePos].fileStartOffset.QuadPart == startOffset.QuadPart) {
		// yes: return pointer
		return *((T *)(m_CacheTable[tablePos].address + (j.LowPart & m_BlockMaskToOffset)));
	}
	else {
#ifdef OOCFILE_PROFILE
		cacheMisses++;
#endif

		// no: unmap and map new
		if (m_CacheTable[tablePos].address)
			unload(tablePos);
		m_CacheTable[tablePos].fileStartOffset.QuadPart = startOffset.QuadPart;
		load(tablePos);

		return *((T *)(m_CacheTable[tablePos].address + (j.LowPart & m_BlockMaskToOffset)));
	}	
}
*/

#ifdef USE_SEPARATE_TYPE
// main access operator, i is array offset (i.e. depends on sizeof(T))!
FORCEINLINE const _Vector4& OOCFile64<_Vector4>::operator[](unsigned int i) {
	OOCSize64 j, startOffset;

	// find pos in table for this i
	j.QuadPart = (__int64)i * (__int64)sizeof(_Vector4);

	startOffset.QuadPart = (j.QuadPart >> (__int64)m_BlockSizePowerTwo);
	unsigned int tablePos = startOffset.LowPart & m_CacheNumMask;

	#ifdef WORKING_SET_COMPUTATION
	unsigned int Idx = j.QuadPart / sizeof (_Vector4);
	g_TriSim.Access (Idx);
	#endif

#ifdef OOCFILE_DEBUG
	assert(j.QuadPart < m_fileSize.QuadPart);
	assert(j.QuadPart % sizeof(_Vector4) == 0);

	//cout << "access(" << i << "): tablePos=" << tablePos << ", startOffset=" << startOffset << " in Table: " << m_CacheTable[tablePos].fileStartOffset <<  endl;
#endif

#ifdef OOCFILE_PROFILE
	cacheAccesses++;
#endif

#ifdef USE_OOC_FILE_MOD
	// check if cache entry valid for this i:
	if (m_CacheTable[tablePos].address != NULL && m_CacheTable[tablePos].fileStartOffset.QuadPart == startOffset.QuadPart) {
		// yes: return pointer
		return *((_Vector4 *)(m_CacheTable[tablePos].address + (j.LowPart & m_BlockMaskToOffset)));
	}
	else {
#ifdef OOCFILE_PROFILE
		cacheMisses++;
#endif

		// no: unmap and map new
		if (m_CacheTable[tablePos].address)
			unload(tablePos);
		m_CacheTable[tablePos].fileStartOffset.QuadPart = startOffset.QuadPart;
		load(tablePos);

		return *((_Vector4 *)(m_CacheTable[tablePos].address + (j.LowPart & m_BlockMaskToOffset)));
	}	
#endif
#ifdef USE_OOC_FILE_LRU
	int PageID = startOffset.QuadPart;
	tablePos = m_Loaded [PageID];
	if (m_Loaded [PageID] == UNLOADED)
	{
		if (m_CurNumCachedPage < m_NumCachedPage)
		{
			tablePos = m_CurNumCachedPage;
			m_Loaded [PageID] = tablePos;
			LRUEntry *newLRUEntry = new LRUEntry;
			m_CacheTable[tablePos].entryLRU = newLRUEntry;
			newLRUEntry->tablePos = tablePos;
			newLRUEntry->pageID = PageID;
			m_CacheTable[tablePos].fileStartOffset.QuadPart = startOffset.QuadPart;
			load(tablePos);
			m_LRUList.ForceAdd(newLRUEntry);
			m_CurNumCachedPage++;
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
	return *((_Vector4 *)(m_CacheTable[tablePos].address + (j.LowPart & m_BlockMaskToOffset)));
#endif
}

// main access operator, i is array offset (i.e. depends on sizeof(T))!
FORCEINLINE const Triangle& OOCFile64<Triangle>::operator[](unsigned int i) {
	OOCSize64 j, startOffset;

	// find pos in table for this i
	j.QuadPart = (__int64)i * (__int64)sizeof(Triangle);

	startOffset.QuadPart = (j.QuadPart >> (__int64)m_BlockSizePowerTwo);
	unsigned int tablePos = startOffset.LowPart & m_CacheNumMask;

	#ifdef WORKING_SET_COMPUTATION
	unsigned int Idx = j.QuadPart / sizeof (Triangle);
	g_TriSim.Access (Idx);
	#endif

#ifdef OOCFILE_DEBUG
	assert(j.QuadPart < m_fileSize.QuadPart);
	assert(j.QuadPart % sizeof(Triangle) == 0);

	//cout << "access(" << i << "): tablePos=" << tablePos << ", startOffset=" << startOffset << " in Table: " << m_CacheTable[tablePos].fileStartOffset <<  endl;
#endif

#ifdef OOCFILE_PROFILE
	cacheAccesses++;
#endif

#ifdef USE_OOC_FILE_MOD
	// check if cache entry valid for this i:
	if (m_CacheTable[tablePos].address != NULL && m_CacheTable[tablePos].fileStartOffset.QuadPart == startOffset.QuadPart) {
		// yes: return pointer
		return *((Triangle *)(m_CacheTable[tablePos].address + (j.LowPart & m_BlockMaskToOffset)));
	}
	else {
#ifdef OOCFILE_PROFILE
		cacheMisses++;
#endif

		// no: unmap and map new
		if (m_CacheTable[tablePos].address)
			unload(tablePos);
		m_CacheTable[tablePos].fileStartOffset.QuadPart = startOffset.QuadPart;
		load(tablePos);

		return *((Triangle *)(m_CacheTable[tablePos].address + (j.LowPart & m_BlockMaskToOffset)));
	}	
#endif
#ifdef USE_OOC_FILE_LRU
	int PageID = startOffset.QuadPart;
	tablePos = m_Loaded [PageID];
	if (m_Loaded [PageID] == UNLOADED)
	{
		if (m_CurNumCachedPage < m_NumCachedPage)
		{
			tablePos = m_CurNumCachedPage;
			m_Loaded [PageID] = tablePos;
			LRUEntry *newLRUEntry = new LRUEntry;
			m_CacheTable[tablePos].entryLRU = newLRUEntry;
			newLRUEntry->tablePos = tablePos;
			newLRUEntry->pageID = PageID;
			m_CacheTable[tablePos].fileStartOffset.QuadPart = startOffset.QuadPart;
			load(tablePos);
			m_LRUList.ForceAdd(newLRUEntry);
			m_CurNumCachedPage++;
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
#endif
}

// main access operator, i is array offset (i.e. depends on sizeof(T))!
FORCEINLINE const unsigned int & OOCFile64<unsigned int >::operator[](unsigned int i) {
	OOCSize64 j, startOffset;

	// find pos in table for this i
	//j.QuadPart = (__int64)i * (__int64)sizeof(unsigned int );
	j.QuadPart = (__int64)i << 2;

	startOffset.QuadPart = (j.QuadPart >> (__int64)m_BlockSizePowerTwo);
	unsigned int tablePos = startOffset.LowPart & m_CacheNumMask;

	#ifdef WORKING_SET_COMPUTATION
	unsigned int Idx = j.QuadPart / sizeof (unsigned int);
	g_kdIdxSim.Access (Idx);
	#endif


#ifdef OOCFILE_DEBUG
	assert(j.QuadPart < m_fileSize.QuadPart);
	assert(j.QuadPart % sizeof(unsigned int) == 0);

	//cout << "access(" << i << "): tablePos=" << tablePos << ", startOffset=" << startOffset << " in Table: " << m_CacheTable[tablePos].fileStartOffset <<  endl;
#endif

#ifdef OOCFILE_PROFILE
	cacheAccesses++;
#endif

#ifdef USE_OOC_FILE_MOD
	// check if cache entry valid for this i:
	if (m_CacheTable[tablePos].address != NULL && m_CacheTable[tablePos].fileStartOffset.QuadPart == startOffset.QuadPart) {
		// yes: return pointer
		return *((unsigned int *)(m_CacheTable[tablePos].address + (j.LowPart & m_BlockMaskToOffset)));
	}
	else {
#ifdef OOCFILE_PROFILE
		cacheMisses++;
#endif

		// no: unmap and map new
		if (m_CacheTable[tablePos].address)
			unload(tablePos);
		m_CacheTable[tablePos].fileStartOffset.QuadPart = startOffset.QuadPart;
		load(tablePos);

		return *((unsigned int *)(m_CacheTable[tablePos].address + (j.LowPart & m_BlockMaskToOffset)));
	}	
#endif
#ifdef USE_OOC_FILE_LRU
	int PageID = startOffset.QuadPart;
	tablePos = m_Loaded [PageID];
	if (m_Loaded [PageID] == UNLOADED)
	{
		if (m_CurNumCachedPage < m_NumCachedPage)
		{
			tablePos = m_CurNumCachedPage;
			m_Loaded [PageID] = tablePos;
			LRUEntry *newLRUEntry = new LRUEntry;
			m_CacheTable[tablePos].entryLRU = newLRUEntry;
			newLRUEntry->tablePos = tablePos;
			newLRUEntry->pageID = PageID;
			m_CacheTable[tablePos].fileStartOffset.QuadPart = startOffset.QuadPart;
			load(tablePos);
			m_LRUList.ForceAdd(newLRUEntry);
			m_CurNumCachedPage++;
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
#endif
}

#ifndef _USE_QBVH
FORCEINLINE const BSPArrayTreeNode& OOCFile64<BSPArrayTreeNode>::operator[](unsigned int i) {
	OOCSize64 j, startOffset;

	// find pos in table for this i

#if HIERARCHY_TYPE == TYPE_BVH
			j.QuadPart = (__int64)i << (__int64)5;
#else
	#ifdef KDTREENODE_16BYTES
		#ifdef FOUR_BYTE_FOR_KD_NODE
			j.QuadPart = (__int64)i << (__int64)4;
		#else
			j.QuadPart = (__int64)i << (__int64)3;
		#endif
	#else
		j.QuadPart = (__int64)i << (__int64)2;
	#endif

#endif
	startOffset.QuadPart = j.QuadPart >> m_BlockSizePowerTwo;	
	unsigned int tablePos = startOffset.LowPart & m_CacheNumMask;

#ifdef WORKING_SET_COMPUTATION
	unsigned int Idx = j.QuadPart / sizeof (BSPArrayTreeNode);
	g_kdTreeSim.Access (Idx);
	g_kdTreeSimL2.Access (Idx);
#endif

#ifdef OOCFILE_DEBUG
	assert(j.QuadPart < m_fileSize.QuadPart);
	assert(j.QuadPart % sizeof(T) == 0);

	cout << "access(" << i << " | " << j.QuadPart << "): tablePos=" << tablePos << ", startOffset=" << startOffset.HighPart << " " << startOffset.LowPart << "(=" << startOffset.QuadPart << ")  in Table: " << m_CacheTable[tablePos].fileStartOffset.HighPart << " " << m_CacheTable[tablePos].fileStartOffset.LowPart <<  endl;
#endif

#ifdef OOCFILE_PROFILE
	cacheAccesses++;
#endif

#ifdef USE_OOC_FILE_MOD
	// check if cache entry valid for this i:
	if (m_CacheTable[tablePos].address != NULL && m_CacheTable[tablePos].fileStartOffset.QuadPart == startOffset.QuadPart) {
		// yes: return pointer
		return *((BSPArrayTreeNode *)(m_CacheTable[tablePos].address + (j.LowPart & m_BlockMaskToOffset)));
	}
	else {
#ifdef OOCFILE_PROFILE
		cacheMisses++;
#endif

		// no: unmap and map new
		if (m_CacheTable[tablePos].address)
			unload(tablePos);
		m_CacheTable[tablePos].fileStartOffset.QuadPart = startOffset.QuadPart;
		load(tablePos);

		return *((BSPArrayTreeNode *)(m_CacheTable[tablePos].address + (j.LowPart & m_BlockMaskToOffset)));
	}	
#endif
#ifdef USE_OOC_FILE_LRU
	int PageID = startOffset.QuadPart;
	tablePos = m_Loaded [PageID];
	
	if (m_Loaded [PageID] == UNLOADED)
	{
		if (m_CurNumCachedPage < m_NumCachedPage)
		{
			tablePos = m_CurNumCachedPage;
			m_Loaded [PageID] = tablePos;
			LRUEntry *newLRUEntry = new LRUEntry;
			m_CacheTable[tablePos].entryLRU = newLRUEntry;
			newLRUEntry->tablePos = tablePos;
			newLRUEntry->pageID = PageID;
			m_CacheTable[tablePos].fileStartOffset.QuadPart = startOffset.QuadPart;
			load(tablePos);
			m_LRUList.ForceAdd(newLRUEntry);
			m_CurNumCachedPage++;
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
#endif
}
#endif

#ifdef _USE_QBVH
extern BSPArrayTreeNode tempNode;
FORCEINLINE const BSPArrayTreeNode& OOCFile64<BSPArrayTreeNode>::operator[](unsigned int i) {
	OOCSize64 j, startOffset;

	// find pos in table for this i

	j.QuadPart = (__int64)i * (__int64)5;

	startOffset.QuadPart = j.QuadPart >> m_BlockSizePowerTwo;	
	unsigned int tablePos = startOffset.LowPart & m_CacheNumMask;

#ifdef WORKING_SET_COMPUTATION
	unsigned int Idx = j.QuadPart / sizeof (BSPArrayTreeNode);
	g_kdTreeSim.Access (Idx);
	g_kdTreeSimL2.Access (Idx);
#endif

#ifdef OOCFILE_DEBUG
	assert(j.QuadPart < m_fileSize.QuadPart);
	assert(j.QuadPart % sizeof(T) == 0);

	cout << "access(" << i << " | " << j.QuadPart << "): tablePos=" << tablePos << ", startOffset=" << startOffset.HighPart << " " << startOffset.LowPart << "(=" << startOffset.QuadPart << ")  in Table: " << m_CacheTable[tablePos].fileStartOffset.HighPart << " " << m_CacheTable[tablePos].fileStartOffset.LowPart <<  endl;
#endif

#ifdef OOCFILE_PROFILE
	cacheAccesses++;
#endif

#ifdef USE_OOC_FILE_MOD
	// check if cache entry valid for this i:
	if (m_CacheTable[tablePos].address != NULL && m_CacheTable[tablePos].fileStartOffset.QuadPart == startOffset.QuadPart) {
		// yes: return pointer
		return *((BSPArrayTreeNode *)(m_CacheTable[tablePos].address + (j.LowPart & m_BlockMaskToOffset)));
	}
	else {
#ifdef OOCFILE_PROFILE
		cacheMisses++;
#endif

		// no: unmap and map new
		if (m_CacheTable[tablePos].address)
			unload(tablePos);
		m_CacheTable[tablePos].fileStartOffset.QuadPart = startOffset.QuadPart;
		load(tablePos);

		return *((BSPArrayTreeNode *)(m_CacheTable[tablePos].address + (j.LowPart & m_BlockMaskToOffset)));
	}	
#endif
#ifdef USE_OOC_FILE_LRU
	int PageID = startOffset.QuadPart;
	tablePos = m_Loaded [PageID];
	
	if (m_Loaded [PageID] == UNLOADED)
	{
		if (m_CurNumCachedPage < m_NumCachedPage)
		{
			tablePos = m_CurNumCachedPage;
			m_Loaded [PageID] = tablePos;
			LRUEntry *newLRUEntry = new LRUEntry;
			m_CacheTable[tablePos].entryLRU = newLRUEntry;
			newLRUEntry->tablePos = tablePos;
			newLRUEntry->pageID = PageID;
			m_CacheTable[tablePos].fileStartOffset.QuadPart = startOffset.QuadPart;
			load(tablePos);
			m_LRUList.ForceAdd(newLRUEntry);
			m_CurNumCachedPage++;
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
	tempNode = *((BSPArrayTreeNode *)(m_CacheTable[tablePos].address + (j.LowPart & m_BlockMaskToOffset)));
	I32 min[3], max[3];
	unsigned int code;
	memcpy(&code, &tempNode.min[0], sizeof(unsigned int));
	min[0] = code >> 16;
	max[0] = code & 0xFFFF;
	memcpy(&code, &tempNode.min[1], sizeof(unsigned int));
	min[1] = code >> 16;
	max[1] = code & 0xFFFF;
	memcpy(&code, &tempNode.min[2], sizeof(unsigned int));
	min[2] = code >> 16;
	max[2] = code & 0xFFFF;
	pq.DeQuantize(min, tempNode.min.e);
	pq.DeQuantize(max, tempNode.max.e);

	return tempNode;
#endif
}
#endif

#ifdef USE_LOD
FORCEINLINE const LODNode& OOCFile64<LODNode>::operator[](unsigned int i) {
	OOCSize64 j, startOffset;

	// find pos in table for this i
	j.QuadPart = (__int64)i << (__int64)5;

	startOffset.QuadPart = j.QuadPart >> m_BlockSizePowerTwo;	
	unsigned int tablePos = startOffset.LowPart & m_CacheNumMask;

#ifdef WORKING_SET_COMPUTATION
	unsigned int Idx = j.QuadPart / sizeof (LODNode);
	//g_kdTreeSim.Access (Idx);
	//g_kdTreeSimL2.Access (Idx);
#endif

#ifdef OOCFILE_DEBUG
	assert(j.QuadPart < m_fileSize.QuadPart);
	assert(j.QuadPart % sizeof(T) == 0);

	cout << "access(" << i << " | " << j.QuadPart << "): tablePos=" << tablePos << ", startOffset=" << startOffset.HighPart << " " << startOffset.LowPart << "(=" << startOffset.QuadPart << ")  in Table: " << m_CacheTable[tablePos].fileStartOffset.HighPart << " " << m_CacheTable[tablePos].fileStartOffset.LowPart <<  endl;
#endif

#ifdef OOCFILE_PROFILE
	cacheAccesses++;
#endif

#ifdef USE_OOC_FILE_MOD
	// check if cache entry valid for this i:
	if (m_CacheTable[tablePos].address != NULL && m_CacheTable[tablePos].fileStartOffset.QuadPart == startOffset.QuadPart) {
		// yes: return pointer
		return *((LODNode *)(m_CacheTable[tablePos].address + (j.LowPart & m_BlockMaskToOffset)));
	}
	else {
#ifdef OOCFILE_PROFILE
		cacheMisses++;
#endif

		// no: unmap and map new
		if (m_CacheTable[tablePos].address)
			unload(tablePos);
		m_CacheTable[tablePos].fileStartOffset.QuadPart = startOffset.QuadPart;
		load(tablePos);

		return *((LODNode *)(m_CacheTable[tablePos].address + (j.LowPart & m_BlockMaskToOffset)));
	}	
#endif
#ifdef USE_OOC_FILE_LRU
	int PageID = startOffset.QuadPart;
	tablePos = m_Loaded [PageID];
	
	if (m_Loaded [PageID] == UNLOADED)
	{
		if (m_CurNumCachedPage < m_NumCachedPage)
		{
			tablePos = m_CurNumCachedPage;
			m_Loaded [PageID] = tablePos;
			LRUEntry *newLRUEntry = new LRUEntry;
			m_CacheTable[tablePos].entryLRU = newLRUEntry;
			newLRUEntry->tablePos = tablePos;
			newLRUEntry->pageID = PageID;
			m_CacheTable[tablePos].fileStartOffset.QuadPart = startOffset.QuadPart;
			load(tablePos);
			m_LRUList.ForceAdd(newLRUEntry);
			m_CurNumCachedPage++;
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
#endif
}
#endif
#else
template <class T>
FORCEINLINE const T& OOCFile64<T>::operator[](unsigned int i) {
	OOCSize64 j, startOffset;

	// find pos in table for this i
	j.QuadPart = (__int64)i * sizeof(T);

	startOffset.QuadPart = j.QuadPart >> m_BlockSizePowerTwo;	
	unsigned int tablePos = startOffset.LowPart & m_CacheNumMask;

	int PageID = startOffset.QuadPart;
	tablePos = m_Loaded [PageID];
	
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
	return *((T *)(m_CacheTable[tablePos].address + (j.LowPart & m_BlockMaskToOffset)));
}
#endif // USE_SEPARATE_TYPE
// unload the specified cache entry
template <class T>
void FORCEINLINE OOCFile64<T>::unload(unsigned int tablePos) {

#ifdef OOCFILE_DEBUG
	if (!UnmapViewOfFile(m_CacheTable[tablePos].address)) {
		cerr << "UnmapViewOfFile(" << (unsigned int)m_CacheTable[tablePos].address << ") failed:" << endl;
		outputWindowsErrorMessage();
	}
	//cout << "UnmapViewOfFile(" << (unsigned int)m_CacheTable[tablePos].address << ")" << endl;
#else
	UnmapViewOfFile(m_CacheTable[tablePos].address);
#endif
}

// load the specified cache entry into mapped memory
template <class T>
void FORCEINLINE OOCFile64<T>::load(unsigned int tablePos) {
	unsigned int blockSize;
	OOCSize64 startAddr;
	startAddr.QuadPart = m_CacheTable[tablePos].fileStartOffset.QuadPart << m_BlockSizePowerTwo;

	if (startAddr.QuadPart + m_BlockSize > m_fileSize.QuadPart)
		blockSize = 0;
	else
		blockSize = m_BlockSize;

#ifdef OOCFILE_DEBUG

	if (!(m_CacheTable[tablePos].address = (char *)MapViewOfFile(m_hMapping, FILE_MAP_READ, startAddr.HighPart, startAddr.LowPart, blockSize))) {
		cerr << "MapViewOfFile(" << (m_CacheTable[tablePos].fileStartOffset.QuadPart << m_BlockSizePowerTwo) << ", " << blockSize << ") failed:" << endl;
		outputWindowsErrorMessage();
	}

	cout << "MapViewOfFile(" << (m_CacheTable[tablePos].fileStartOffset.QuadPart << m_BlockSizePowerTwo) << ", " << blockSize << ") = " << (unsigned int)m_CacheTable[tablePos].address << endl;

#else

	if(!(m_CacheTable[tablePos].address = (char *)MapViewOfFile(m_hMapping, FILE_MAP_READ, startAddr.HighPart, startAddr.LowPart, blockSize)))
	{
		cerr << "MapViewOfFile(" << (m_CacheTable[tablePos].fileStartOffset.QuadPart << m_BlockSizePowerTwo) << ", " << blockSize << ") failed:" << endl;
		outputWindowsErrorMessage();
	}

#endif
}

#undef OOCFILE_PROFILE
#undef OOCFILE_DEBUG

#endif