#ifndef COMMON_OOCDATA_H
#define COMMON_OOCDATA_H

#include "LogManager.h"
//#include "common.h"
#include "windows.h"

#include "VDTActiveList.h"

#include "positionquantizer_new.h"

// 64 bit addressing structure
typedef ULARGE_INTEGER OOCSize64;

template <class T>
class OOCData {

public:
	OOCData(const char * pFileName, int maxAllowedMem, int blockSize);
	FORCEINLINE const T &operator[](unsigned int i); 
	~OOCData();
	OOCSize64 m_fileSize;

protected:

	void FORCEINLINE unload(unsigned int tablePos);
	void FORCEINLINE load(unsigned int tablePos);

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

// constructor
template <class T>
OOCData<T>::OOCData(const char * pFileName, int maxAllowedMem, int blockSize) {
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

			cerr << "OOCData: Cannot open file: " << pFileName << endl;			
			exit (-1);
	}

	// get file size:
	GetFileInformationByHandle(m_hFile, &fileInfo);
	m_fileSize.LowPart = fileInfo.nFileSizeLow; 
	m_fileSize.HighPart = fileInfo.nFileSizeHigh;

	if (!(m_hMapping = CreateFileMapping(m_hFile, NULL, PAGE_READONLY, 
		m_fileSize.HighPart, m_fileSize.LowPart, NULL))) {
			cerr << "OOCData<" << sizeof(T) << ">: CreateFileMapping() failed" << endl;
			exit (-1);
	}

	//
	// determine block size:	
	//	

	m_BlockSize = blockSize; 

	if (m_BlockSize % sizeof(T) != 0) {
		sprintf(output, "OOCData: cache block size need to be multiple of structure size! sizeof(T) = %d, block size = %d", sizeof(T), m_BlockSize);
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

	sprintf(output, "OOCData: total:%u bytes (%u KB) in %d entries of %d KB\n", m_MaxAllowedMem, m_MaxAllowedMem/1024, m_NumCacheEntries, m_BlockSize / 1024);
	log->logMessage(LOG_DEBUG, output);

	m_NumCacheEntriesPowerTwo = (unsigned int)(log10((double)m_NumCacheEntries) / log10(2.0));
	m_CacheNumMask = m_NumCacheEntries - 1;

	//
	// init cache table:
	//
	m_CacheTable = new CacheEntry[m_NumCacheEntries];
	memset(m_CacheTable, 0, sizeof(CacheEntry) * m_NumCacheEntries);

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

template <class T> 
OOCData<T>::~OOCData() {

	// unmap all existing cache entries
	for (unsigned int i = 0; i < m_NumCacheEntries; i++) {
		if (m_CacheTable[i].address != NULL)
			unload(i);
	}

	// close file mapping and file:
	CloseHandle (m_hMapping);
	CloseHandle (m_hFile);
#ifdef USE_OOC_FILE_LRU
	if (m_Loaded) {
		delete [] m_Loaded;
		m_Loaded = NULL;
	}
#endif
}

template <class T>
FORCEINLINE const T& OOCData<T>::operator[](unsigned int i) {
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
// unload the specified cache entry
template <class T>
void FORCEINLINE OOCData<T>::unload(unsigned int tablePos) {
	UnmapViewOfFile(m_CacheTable[tablePos].address);
}

// load the specified cache entry into mapped memory
template <class T>
void FORCEINLINE OOCData<T>::load(unsigned int tablePos) {
	unsigned int blockSize;
	OOCSize64 startAddr;
	startAddr.QuadPart = m_CacheTable[tablePos].fileStartOffset.QuadPart << m_BlockSizePowerTwo;

	if (startAddr.QuadPart + m_BlockSize > m_fileSize.QuadPart)
		blockSize = 0;
	else
		blockSize = m_BlockSize;
	m_CacheTable[tablePos].address = (char *)MapViewOfFile(m_hMapping, FILE_MAP_READ, startAddr.HighPart, startAddr.LowPart, blockSize+sizeof(T));	
}

#endif