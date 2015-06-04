#ifndef COMMON_OOCFILE64_H
#define COMMON_OOCFILE64_H

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
class OOCFile64 {

public:
	OOCFile64(const char * pFileName, int maxAllowedMem, int blockSize, 
		char * pMappingMode = NULL, int MaxNumElements = 0);
	FORCEINLINE const T &operator[](unsigned int i); 
	FORCEINLINE T & GetRef (unsigned int i); 
	~OOCFile64();

	OOCSize64 m_fileSize;

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

	/**
    * @fn IsPowerOfTwo(int n)
    * @brief Returns true if /param n is an integer power of 2.
    * 
    * Taken from Steve Baker's Cute Code Collection. 
    * http://www.sjbaker.org/steve/software/cute_code.html
    */ 
    static bool IsPowerOfTwo(int n) { return ((n&(n-1))==0); }

	// handles for memory-mapped file
#ifdef WIN32
	HANDLE m_hFile, m_hMapping;
#else
	int m_FD;
#endif

	char * m_pMappingMode;
	DWORD m_FileMappingMode;
	DWORD m_FileShareMode;
	DWORD m_FileOpenMode;


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

	// cache entry in table:
	typedef struct CacheEntry_t {
		char *address;
		OOCSize64 fileStartOffset;
	} CacheEntry;

	// main table of cache entries
	CacheEntry *m_CacheTable;

	// for profiling
	unsigned int cacheAccesses;
	unsigned int cacheMisses;

};

#include "TreeLayout.h"

// constructor
// MaxNumElements: maximum number of nodes of T when we need to create the file
template <class T>
OOCFile64<T>::OOCFile64(const char * pFileName, int maxAllowedMem, int blockSize, 
						char * pMappingMode, int MaxNumElements)
{
	SYSTEM_INFO systemInfo;
	BY_HANDLE_FILE_INFORMATION fileInfo;
	LogManager *log = LogManager::getSingletonPtr();
	char output[200];

	m_MaxAllowedMem = maxAllowedMem;

	// get windows allocation granularity:
	GetSystemInfo(&systemInfo);
	m_OSMappingGranularity = systemInfo.dwAllocationGranularity;


	if (pMappingMode == NULL)
		m_pMappingMode = NULL;
	else {
		m_pMappingMode = new char [10];
		strcpy (m_pMappingMode, pMappingMode);
	}

	if (pMappingMode == NULL) {
		m_FileMappingMode = GENERIC_READ;
		m_FileShareMode = FILE_SHARE_READ;
	}
	else if (strchr (pMappingMode, 'w')) {
		m_FileMappingMode = GENERIC_WRITE | GENERIC_READ;
		m_FileShareMode = FILE_SHARE_READ | FILE_SHARE_WRITE;
	}
	else {
		m_FileMappingMode = GENERIC_READ;
		m_FileShareMode = FILE_SHARE_READ;
	}

	if (pMappingMode == NULL) 
		m_FileOpenMode = OPEN_EXISTING;
	else if (strchr (pMappingMode, 'c'))
		m_FileOpenMode = CREATE_NEW;
	else
		m_FileOpenMode = OPEN_EXISTING;

	// open file:
	if (! (m_hFile = CreateFile(pFileName, m_FileMappingMode, m_FileShareMode, NULL, m_FileOpenMode,
		FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, NULL))) {

			cerr << "OOCFile64: Cannot open file: " << pFileName << endl;			
			outputWindowsErrorMessage();
			exit (-1);
		}

		// get file size:
		GetFileInformationByHandle(m_hFile, &fileInfo);
		m_fileSize.LowPart = fileInfo.nFileSizeLow; 
		m_fileSize.HighPart = fileInfo.nFileSizeHigh;

		if (pMappingMode && strchr (pMappingMode, 'c')) {
			// compute filesize
			m_fileSize.QuadPart = (__int64) MaxNumElements * (__int64) sizeof (T);

		}


		DWORD FileMappingMode;
		if (pMappingMode == NULL)
			FileMappingMode = PAGE_READONLY;
		else if (strchr (pMappingMode, 'w')) 
			FileMappingMode = PAGE_READWRITE;
		else
			FileMappingMode = PAGE_READONLY;

		if (!(m_hMapping = CreateFileMapping(m_hFile, NULL, FileMappingMode, 
			m_fileSize.HighPart, m_fileSize.LowPart, NULL))) {
				cerr << "OOCFile64<" << sizeof(T) << ">: CreateFileMapping() failed" << endl;
				outputWindowsErrorMessage();
				exit (-1);
			}

			//
			// determine block size:	
			//	

			m_BlockSize = blockSize; 

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

			if (!IsPowerOfTwo (m_NumCacheEntries)) {
				sprintf(output, "OOCFile: num of cache entries need to be power of two (current num = %d)", 
					m_NumCacheEntries);
				log->logMessage(LOG_ERROR, output);
			}

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

			if (pMappingMode == NULL)
				m_FileMappingMode = FILE_MAP_READ;
			else if (strchr (pMappingMode, 'w')) 
				m_FileMappingMode = FILE_MAP_WRITE;
			else
				m_FileMappingMode = FILE_MAP_READ;

				
}

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

	if (m_pMappingMode)
		delete [] m_pMappingMode;

#ifdef OOCFILE_PROFILE
	dumpCacheStats();
#endif
}

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

// main access operator, i is array offset (i.e. depends on sizeof(T))!
FORCEINLINE const CNodeLayout& OOCFile64<CNodeLayout>::operator[](unsigned int i) {
	OOCSize64 j, startOffset;

	// find pos in table for this i
#if HIERARCHY_TYPE == TYPE_BVH
	j.QuadPart = (__int64)(i)  * (__int64)sizeof(CNodeLayout);
#endif
	startOffset.QuadPart = (j.QuadPart >> (__int64)m_BlockSizePowerTwo);
	unsigned int tablePos = startOffset.LowPart & m_CacheNumMask;

#ifdef OOCFILE_DEBUG
	assert(j.QuadPart < m_fileSize.QuadPart);
	assert(j.QuadPart % sizeof(CNodeLayout) == 0);

	//cout << "access(" << i << "): tablePos=" << tablePos << ", startOffset=" << startOffset << " in Table: " << m_CacheTable[tablePos].fileStartOffset <<  endl;
#endif

#ifdef OOCFILE_PROFILE
	cacheAccesses++;
#endif

	// check if cache entry valid for this i:
	if (m_CacheTable[tablePos].address != NULL && m_CacheTable[tablePos].fileStartOffset.QuadPart == startOffset.QuadPart) {
		// yes: return pointer
		return *((CNodeLayout *)(m_CacheTable[tablePos].address + (j.LowPart & m_BlockMaskToOffset)));
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

		return *((CNodeLayout *)(m_CacheTable[tablePos].address + (j.LowPart & m_BlockMaskToOffset)));
	}	
}

// main access operator, i is array offset (i.e. depends on sizeof(T))!

FORCEINLINE CNodeLayout& OOCFile64<CNodeLayout>::GetRef (unsigned int i) {
	OOCSize64 j, startOffset;

	// find pos in table for this i
#if HIERARCHY_TYPE == TYPE_BVH
	j.QuadPart = (__int64)(i)  * (__int64)sizeof(CNodeLayout);
#endif


	startOffset.QuadPart = (j.QuadPart >> (__int64)m_BlockSizePowerTwo);
	unsigned int tablePos = startOffset.LowPart & m_CacheNumMask;

#ifdef OOCFILE_DEBUG
	assert(j.QuadPart < m_fileSize.QuadPart);
	assert(j.QuadPart % sizeof(CNodeLayout) == 0);

	//cout << "access(" << i << "): tablePos=" << tablePos << ", startOffset=" << startOffset << " in Table: " << m_CacheTable[tablePos].fileStartOffset <<  endl;
#endif

#ifdef OOCFILE_PROFILE
	cacheAccesses++;
#endif

	// check if cache entry valid for this i:
	if (m_CacheTable[tablePos].address != NULL && m_CacheTable[tablePos].fileStartOffset.QuadPart == startOffset.QuadPart) {
		// yes: return pointer
		return *((CNodeLayout *)(m_CacheTable[tablePos].address + (j.LowPart & m_BlockMaskToOffset)));
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

		return *((CNodeLayout *)(m_CacheTable[tablePos].address + (j.LowPart & m_BlockMaskToOffset)));
	}	
}

// main access operator, i is array offset (i.e. depends on sizeof(T))!
template <class T>
FORCEINLINE T& OOCFile64<T>::GetRef (unsigned int i) {
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



FORCEINLINE const BSPArrayTreeNode& OOCFile64<BSPArrayTreeNode>::operator[](unsigned int i) {
	OOCSize64 j, startOffset;
	
	// find pos in table for this i
	#ifdef FOUR_BYTE_FOR_KD_NODE
		j.QuadPart = (__int64)i << (__int64)4;
	#else
		j.QuadPart = (__int64)i << (__int64)5;
	#endif

	startOffset.QuadPart = j.QuadPart >> m_BlockSizePowerTwo;	
	unsigned int tablePos = startOffset.LowPart & m_CacheNumMask;

#ifdef OOCFILE_DEBUG
	assert(j.QuadPart < m_fileSize.QuadPart);
	assert(j.QuadPart % sizeof(T) == 0);

	cout << "access(" << i << " | " << j.QuadPart << "): tablePos=" << tablePos << ", startOffset=" << startOffset.HighPart << " " << startOffset.LowPart << "(=" << startOffset.QuadPart << ")  in Table: " << m_CacheTable[tablePos].fileStartOffset.HighPart << " " << m_CacheTable[tablePos].fileStartOffset.LowPart <<  endl;
#endif

#ifdef OOCFILE_PROFILE
	cacheAccesses++;
#endif

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
}

FORCEINLINE BSPArrayTreeNode & OOCFile64<BSPArrayTreeNode>::GetRef (unsigned int i) {
	OOCSize64 j, startOffset;
	
	// find pos in table for this i
	#ifdef FOUR_BYTE_FOR_KD_NODE
		j.QuadPart = (__int64)i << (__int64)4;
	#else
		j.QuadPart = (__int64)i << (__int64)5;
	#endif

	startOffset.QuadPart = j.QuadPart >> m_BlockSizePowerTwo;	
	unsigned int tablePos = startOffset.LowPart & m_CacheNumMask;

#ifdef OOCFILE_DEBUG
	assert(j.QuadPart < m_fileSize.QuadPart);
	assert(j.QuadPart % sizeof(T) == 0);

	cout << "access(" << i << " | " << j.QuadPart << "): tablePos=" << tablePos << ", startOffset=" << startOffset.HighPart << " " << startOffset.LowPart << "(=" << startOffset.QuadPart << ")  in Table: " << m_CacheTable[tablePos].fileStartOffset.HighPart << " " << m_CacheTable[tablePos].fileStartOffset.LowPart <<  endl;
#endif

#ifdef OOCFILE_PROFILE
	cacheAccesses++;
#endif

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
}
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

	if (!(m_CacheTable[tablePos].address = (char *)MapViewOfFile(m_hMapping, m_FileMappingMode, startAddr.HighPart, startAddr.LowPart, blockSize))) {
		cerr << "MapViewOfFile(" << (m_CacheTable[tablePos].fileStartOffset.QuadPart << m_BlockSizePowerTwo) << ", " << blockSize << ") failed:" << endl;
		outputWindowsErrorMessage();
	}

	cout << "MapViewOfFile(" << (m_CacheTable[tablePos].fileStartOffset.QuadPart << m_BlockSizePowerTwo) << ", " << blockSize << ") = " << (unsigned int)m_CacheTable[tablePos].address << endl;

#else

	m_CacheTable[tablePos].address = (char *)MapViewOfFile(m_hMapping, m_FileMappingMode, startAddr.HighPart, startAddr.LowPart, blockSize);	

#endif
}

#undef OOCFILE_PROFILE
#undef OOCFILE_DEBUG

#endif