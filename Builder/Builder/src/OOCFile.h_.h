#ifndef COMMON_OOCFILE_H
#define COMMON_OOCFILE_H

// debugging mode, tests for errors on functions calls and
// has some asserts to test correctness in debug build
//#define OOCFILE_DEBUG

// measure approximate cache hit rate
//#define OOCFILE_PROFILE

#include "LogManager.h"

typedef unsigned long OOCSize;

template <class T>
class OOCFile {

public:
	OOCFile(const char * pFileName, int maxAllowedMem, int blockSize);
	FORCEINLINE const T &operator[](unsigned int i); 
	~OOCFile();

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

	OOCSize m_fileSize;

	// size of individual mappings:
	OOCSize m_BlockSize;
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
		OOCSize fileStartOffset;
	} CacheEntry;

	// main table of cache entries
	CacheEntry *m_CacheTable;	

	// for profiling
	unsigned int cacheAccesses;
	unsigned int cacheMisses;

};

// constructor
template <class T>
OOCFile<T>::OOCFile(const char * pFileName, int maxAllowedMem, int blockSize) {
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

			cerr << "OOCFile: Cannot open file: " << pFileName << endl;			
			outputWindowsErrorMessage();
			exit (-1);
		}

		// get file size:
		GetFileInformationByHandle(m_hFile, &fileInfo);
		m_fileSize = fileInfo.nFileSizeLow; /// TODO: >4GB files!										

		if (!(m_hMapping = CreateFileMapping(m_hFile, NULL, PAGE_READONLY, 
			0, m_fileSize, NULL))) {
				cerr << "CreateFileMapping() failed" << endl;
				outputWindowsErrorMessage();
				exit (-1);
			}

			//
			// determine block size:	
			//	

			m_BlockSize = blockSize; 

			if (m_BlockSize % sizeof(T) != 0) {
				sprintf(output, "OOCFile: cache block size need to be multiple of structure size! sizeof(T) = %d, block size = %d", sizeof(T), m_BlockSize);
				log->logMessage(LOG_ERROR, output);
			}

			m_BlockSizePowerTwo = log10((double)m_BlockSize) / log10(2.0);
			m_BlockMaskToOffset = (1<<m_BlockSizePowerTwo) - 1;	

			//
			// determine number of blocks:
			//

			// make max mem a multiple of block size:
			m_MaxAllowedMem = (m_MaxAllowedMem >> m_BlockSizePowerTwo) << m_BlockSizePowerTwo;
			m_NumCacheEntries = m_MaxAllowedMem / m_BlockSize;

			sprintf(output, "OOCFile: total:%u bytes (%u KB) in %d entries of %d KB\n", m_MaxAllowedMem, m_MaxAllowedMem/1024, m_NumCacheEntries, m_BlockSize / 1024);
			log->logMessage(LOG_DEBUG, output);

			m_NumCacheEntriesPowerTwo = log10((double)m_NumCacheEntries) / log10(2.0);
			m_CacheNumMask = m_NumCacheEntries - 1;

			//
			// init cache table:
			//
			m_CacheTable = new CacheEntry[m_NumCacheEntries];
			memset(m_CacheTable, 0, sizeof(CacheEntry) * m_NumCacheEntries);


			cacheAccesses = 0;
			cacheMisses = 0;
}

template <class T> 
OOCFile<T>::~OOCFile() {

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
}

// main access operator, i is byte offset!
template <class T>
FORCEINLINE const T& OOCFile<T>::operator[](unsigned int i) {

	// find pos in table for this i
	unsigned int startOffset = (i>>m_BlockSizePowerTwo);
	unsigned int tablePos = startOffset & m_CacheNumMask;

#ifdef OOCFILE_DEBUG
	assert(i < m_fileSize);
	assert(i % sizeof(T) == 0);

	cout << "access(" << i << "): tablePos=" << tablePos << ", startOffset=" << startOffset << " in Table: " << m_CacheTable[tablePos].fileStartOffset <<  endl;
#endif

#ifdef OOCFILE_PROFILE
	cacheAccesses++;
#endif

	// check if cache entry valid for this i:
	if (m_CacheTable[tablePos].address != NULL && m_CacheTable[tablePos].fileStartOffset == startOffset) {
		// yes: return pointer
		return *((T *)(m_CacheTable[tablePos].address + (i & m_BlockMaskToOffset)));
	}
	else {
#ifdef OOCFILE_PROFILE
		cacheMisses++;
#endif

		// no: unmap and map new
		if (m_CacheTable[tablePos].address)
			unload(tablePos);
		m_CacheTable[tablePos].fileStartOffset = startOffset;
		load(tablePos);

		return *((T *)(m_CacheTable[tablePos].address + (i & m_BlockMaskToOffset)));
	}	
}

// specialized access for unsigned int, index is integer and not byte offset!
// (i.e. 4-byte stride)
FORCEINLINE const unsigned int& OOCFile<unsigned int>::operator[](unsigned int i) {

	// find pos in table for this i
	i <<= 2;
	unsigned int startOffset = (i >> m_BlockSizePowerTwo);
	unsigned int tablePos = startOffset & m_CacheNumMask;

#ifdef OOCFILE_PROFILE
	cacheAccesses++;
#endif

	// check if cache entry valid for this i:
	if (m_CacheTable[tablePos].address != NULL && m_CacheTable[tablePos].fileStartOffset == startOffset) {
		// yes: return pointer
		return *((unsigned int *)(m_CacheTable[tablePos].address + (i & m_BlockMaskToOffset)));
	}
	else {
#ifdef OOCFILE_PROFILE
		cacheMisses++;
#endif
		// no: unmap and map new
		if (m_CacheTable[tablePos].address)
			unload(tablePos);
		m_CacheTable[tablePos].fileStartOffset = startOffset;
		load(tablePos);

		return *((unsigned int *)(m_CacheTable[tablePos].address + (i & m_BlockMaskToOffset)));
	}	
}

// specialized access for vector3 with 12-byte size
FORCEINLINE const Vector3& OOCFile<Vector3>::operator[](unsigned int i) {

	// find pos in table for this i
	unsigned int startOffset = i / m_BlockSize;
	unsigned int tablePos = startOffset % m_NumCacheEntries;

#ifdef OOCFILE_DEBUG
	assert(i < m_fileSize);
	assert(i % sizeof(T) == 0);

	cout << "access(" << i << "): tablePos=" << tablePos << ", startOffset=" << startOffset << " in Table: " << m_CacheTable[tablePos].fileStartOffset <<  endl;
#endif

#ifdef OOCFILE_PROFILE
	cacheAccesses++;
#endif

	// check if cache entry valid for this i:
	if (m_CacheTable[tablePos].address != NULL && m_CacheTable[tablePos].fileStartOffset == startOffset) {
		// yes: return pointer
		return *((Vector3 *)(m_CacheTable[tablePos].address + (i % m_BlockSize)));
	}
	else {
#ifdef OOCFILE_PROFILE
		cacheMisses++;
#endif
		// no: unmap and map new
		if (m_CacheTable[tablePos].address)
			unload(tablePos);
		m_CacheTable[tablePos].fileStartOffset = startOffset;
		load(tablePos);

		return *((Vector3 *)(m_CacheTable[tablePos].address + (i % m_BlockSize)));
	}	
}


// unload the specified cache entry
template <class T>
void FORCEINLINE OOCFile<T>::unload(unsigned int tablePos) {

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

template <class T>
void FORCEINLINE OOCFile<T>::load(unsigned int tablePos) {
	unsigned int blockSize;
	unsigned int startAddr = m_CacheTable[tablePos].fileStartOffset << m_BlockSizePowerTwo;

	if (startAddr + m_BlockSize > m_fileSize)
		blockSize = 0;
	else
		blockSize = m_BlockSize;

#ifdef OOCFILE_DEBUG

	if (!(m_CacheTable[tablePos].address = (char *)MapViewOfFile(m_hMapping, FILE_MAP_READ, 0, startAddr, blockSize))) {
		cerr << "MapViewOfFile(" << (m_CacheTable[tablePos].fileStartOffset << m_BlockSizePowerTwo) << ", " << blockSize << ") failed:" << endl;
		outputWindowsErrorMessage();
	}

	//cout << "MapViewOfFile(" << (m_CacheTable[tablePos].fileStartOffset << m_BlockSizePowerTwo) << ", " << blockSize << ") = " << (unsigned int)m_CacheTable[tablePos].address << endl;
#else
	m_CacheTable[tablePos].address = (char *)MapViewOfFile(m_hMapping, FILE_MAP_READ, 0, startAddr, blockSize);	
#endif
}

#undef OOCFILE_PROFILE
#undef OOCFILE_DEBUG

#endif
