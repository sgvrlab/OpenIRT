#ifndef _MEM_NANAGER_H
#define _MEM_NANAGER_H

#include "App.h"
#include "VDTActiveList.h"
#include <math.h>
#include "memory_map.h"

#include "gpu_cache_sim.h"
extern CGPUCacheSim g_TriMemCache;
extern CGPUCacheSim g_VerMemCache;


// Application specific part
class CCompressedMesh;
extern bool UnloadPageData (CCompressedMesh * pMesh, int WhichPart, int PageIdx);


// read only memory manager based on LRU

template <class T>
class CMemElement{
public:
	int m_PageID;           // Page idx in the original data
	int m_CachedPageID;     // idx of cached Page

	T * m_Element;
	int m_NumElement;
	CMemElement <T> * m_pNext, * m_pPrev;

	CMemElement (void) 
	{
		m_CachedPageID = m_PageID = -1;
		m_Element = NULL;
		m_pNext = m_pPrev = NULL;
	}

	CMemElement (int PageID, int CachedPageID, int NumElement, bool UseFileMap)
	{
		m_PageID = PageID;
		m_CachedPageID = CachedPageID;

		m_NumElement = NumElement;

		if (UseFileMap == false)
			m_Element = new T [NumElement];

		m_pNext = m_pPrev = NULL;
	}
	~CMemElement (void)
	{
		if (m_Element) {
			delete [] m_Element;
			m_Element = NULL;
		}
	}



};

static bool IsPowerOfTwo (unsigned int Src, int & Power)
{
	const int NumTests = 32;
	static const unsigned int powConst[NumTests] = 
	{ 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648U };

	int i;
	for (i = 0;i < NumTests;i++)
		if (Src == powConst [i]) {
			Power = i;
			return true;
		}
		else if (Src < powConst [i]) {
			return false;
		}

		return false;

}

static float _log_2 (float x)
{
	float Result = log (x) / log (float (2));
	return Result;
}

const int UNLOADED = -1;


template <class T>
class CMemManager : public CMemoryMappedFile <T> 
{
public:
	char m_ObjName [255];       // Manager Name, for each debug
	int m_ObjID;                // Manager ID
	int m_MaxNumPage;           // maximum different Pages in the original data
	int m_NumCachedPage;        // maximum cached Pages in the manager
	int m_CurNumCachedPage;     // current # of cached Pages
	int m_PageSize;             // Page size in terms of element
	int m_LocalIDMask, m_PageLocalBit;  // bit mask and # of bit corresponding to slot size

	int m_LastAccessedPage;

	int * m_Loaded;             // indicate idx of cached Page if loaded
	CMemElement <T> ** m_pPages;

	CActiveList <CMemElement <T> * > m_LRUList;
	bool m_UseFileMap;

#ifdef _USE_OPENMP
	omp_lock_t lck;
#endif


	CMemManager (void) 
	{
		m_Loaded = NULL;
		m_pPages = NULL;
		m_pMesh = NULL;
	} 

	// PageSize should be power of two for efficiency
	bool Init (char * pName, int NumElement, int NumCachedPage, int PageSize, 
		bool UseFileMap = false, char * pMappingMode = NULL) 
	{
		m_UseFileMap = UseFileMap;

		bool Result = IsPowerOfTwo (PageSize, m_PageLocalBit);
		if (Result == false) {
			printf ("Page size (%d) is not power of two\n", PageSize);
			exit (-1);
		}

		m_NumCachedPage = NumCachedPage;
		m_MaxNumPage = int (ceil (float (NumElement) / float (PageSize)));
		if (m_MaxNumPage < m_NumCachedPage)
			m_NumCachedPage = m_MaxNumPage;

		m_LocalIDMask = PageSize - 1;
		m_CurNumCachedPage = 0; 
		m_PageSize = PageSize;
		m_LastAccessedPage = -1;
		strcpy (m_ObjName, pName);

		m_Loaded = new int [m_MaxNumPage];
		int i;
		for (i = 0;i < m_MaxNumPage;i++)
			m_Loaded [i] = UNLOADED;

		m_pPages = new CMemElement <T> * [m_NumCachedPage];
		{ 
			// init LRU list
			CMemElement <T> * pStartHead = new CMemElement <T>;
			CMemElement <T> * pEndHead = new CMemElement <T>;

			m_LRUList.InitList (pStartHead, pEndHead);
		}

		// initialize file open if memory mapped enalbed
		if (UseFileMap) {
			strcpy (m_FileName, pName);
			m_FileSize = (__int64) NumElement * sizeof (T);
			strcpy (m_MappingMode, pMappingMode);
			m_pFileData = NULL;

#ifdef WIN32
			{ 
				SYSTEM_INFO  Info;
				GetSystemInfo (& Info);
				m_MemoryAllGra = Info.dwAllocationGranularity;

				DWORD FileMappingMode;
				DWORD FileShareMode = 0;
				DWORD FileOpenMode;

				if (strchr (pMappingMode, 'w')) {
					FileMappingMode = GENERIC_WRITE | GENERIC_READ;
					FileShareMode = FILE_SHARE_READ | FILE_SHARE_WRITE;
				}
				else {
					FileMappingMode = GENERIC_READ;
					FileShareMode = FILE_SHARE_READ;    			
				}

				if (strchr (pMappingMode, 'c'))
					FileOpenMode = CREATE_ALWAYS;
				else
					FileOpenMode = OPEN_EXISTING;

				if (! (m_hFile = CreateFile(
					m_FileName, FileMappingMode, FileShareMode, NULL, FileOpenMode,
					FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, NULL))) {
						cerr << "can't open file: " << m_FileName << endl;
						DWORD  ErrorCode = GetLastError ();
						LPVOID lpMsgBuf;
						FormatMessage (  FORMAT_MESSAGE_ALLOCATE_BUFFER | 
							FORMAT_MESSAGE_FROM_SYSTEM | 
							FORMAT_MESSAGE_IGNORE_INSERTS, 0, ErrorCode, 0, // Default language
							(LPTSTR) &lpMsgBuf,   0,   NULL );
						printf ("%s\n", lpMsgBuf);
						exit (-1);
				}

				if (strchr (pMappingMode, 'w'))
					FileMappingMode = PAGE_READWRITE;
				else
					FileMappingMode = PAGE_READONLY;

				DWORD StartPos [2];
				StartPos [0] = (DWORD) (m_FileSize >> 32);
				StartPos [1] = (DWORD) (m_FileSize & UINT_MAX);

				if (! (m_hMapping = CreateFileMapping (m_hFile, NULL, FileMappingMode, 
					StartPos [0], StartPos [1], NULL))) {
						cerr << "CreateFileMapping() failed" << endl;
						DWORD  ErrorCode = GetLastError ();
						LPVOID lpMsgBuf;
						FormatMessage (  FORMAT_MESSAGE_ALLOCATE_BUFFER | 
							FORMAT_MESSAGE_FROM_SYSTEM | 
							FORMAT_MESSAGE_IGNORE_INSERTS, 0, ErrorCode, 0, // Default language
							(LPTSTR) &lpMsgBuf,   0,   NULL );
						printf ("%s\n", lpMsgBuf);
						exit (-1);
				} 
			}
#else
			// not supported yet.
#endif



		}

		fprintf (stderr, "%d (among %d) Pages created (total size = %dK)\n", 
			m_NumCachedPage, m_MaxNumPage,
			PageSize * m_NumCachedPage * sizeof (T) / 1024);

		m_pMesh = NULL;

#ifdef _USE_OPENMP
		omp_init_lock(&lck);
#endif

		return true;
	}

	~CMemManager (void) {
		if (m_Loaded) {
			delete [] m_Loaded;
			m_Loaded = NULL;
		}

		if (m_pPages) {
			if (m_UseFileMap) {
				int i;
				for (i = 0;i < m_CurNumCachedPage;i++)
					Unload (m_pPages [i]);
			}

			delete [] m_pPages;
			m_pPages = NULL;
		}

#ifdef _USE_OPENMP
		omp_destroy_lock(&lck);
#endif

	}

	const bool IsElementLoaded (unsigned int i)
	{
		int PageID = i >> m_PageLocalBit;

		if (m_Loaded [PageID] == UNLOADED)
			return false;
		return true;
	}

	bool SetPageAccessed (unsigned int PageID)
	{
		assert (PageID < m_MaxNumPage);
		assert (m_Loaded [PageID] != UNLOADED);

		int CachedPageID = m_Loaded [PageID];
		CMemElement <T> * pPage = m_pPages [CachedPageID];

		/*
		if (PageID != m_LastAccessedPage) {
			// manage LRU list, already loaded. So put it front.
			m_LRUList.ForceAdd (pPage);
			m_LastAccessedPage = PageID;
		}
		*/

		return true;
	}

	const T & operator [] (unsigned int i) 
	{
		int PageID = i >> m_PageLocalBit;
		int LocalID = i & m_LocalIDMask;

#ifdef WORKING_SET
		if (sizeof (T) == sizeof (CGeomVertex))
			g_VerMemCache.Access (i);
		else if (sizeof (T) == sizeof (COutTriangle))
			g_TriMemCache.Access (i);
#endif

		if (m_Loaded [PageID] == UNLOADED) {
#ifdef _USE_OPENMP
			omp_set_lock(&lck);
#endif

			if (m_Loaded [PageID] == UNLOADED) {
				if (m_CurNumCachedPage < m_NumCachedPage) {
					m_pPages [m_CurNumCachedPage] = new CMemElement <T> (PageID, m_CurNumCachedPage, m_PageSize, m_UseFileMap);

					// require application specific load job
					Load (m_pPages [m_CurNumCachedPage], PageID);

					m_LRUList.ForceAdd (m_pPages [m_CurNumCachedPage]);

					m_Loaded [PageID] = m_CurNumCachedPage++;
				}
				else {
					CMemElement <T> * pLeastUsed = m_LRUList.m_pEnd->m_pPrev;
					Unload (pLeastUsed);
					m_Loaded [pLeastUsed->m_PageID] = -1;

					// require application specific load job
					// Map.Load (StartPos, m_AccessibleSize, m_FileSize);
					Load (pLeastUsed, PageID);

					m_LRUList.ForceAdd (pLeastUsed);
					m_Loaded [PageID] = pLeastUsed->m_CachedPageID;
				}

			}
#ifdef _USE_OPENMP
			omp_unset_lock(&lck);
#endif
		}
		int CachedPageID = m_Loaded [PageID];
		CMemElement <T> * pPage = m_pPages [CachedPageID];

		if (PageID != m_LastAccessedPage) {
			// manage LRU list, already loaded. So put it front.
			m_LRUList.ForceAdd (pPage);
			m_LastAccessedPage = PageID;
		}

		return pPage->m_Element [LocalID];
	}

	T & GetReference (unsigned int i) 
	{
		int PageID = i >> m_PageLocalBit;
		int LocalID = i & m_LocalIDMask;

#ifdef WORKING_SET
		if (sizeof (T) == sizeof (CGeomVertex))
			g_VerMemCache.Access (i);
		else if (sizeof (T) == sizeof (COutTriangle))
			g_TriMemCache.Access (i);
#endif

		if (m_Loaded [PageID] == UNLOADED) {
#ifdef _USE_OPENMP
			omp_set_lock(&lck);
#endif

			if (m_Loaded [PageID] == UNLOADED) {
				if (m_CurNumCachedPage < m_NumCachedPage) {
					m_pPages [m_CurNumCachedPage] = new CMemElement <T> (PageID, m_CurNumCachedPage, m_PageSize, m_UseFileMap);

					// require application specific load job
					Load (m_pPages [m_CurNumCachedPage], PageID);

					m_LRUList.ForceAdd (m_pPages [m_CurNumCachedPage]);

					m_Loaded [PageID] = m_CurNumCachedPage++;
				}
				else {
					CMemElement <T> * pLeastUsed = m_LRUList.m_pEnd->m_pPrev;
					Unload (pLeastUsed);
					m_Loaded [pLeastUsed->m_PageID] = -1;

					// require application specific load job
					// Map.Load (StartPos, m_AccessibleSize, m_FileSize);
					Load (pLeastUsed, PageID);

					m_LRUList.ForceAdd (pLeastUsed);			  
					m_Loaded [PageID] = pLeastUsed->m_CachedPageID;
				}
			}
#ifdef _USE_OPENMP
			omp_unset_lock(&lck);
#endif

		}

		int CachedPageID = m_Loaded [PageID];
		CMemElement <T> * pPage = m_pPages [CachedPageID];

		if (PageID != m_LastAccessedPage) {
			// manage LRU list: already loaded. So put it front.
			m_LRUList.ForceAdd (pPage);
			m_LastAccessedPage = PageID;
		}

		return pPage->m_Element [LocalID];
	}

	const T & GetConstRefWithoutLRU (unsigned int i) 
	{
		int PageID = i >> m_PageLocalBit;
		int LocalID = i & m_LocalIDMask;


		if (m_Loaded [PageID] == UNLOADED) {
			fprintf (stderr, "GetConstRefWithoutLRU should not be called here\n");
			exit (-1);
		}
		int CachedPageID = m_Loaded [PageID];
		CMemElement <T> * pPage = m_pPages [CachedPageID];

		return pPage->m_Element [LocalID];
	}

	T & GetReferenceWithoutLRU (unsigned int i) {
		int PageID = i >> m_PageLocalBit;
		int LocalID = i & m_LocalIDMask;

		if (m_Loaded [PageID] == UNLOADED) {
			fprintf (stderr, "GetReferenceWithoutLRU should not be called here\n");
			exit (-1);
		}
		int CachedPageID = m_Loaded [PageID];
		CMemElement <T> * pPage = m_pPages [CachedPageID];

		return pPage->m_Element [LocalID];
	}


	// application specific data and functions
	// TODO, we can do this by inheriting and virtualization
	CCompressedMesh * m_pMesh;    // class holding data

	bool Unload (CMemElement <T> * pElement) {

		/*
		if (m_ObjID == 0)
		printf ("Obj ID = %d, Unload %d Page\n", m_ObjID, pElement->m_PageID);
		*/

		if (m_pMesh)
			UnloadPageData (m_pMesh, m_ObjID, pElement->m_PageID);
		else {
			UnloadMapPage ((char *) pElement->m_Element);
			pElement->m_Element = NULL;
		}



		return true;
	}
	bool Load (CMemElement <T> * pElement, int PageID)
	{
		pElement->m_PageID = PageID;

		if (m_UseFileMap) {
			__int64 StartPos;
			StartPos = (__int64) PageID * m_PageSize * sizeof (T);
			//printf ("load %d unit\n", WhichMap);
			pElement->m_Element = (T *) LoadPage (StartPos, m_PageSize * sizeof (T), m_FileSize, m_MappingMode);
		}

		return true;
	}

	bool Flush (void)
	{
		if (m_UseFileMap) {
			int i;
			for (i = 0;i < m_CurNumCachedPage;i++)
				FlushPage ((char *) m_pPages [i]->m_Element);
		}
		return true;
	}
};



#endif


