#ifndef MEM_NANAGER_RACBVH_H
#define MEM_NANAGER_RACBVH_H

#include "App.h"
#include "VDTActiveList.h"
#include <math.h>
#include "memory_map.h"

// Application specific part
template <class T>
class RACBVH;

template <class T>
extern bool loadCluster(RACBVH<T> *pBVH, unsigned int CN, T* posCluster, long diskClusterOffset, int threadNum);
// read only memory manager based on LRU

template <class T>
class CMemElementRACBVH{
public:
	int m_PageID;           // Page idx in the original data
	int m_CachedPageID;     // idx of cached Page

	T * m_Element;
	int m_NumElement;
	CMemElementRACBVH <T> * m_pNext, * m_pPrev;

	CMemElementRACBVH (void) 
	{
		m_CachedPageID = m_PageID = -1;
		m_Element = NULL;
		m_pNext = m_pPrev = NULL;
	}

	CMemElementRACBVH (int PageID, int CachedPageID, int NumElement)
	{
		m_PageID = PageID;
		m_CachedPageID = CachedPageID;

		m_NumElement = NumElement;

		m_Element = new T [NumElement];

		m_pNext = m_pPrev = NULL;
	}
	~CMemElementRACBVH (void)
	{
		if (m_Element) {
			delete [] m_Element;
			m_Element = NULL;
		}
	}
};

class CMemElementRACBVHCompressedCluster{
public:
	int m_PageID;           // Page idx in the original data
	unsigned char *m_CachedCluster;     // Pointer of cached compressed cluster
	int m_ClusterSize;

	CMemElementRACBVHCompressedCluster * m_pNext, * m_pPrev;

	CMemElementRACBVHCompressedCluster (void) 
	{
		m_CachedCluster = (unsigned char*)-1;
		m_PageID = -1;
		m_pNext = m_pPrev = NULL;
		m_ClusterSize = 0;
	}

	CMemElementRACBVHCompressedCluster (int PageID, int ClusterSize)
	{
		m_PageID = PageID;

		m_ClusterSize = ClusterSize;

		m_CachedCluster = new unsigned char[ClusterSize];

		m_pNext = m_pPrev = NULL;
	}
	~CMemElementRACBVHCompressedCluster (void)
	{
		if (m_CachedCluster) {
			delete [] m_CachedCluster;
			m_CachedCluster = NULL;
		}
	}
};

template <class T>
class CMemManagerRACBVH// : public CMemoryMappedFile <T> 
{
public:
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

	int UNLOADED;

	char m_ObjName [255];       // Manager Name, for each debug
	int m_ObjID;                // Manager ID
	int m_MaxNumPage;           // maximum different Pages in the original data
	int m_NumCachedPage;        // maximum cached Pages in the manager
	int m_CurNumCachedPage;     // current # of cached Pages
	int m_PageSize;             // Page size in terms of element
	int m_LocalIDMask, m_PageLocalBit;  // bit mask and # of bit corresponding to slot size

	int m_LastAccessedPage[NUM_THREADS];

	int * m_Loaded;             // indicate idx of cached Page if loaded
	long * m_DiskClusterOffset;	// disk offsets of compressed clusters

	CMemElementRACBVH <T> ** m_pPages;

	CActiveList <CMemElementRACBVH <T> * > m_LRUList;

	#ifdef USE_DM
	int m_MaxCachedMemCCluster;
	int m_UsedCacedMemCCluster;

	unsigned char **m_LoadedCCluster;

	CMemElementRACBVHCompressedCluster ** m_pPagesCCluster;

	CActiveList <CMemElementRACBVHCompressedCluster *> m_LRUListCCluster;
	#endif

#ifdef _USE_OPENMP
	omp_lock_t *lck;
#endif

	CMemManagerRACBVH (void) 
	{
		m_Loaded = NULL;
		m_DiskClusterOffset = NULL;
		m_pPages = NULL;
		m_pRACBVH = NULL;
		UNLOADED = -1;
	} 

	// PageSize should be power of two for efficiency
	bool Init (char * pName, int NumElement, int NumCachedPage, int PageSize) 
	{
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
		m_CurNumCachedPage = -1; 
		m_PageSize = PageSize;

		int i;
		for(i=0;i<NUM_THREADS;i++)
			m_LastAccessedPage[i] = -1;
		strcpy (m_ObjName, pName);

		m_Loaded = new int [m_MaxNumPage];
		m_DiskClusterOffset = new long [m_MaxNumPage];

		for (i = 0;i < m_MaxNumPage;i++)
		{
			m_Loaded [i] = UNLOADED;
			m_DiskClusterOffset [i] = 0;
		}

		m_pPages = new CMemElementRACBVH <T> * [m_NumCachedPage];
		{ 
			// init LRU list
			CMemElementRACBVH <T> * pStartHead = new CMemElementRACBVH <T>;
			CMemElementRACBVH <T> * pEndHead = new CMemElementRACBVH <T>;

			m_LRUList.InitList (pStartHead, pEndHead);
		}

		fprintf (stderr, "%d (among %d) Pages created (total size = %dK)\n", 
			m_NumCachedPage, m_MaxNumPage,
			PageSize * m_NumCachedPage * sizeof (T) / 1024);

		m_pRACBVH = NULL;

#ifdef _USE_OPENMP
		lck = new omp_lock_t[m_MaxNumPage];
		for(i=0;i<m_MaxNumPage;i++)
		{
			omp_init_lock(&lck[i]);
		}
#endif

		return true;
	}

	~CMemManagerRACBVH (void) {
		if (m_Loaded) {
			delete [] m_Loaded;
			m_Loaded = NULL;
		}

		if (m_DiskClusterOffset) {
			delete [] m_DiskClusterOffset;
			m_DiskClusterOffset = NULL;
		}

		if (m_pPages) {
			delete [] m_pPages;
			m_pPages = NULL;
		}

#ifdef _USE_OPENMP
		int i;
		for(i=0;i<m_MaxNumPage;i++)
		{
			omp_destroy_lock(&lck[i]);
		}
		delete[] lck;
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
		CMemElementRACBVH <T> * pPage = m_pPages [CachedPageID];

		/*
		if (PageID != m_LastAccessedPage) {
			// manage LRU list, already loaded. So put it front.
			m_LRUList.ForceAdd (pPage);
			m_LastAccessedPage = PageID;
		}
		*/

		return true;
	}

	T & operator [] (unsigned int i) 
	{
		int PageID = i >> m_PageLocalBit;
		int LocalID = i & m_LocalIDMask;

		if (m_Loaded [PageID] == UNLOADED) {

#ifdef _USE_OPENMP
			omp_set_lock(&lck[PageID]);
			//cout << "[" << omp_get_thread_num() << "] " << "lock setted (" << lck << ")" << endl;
#endif
			if (m_Loaded [PageID] == UNLOADED) {
				if (m_CurNumCachedPage < m_NumCachedPage) {
					m_CurNumCachedPage++;
					int curNumCachedPage = m_CurNumCachedPage;
					m_pPages [curNumCachedPage] = new CMemElementRACBVH <T> (PageID, curNumCachedPage, m_PageSize);

					// require application specific load job
					Load (m_pPages [curNumCachedPage], PageID);

					m_LRUList.ForceAdd (m_pPages [curNumCachedPage]);

					m_Loaded [PageID] = curNumCachedPage;
				}
				else {
					CMemElementRACBVH <T> * pLeastUsed;
					#ifdef _USE_OPENMP
					#pragma omp critical
					#endif
					{
					pLeastUsed = m_LRUList.m_pEnd->m_pPrev;
					Unload (pLeastUsed);
					m_Loaded [pLeastUsed->m_PageID] = -1;
					}

					m_LRUList.ForceAdd (pLeastUsed);

					// require application specific load job
					// Map.Load (StartPos, m_AccessibleSize, m_FileSize);
					Load (pLeastUsed, PageID);

					m_Loaded [PageID] = pLeastUsed->m_CachedPageID;
				}
			}
#ifdef _USE_OPENMP
			//cout << "[" << omp_get_thread_num() << "] " << "lock unsetted (" << lck << ")" << endl;
			omp_unset_lock(&lck[PageID]);
#endif
		}
		int CachedPageID = m_Loaded [PageID];
		CMemElementRACBVH <T> * pPage = m_pPages [CachedPageID];
#ifdef _USE_OPENMP
		int thread_num = omp_get_thread_num();
#else
		int thread_num = 0;
#endif
		if (PageID != m_LastAccessedPage[thread_num]) {
			// manage LRU list, already loaded. So put it front.
			m_LRUList.ForceAdd (pPage);
			m_LastAccessedPage[thread_num] = PageID;
		}

		return pPage->m_Element [LocalID];
	}

	T & GetReference (unsigned int i) 
	{
		int PageID = i >> m_PageLocalBit;
		int LocalID = i & m_LocalIDMask;

		if (m_Loaded [PageID] == UNLOADED) {
#ifdef _USE_OPENMP
			omp_set_lock(&lck[PageID]);
#endif
			if (m_Loaded [PageID] == UNLOADED) {
				if (m_CurNumCachedPage < m_NumCachedPage) {
					m_CurNumCachedPage++;
					int curNumCachedPage = m_CurNumCachedPage;
					m_pPages [curNumCachedPage] = new CMemElementRACBVH <T> (PageID, curNumCachedPage);

					// require application specific load job
					Load (m_pPages [curNumCachedPage], PageID);

					m_LRUList.ForceAdd (m_pPages [curNumCachedPage]);

					m_Loaded [PageID] = curNumCachedPage;
				}
				else {
					CMemElementRACBVH <T> * pLeastUsed;
					#ifdef _USE_OPENMP
					#pragma omp critical
					#endif
					{
					pLeastUsed = m_LRUList.m_pEnd->m_pPrev;
					Unload (pLeastUsed);
					m_Loaded [pLeastUsed->m_PageID] = -1;
					}

					// require application specific load job
					// Map.Load (StartPos, m_AccessibleSize, m_FileSize);
					Load (pLeastUsed, PageID);

					m_LRUList.ForceAdd (pLeastUsed);			  

					m_Loaded [PageID] = pLeastUsed->m_CachedPageID;
				}
			}
#ifdef _USE_OPENMP
			omp_unset_lock(&lck[PageID]);
#endif
		}

		int CachedPageID = m_Loaded [PageID];
		CMemElementRACBVH <T> * pPage = m_pPages [CachedPageID];

#ifdef _USE_OPENMP
		int thread_num = omp_get_thread_num();
#else
		int thread_num = 0;
#endif
		if (PageID != m_LastAccessedPage[thread_num]) {
			// manage LRU list, already loaded. So put it front.
			m_LRUList.ForceAdd (pPage);
			m_LastAccessedPage[thread_num] = PageID;
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
		CMemElementRACBVH <T> * pPage = m_pPages [CachedPageID];

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
		CMemElementRACBVH <T> * pPage = m_pPages [CachedPageID];

		return pPage->m_Element [LocalID];
	}


	// application specific data and functions
	// TODO, we can do this by inheriting and virtualization
	RACBVH<T> * m_pRACBVH;    // class holding data

	bool Unload (CMemElementRACBVH <T> * pElement) {
		/*
		if (m_ObjID == 0)
		printf ("Obj ID = %d, Unload %d Page\n", m_ObjID, pElement->m_PageID);
		*/
		if (m_pRACBVH);
		else {
			//UnloadMapPage ((char *) pElement->m_Element);
			pElement->m_Element = NULL;
		}

		return true;
	}
	bool Load (CMemElementRACBVH <T> * pElement, int PageID)
	{
		pElement->m_PageID = PageID;

		if(m_pRACBVH)
		{
			int threadNum = 0;
			#ifdef _USE_OPENMP
			threadNum = omp_get_thread_num();
			#endif
			loadCluster(m_pRACBVH, PageID, pElement->m_Element, m_DiskClusterOffset[PageID], threadNum);
		}
		else
		{
			/*
			if (m_UseFileMap) {
			__int64 StartPos;
			StartPos = (__int64) PageID * m_PageSize * sizeof (T);
			//printf ("load %d unit\n", WhichMap);
			pElement->m_Element = (T *) LoadPage (StartPos, m_PageSize * sizeof (T), m_FileSize, m_MappingMode);
			}
			*/
		}

		return true;
	}

	bool Flush (void)
	{
		return true;
	}
};



#endif


