
#define _CRT_SECURE_NO_DEPRECATE

#include "memory_map.h"
#include <sys/types.h>
#include <sys/stat.h>

#ifndef WIN32
#include <unistd.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/mman.h>
#endif

#include <limits.h>
#include <fcntl.h>
#include <errno.h>

//#include "TypeDef.h"
#include "stopwatch.hpp"
#include "fstream"
#include "assert.h"
#include "iostream"
using namespace std;

//#include "CashedBoxFiles.h"
//#include "OutCorner.h"

//#include "Cluster.h"
#include "OutCorner.h"

// Copied from Triangle.h from RT project

//#include "Triangle.h"
//#include "Vector3.h"



class Vector3  {
public:
    
    float e[3];
};

class _Vector4: public Vector3
{
	float m_alpha;			// it's dummy now


};


typedef struct Triangle_t {
	unsigned int p[3];		// vertex indices
	Vector3 n;			    // normal vector (normalized)
	float d;				// d from plane equation
	unsigned char  i1,i2;	// planes to be projected to

#ifdef _USE_TRI_MATERIALS
	unsigned short material;	// Index of material in list
#endif

#ifdef _USE_VERTEX_NORMALS
	Vector3 normals[3];		// Vertex normals
#endif

#ifdef _USE_TEXTURING
	Vector2 uv[3];			// Tex coords for each vertex
#endif

} Triangle, *TrianglePtr;


template class CMemoryMappedFile<Triangle>;
template class CMemoryMappedFile<_Vector4>;

template class CMemoryMappedFile<unsigned char>;
template class CMemoryMappedFile<CGeomVertex>;
template class CMemoryMappedFile<COutTriangle>;






/*
template class CMemoryMappedFile<CBV>;
template class CMemoryMappedFile<CInterBV>;
template class CMemoryMappedFile<CIncoreBV>;
template class CMemoryMappedFile<COBBTri>;
*/


//template class CMemoryMappedFile<COutTriangle>;

__int64 NULL_FILE_SIZE = 0;


#include <xmmintrin.h>
#include "gpu_cache_sim.h"


/*
CGPUCacheSim g_TriMemCache (256 * 1024, 4096 / sizeof (COutTriangle), 1, true);
CGPUCacheSim g_VerMemCache (256 * 1024, 4096 / sizeof (CGeomVertex), 1, true);
*/

template <class T>
CMemoryMappedFile<T>::CMemoryMappedFile (void)
{
//	m_pMappingMode = NULL;

  m_CurrentPointer = 0;
}

#ifdef WIN32
template <class T>
CMemoryMappedFile<T>::CMemoryMappedFile (HANDLE hFile, HANDLE hMapping)
{

	m_hFile = hFile;
	m_hMapping = hMapping;

  m_CurrentPointer = 0;
}
#else
template <class T>
CMemoryMappedFile<T>::CMemoryMappedFile (int FD)
{
	m_FD = FD;

  m_CurrentPointer = 0;
}
#endif



template <class T>
CMemoryMappedFile<T>::CMemoryMappedFile (const char * pFileName, char * pMappingMode, int MappingSize,
	       	__int64 & FileSize)
{
  Init (pFileName, pMappingMode, MappingSize, FileSize);
}

template <class T>
void CMemoryMappedFile<T>::Init (const char * pFileName, char * pMappingMode, int MappingSize,
	       	__int64 & FileSize)
{

  strcpy (m_FileName, pFileName);

  m_CurrentPointer = 0;

	//m_MaxNodeLowerInt = int (double (4*1024*1024) * double (1024)/ double (sizeof (T)));
	//m_MaxIncrement = m_MaxNodeLowerInt * sizeof (T);


	strcpy (m_MappingMode, pMappingMode);

	//m_MappingSize *= 1024 * 1024;
	m_FileSize = FileSize;
	m_MemoryAllGra = 4 * 1024; // 64k

	printf ("Initial Mapping size = %d\n", MappingSize);

#ifdef WIN32
	// Windows option; memory map file.
	{
		SYSTEM_INFO  Info;
		GetSystemInfo (& Info);
		m_MemoryAllGra = Info.dwAllocationGranularity;

		// when we divide this with 2, it should be multiple of the granul.
		// because we move the base with max /2.
		//m_MappingSize = m_MemoryAllGra * 16 * MappingSize;
		m_MappingSize = m_MemoryAllGra * MappingSize;

		DWORD FileMappingMode;
		DWORD FileShareMode = 0;
		DWORD FileOpenMode;

	
		if (pMappingMode == NULL) {
			FileMappingMode = GENERIC_READ;
			FileShareMode = FILE_SHARE_READ;
		}
		else if (strchr (pMappingMode, 'w')) {
			FileMappingMode = GENERIC_WRITE | GENERIC_READ;
			FileShareMode = FILE_SHARE_READ | FILE_SHARE_WRITE;
		}
		else {
			FileMappingMode = GENERIC_READ;
			FileShareMode = FILE_SHARE_READ;
			
		}

		if (pMappingMode == NULL) 
			FileOpenMode = OPEN_EXISTING;
		else if (strchr (pMappingMode, 'c'))
			//FileOpenMode = CREATE_NEW;
      FileOpenMode = CREATE_ALWAYS;
		else
			FileOpenMode = OPEN_EXISTING;

		if (! (m_hFile = CreateFile(
				pFileName, FileMappingMode, FileShareMode, NULL, FileOpenMode,
				FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, NULL))) {
			cerr << "can't open file: " << pFileName << endl;
			//return FALSE;

			DWORD  ErrorCode = GetLastError ();
			LPVOID lpMsgBuf;

			FormatMessage (  FORMAT_MESSAGE_ALLOCATE_BUFFER | 
					FORMAT_MESSAGE_FROM_SYSTEM | 
					FORMAT_MESSAGE_IGNORE_INSERTS, 0, ErrorCode, 0, // Default language
					(LPTSTR) &lpMsgBuf,   0,   NULL );

			printf ("%s\n", lpMsgBuf);

			exit (-1);
		}

		/*
			DWORD  ErrorCode = GetLastError ();
			LPVOID lpMsgBuf;

			FormatMessage (  FORMAT_MESSAGE_ALLOCATE_BUFFER | 
					FORMAT_MESSAGE_FROM_SYSTEM | 
					FORMAT_MESSAGE_IGNORE_INSERTS, 0, ErrorCode, 0, // Default language
					(LPTSTR) &lpMsgBuf,   0,   NULL );

			printf ("%s\n", lpMsgBuf);

			//CloseHandle (g_hOneBigFile);
		*/

		if (m_FileSize == NULL_FILE_SIZE) {	// we need to get file size
			// this have to be called for files that has lower than 4GB size

			// Not supported yet.


			//m_FileSize.m_Low = GetFileSize(file, NULL);
      printf ("Please specify file size\n");
		}
    if (m_MappingSize == 0) {
      assert (m_FileSize < UINT_MAX);
			m_MappingSize = (unsigned int) m_FileSize;
    }
		if (m_FileSize < m_MappingSize)
			m_MappingSize = (unsigned int) m_FileSize;

		printf ("- Mapping granularity = %dK\n", m_MappingSize/1024);


		if (pMappingMode == NULL) {
			FileMappingMode = PAGE_READONLY;
			//FileShareMode = FILE_SHARE_READ;
		}
		else if (strchr (pMappingMode, 'w')) {
			FileMappingMode = PAGE_READWRITE;
			//FileShareMode = FILE_SHARE_READ | FILE_SHARE_WRITE;
		}
		else {
			FileMappingMode = PAGE_READONLY;
			//FileShareMode = FILE_SHARE_READ;
			
		}

    DWORD StartPos [2];
    StartPos [0] = DWORD (m_FileSize >> 32);
    StartPos [1] = DWORD (m_FileSize & UINT_MAX);

		if (!(m_hMapping = CreateFileMapping (m_hFile, NULL, FileMappingMode, 
			StartPos [0], StartPos [1], NULL))) {
			cerr << "CreateFileMapping() failed" << endl;

			DWORD  ErrorCode = GetLastError ();
			LPVOID lpMsgBuf;

			FormatMessage (  FORMAT_MESSAGE_ALLOCATE_BUFFER | 
					FORMAT_MESSAGE_FROM_SYSTEM | 
					FORMAT_MESSAGE_IGNORE_INSERTS, 0, ErrorCode, 0, // Default language
					(LPTSTR) &lpMsgBuf,   0,   NULL );

			printf ("%s\n", lpMsgBuf);

			//CloseHandle (g_hOneBigFile);
			//return FALSE;
			exit (-1);
		} 

		DWORD MappingMode;
		if (pMappingMode == NULL) {
			MappingMode = FILE_MAP_READ;
			//printf ("mapping mode is read only\n");
		}
		else if (strchr (pMappingMode, 'w')) {
			MappingMode = FILE_MAP_WRITE;
			//printf ("mapping mode is read and write\n");
		}
		else {
			MappingMode = FILE_MAP_READ;
			//printf ("mapping mode is read only\n");
		}

			 
		if (!(m_pFileData = (char *) MapViewOfFile (m_hMapping, MappingMode, 0, 0, m_MappingSize))) {
			//if (! (g_pFileData = (char *) MapViewOfFile (mapping, FILE_MAP_READ, 0, 0, 1024*1024*600))) {
			cerr << "MapViewOfFile() failed" << endl;
			DWORD  ErrorCode = GetLastError ();
			LPVOID lpMsgBuf;

			FormatMessage (  FORMAT_MESSAGE_ALLOCATE_BUFFER | 
					FORMAT_MESSAGE_FROM_SYSTEM | 
					FORMAT_MESSAGE_IGNORE_INSERTS, 0, ErrorCode, 0, // Default language
					(LPTSTR) &lpMsgBuf,   0,   NULL );

			printf ("%s\n", lpMsgBuf);

			CloseHandle (m_hFile);
			CloseHandle (m_hMapping);
			//return NULL;
			exit (-1);
		}
	}
#else
	// Unix option; memory map file.
	{
		//m_MappingSize = m_MemoryAllGra * 16 * MappingSize;
		m_MappingSize = m_MemoryAllGra * 1 * MappingSize;

		if ((m_FD = open (pFileName, O_RDONLY)) == -1) {
			fprintf(stderr, "can't open '%s': %s\n", pFileName, strerror(errno));
			exit (-1);
		}

		if (m_FileSize == NULL_FILE_SIZE) {	// we need to get file size
			// this have to be called for files that has lower than 4GB size
			struct stat Buf;
			if (fstat (m_FD, & Buf) == -1) {
				fprintf(stderr, "can't get fstat '%s': %s\n", pFileName, strerror(errno));
				exit (-1);
			}

			m_FileSize.m_Low = Buf.st_size;
		}
		if (m_MappingSize == 0)
			m_MappingSize = m_FileSize.m_Low;

		printf ("%d %d\n", m_MappingSize, m_FileSize.m_Low);
		if (m_FileSize.m_High == 0 && 
		    m_FileSize.m_Low < m_MappingSize)
			m_MappingSize = m_FileSize.m_Low;

		printf ("Mapping granularity = %dK\n", m_MappingSize/1024);

		//if ((m_pFileData = (char *) mmap (NULL, m_MappingSize + 100, PROT_READ, MAP_SHARED, m_FD, 0)) == MAP_FAILED) {
		if ((m_pFileData = (char *) mmap (NULL, m_MappingSize, PROT_READ, MAP_SHARED, m_FD, 0)) == MAP_FAILED) {
			fprintf(stderr, "mmap() failed: %s\n", strerror(errno));
			close (m_FD);
		}
	}
#endif
	//*/

	// initialize the current mapping range
	m_StartLoc = 0;
	m_EndLoc = m_MappingSize;
	m_MaxMappingSize = m_MappingSize;
}

template <class T>
CMemoryMappedFile<T>::~CMemoryMappedFile (void)
{
  Flush ();

  #ifdef WIN32
    if (m_pFileData)
	    if (!UnmapViewOfFile (m_pFileData))
		    printf ("Something is wrong in unmapping\n");

	  CloseHandle (m_hMapping);
	  CloseHandle (m_hFile);
  #else
	  if (munmap (m_pFileData, m_MappingSize))
		  printf ("Something is wrong in unmapping\n");

	  close (m_FD);
  #endif


}

// If there is modified stuff, write it into the file
template <class T>
bool CMemoryMappedFile<T>::FlushPage (char * pFileData)
{
  #ifdef WIN32

    if (strchr (m_MappingMode, 'w'))
      if (!FlushViewOfFile (pFileData, 0)) {
        printf("Could not flush memory (%s)to disk.\n", m_FileName);
        DWORD  ErrorCode = GetLastError ();
			  LPVOID lpMsgBuf;

			  FormatMessage (  FORMAT_MESSAGE_ALLOCATE_BUFFER | 
					  FORMAT_MESSAGE_FROM_SYSTEM | 
					  FORMAT_MESSAGE_IGNORE_INSERTS, 0, ErrorCode, 0, // Default language
					  (LPTSTR) &lpMsgBuf,   0,   NULL );

			  printf ("%s\n", lpMsgBuf);
      }

  #else

    printf ("Not yet implemented!\n");
    exit (-1);

  #endif

   return true;
}

template <class T>
bool CMemoryMappedFile<T>::Flush (void)
{
  if (m_pFileData)
    FlushPage (m_pFileData);

  return true;
}

int g_Garbage = 0;
int g_NumDiffVertex = 0;
int g_NumDiffTri = 0;
int g_NumVertexFault = 0;
int g_NumTriFault = 0;

#include <set>
/*
template <class T>
T & CMemoryMappedFile<T>::operator [] (int i)
{
	static set <int, ltstr_Vertex> DiffVertex;
	struct rusage Usage, After;
	getrusage (RUSAGE_SELF, & Usage);


	if (sizeof (T) == sizeof (CGeomVertex)) {
		DiffVertex.insert (i);
		g_NumDiffVertex = DiffVertex.size ();
	}
	else {
		DiffVertex.insert (i);
		g_NumDiffTri = DiffVertex.size ();
	}


	// ASSUMP: whole file is mapped
	long Idx = i * sizeof (T);		
	T & Data = (T &) * (m_pFileData + Idx);

#if T == CGeomVertex
		int temp = (int) Data.m_c + int (Data.x);
		g_Garbage += temp;
		getrusage (RUSAGE_SELF, & After);
		g_NumVertexFault += (After.ru_majflt - Usage.ru_majflt);
#else
	//else {
		int temp = Data.m_c [0].m_v + Data.m_c [1].m_v + Data.m_c [2].m_v;
		g_Garbage += temp;
		getrusage (RUSAGE_SELF, & After);
		g_NumTriFault += (After.ru_majflt - Usage.ru_majflt);

//	}
#endif
	return Data;
}
*/

#ifndef NO_TEST
struct ltstr_Vertex
{
        bool operator()(int V1, int V2) const
        {
                return (V1 < V2);
        }
};

// hashmap to detect unique vertex
struct eqIndex
{
        bool operator()(int p1, int p2) const
        {
                if (p1 ==  p2)
                        return 1;
                return 0;
        }
};

struct PFTable {
        int m_Num;
        int m_NumFault;
        int m_SumFault;
};

typedef hash_map <int, PFTable, hash <int>, eqIndex> CPFHashMap;




CPFHashMap g_PFHashMap;
#include <set>
#endif

int g_NumEvents = 0;
int * g_pPdf = NULL;


const CGeomVertex & CMemoryMappedFile<CGeomVertex>::operator [] (int i)
{
	long Idx = i * sizeof (CGeomVertex);		

	// remapping
	//CPairMem PairIdx (0, Idx);
  __int64 PairIdx = (__int64) Idx;
	RemapFileBase (PairIdx);
	long CurrOffset = ComputeOffset (PairIdx);

	CGeomVertex & Data = (CGeomVertex &) * (m_pFileData + CurrOffset);

  /*
  #ifdef WORKING_SET
    g_VerMemCache.Access (i);
  #endif
  */
	return Data;
}

const COutTriangle & CMemoryMappedFile<COutTriangle>::operator [] (int i)
{
	long Idx = i * sizeof (COutTriangle);		

	// remapping
	//CPairMem PairIdx (0, Idx);
   __int64 PairIdx = (__int64) Idx;
	RemapFileBase (PairIdx);
	long CurrOffset = ComputeOffset (PairIdx);

	COutTriangle & Data = (COutTriangle &) * (m_pFileData + CurrOffset);

  /*
  #ifdef WORKING_SET
    g_TriMemCache.Access (i);
  #endif
  */
	return Data;
}
  

template <class T>
const T & CMemoryMappedFile<T>::operator [] (int i)
{
	long Idx = i * sizeof (T);		

	// remapping
	//CPairMem PairIdx (0, Idx);
   __int64 PairIdx = (__int64) Idx;
	RemapFileBase (PairIdx);
	long CurrOffset = ComputeOffset (PairIdx);

	T & Data = (T &) * (m_pFileData + CurrOffset);

	return Data;
}




CGeomVertex & CMemoryMappedFile<CGeomVertex>::GetReference (int i)
{
	long Idx = i * sizeof (CGeomVertex);		
	
	// remapping
	//CPairMem PairIdx (0, Idx);
   __int64 PairIdx = (__int64) Idx;
	RemapFileBase (PairIdx);

	long CurrOffset = ComputeOffset (PairIdx);

	char * pPos = m_pFileData + CurrOffset;
	CGeomVertex & Data = (CGeomVertex &) * (pPos);

  /*
  #ifdef WORKING_SET
    g_VerMemCache.Access (i);
  #endif
  */
  return Data;
}

COutTriangle & CMemoryMappedFile<COutTriangle>::GetReference (int i)
{
	long Idx = i * sizeof (COutTriangle);		
	
	// remapping
	//CPairMem PairIdx (0, Idx);
   __int64 PairIdx = (__int64) Idx;
	RemapFileBase (PairIdx);

	long CurrOffset = ComputeOffset (PairIdx);

	char * pPos = m_pFileData + CurrOffset;
	COutTriangle & Data = (COutTriangle &) * (pPos);

  /*
  #ifdef WORKING_SET
    g_TriMemCache.Access (i);
  #endif
  */
  return Data;
}
 

template <class T>
T & CMemoryMappedFile<T>::GetReference (int i)
{
	long Idx = i * sizeof (T);		
	
	// remapping
	//CPairMem PairIdx (0, Idx);
   __int64 PairIdx = (__int64) Idx;
	RemapFileBase (PairIdx);

	long CurrOffset = ComputeOffset (PairIdx);

	char * pPos = m_pFileData + CurrOffset;
	T & Data = (T &) * (pPos);

  return Data;
}

template <class T>
const T & CMemoryMappedFile<T>::GetNextElement (void)
{
	long Idx = m_CurrentPointer * sizeof (T);		

	// remapping
	//CPairMem PairIdx (0, Idx);
   __int64 PairIdx = (__int64) Idx;
	RemapFileBase (PairIdx);

	long CurrOffset = ComputeOffset (PairIdx);

	char * pPos = m_pFileData + CurrOffset;
	T & Data = (T &) * (pPos);

  m_CurrentPointer++;

  return Data;
}

template <class T>
T & CMemoryMappedFile<T>::RefNextElement (void)
{
	long Idx = m_CurrentPointer * sizeof (T);		

	// remapping
	//CPairMem PairIdx (0, Idx);
   __int64 PairIdx = (__int64) Idx;
	RemapFileBase (PairIdx);

	long CurrOffset = ComputeOffset (PairIdx);

	char * pPos = m_pFileData + CurrOffset;
	T & Data = (T &) * (pPos);

  m_CurrentPointer++;

  return Data;
}


// NOTE: input idx should be extended into CPairMem
template <class T>
inline 
void CMemoryMappedFile<T>::RemapFileBase (__int64 & FileOffset)
{
#ifndef NO_TEST
	Stopwatch RemapT ("Remapping the basemap");
	RemapT.Start ();
#endif

	__int64 EndOffset = FileOffset + sizeof (T);
	bool Changed = false;

	if (FileOffset < m_StartLoc || EndOffset > m_EndLoc) {
	  //	printf ("remap\n");
		m_StartLoc = FileOffset;
		
		// align
		m_StartLoc = (m_StartLoc / m_MemoryAllGra) * m_MemoryAllGra;
		m_EndLoc = m_StartLoc + m_MaxMappingSize;

		Changed = true;
	}

 

	if (FileOffset < m_StartLoc ||
	    EndOffset > m_EndLoc) {
		printf ("Out of range of mapping\n");
		printf ("mapping Start %I64d\n", m_StartLoc);
		printf ("access Start %I64d, End %I64d\n", FileOffset, EndOffset);
		exit (-1);
	}


	// remap by moving base forward with half of max_size
	if (! Changed)
		return;  

	//printf ("Try to unmap the file\n");
	//if (!UnmapViewOfFile (g_pFileData))
	//if (munmap (m_pFileData, m_MappingSize + 100))

#ifdef WIN32
	if (!UnmapViewOfFile (m_pFileData))
		printf ("Something is wrong in unmapping\n");
#else

	if (munmap (m_pFileData, m_MappingSize))
		printf ("Something is wrong in unmapping\n");
#endif


	m_MappingSize = ComputeMappingSize (m_StartLoc);

	if (m_StartLoc % (m_MemoryAllGra) != 0) {
		printf ("granuality is wrong in unmapping\n");
		exit (-1);
	}



	//printf ("Try to remap the file\n");
	//while (! (m_pFileData = (char *) MapViewOfFile (g_hOneBigMapping, FILE_MAP_READ, 
	//	NewBaseMap.m_High, NewBaseMap.m_Low, MappingSize))) {
	//while ((m_pFileData = (char *) mmap (NULL, m_MappingSize + 100, PROT_READ, MAP_SHARED, m_FD, m_StartLoc.m_Low)) == MAP_FAILED) {
  #ifdef WIN32
		DWORD MappingMode;
		if (strchr (m_MappingMode, 'w')) {
			MappingMode = FILE_MAP_WRITE;
			//printf ("mapping mode is read and write\n");
		}
		else {
			MappingMode = FILE_MAP_READ;
			//printf ("mapping mode is read only\n");
		}

    DWORD StartPos [2];
    StartPos [0] = DWORD (m_StartLoc >> 32);
    StartPos [1] = DWORD (m_StartLoc & UINT_MAX);

		while (! (m_pFileData = (char *) MapViewOfFile (m_hMapping, MappingMode, 
		               StartPos [0], StartPos [1], m_MappingSize))) {
#else
	while ((m_pFileData = (char *) mmap (NULL, m_MappingSize, PROT_READ, MAP_SHARED, m_FD, 
		m_StartLoc.m_Low)) == MAP_FAILED) {
#endif
		m_MappingSize /= 2;
		m_EndLoc = m_StartLoc + m_MappingSize;
		printf ("Fail New mapping %d\n", m_MappingSize / 2 /2);
		printf ("Start %I64d\n", m_StartLoc);
		/*
		CPairMem EndAddress = NewBaseMap + MappingSize;
		printf ("Start Address %d, %d\n", NewBaseMap.m_High, NewBaseMap.m_Low);
		printf ("End Address   %d, %d\n", EndAddress.m_High, EndAddress.m_Low);
		printf ("File end      %d, %d\n", g_OneBigFileSize.m_High, g_OneBigFileSize.m_Low);

		cerr << "MapViewOfFile() failed" << endl;
		DWORD  ErrorCode = GetLastError ();
		LPVOID lpMsgBuf;

		FormatMessage (  FORMAT_MESSAGE_ALLOCATE_BUFFER | 
		FORMAT_MESSAGE_FROM_SYSTEM | 
		FORMAT_MESSAGE_IGNORE_INSERTS, 0, ErrorCode, 0, // Default language
							(LPTSTR) &lpMsgBuf,   0,   NULL );
		
		printf ("%s\n", lpMsgBuf);
		*/

		//CloseHandle (g_hOneBigFile);
		//CloseHandle (g_hOneBigMapping);
	}

#ifndef NO_TEST
	RemapT.Stop ();
#endif
	//cout << " ------------ " <<  RemapT << endl;

	/*
			CPairMem EndAddress = NewBaseMap + MappingSize;

	printf ("Start Address %d, %d\n", NewBaseMap.m_High, NewBaseMap.m_Low);
		printf ("End Address %d, %d\n", EndAddress.m_High, EndAddress.m_Low);
		printf ("File end %d, %d\n", g_OneBigFileSize.m_High, g_OneBigFileSize.m_Low);

	*/

}

template <class T>
long CMemoryMappedFile<T>::ComputeMappingSize (__int64 & BaseAddress)
{
	__int64 Span = m_FileSize - BaseAddress;

  if (Span < m_MaxMappingSize) {
			m_EndLoc = BaseAddress + Span;
			return (long) Span;
	}
	else
		return m_MaxMappingSize;
}


template <class T>
long CMemoryMappedFile<T>::ComputeOffset (__int64 & FileOffset)
{
	assert (FileOffset >= m_StartLoc);

	long Offset = FileOffset - m_StartLoc;
	return Offset;
}

template <class T>
char * CMemoryMappedFile<T>::LoadPage (__int64 & StartPos, int MappingSize, __int64 & FileSize, char * pMappingMode)
{
	m_FileSize = FileSize;
  strcpy (m_MappingMode, pMappingMode);
  char * pFileData = NULL;

  #ifdef WIN32
	// Windows option; memory map file.
	{
		// when we divide this with 2, it should be multiple of the granul.
		// because we move the base with max /2.
		//m_MappingSize = m_MemoryAllGra * 16 * MappingSize;
		m_MappingSize = MappingSize;
		__int64 EndPos = StartPos + m_MappingSize;

		if (EndPos > m_FileSize) {
			__int64 Size = m_FileSize - StartPos;
			m_MappingSize = (long) Size;
		}


		//printf ("- Mapping granularity = %dK\n", m_MappingSize/1024);

	  DWORD MappingMode;
		if (strchr (pMappingMode, 'w')) {
			MappingMode = FILE_MAP_WRITE;
			//printf ("mapping mode is read and write\n");
		}
		else {
			MappingMode = FILE_MAP_READ;
			//printf ("mapping mode is read only\n");
		}

    DWORD HIGH_StartPos, LOW_StartPos;
    HIGH_StartPos = DWORD (StartPos >> 32);
    LOW_StartPos = DWORD (StartPos & UINT_MAX);

    if (LOW_StartPos % m_MemoryAllGra != 0) {
      printf ("Starting position in Load is not aligned (%d).\n", m_MemoryAllGra);
      exit (-1);
    }

		if (!(pFileData = (char *) MapViewOfFile (m_hMapping, MappingMode, HIGH_StartPos, LOW_StartPos, m_MappingSize))) {
			cerr << "MapViewOfFile() failed in Load" << endl;
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
	// Unix option; memory map file.
	{
		//m_MappingSize = m_MemoryAllGra * 16 * MappingSize;
		m_MappingSize = m_MemoryAllGra * 1 * MappingSize;

		if ((m_FD = open (pFileName, O_RDONLY)) == -1) {
			fprintf(stderr, "can't open '%s': %s\n", pFileName, strerror(errno));
			exit (-1);
		}

		if (m_FileSize == NULL_FILE_SIZE) {	// we need to get file size
			// this have to be called for files that has lower than 4GB size
			struct stat Buf;
			if (fstat (m_FD, & Buf) == -1) {
				fprintf(stderr, "can't get fstat '%s': %s\n", pFileName, strerror(errno));
				exit (-1);
			}

			m_FileSize.m_Low = Buf.st_size;
		}
		if (m_MappingSize == 0)
			m_MappingSize = m_FileSize.m_Low;

		printf ("%d %d\n", m_MappingSize, m_FileSize.m_Low);
		if (m_FileSize.m_High == 0 && 
		    m_FileSize.m_Low < m_MappingSize)
			m_MappingSize = m_FileSize.m_Low;

		printf ("Mapping granularity = %dK\n", m_MappingSize/1024);

		//if ((m_pFileData = (char *) mmap (NULL, m_MappingSize + 100, PROT_READ, MAP_SHARED, m_FD, 0)) == MAP_FAILED) {
		if ((pFileData = (char *) mmap (NULL, m_MappingSize, PROT_READ, MAP_SHARED, m_FD, 0)) == MAP_FAILED) {
			fprintf(stderr, "mmap() failed: %s\n", strerror(errno));
			close (m_FD);
		}
	}
#endif
	//*/

  return pFileData;

 
}

template <class T>
void CMemoryMappedFile<T>::UnloadMapPage (char * pFileData)
{

  FlushPage (pFileData);

  #ifdef WIN32
	  if (!UnmapViewOfFile (pFileData))
		  printf ("Something is wrong in unmapping (%s)\n", m_FileName);
  #else
	  if (munmap (pFileData, m_MappingSize))
		  printf ("Something is wrong in unmapping (%s)\n", m_FileName);
  #endif

}

template <class T>
void CMemoryMappedFile<T>::Unload (void)
{
  UnloadMapPage (m_pFileData);
}


template <class T>
bool CMemoryMappedFile<T>::ReserveMemorySpace (int Start, int End)
{

  return true;
}
