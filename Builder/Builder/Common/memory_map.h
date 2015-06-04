#ifndef _Layout_map_file_
#define _Layout_map_file_

//#include "pair_mem.h"
#include "limits.h"

#ifdef WIN32

#include <windows.h>
#endif

#define NO_TEST

extern __int64 NULL_FILE_SIZE;


template <class T>
class CMemoryMappedFile
{
public :
	//int m_MaxNodeLowerInt;		// the maximum number of node within 4GB
	//unsigned long m_MaxIncrement;	// increment when we have maximum nodes

	int m_Idx;			// just index to see which model use which map.

	long m_MappingSize, m_MaxMappingSize;			// mapping size
	__int64 m_StartLoc, m_EndLoc;		          // current mapping area
	__int64 m_FileSize;			                  // input file size	
	long m_MemoryAllGra;
  char m_FileName [255];
  char * m_pFileData;

  // handles for memory-mapped file

	#ifdef WIN32

	  HANDLE m_hFile, m_hMapping;

	#else

	  int m_FD;

	#endif

  char m_MappingMode [20];
	
  unsigned int m_CurrentPointer;            //  used for GetNextElement

	CMemoryMappedFile (void);
	#ifdef WIN32
	CMemoryMappedFile (HANDLE hFile, HANDLE hMapping);
	#else
	CMemoryMappedFile<T>::CMemoryMappedFile (int FD);
	#endif

	CMemoryMappedFile (const char * pFileName, char * pMappingMode, int MappingSize = 0, 
				__int64 & FileSize = NULL_FILE_SIZE); 
	~CMemoryMappedFile (void);
  

	
  void Init (const char * pFileName, char * pMappingMode, int MappingSize = 0, 
				__int64 & FileSize = NULL_FILE_SIZE);

	inline void RemapFileBase (__int64 & FileOffset);
	long ComputeMappingSize (__int64 & BaseAddress);
	long ComputeOffset (__int64 & FileOffset);
	
	const T & operator [] (int i); 

	T & GetReference (int i);

  bool ReserveMemorySpace (int Start, int End);

	char * LoadPage (__int64 & StartPos, int MappingSize, __int64 & FileSize, char * pMappingMode);
	void Unload (void);
  void UnloadMapPage (char * pFileData );

  bool Flush (void);
  bool FlushPage (char * pFileData);

  // sequential access
  bool SetCurrentPointer (int Pointer) { m_CurrentPointer = Pointer; return true;}
  const T & GetNextElement (void);
  T & RefNextElement (void);
  
};

#endif
