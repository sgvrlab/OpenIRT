#ifndef CACHE_SIMULATION_H_
#define CACHE_SIMULATION_H_

//#include "App.h"
//#include "Cluster.h"
//#include "TypeDef.h"
#include "VDTActiveList.h"
//#include <ext/hash_map>
//using __gnu_cxx::hash_map; 
//using __gnu_cxx::hash; 
//using namespace std;
#include <hash_map>
using namespace std;

#define LRU	
//#define FIFO	

///*

// hashmap to detect unique vertex
struct eqInt2
{
	bool operator()(int V1, int V2) const
	{
		if (V1 ==  V2)
			return 1;
		return 0;
	}
};

struct greater_strgpusim {
   bool operator()(const char* x, const char* y) const {
      if ( strcmp(x, y) < 0)
         return true;

      return false;
   }
};

typedef stdext::hash_map <int, int> CIntHashMap2;
//*/

// hashmap to detect unique vertex
/*
struct eqIdx
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

typedef hash_map <int, PFTable, hash <int>, eqIdx> CPFHashMap;
*/

class CCacheEle
{
public:
	int m_Idx;
	CCacheEle * m_pPrev, * m_pNext;

	CCacheEle (void);
};

// caches that map to one cache Idx
// constains one elemnts of each set
class CCacheMap
{
	public:
		int m_NumAssociativity;
		CCacheEle * m_Elements;
		CActiveList <CCacheEle *> m_LRUList;		// LRU list

		
		~CCacheMap (void);
		void Setup (int Associativity);
		
		void InitLRUList (void);
};


class CGPUCacheSim
{
public:
	bool m_UseWorkingSet;		// compute working set based granuality
	CIntHashMap2 m_WS;			// working set
	
	int m_NumAssociativity;
	int m_NumMap;
	int m_NumCacheEle;		// m_NumMap * m_NumAssociativity
	
	CCacheMap * m_CacheMap;
	int m_NumAccess, m_NumHit;
	int m_NumTotalElements;
	bool m_Enabled;
	int m_Granul;			// one cache can have several vertices

	// collecting data
	int m_PreIdx;
	int m_MaxEdgeSpan;
	//CIntHMap m_CMRF;		// cache miss ratio given dist
	//CPFHashMap m_CMRF;

	CGPUCacheSim (int CacheSize, int Granul, int Associativity, bool WorkSet = false);
	~CGPUCacheSim (void);

	void InitLRUList (void);
	void InitCache (void);
	void Access (unsigned int Idx);

	float GetCacheHit (void);
	int GetNumFault (void);
	float GetACMR (void);

	void UpdateCMRF (int SrcIdx, bool hit);
	void PrintCMRF (void);

};

#endif
