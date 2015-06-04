//#include "math.h"


#include "stdafx.h"
#include "gpu_cache_sim.h"


CCacheEle::CCacheEle (void)
{
	m_Idx = -1;
	m_pNext = m_pPrev = NULL;
}


void CCacheMap::Setup (int Associativity)
{
	m_NumAssociativity = Associativity;

	m_Elements = new CCacheEle [m_NumAssociativity];

	CCacheEle * m_pStart = new CCacheEle;
	CCacheEle * m_pEnd = new CCacheEle;

	m_LRUList.InitList (m_pStart, m_pEnd);
}

CCacheMap::~CCacheMap (void)
{

}
		  
void CCacheMap::InitLRUList (void)
{
	m_LRUList.Clear (false);
	for (int i = 0;i < m_NumAssociativity;i++)
	{
		m_Elements [i].m_pNext = NULL;
		m_Elements [i].m_pPrev = NULL;
	}
}



// we only create CacheSize * Associativity cache elements
// assume Granu is block transfer.
CGPUCacheSim::CGPUCacheSim (int CacheSize, int Granul, int Associativity,
							bool WorkingSet)
{

	m_UseWorkingSet = WorkingSet;


	m_NumAssociativity = Associativity;
	m_NumMap = CacheSize;
	//m_Elements = new CCacheEle [MAX_CACHE_ELE];
		
	m_NumCacheEle = CacheSize * Associativity;


	m_CacheMap = new CCacheMap [m_NumMap];
	int i;
	for (i = 0;i < m_NumMap;i++)
		m_CacheMap [i].Setup (Associativity);

	m_NumHit = 0;
	m_NumAccess = 0;
	m_NumTotalElements = 0;
	m_Enabled = false;
	m_Granul = Granul;	


	m_PreIdx = -1;
	m_MaxEdgeSpan = 0;
}
CGPUCacheSim::~CGPUCacheSim (void)
{
	//delete [] m_Elements;
	delete [] m_CacheMap;
}

void CGPUCacheSim::InitLRUList (void)
{
	int i;
	for (i = 0;i < m_NumMap;i++)
		m_CacheMap [i].InitLRUList ();

	/*
	m_LRUList.Clear (false);
	for (int i = 0;i < m_NumCacheEle;i++)
	{
		m_Elements [i].m_pNext = NULL;
		m_Elements [i].m_pPrev = NULL;
	}
	*/

}

void CGPUCacheSim::InitCache (void)
{
	InitLRUList ();
	m_NumHit = 0;
	m_NumAccess = 0;
	m_NumTotalElements = 0;

	if (m_UseWorkingSet)
		m_WS.clear ();
}

void CGPUCacheSim::PrintCMRF (void)
{
	/*
	printf ("Probability\n");

	int i, NumEle = 0;
	for (i = 0;i <= m_MaxEdgeSpan;i++)
	{
		CPFHashMap::iterator Iter = m_CMRF.find (i);

		if (Iter == m_CMRF.end ()) {
			//printf ("%d %d %d %d\n",  i, 0, 0, 0); 
			continue;
		}

		NumEle++;
		PFTable Table = Iter->second;


	
		printf ("%d %d %d\n",  Iter->first, 
				Table.m_Num, Table.m_NumFault);
	}

	
	assert (NumEle == m_CMRF.size ());
	*/

}

void CGPUCacheSim::UpdateCMRF (int SrcIdx, bool hit)
{
	int EdgeSpan = abs (long (SrcIdx - m_PreIdx));
	m_PreIdx = SrcIdx;

	if (EdgeSpan > m_MaxEdgeSpan)
		m_MaxEdgeSpan = EdgeSpan;

	/*
	CPFHashMap::iterator Iter = m_CMRF.find (EdgeSpan);
	if (Iter != m_CMRF.end ()) {
		PFTable & Table = Iter->second;
		assert (Table.m_Num > 0);
		Table.m_Num++;

		if (hit == false) 
			Table.m_NumFault++;	
	}
	else {
		PFTable Table;
		Table.m_Num = 1;
		Table.m_NumFault = 1;	
		
		CPFHashMap::value_type NewMap (EdgeSpan, Table);
		m_CMRF.insert (NewMap);
	}
	*/

}


void CGPUCacheSim::Access (unsigned int Idx)
{

	/*
	if (!m_Enabled)
		return;
	*/

	unsigned int RawIdx = Idx;
	bool IsCacheMiss = false;

	if (m_Granul != 1)
		Idx = unsigned int (Idx / m_Granul); 	// Idx indicates block now

	m_NumAccess++;
	// find the Idx in the List.


	if (m_UseWorkingSet) {
		// maintain working set
		CIntHashMap2::iterator Iter = m_WS.find (Idx);
		if (Iter == m_WS.end ()) {	
			CIntHashMap2::value_type NewElement (Idx, 1);
			m_WS.insert (NewElement);
		}
	}


	int WhichMap = Idx % m_NumMap;

	CCacheMap & Map = m_CacheMap [WhichMap];

	CCacheEle * pNode = Map.m_LRUList.Head ();
	bool Found = false;

	while (pNode->m_pNext != NULL) {
		if (pNode->m_Idx == Idx) {

#ifdef LRU
			Map.m_LRUList.ForceAdd (pNode);
#endif

			Found = true;
			m_NumHit++;

			UpdateCMRF (RawIdx, true);
			return;
		}

		pNode = pNode->m_pNext;
	}


	// cache miss
	UpdateCMRF (RawIdx, false);





	// put Idx into the head
	int SizeEle = Map.m_LRUList.Size ();
	if (SizeEle < m_NumAssociativity) {
		// get element from array and put it in the head
		Map.m_Elements [SizeEle].m_Idx = Idx;

		Map.m_LRUList.Add (& Map.m_Elements [SizeEle]);
	}
	else {
		// get last one and put it in the head.

		pNode = Map.m_LRUList.m_pEnd->m_pPrev;
		pNode->m_Idx = Idx;

		Map.m_LRUList.ForceAdd (pNode);
	}
}

float CGPUCacheSim::GetCacheHit (void)
{
	return float (m_NumHit) / float (m_NumAccess);
}

int CGPUCacheSim::GetNumFault (void)
{
	return m_NumAccess - m_NumHit;
}


float CGPUCacheSim::GetACMR (void)
{
	return float (m_NumAccess - m_NumHit) / float (m_NumTotalElements);
}
