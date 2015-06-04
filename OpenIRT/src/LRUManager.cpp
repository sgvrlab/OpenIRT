#include "LRUManager.h"
#include <string>

template class LRUManager<unsigned char*>;

template <class ElemPtr>
LRUManager<ElemPtr>::LRUManager(int numElems, int sizeElem, ProcessCacheMissCallBack function, __int64 allowedMem)
{
	m_numElems = numElems;
	m_sizeElem = sizeElem;
	m_processCacheMissCallBack = function;

	m_cacheSize = allowedMem / sizeElem;

	m_cachedData = new unsigned char[(__int64)m_cacheSize*sizeElem];

	m_loaded = new int[numElems];
	memset(m_loaded, -1, sizeof(int)*numElems);

	m_assigned = new int[m_cacheSize];
	memset(m_assigned, -1, sizeof(int)*m_cacheSize);

	m_numUsedCache = 0;

	m_clockCount = new int[m_cacheSize];
	memset(m_clockCount, 0, sizeof(int)*m_cacheSize);
	m_curClock = 0;
}

template <class ElemPtr>
LRUManager<ElemPtr>::~LRUManager()
{
	delete[] m_cachedData;
	delete[] m_loaded;
	delete[] m_assigned;
	delete[] m_clockCount;
}

template <class ElemPtr>
ElemPtr LRUManager<ElemPtr>::operator[](unsigned int idx)
{
	int tablePos = m_loaded[idx];
	if(tablePos < 0)
	{
		// Cache miss
		//printf("%d\t", idx);
		m_lock.lock();
		if(m_loaded[idx] < 0)
		{
			if(m_numUsedCache < m_cacheSize)
			{
				// Cache is not full
				tablePos = m_numUsedCache++;
			}
			else
			{
				// Cache is full
				while(m_clockCount[m_curClock] > 0)
				{
					m_clockCount[m_curClock]--;
					m_curClock = (m_curClock + 1) % m_cacheSize;
				}
				m_loaded[m_assigned[m_curClock]] = -1;
				tablePos = m_curClock;
				m_curClock = (m_curClock + 1) % m_cacheSize;
			}
			m_assigned[tablePos] = idx;
			ElemPtr address = (ElemPtr)(((unsigned char*)m_cachedData) + tablePos * m_sizeElem);
			m_processCacheMissCallBack(idx, address);
			m_loaded[idx] = tablePos;
		}
		m_lock.unlock();
	}
	m_clockCount[tablePos] = 1;
	return (ElemPtr)(((unsigned char*)m_cachedData) + tablePos * m_sizeElem);
}