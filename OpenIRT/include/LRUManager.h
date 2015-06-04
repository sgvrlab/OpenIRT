/********************************************************************
	created:	2012/10/23
	file path:	d:\Projects\Redering\OpenIRT\include
	file base:	LRUManager
	file ext:	h
	author:		Tae-Joon Kim (tjkim.kaist@gmail.com)
	
	comment:	LRU based cache manager. Size of element should be fixed. Clock algorithm is used.
*********************************************************************/

#pragma once

#include <Windows.h>
#include <WinLock.h>

template <class ElemPtr>
class LRUManager
{
public:
	typedef void (*ProcessCacheMissCallBack)(unsigned int idx, ElemPtr address);

// Member variables
protected:
	ElemPtr m_cachedData;

	__int64 m_cacheSize;

	int m_numElems;
	int m_sizeElem;

	int *m_loaded;
	int *m_assigned;
	int m_numUsedCache;

	int *m_clockCount;
	int m_curClock;

	WinLock m_lock;

	ProcessCacheMissCallBack m_processCacheMissCallBack;

// Member functions
public:

	LRUManager(int numElems, int sizeElem, ProcessCacheMissCallBack function, __int64 allowedMem = 256*1024*1024);
	~LRUManager();

	ElemPtr operator[](unsigned int idx);
};