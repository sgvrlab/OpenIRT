/********************************************************************
	created:	2014/02/18
	file path:	Common\include
	file base:	MemoryCleaner
	file ext:	h
	author:		Tae-Joon Kim (tjkim.kaist@gmail.com)
	
	comment:	Simple memory manager.
*********************************************************************/

#pragma once

#include <vector>

class MemoryCleaner
{
public:
	enum MemoryType
	{
		MALLOC,
		NEW,
		NEW_ARRAY
	};

	typedef struct Elem_t
	{
		Elem_t(MemoryType type, void *mem) : type(type), mem(mem) {}

		MemoryType type;
		void *mem;
	} Elem;

	static MemoryCleaner *getSingletonPtr(void)
	{
		static MemoryCleaner cleaner;
		return &cleaner;
	}

	void push(void *mem, MemoryType type = NEW)
	{
		m_list.push_back(Elem(type, mem));
	}

	void ignore(void *mem)
	{
		for(size_t i=0;i<m_list.size();i++)
		{
			if(m_list[i].mem == mem) m_list[i].mem = NULL;
		}
	}

	void clear(void)
	{
		for(size_t i=0;i<m_list.size();i++)
		{
			if(!m_list[i].mem) continue;
			switch(m_list[i].type)
			{
			case MALLOC : free(m_list[i].mem); break;
			case NEW : delete m_list[i].mem; break;
			case NEW_ARRAY : delete[] m_list[i].mem; break;
			}
		}

		m_list.clear();
	}
protected:

	std::vector<Elem> m_list;
};
