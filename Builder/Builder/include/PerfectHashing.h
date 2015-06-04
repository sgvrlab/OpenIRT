#pragma once

#include <vector>
#include <limits.h>
#include <math.h>

class Position
{
public:
	int x, y, z;

	Position() : x(INT_MAX), y(INT_MAX), z(INT_MAX) {}
	Position(int x, int y, int z) : x(x), y(y), z(z) {}

	int& operator [] (int i) {return i == 0 ? x : (i == 1 ? y : z);}
	Position operator % (int m) {return Position(x%m, y%m, z%m);}
	Position operator + (const Position& p) {return Position(x+p.x, y+p.y, z+p.z);}

	bool isValid() {x != INT_MAX;}
};

template <class T>
class HashElem
{
public:
	Position pos;
	T data;

	HashElem(const Position &pos, const T& data) : pos(pos), data(data) {}

	bool isValid() {return pos.isValid();}
};

template <class T>
class PerfectHashing
{
protected:
	std::vector<HashElem<T> > m_data;	// sparse data
	T ***m_hashTable;
	char ***m_hashValidTable;
	Position ***m_offsetTable;

	int m_dimU;		// dimension of original data
	int m_dimM;		// dimension of hash table
	int m_dimR;		// dimension of offset table

	int pgcd(int a, int b){
		return b ?  pgcd(b,a%b) : a;
	}

	void allocHashTable();
	void clearHashTable();
	void allocOffsetTable();
	void clearOffsetTable();

	bool isValidHash(const Position &pos) {return m_hashValidTable[pos.x%m_dimM][pos.y%m_dimM][pos.z%m_dimM] != 0;}

public:
	PerfectHashing(void);
	~PerfectHashing(void);

	void clear() {m_data.clear();}
	void push(const HashElem<T>& elem)
	{
		m_dimU = max(max(max(m_dimU, elem.pos.x), elem.pos.y), elem.pos.z);
		m_data.push_back(elem);
	}

	bool tryHashing();
	void hashing();

	T& get(int x, int y, int z);

	bool test();
};

template <class T>
PerfectHashing<T>::PerfectHashing(void)
	: m_hashTable(0), m_offsetTable(0), m_dimU(0), m_dimM(0), m_dimR(0)
{
}

template <class T>
PerfectHashing<T>::~PerfectHashing(void)
{
	clearHashTable();
	clearOffsetTable();
}

template <class T>
void PerfectHashing<T>::allocHashTable()
{
	clearHashTable();
	m_hashTable = new T**[m_dimM];
	m_hashValidTable = new char**[m_dimM];
	for(int i=0;i<m_dimM;i++)
	{
		m_hashTable[i] = new T*[m_dimM];
		m_hashValidTable[i] = new char*[m_dimM];
		for(int j=0;j<m_dimM;j++)
		{
			m_hashTable[i][j] = new T[m_dimM];
			m_hashValidTable[i][j] = new char[m_dimM];
			memset(m_hashValidTable[i][j], 0, m_dimM);
		}
	}
}

template <class T>
void PerfectHashing<T>::clearHashTable()
{
	if(m_hashTable)
	{
		for(int i=0;i<m_dimM;i++)
		{
			for(int j=0;j<m_dimM;j++)
			{
				delete[] m_hashTable[i][j];
				delete[] m_hashValidTable[i][j];
			}
			delete[] m_hashTable[i];
			delete[] m_hashValidTable[i];
		}
		delete[] m_hashTable;
		delete[] m_hashValidTable;
	}
	m_hashTable = 0;
}

template <class T>
void PerfectHashing<T>::allocOffsetTable()
{
	clearOffsetTable();
	m_offsetTable = new Position**[m_dimR];
	for(int i=0;i<m_dimR;i++)
	{
		m_offsetTable[i] = new Position*[m_dimR];
		for(int j=0;j<m_dimR;j++)
		{
			m_offsetTable[i][j] = new Position[m_dimR];
		}
	}
}

template <class T>
void PerfectHashing<T>::clearOffsetTable()
{
	if(m_offsetTable)
	{
		for(int i=0;i<m_dimR;i++)
		{
			for(int j=0;j<m_dimR;j++)
			{
				delete[] m_offsetTable[i][j];
			}
			delete[] m_offsetTable[i];
		}
		delete[] m_offsetTable;
	}
	m_offsetTable = 0;
}

template <class T>
bool PerfectHashing<T>::tryHashing()
{
	allocOffsetTable();

	Position hashPos, offsetPos;
	for(size_t i=0;i<m_data.size();i++)
	{
		HashElem<T> &elem = m_data[i];

		hashPos = elem.pos % m_dimM;
		offsetPos = elem.pos % m_dimR;

		//if(hashPos.x == 0 && hashPos.y == 2 && hash

		Position &offset = m_offsetTable[offsetPos.x][offsetPos.y][offsetPos.z];

		for(int i0=0;;i0++)
		{
			for(int i1=0;i1<=i0;i1++)
			{
				for(int i2=0;i2<=i0-i1;i2++)
				{
					int i3 = i0-i1-i2;
					offset = Position(i1, i2, i3);
					if(!isValidHash(hashPos + offset)) goto POSITION_FOUND;
				}
			}
		}
		POSITION_FOUND:

		hashPos = (hashPos + offset) % m_dimM;
		m_hashTable[hashPos.x][hashPos.y][hashPos.z] = elem.data;
		m_hashValidTable[hashPos.x][hashPos.y][hashPos.z] = 1;
	}

	return true;
}

template <class T>
void PerfectHashing<T>::hashing()
{
	//Hashing table declaration
	m_dimM = (int)ceil(pow((double)m_data.size(), 1/3.0));
	allocHashTable();

	//Initialization of r
	m_dimR = (int)ceil(pow(m_data.size()/8.0, 1/3.0));
	while(pgcd(m_dimR, m_dimM) != 1)
		m_dimR++;

	while(!tryHashing())
	{
		m_dimR = (int)ceil(m_dimR*1.2);
		while (pgcd(m_dimR, m_dimM) != 1)
			m_dimR++;
	}
}

template <class T>
T& PerfectHashing<T>::get(int x, int y, int z)
{
	const Position &offset = m_offsetTable[x%m_dimR][y%m_dimR][z%m_dimR];
	return m_hashTable[(x+offset.x)%m_dimM][(y+offset.y)%m_dimM][(z+offset.z)%m_dimM];
}

template <class T>
bool PerfectHashing<T>::test()
{
	for(size_t i=0;i<m_data.size();i++)
	{
		const HashElem<T> &elem = m_data[i];
		if(elem.data != get(elem.pos.x, elem.pos.y, elem.pos.z)) 
			return false;
	}
	return true;
}
