#pragma once

template <class T>
class HashElem
{
public:
	int x, y, z;
	T data;
};

template <class T>
class PerfectHashingAPI
{
public:
	virtual void clear() = 0;
	virtual void push(const HashElem& elem)

	virtual HashElem& begin() = 0;
	virtual bool hasNext() = 0;
	virtual HashElem& next() = 0;
};

