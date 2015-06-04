#pragma once

#include "mydefs.h"

class DirectTriIndex
{
public:
	DirectTriIndex() {}
public:
	~DirectTriIndex(void) {}
public:
	int Do(const char* filepath);
	int DoMulti(const char* filepath);
};
