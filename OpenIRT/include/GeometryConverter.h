#ifndef TRIANGLE_CONVERTER_H
#define TRIANGLE_CONVERTER_H
/********************************************************************
	created:	2011/08/09
	file path:	d:\Projects\Redering\OpenIRT\include
	file base:	GeometryConverter
	file ext:	h
	author:		Tae-Joon Kim (tjkim.kaist@gmail.com)
	
	comment:	Convert triangle file into one that suitable for GPU
*********************************************************************/

#include "Vertex.h"
#include "Triangle.h"

class GeometryConverter {
public:
	enum TCReturnType
	{
		ERR = 0,
		SUCCESS,
		TYPE_NO_NEED
	};

	static TCReturnType convertTri(const char *fullFileName);
	static TCReturnType convertVert(const char *fullFIleName);
};

#endif