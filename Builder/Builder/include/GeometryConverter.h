#ifndef GEOMETRY_CONVERTER_H
#define GEOMETRY_CONVERTER_H
/********************************************************************
	created:	2011/08/09
	file path:	d:\Projects\Redering\OpenIRT\include
	file base:	GeometryConverter
	file ext:	h
	author:		Tae-Joon Kim (tjkim.kaist@gmail.com)
	
	comment:	Convert triangle and vertex files into files that suitable for GPU
*********************************************************************/

#include "Vertex.h"
#include "Triangle.h"
#include "BV.h"

class GeometryConverter {
public:
	enum TCReturnType
	{
		ERR = 0,
		SUCCESS,
		TYPE_NO_NEED
	};

	enum GeomType
	{
		OLD_OOC = 0,
		NEW_OOC,
		SIMP
	};

	static TCReturnType convertTri(const char *fullFileName, bool useBackup = true);
	static TCReturnType convertVert(const char *fullFIleName, bool useBackup = true);
	static TCReturnType convert(const char *filePath);
	static TCReturnType convert(const char *filePath, GeomType from, GeomType to, int cluster = -1);
	static TCReturnType convertCluster(const char *filePath);
};

#endif