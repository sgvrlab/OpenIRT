/********************************************************************
	created:	2013/11/05
	file path:	d:\Projects\Redering\OpenIRT\include
	file base:	Face
	file ext:	h
	author:		Tae-Joon Kim (tjkim.kaist@gmail.com)
	
	comment:	Face class mostly used for rasterization
*********************************************************************/

#pragma once

namespace irt
{

class Face {
public:
	unsigned char n;		// number of vertex indices
	int *verts;				// vertex indices
};

typedef struct FaceStr_t
{
	int n;
	std::string *f;
} FaceStr;

};