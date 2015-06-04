/********************************************************************
	created:	2014/03/11
	file path:	d:\Projects\Redering\OpenIRT\include
	file base:	PLYLoader
	file ext:	h
	author:		Tae-Joon Kim (tjkim.kaist@gmail.com)
	
	comment:	PLYLoader class, load Polygon file format (.ply)
*********************************************************************/

#pragma once

#include <vector>
#include <map>

#include "Vertex.h"
#include "Face.h"

using namespace std;

namespace irt
{

class PLYLoader
{
public:
	PLYLoader(void);
	~PLYLoader(void);

	void clear(void);

	PLYLoader(const char *fileName);

	bool load(const char *fileName);

	bool hasTextureCoordinates(void);
	bool hasNormals(void);

	void getModelBB(Vector3 &min, Vector3 &max);

	Vertex *getVertex();
	Face *getFaces();

	int getNumVertexs();
	int getNumFaces();

private:
	bool m_hasTextureCoordinates;
	bool m_hasNormals;

	vector<Face> m_faceList;
	vector<Vertex> m_vertList;

	Vector3 m_BBMin, m_BBMax;
};

};