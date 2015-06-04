#include <stdlib.h>
#include <string>
#include <direct.h>
#include "defines.h"
#include "PLYLoader.h"
#include "ply.h"

using namespace irt;

static PlyProperty vertex_props[] = { /* list of property information for a vertex */
	{"x", Float32, Float32, offsetof(Vertex,v) + sizeof(float) * 0, 0, 0, 0, 0},
	{"y", Float32, Float32, offsetof(Vertex,v) + sizeof(float) * 1, 0, 0, 0, 0},
	{"z", Float32, Float32, offsetof(Vertex,v) + sizeof(float) * 2, 0, 0, 0, 0},

	{"nx", Float32, Float32, offsetof(Vertex,n) + sizeof(float) * 0, 0, 0, 0, 0},
	{"ny", Float32, Float32, offsetof(Vertex,n) + sizeof(float) * 1, 0, 0, 0, 0},
	{"nz", Float32, Float32, offsetof(Vertex,n) + sizeof(float) * 2, 0, 0, 0, 0},

	{"s", Float32, Float32, offsetof(Vertex,uv) + sizeof(float) * 0, 0, 0, 0, 0},
	{"t", Float32, Float32, offsetof(Vertex,uv) + sizeof(float) * 1, 0, 0, 0, 0},
};

static PlyProperty face_props[] = { /* list of property information for a face */
	{"vertex_indices", Int32, Int32, offsetof(Face,verts), 1, Uint8, Uint8, offsetof(Face,n)},
};

using namespace irt;

PLYLoader::PLYLoader(void)
	: m_hasTextureCoordinates(0), m_hasNormals(0)
{
	m_BBMin.set(FLT_MAX);
	m_BBMax.set(-FLT_MAX);
}

PLYLoader::~PLYLoader(void)
{
	clear();
}

void PLYLoader::clear(void)
{
	for(size_t i=0;i<m_faceList.size();i++)
	{
		if(m_faceList[i].verts) delete[] m_faceList[i].verts;
	}

	m_faceList.clear();
	m_vertList.clear();

	m_BBMin.set(FLT_MAX);
	m_BBMax.set(-FLT_MAX);
}

PLYLoader::PLYLoader(const char *fileName)
{
	load(fileName);
}

void PLYLoader::getModelBB(Vector3 &min, Vector3 &max)
{
	min = m_BBMin;
	max = m_BBMax;
}

Vertex *PLYLoader::getVertex()
{
	if(m_vertList.size() == 0) return NULL;

	return &m_vertList[0];
}

Face *PLYLoader::getFaces()
{
	if(m_faceList.size() == 0) return NULL;

	return &m_faceList[0];
}

int PLYLoader::getNumVertexs()
{
	return (int)m_vertList.size();
}

int PLYLoader::getNumFaces()
{
	return (int)m_faceList.size();
}

bool PLYLoader::load(const char *fileName)
{
	PlyFile *plyFile = open_ply_for_read((char*)fileName);

	if(!plyFile)
	{
		printf("Failed: PLYLoader::load(%s)\n", fileName);
		return false;
	}

	int elementType = 0;
	int elementLeft = 0;
	int numElems, nprops;
	PlyProperty **plist;

	for(int elem=0;elem<plyFile->num_elem_types;elem++) 
	{
		/* get the description of the first element */
		char* elemName = setup_element_read_ply (plyFile, elementType, &elementLeft);
		plist = get_element_description_ply(plyFile, elemName, &numElems, &nprops);

		/* if we're on vertex elements, read them in */
		if(equal_strings("vertex", elemName)) 
		{
			/* create a vertex list to hold all the vertices */
			m_vertList.clear();
			m_vertList.resize(numElems);

			/* set up for getting vertex elements */
			setup_property_ply(plyFile, &vertex_props[0]);
			setup_property_ply(plyFile, &vertex_props[1]);
			setup_property_ply(plyFile, &vertex_props[2]);

			for(int i=3;i<nprops;i++)
			{
				if(equal_strings("nx", plist[i]->name))
				{
					m_hasNormals = true;
					setup_property_ply(plyFile, &vertex_props[3]);
				}
				if(equal_strings("ny", plist[i]->name))
					setup_property_ply(plyFile, &vertex_props[4]);
				if(equal_strings("nz", plist[i]->name))
					setup_property_ply(plyFile, &vertex_props[5]);
				if(equal_strings("s", plist[i]->name))
				{
					m_hasTextureCoordinates = true;
					setup_property_ply(plyFile, &vertex_props[6]);
				}
				if(equal_strings("t", plist[i]->name))
					setup_property_ply(plyFile, &vertex_props[7]);
			}

			/* grab all the vertex elements */
			for(int i=0;i<numElems;i++)
				get_element_ply(plyFile, (void *)&m_vertList[i]);

		}

		/* if we're on face elements, read them in */
		if(equal_strings("face", elemName)) 
		{
			/* create a list to hold all the face elements */
			m_faceList.clear();
			m_faceList.resize(numElems);

			/* set up for getting face elements */
			setup_property_ply(plyFile, &face_props[0]);

			/* grab all the face elements */
			for(int i=0;i<numElems;i++)
				get_element_ply(plyFile, (void *)&m_faceList[i]);
		}
		for(int i=0;i<nprops;i++)
		{
			if(!plist[i]) continue;
			if(plist[i]->name) free(plist[i]->name);
			free(plist[i]);
		}
		if(nprops > 0 && plist) free(plist);

		elementType++;
	}
	/* close the PLY file */
	close_ply (plyFile);
	free_ply (plyFile);
	return true;
}

bool PLYLoader::hasTextureCoordinates(void)
{
	return m_hasTextureCoordinates;
}

bool PLYLoader::hasNormals(void)
{
	return m_hasNormals;
}
