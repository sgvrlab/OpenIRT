#include "CommonOptions.h"
#include "defines.h"
#include "CommonHeaders.h"
#include "handler.h"

#include "OpenGLModel.h"
#include "OpenIRT.h"

using namespace irt;

OpenGLModel::OpenGLModel(void) : 
m_vList(0),
m_vnList(0),
m_vtList(0),
m_indexList(0),
m_vUpdated(0),
m_vnUpdated(0),
m_vtUpdated(0),
m_idxUpdated(0),
m_numIndices(0),
m_attached(0)
{

}

OpenGLModel::~OpenGLModel(void)
{
	Model::unload();
	unload();
}

bool OpenGLModel::load(Vector3 *vList, Vector3 *vnList, Vector2 *vtList, int numVerts, Index_t *indexList, int numIndices, const Material &mat, bool attached)
{
	m_matList.clear();
	m_matList.push_back(mat);

	m_BB.min.set(FLT_MAX);
	m_BB.max.set(-FLT_MAX);

	if(vList)
	{
		for(int i=0;i<numVerts;i++)
			m_BB.update(vList[i]);
	}

	m_numVerts = numVerts;
	m_numIndices = numIndices;
	m_numTris = 0;

	if(attached)
	{
		m_vList = vList;
		m_vnList = vnList;
		m_vtList = vtList;
		m_indexList = indexList;
	}
	else
	{
		if(!m_attached)
		{
			if(m_vList) delete[] m_vList;
			if(m_vnList) delete[] m_vnList;
			if(m_vtList) delete[] m_vtList;
			if(m_indexList) delete[] m_indexList;
		}

		m_vList = new Vector3[numVerts];
		m_vnList = new Vector3[numVerts];
		m_vtList = new Vector2[numVerts];
		m_indexList = new Index_t[m_numIndices];

		if(vList) memcpy_s(m_vList, sizeof(Vector3)*numVerts, vList, sizeof(Vector3)*numVerts);
		if(vnList) memcpy_s(m_vnList, sizeof(Vector3)*numVerts, vnList, sizeof(Vector3)*numVerts);
		if(vtList) memcpy_s(m_vtList, sizeof(Vector2)*numVerts, vtList, sizeof(Vector2)*numVerts);
		if(indexList) memcpy_s(m_indexList, sizeof(Index_t)*m_numIndices, indexList, sizeof(Index_t)*m_numIndices);
	}

	m_attached = attached;

	return true;
}

void OpenGLModel::unload()
{
#	ifdef USE_MM
	if(!m_attached)
	{
		if(m_vList) FileMapper::unmap(m_vList);
		if(m_vnList) FileMapper::unmap(m_vnList);
		if(m_vtList) FileMapper::unmap(m_vtList);
		if(m_indexList) FileMapper::unmap(m_indexList);
	}
#	else
	if(!m_attached)
	{
		if(m_vList) delete[] m_vList;
		if(m_vnList) delete[] m_vnList;
		if(m_vtList) delete[] m_vtList;
		if(m_indexList) delete[] m_indexList;
	}
#	endif
	m_vList = NULL;
	m_vnList = NULL;
	m_vtList = NULL;
	m_indexList = NULL;

	m_numVerts = m_numTris = m_numNodes = m_numIndices = 0;
}

Vector3 *OpenGLModel::getPosition(const Index_t n)
{
	return &m_vList[n];
}

Vector3 *OpenGLModel::getNormal(const Index_t n)
{
	return &m_vnList[n];
}

Vector2 *OpenGLModel::getTextureCoordinate(const Index_t n)
{
	return &m_vtList[n];
}

Index_t *OpenGLModel::getIndex(const Index_t n)
{
	return &m_indexList[n];
}

int OpenGLModel::getNumIndices()
{
	return m_numIndices;
}

void OpenGLModel::updatePositions(Vector3 *positions)
{
	if(!m_attached) memcpy_s(&m_vList[0], sizeof(Vector3)*m_numVerts, positions, sizeof(Vector3)*m_numVerts);
	m_vUpdated = true;
}

void OpenGLModel::updateNormals(Vector3 *normals)
{
	if(!m_attached) memcpy_s(&m_vnList[0], sizeof(Vector3)*m_numVerts, normals, sizeof(Vector3)*m_numVerts);
	m_vnUpdated = true;
}

void OpenGLModel::updateTextureCoordinates(Vector2 *tex)
{
	if(!m_attached) memcpy_s(&m_vtList[0], sizeof(Vector2)*m_numVerts, tex, sizeof(Vector2)*m_numVerts);
	m_vtUpdated = true;
}

void OpenGLModel::updateIndexList(unsigned int *idx)
{
	if(!m_attached) memcpy_s(&m_indexList[0], sizeof(unsigned int)*m_numIndices, idx, sizeof(unsigned int)*m_numIndices);
	m_idxUpdated = true;
}

bool OpenGLModel::isPositionUpdated()
{
	bool ret = m_vUpdated; 
	m_vUpdated = false;
	return ret;
}

bool OpenGLModel::isNormalUpdated()
{
	bool ret = m_vnUpdated; 
	m_vnUpdated = false;
	return ret;
}

bool OpenGLModel::isTextureCoordinateUpdated()
{
	bool ret = m_vtUpdated; 
	m_vtUpdated = false;
	return ret;
}

bool OpenGLModel::isIndexListUpdated()
{
	bool ret = m_idxUpdated; 
	m_idxUpdated = false;
	return ret;
}
