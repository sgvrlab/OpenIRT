/********************************************************************
	created:	2014/02/28
	file path:	d:\Projects\Redering\OpenIRT\include
	file base:	OpenGLModel
	file ext:	h
	author:		Tae-Joon Kim (tjkim@tclab.kaist.ac.kr)
	
	comment:	OpenGLModel class, has OpenGL friendly instances and access functions
*********************************************************************/

#pragma once

#include "Model.h"

namespace irt
{

class OpenGLModel : public Model
{
public:
// Member variables
protected:
	Vector3 *m_vList;		// vertex positions
	Vector3 *m_vnList;		// vertex normals
	Vector2 *m_vtList;		// vertex texture coordinates
	Index_t *m_indexList;	// only has vertex indices of triangles
	int m_numIndices;		// = 3 * m_numTris;
	bool m_attached;		// if true, above variables have only reference; the vertices and indices will not be copyed
	bool m_vUpdated;
	bool m_vnUpdated;
	bool m_vtUpdated;
	bool m_idxUpdated;

	// Member functions
public:
	OpenGLModel(void);
	virtual ~OpenGLModel(void);

	bool load(Vector3 *vList, Vector3 *vnList, Vector2 *vtList, int numVerts, Index_t *indexList, int numIndices, const Material &mat, bool attached = false);
	void unload();

	Vector3 *getPosition(const Index_t n);
	Vector3 *getNormal(const Index_t n);
	Vector2 *getTextureCoordinate(const Index_t n);
	Index_t *getIndex(const Index_t n);
	int getNumIndices();
	void updatePositions(Vector3 *positions);
	void updateNormals(Vector3 *normals);
	void updateTextureCoordinates(Vector2 *tex);
	void updateIndexList(unsigned int *idx);
	bool isPositionUpdated();
	bool isNormalUpdated();
	bool isTextureCoordinateUpdated();
	bool isIndexListUpdated();
};

};