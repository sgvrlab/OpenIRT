/********************************************************************
	created:	2011/07/29
	file path:	d:\Projects\Redering\OpenIRT\include
	file base:	SimpleRasterizer
	file ext:	h
	author:		Tae-Joon Kim (tjkim.kaist@gmail.com)
	
	comment:	Simple rasterizer using OpenGL
*********************************************************************/

#pragma once

#include "Renderer.h"
#include "GLAdapter.h"
#include "Vertex.h"

namespace irt
{

class SimpleRasterizer :
	public Renderer, public GLAdapter
{
protected:
	int m_numModels;

	unsigned int **m_indexList;
	int *m_numTris;

	int m_GLVersion;

	unsigned int *m_vertexVBOID;
	unsigned int *m_indexVBOID;

public:
	SimpleRasterizer(void);
	virtual ~SimpleRasterizer(void);

	virtual void init(Scene *scene);
	virtual void done();

	virtual void resized(int width, int height);

	virtual void sceneChanged();

	// renderer
	virtual void render(Camera *camera, Image *image, unsigned int seed = UINT_MAX);

	void render(SceneNode *sceneNode);
};

};