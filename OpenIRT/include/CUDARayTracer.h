/********************************************************************
	created:	2011/08/15
	file path:	d:\Projects\Redering\OpenIRT\include
	file base:	CUDARayTracer
	file ext:	h
	author:		Tae-Joon Kim (tjkim.kaist@gmail.com)
	
	comment:	Ray tracer using CUDA
*********************************************************************/

#pragma once

#include "Renderer.h"

namespace irt
{

class CUDARayTracer :
	public Renderer
{
public:
	CUDARayTracer(void);
	virtual ~CUDARayTracer(void);

	virtual void init(Scene *scene);
	virtual void done();

	virtual void resized(int width, int height);

	virtual void sceneChanged();
	virtual void materialChanged();

	// renderer
	virtual void render(Camera *camera, Image *image, unsigned int seed = UINT_MAX);

	virtual void clearResult();

	unsigned int makeCUDASceneGraph(void *srcSceneNode, void *dstSceneNode, void *dstScene, int &Index);

	void *m_dstScene;
	bool m_materialUpdated;
	virtual int adaptScene(void *srcScene, void *dstScene);
	int adaptCamera(void *srcCamera, void *dstCamera);
	int adaptDataStructures(void *srcScene, void *srcCamera, void *dstScene, void *dstCamera);
	void clearScene();
	void applyChangedMaterial();
};

};