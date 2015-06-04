/********************************************************************
	created:	2011/08/15
	file path:	d:\Projects\Redering\OpenIRT\include
	file base:	CUDAPathTracer
	file ext:	h
	author:		Tae-Joon Kim (tjkim.kaist@gmail.com)
	
	comment:	Path tracer using CUDA
*********************************************************************/

#pragma once

#include "CUDARayTracer.h"

namespace irt
{

class CUDAPathTracer :
	public CUDARayTracer
{
public:
	CUDAPathTracer(void);
	virtual ~CUDAPathTracer(void);

	virtual void init(Scene *scene);
	virtual void done();

	virtual void resized(int width, int height);

	virtual void sceneChanged();
	virtual void lightChanged(bool soft = false);
	virtual void materialChanged();

	virtual void getCurrentColorImage(Image *image);
	virtual void getCurrentDepthImage(Image *image);
	virtual void getCurrentNormalImage(Image *image);

	virtual void filter(Image *image);

	// renderer
	virtual void render(Camera *camera, Image *image, unsigned int seed = UINT_MAX);
};

};