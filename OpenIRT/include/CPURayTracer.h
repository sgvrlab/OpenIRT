/********************************************************************
	created:	2012/08/29
	file path:	d:\Projects\Redering\OpenIRT\include
	file base:	CPURayTracer
	file ext:	h
	author:		Tae-Joon Kim (tjkim.kaist@gmail.com)
	
	comment:	Ray tracer on CPU
*********************************************************************/

#pragma once

#include "Renderer.h"

namespace irt
{

class CPURayTracer :
	public Renderer
{
public:
	CPURayTracer(void);
	virtual ~CPURayTracer(void);

	virtual void init(Scene *scene);
	virtual void done();

	virtual void resized(int width, int height);

	virtual void sceneChanged();
	virtual void materialChanged();

	// renderer
	virtual void render(Camera *camera, Image *image, unsigned int seed = UINT_MAX);
};

};
