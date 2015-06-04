/********************************************************************
	created:	2011/08/28
	file path:	d:\Projects\Redering\OpenIRT\include
	file base:	CUDAPhotonMapping
	file ext:	h
	author:		Tae-Joon Kim (tjkim.kaist@gmail.com)
	
	comment:	Photon Mapping using CUDA
*********************************************************************/

#pragma once

#include "CUDARayTracer.h"
#include "Photon.h"

namespace irt
{

class CUDAPhotonMapping :
	public CUDARayTracer
{
protected:
	Photon *m_photons;
public:
	CUDAPhotonMapping(void);
	virtual ~CUDAPhotonMapping(void);

	virtual void init(Scene *scene);
	virtual void done();

	virtual void resized(int width, int height);

	virtual void sceneChanged();

	// renderer
	virtual void render(Camera *camera, Image *image, unsigned int seed = UINT_MAX);
};

};