/********************************************************************
	created:	2012/01/31
	file path:	d:\Projects\Redering\OpenIRT\include
	file base:	GBufferFilter
	file ext:	h
	author:		Tae-Joon Kim (tjkim.kaist@gmail.com)
	
	comment:	G-Buffer filter (reference?)
*********************************************************************/

#pragma once

#include "ImageFilter.h"

namespace irt
{

class GBufferFilter : public ImageFilter {
public:
	virtual void filter(Image *source, Image *target, int startX, int startY, int width, int height, int filterSize, ...);
	virtual void filter(Image *source, Image *target, int count, int *startX, int *startY, int width, int height, int filterSize, ...);

	static void simpleFilter(Image *source, Image *target, int startX, int startY, int width, int height, int filterSize, ...);
};

};