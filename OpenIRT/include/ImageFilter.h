/********************************************************************
	created:	2012/01/30
	file path:	d:\Projects\Redering\OpenIRT\include
	file base:	Filter
	file ext:	h
	author:		Tae-Joon Kim (tjkim.kaist@gmail.com)
	
	comment:	Box filter of given image
*********************************************************************/

#pragma once

#include "Image.h"

namespace irt
{

	class ImageFilter {
public:
	enum KernelType
	{
		ORIGINAL,
		SMOOTHING,
		SELECTIVE_SMOOTING,
		SHARPENING,
		RAISED,
		MOTION_BLUR,
		EDGE_DETECTION,
		MANUAL
	};

	static void boxFilter(Image *source, Image *target, int startX, int startY, int width, int height, KernelType type, float *kernel = 0, Image *selector = 0);

	virtual void filter(Image *source, Image *target, int startX, int startY, int width, int height, int filterSize, ...) {}
};

};