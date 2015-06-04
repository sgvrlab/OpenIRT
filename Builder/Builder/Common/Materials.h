#ifndef COMMON_MATERIALS_H
#define COMMON_MATERIALS_H

/********************************************************************
	created:	2004/10/05
	created:	5.10.2004   21:04
	filename: 	c:\MSDev\MyProjects\Renderer\Common\Materials.h
	file base:	Materials
	file ext:	h
	author:		Christian Lauterbach (lauterb@informatik.uni-bremen.de)
	
	purpose:	Include for all material classes' headers. Only include
	            this file to use all materials.
*********************************************************************/

#include "Random.h"
#include "Hitpoint.h"

#include "Material.h"
#include "MaterialDiffuse.h"
#include "MaterialEmitter.h"
#include "MaterialSpecular.h"

//#ifdef _USE_TEXTURING
#include "BitmapTexture.h"
#include "MaterialBitmapTexture.h"
#include "MaterialSpecularAndBitmap.h"
#include "MaterialSolidNoise.h"
//#endif

typedef std::vector<Material *> MaterialList;
typedef MaterialList::iterator MaterialListIterator;

#endif