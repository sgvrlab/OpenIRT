#include "stdafx.h"
#include "common.h"

void MaterialBitmapTexture::shade(Vector3 &viewer, Hitpoint &p, rgb &result) {
	// do we actually have texture coords in the triangles?
#ifndef _USE_TEXTURING		
	// no, fake them via the object BB!
	char i1 = p.objectPtr->i1;
	char i2 = p.objectPtr->i2;
	p.uv[0] = (p.x[i1] - p.objectPtr->bb[0][i1]) * p.objectPtr->coordToU;
	p.uv[1] = (p.x[i2] - p.objectPtr->bb[0][i2]) * p.objectPtr->coordToV;
#endif

	if (texPtr != 0)
		texPtr->getTexValue(result, p.uv[0], p.uv[1]);
}

void MaterialBitmapTexture::shade(Vector3 &viewer, SIMDHitpoint &p, int idx, rgb& result) {
	if (texPtr != 0) {
		// do we actually have texture coords in the triangles?
#ifndef _USE_TEXTURING
		// no, fake them via the object BB!

#endif

		texPtr->getTexValue(result, p.uv[idx][0], p.uv[idx][1]);
	}
}

void MaterialSpecularAndBitmap::shade(Vector3 &viewer, Hitpoint &p, rgb &result) {
	// do we actually have texture coords in the triangles?
#ifndef _USE_TEXTURING		
	// no, fake them via the object BB!
	char i1 = p.objectPtr->i1;
	char i2 = p.objectPtr->i2;
	p.uv[0] = (p.x[i1] - p.objectPtr->bb[0][i1]) * p.objectPtr->coordToU;
	p.uv[1] = (p.x[i2] - p.objectPtr->bb[0][i2]) * p.objectPtr->coordToV;
#endif

	if (texPtr != 0)
		texPtr->getTexValue(result, p.uv[0], p.uv[1]);
}

void MaterialSpecularAndBitmap::shade(Vector3 &viewer, SIMDHitpoint &p, int idx, rgb& result) {
	if (texPtr != 0) {
		// do we actually have texture coords in the triangles?
#ifndef _USE_TEXTURING
		// no, fake them via the object BB!

#endif

		texPtr->getTexValue(result, p.uv[idx][0], p.uv[idx][1]);
	}
}