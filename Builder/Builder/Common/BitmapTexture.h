#ifndef COMMON_BITMAPTEXTURE_H
#define COMMON_BITMAPTEXTURE_H

#include "Image.h"
#include "OptionManager.h"

// texturing mode for coordinates <0.0 or >1.0 :
//
// wrap around 
#define TEXTURE_MODE_WRAP  0
// continue border value
#define TEXTURE_MODE_CLAMP 1
// mirror
#define TEXTURE_MODE_MIRROR 2

class TextureManager;
class BitmapTexture
{
public:
	BitmapTexture() {
		refCount = 0;
		lastError[0] = 0;
		width = 0;
		height = 0;
		textureImage = NULL;
		data = NULL;
	}

	~BitmapTexture() {
		if (textureImage)
			delete textureImage;
	} 

	bool loadFromFile(const char *fileName, const char *basePath = NULL);
	// TODO: read from memory

	/**
	 * Get the texture value at a certain texture position
	 */	
	void getTexValue(rgb &color, float u, float v, int mode = TEXTURE_MODE_WRAP);

	const char *getLastErrorString() {
		return lastError;
	}

	friend TextureManager;
protected:
	// reference counter of this bitmap texture
	int refCount;

	// image this texture uses
	Image *textureImage;
	// .. and its dimensions
	int width, height;

	ILubyte *data;

	char lastError[200];
	
private:
};

FORCEINLINE void BitmapTexture::getTexValue(rgb &color, float u, float v, int mode) {
#if TEXTURE_FILTER == TEXTURE_FILTER_POINT
	// Simplest and fastest texture filtering: point filter

	// TODO: use texturing mode !
	int x = (int)(u*(float)width) % width;
	int y = (int)(v*(float)height) % height;

	if (x < 0)
		x = width + x;
	if (y < 0)
		y = height + y;
	
	textureImage->getPixel(x,y,color);

#else if TEXTURE_FILTER == TEXTURE_FILTER_BILINEAR
	// Bilinear filtering, take 4 pixels into account and interpolate

	float u0to1 = u - (int)u;
	float v0to1 = v - (int)v;

	assert(u0to1 <= 1.0f && u0to1 >= -1.0f);
	assert(v0to1 <= 1.0f && u0to1 >= -1.0f);

	if (u0to1 < 0.0f)
		u0to1 = 1.0f + u0to1;
	if (v0to1 < 0.0f)
		v0to1 = 1.0f + v0to1;

	u0to1 *= (float)width;
	v0to1 *= (float)height;

	int x = (int)u0to1;
	int y = (int)v0to1;	

	float tx = u0to1 - x;
	float ty = v0to1 - y;

	color = textureImage->getPixel(x,y)*(1.0f-tx)*(1.0f-ty) 
			+ textureImage->getPixel(x+1,y)*tx*(1.0f-ty) 
			+ textureImage->getPixel(x,y+1)*(1.0f-tx)*ty 
			+ textureImage->getPixel(x+1,y+1)*tx*ty; 
	
#endif
}


#endif