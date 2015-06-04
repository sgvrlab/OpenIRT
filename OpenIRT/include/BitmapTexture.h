#pragma once

#include "Image.h"
#include "Vector2.h"
#include "Vector3.h"

// texturing mode for coordinates <0.0 or >1.0 :
//
// wrap around 
#define TEXTURE_MODE_WRAP  0
// continue border value
#define TEXTURE_MODE_CLAMP 1
// mirror
#define TEXTURE_MODE_MIRROR 2

namespace irt
{

class TextureManager;
class BitmapTexture
{
public:
	BitmapTexture() {
		refCount = 0;
		lastError[0] = 0;
		width = 0;
		height = 0;
		bpp = 0;
		textureImage = NULL;
		data = NULL;
	}

	~BitmapTexture() {
		if (textureImage)
			delete textureImage;
	} 

	bool loadFromFile(const char *fileName);
	// TODO: read from memory

	/**
	 * Get the texture value at a certain texture position
	 */	
	void getTexValue(RGBf &color, float u, float v, int mode = TEXTURE_MODE_WRAP);
	RGBf BitmapTexture::Sample(const Vector2 &tex);
	const char *getLastErrorString() {
		return lastError;
	}
	unsigned char* getData() {
		return data;
	}

	int getWidth() {return width;}
	int getHeight() {return height;}
	int getBpp() {return bpp;}
	const char *getFileName() {return fileName;}

	friend TextureManager;
protected:
	// reference counter of this bitmap texture
	int refCount;

	// image this texture uses
	Image *textureImage;
	// .. and its dimensions
	int width, height;
	int bpp;

	unsigned char *data;

	char lastError[200];

	char fileName[256];
	
private:
};

inline RGBf BitmapTexture::Sample(const Vector2 &tex) {
	float u = tex.x();
	float v = tex.y();

	// Simplest and fastest texture filtering: point filter

	// TODO: use texturing mode !
	int x = (int)(u*(float)width) % width;
	int y = (int)(v*(float)height) % height;

	if (x < 0)
		x = width + x;
	if (y < 0)
		y = height + y;

	return textureImage->getPixel(x,y);
}


inline void BitmapTexture::getTexValue(RGBf &color, float u, float v, int mode) {
	// Simplest and fastest texture filtering: point filter

	// TODO: use texturing mode !
	int x = (int)(u*(float)width) % width;
	int y = (int)(v*(float)height) % height;

	if (x < 0)
		x = width + x;
	if (y < 0)
		y = height + y;
	
	textureImage->getPixel(x,y,color);
}

};