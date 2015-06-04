#include "BitmapTexture.h"
#include "ImageIL.h"

using namespace irt;

bool BitmapTexture::loadFromFile(const char *fileName) 
{
	strcpy_s(this->fileName, 256, fileName);
	// try to load this image
	textureImage = new ImageIL();
	if (textureImage->loadFromFile(fileName)) {
		//width = textureImage->width-1;
		//height = textureImage->height-1;
		width = textureImage->width;
		height = textureImage->height;
		bpp = textureImage->bpp;
		data = textureImage->data;

		return true;
	}
	else { // error loading bitmap file
		delete textureImage;
		textureImage = NULL;
		return false;
	}	
}