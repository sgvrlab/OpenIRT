#include "stdafx.h"
#include "BitmapTexture.h"

bool BitmapTexture::loadFromFile(const char *fileName, const char *basePath) {
	OptionManager *opt = OptionManager::getSingletonPtr();

	// Get default base path and prepend it
	const char *textureBasePath;
	if (basePath)
		textureBasePath = basePath;
	else
		textureBasePath = opt->getOption("global", "texPath");

	char textureFileName[MAX_PATH];

	// append '/' if needed
	if (textureBasePath[strlen(textureBasePath)-1] != '/')
		sprintf(textureFileName, "%s/%s", textureBasePath, fileName);
	else
		sprintf(textureFileName, "%s%s", textureBasePath, fileName);

	// try to load this image
	textureImage = new Image();
	if (textureImage->loadFromFile(textureFileName)) {
		width = textureImage->width-1;
		height = textureImage->height-1;
		data = textureImage->data;

#if TEXTURE_FILTER == TEXTURE_FILTER_BILINEAR
		// if using biliear filtering, we can't access the border pixels
		width -= 2;
		height -= 2;
#endif

		return true;
	}
	else { // error loading bitmap file
		strncpy(lastError, textureImage->getLastError(), sizeof(lastError)-1);
		delete textureImage;
		textureImage = NULL;
		return false;
	}	
}