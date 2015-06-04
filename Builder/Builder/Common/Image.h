#ifndef COMMON_IMAGE_H_
#define COMMON_IMAGE_H_

#include <IL/il.h>
#include <IL/ilu.h>
#include <IL/ilut.h>

#include "Logger.h"
#include "rgb.h"

/**
 * Central class for image representation.
 *
 * Stores pixels as individual instances of color classes in an
 * twodimensional array. Offers gamma correction and other operations
 * on pixels. Most importantly has functions to load and save images
 * to/from files.
 *
 * This class uses the DevIL image library for low-level handling of
 * the image and image files (loading, writing etc.).
 *
 */
class Image
{
public:

	Image() { 
		isinitialized = false;
		
		width = height = bpp = 0;

		logger = LogManager::getSingletonPtr();

		initIL();
	}

	Image(char *filename) {
		logger = LogManager::getSingletonPtr();

		initIL();

		if (loadFromFile(filename)) {
			isinitialized = true;
		}
	}

	Image(int width, int height, int bpp=3);
	~Image();

	void setPixel(int x, int y, const rgb& color);
	void getPixel(int x, int y, rgb& color);
	rgb &getPixel(int x, int y);

	/*
	 * Output Functions:
	 */
	bool writeToFile(char *filename);

	/*
	 * Input Functions
	 */
	bool loadFromFile(char *filename);

	/*
	 *	Error Functions
	 */
	const char* getLastError();

	/*
	 *  Public Variables 
	 */

	int width, height, bpp; // Dimensions	
	ILubyte *data;			// Image Data

	// Reference counter (for multiple instances), so we know when
	// to startup and when to shutdown IL
	static int m_ilUseCount;

protected:

	void initIL() {
		// Is this the first Image instance ? 
		if (m_ilUseCount == 0) {
			//logger->logMessage(LOG_DEBUG, ilGetString(IL_VERSION));

			// yes, then do the DevIL initialization
			ilInit();
			ilEnable(IL_FILE_OVERWRITE); // always overwrite files
			ilEnable(IL_CONV_PAL);		 // convert palette images to bgra images automatically
			
			// Clear colour
			ilClearColor(0, 0, 0, 255);

			// Origin is in upper left corner
			ilEnable(IL_ORIGIN_SET);			
			ilOriginFunc(IL_ORIGIN_UPPER_LEFT);
			
			// force all loaded images to be RGB images with 8 bits per color channel
			ilEnable(IL_TYPE_SET);
			ilEnable(IL_FORMAT_SET);
			ilSetInteger(IL_FORMAT_MODE, IL_RGB);
			ilSetInteger(IL_TYPE_MODE, IL_UNSIGNED_BYTE);
		}

		m_ilUseCount++;
		m_ilImageID = 0;
		m_ilError = IL_NO_ERROR;
	}

	void resetIL() {
		// Image allocated before ?
		if (m_ilImageID)
			ilDeleteImages(1, &m_ilImageID);
		width = 0;
		height = 0;
		m_ilImageID = 0;
		m_ilError = IL_NO_ERROR;
		isinitialized = false;
	}

	bool isinitialized;  // only true when image loaded

	// DevIL Variables:
	ILuint m_ilImageID, 
		   m_ilError;
	
	// Log Manager
	LogManager *logger;

	rgb tempcolor;

};

// TODO: optimize this..
FORCEINLINE void Image::setPixel(int x, int y, const rgb& color) {
	assert(x < width && x >= 0 && y < height && y >= 0);
	assert(data != 0);

	int offset = ((height - y -1)*width + x)*3;

	data[offset] = (unsigned char)(color.r() * 255);
	data[offset + 1] = (unsigned char)(color.g() * 255);
	data[offset + 2] = (unsigned char)(color.b() * 255);
}

FORCEINLINE void Image::getPixel(int x, int y, rgb& color) {
	assert(x < width && x >= 0 && y < height && y >= 0);
	assert(data != 0);

	color.data[0] = data[(width*y + x)*3] / 255.0f;
	color.data[1] = data[(width*y + x)*3 + 1] / 255.0f;
	color.data[2] = data[(width*y + x)*3 + 2] / 255.0f;
}

FORCEINLINE rgb &Image::getPixel(int x, int y) {
	assert(x < width && x >= 0 && y < height && y >= 0);
	assert(data != 0);

	tempcolor.data[0] = data[(width*y + x)*3] / 255.0f;
	tempcolor.data[1] = data[(width*y + x)*3 + 1] / 255.0f;
	tempcolor.data[2] = data[(width*y + x)*3 + 2] / 255.0f;
	return tempcolor;
}

#endif