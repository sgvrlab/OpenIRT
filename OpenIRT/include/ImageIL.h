#pragma once

#include "Image.h"

/**
 *
 * This class uses the DevIL image library for low-level handling of
 * the image and image files (loading, writing etc.).
 *
 */

namespace irt
{

class ImageIL : public Image
{
public:

	ImageIL();
	ImageIL(const char *filename);

	ImageIL(int width, int height, int bpp=3);
	virtual ~ImageIL();

	/*
	 * Output Functions:
	 */
	virtual bool writeToFile(const char *filename);

	/*
	 * Input Functions
	 */
	virtual bool loadFromFile(const char *filename);
	virtual bool loadFromData(unsigned char *data, bool flip = false, bool exchangeRB = false);

	/*
	 * Error Function
	 */
	const char* getLastError();

	// Reference counter (for multiple instances), so we know when
	// to startup and when to shutdown IL
	static int m_ilUseCount;

protected:

	void initIL();
	void resetIL();

	bool isinitialized;  // only true when image loaded

	public:
	// DevIL Variables:
	unsigned int m_ilImageID, 
		   m_ilError;
};

};