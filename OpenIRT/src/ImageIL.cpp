#include "defines.h"

#include <IL/il.h>
#include <IL/ilu.h>
#include <IL/ilut.h>

#include "ImageIL.h"
#include <math.h>
#include <string>

using namespace std;
using namespace irt;

int ImageIL::m_ilUseCount = 0;

ImageIL::ImageIL() 
{
	isinitialized = false;
		
	width = height = bpp = 0;
	rowOffset = 0;

	initIL();
}

ImageIL::ImageIL(const char *filename) 
{
	initIL();

	if (loadFromFile(filename)) {
		isinitialized = true;
	}
}

ImageIL::ImageIL(int w, int h, int bpp)
{
	width = w;
	height = h;

	assert(w > 0 && h > 0 && bpp > 0);
	
	this->bpp = bpp;
	rowOffset = width * bpp;

	initIL();

	ilGenImages(1, &m_ilImageID);
	ilBindImage(m_ilImageID);

	if (!ilTexImage(w, h, 1, bpp, bpp == 4 ? IL_RGBA: IL_RGB, IL_UNSIGNED_BYTE, NULL)) {
		printf("Image: Error creating new image !\n");
		printf("Params: Width %d, Height %d, BPP %d), Error: %s\n", w, h, bpp, getLastError());
		return;
	}
	
	ilClearImage();
	
	data = ilGetData();

	isinitialized = true;

}

/**
 * Destructor, frees image array, if initialized
 */
ImageIL::~ImageIL()
{
	if (m_ilImageID)
		ilDeleteImages(1, &m_ilImageID);
	m_ilUseCount--;
	if (m_ilUseCount == 0)
		ilShutDown();

	data = NULL;
}

void ImageIL::initIL()
{
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
		ilSetInteger(IL_FORMAT_MODE, IL_RGBA);
		ilSetInteger(IL_TYPE_MODE, IL_UNSIGNED_BYTE);
			
	}

	m_ilUseCount++;
	m_ilImageID = 0;
	m_ilError = IL_NO_ERROR;
}

void ImageIL::resetIL()
{
	// Image allocated before ?
	if (m_ilImageID)
		ilDeleteImages(1, &m_ilImageID);
	width = 0;
	height = 0;
	rowOffset = 0;
	m_ilImageID = 0;
	m_ilError = IL_NO_ERROR;
	isinitialized = false;

	ilEnable(IL_TYPE_SET);
	ilEnable(IL_FORMAT_SET);
	ilSetInteger(IL_FORMAT_MODE, IL_RGBA);
	ilSetInteger(IL_TYPE_MODE, IL_UNSIGNED_BYTE);
}


bool ImageIL::loadFromFile(const char *filename){
	resetIL();

	if (m_ilImageID == 0) {
		ilGenImages(1, &m_ilImageID);
		ilBindImage(m_ilImageID);
	}

	if (ilLoadImage((const ILstring)filename)) {		
		width = ilGetInteger(IL_IMAGE_WIDTH);
		height = ilGetInteger(IL_IMAGE_HEIGHT);
		bpp = ilGetInteger(IL_IMAGE_BYTES_PER_PIXEL);
		rowOffset = width * bpp;

		printf("Loaded image \"%s\" (W:%d H:%d bpp:%d)\n", filename, width, height, bpp);

		isinitialized = true;

		data = ilGetData();

		return true;
	}
	else 
		return false;
}

bool ImageIL::loadFromData(unsigned char *inData, bool flip, bool exchangeRB)
{
	memcpy_s(data, width*height*bpp, inData, width*height*bpp);

	if(exchangeRB)
	{
		for(int i=0;i<height;i++)
		{
			for(int j=0;j<width;j++)
			{
				int offset = (j+i*width)*bpp;
				ILubyte temp;
				temp = data[offset];
				data[offset] = data[offset+2];
				data[offset+2] = temp;
			}
		}
	}

	if(flip)
	{
		for(int i=0;i<height/2;i++)
		{
			for(int j=0;j<width;j++)
			{
				int offset = (j+i*width)*bpp;
				int offset2 = (j+(height - 1 - i)*width)*bpp;
				ILubyte temp;
				temp = data[offset];
				data[offset] = data[offset2];
				data[offset2] = temp;
				temp = data[offset+1];
				data[offset+1] = data[offset2+1];
				data[offset2+1] = temp;
				temp = data[offset+2];
				data[offset+2] = data[offset2+2];
				data[offset2+2] = temp;
			}
		}
	}

	return true;
}

bool ImageIL::writeToFile(const char *filename) {
	if (!isinitialized || m_ilImageID<=0 || width <=0 || height <=0 || bpp <= 0)
		return false;

	ilBindImage(m_ilImageID);
	if (ilSaveImage((const ILstring)filename)) {
		
		printf("Saved image \"%s\" (W: %d H: %d bpp: %d)\n", filename, width, height, bpp);

		return true;
	}
	else {
		return false;

		printf("Error saving image \"%s\" (W: %d H: %d bpp: %d)\n", filename, width, height, bpp);
	}
}

const char* ImageIL::getLastError() {
	ILenum ILerror;

	ILerror = ilGetError();

	switch (ILerror) {
		case IL_NO_ERROR:
			return "No detectable error has occured.";
		case IL_INVALID_ENUM:
			return "An unacceptable enumerated value was passed to a function.";
		case IL_OUT_OF_MEMORY:
			return "Could not allocate enough memory in an operation. ";
		case IL_FORMAT_NOT_SUPPORTED:
			return "The format a function tried to use was not able to be used by that function. ";
		case IL_INTERNAL_ERROR:
			return "A serious error has occurred. Please e-mail DooMWiz with the conditions leading up to this error being reported. ";
		case IL_INVALID_VALUE:
			return "An invalid value was passed to a function or was in a file. ";
		case IL_ILLEGAL_OPERATION:
			return "The operation attempted is not allowable in the current state. The function returns with no ill side effects. ";
		case IL_ILLEGAL_FILE_VALUE:
			return "An illegal value was found in a file trying to be loaded. ";
		case IL_INVALID_FILE_HEADER:
			return "A file's header was incorrect. ";
		case IL_INVALID_PARAM:
			return "An invalid parameter was passed to a function, such as a NULL pointer. ";
		case IL_COULD_NOT_OPEN_FILE:
			return "Could not open the file specified. The file may already be open by another app or may not exist. ";
		case IL_INVALID_EXTENSION:
			return "The extension of the specified filename was not correct for the type of image-loading function. ";
		case IL_FILE_ALREADY_EXISTS:
			return "The filename specified already belongs to another file. To overwrite files by default, call ilEnable with the IL_FILE_OVERWRITE parameter. ";
		case IL_OUT_FORMAT_SAME:
			return "Tried to convert an image from its format to the same format. ";
		case IL_STACK_OVERFLOW:
			return "One of the internal stacks was already filled, and the user tried to add on to the full stack. ";
		case IL_STACK_UNDERFLOW:
			return "One of the internal stacks was empty, and the user tried to empty the already empty stack. ";
		case IL_INVALID_CONVERSION:
			return "An invalid conversion attempt was tried. ";
		case IL_LIB_JPEG_ERROR:
			return "An error occurred in the libjpeg library. ";
		case IL_LIB_PNG_ERROR:
			return "An error occurred in the libpng library. ";
		case ILUT_NOT_SUPPORTED:
			return "A type is valid but not supported in the current build.";
		case IL_UNKNOWN_ERROR:
		default:
			return "No function sets this yet, but it is possible (not probable) it may be used in the future. ";
	}


}