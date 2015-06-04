#include "stdafx.h"
#include "Image.h"
#include <math.h>
#include <string>

using namespace std;

int Image::m_ilUseCount = 0;

Image::Image(int w, int h, int bpp) : width(w), height(h) {
	assert(w > 0 && h > 0 && bpp > 0);
	
	this->bpp = bpp;

	logger = LogManager::getSingletonPtr();

	initIL();

	ilGenImages(1, &m_ilImageID);
	ilBindImage(m_ilImageID);

	if (!ilTexImage(w, h, 1, bpp, IL_RGB, IL_UNSIGNED_BYTE, NULL)) {
		logger->logMessage(LOG_ERROR, "Image: Error creating new image !");
		char debugMessage[200];
		sprintf(debugMessage, "Params: Width %d, Height %d, BPP %d), Error: %s", w, h, bpp, getLastError());
		logger->logMessage(LOG_ERROR, debugMessage);
		return;
	}
	
	ilClearImage();
	
	data = ilGetData();

	isinitialized = true;

}

/**
 * Destructor, frees image array, if initialized
 */
Image::~Image()
{
	if (m_ilImageID)
		ilDeleteImages(1, &m_ilImageID);
	m_ilUseCount--;
	if (m_ilUseCount == 0)
		ilShutDown();
}

bool Image::loadFromFile(char *filename){
	resetIL();

	if (m_ilImageID == 0) {
		ilGenImages(1, &m_ilImageID);
		ilBindImage(m_ilImageID);
	}

	if (ilLoadImage(filename)) {		
		width = ilGetInteger(IL_IMAGE_WIDTH);
		height = ilGetInteger(IL_IMAGE_HEIGHT);
		bpp = ilGetInteger(IL_IMAGE_BITS_PER_PIXEL);

		char imageInfo[200];
		sprintf(imageInfo, "Loaded image \"%s\" (W:%d H:%d bpp:%d)", filename, width, height, bpp);
		logger->logMessage(LOG_INFO, imageInfo);

		isinitialized = true;

		data = ilGetData();

		return true;
	}
	else 
		return false;
}

bool Image::writeToFile(char *filename) {
	if (!isinitialized || m_ilImageID<=0 || width <=0 || height <=0 || bpp <= 0)
		return false;

	char imageInfo[200];
	if (ilSaveImage(filename)) {
		
		sprintf(imageInfo, "Saved image \"%s\" (W: %d H: %d bpp: %d)", filename, width, height, bpp);
		logger->logMessage(LOG_INFO, imageInfo);

		return true;
	}
	else {
		return false;

		sprintf(imageInfo, "Error saving image \"%s\" (W: %d H: %d bpp: %d)", filename, width, height, bpp);
		logger->logMessage(LOG_ERROR, imageInfo);
	}
}

const char* Image::getLastError() {
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
