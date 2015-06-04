#include "Image.h"

using namespace irt;

Image::Image() 
{
	width = height = bpp = 0;
	rowOffset = 0;
	data = 0;
}

Image::Image(int w, int h, int bpp)
{
	this->width = w;
	this->height = h;
	this->bpp = bpp;
	rowOffset = width * bpp;

	data = new unsigned char[w*h*bpp];
}

/**
 * Destructor, frees image array, if initialized
 */
Image::~Image()
{
	if(data) delete[] data;
}

bool Image::loadFromData(unsigned char *inData, bool flip, bool exchangeRB)
{
	memcpy_s(data, width*height*bpp, inData, width*height*bpp);

	if(exchangeRB)
	{
		for(int i=0;i<height;i++)
		{
			for(int j=0;j<width;j++)
			{
				int offset = (j+i*width)*bpp;
				unsigned char temp;
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
				unsigned char temp;
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