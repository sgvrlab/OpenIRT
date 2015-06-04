#pragma once

#include "rgb.h"

/**
 * Central class for image representation.
 *
 * Stores pixels as individual instances of color classes in an
 * twodimensional array. Offers gamma correction and other operations
 * on pixels. Most importantly has functions to load and save images
 * to/from files.
 *
 */

namespace irt
{

class Image
{
public:

	Image();
	Image(int width, int height, int bpp=3);
	virtual ~Image();

	void setPixel(int x, int y, const RGBf& color);
	void getPixel(int x, int y, RGBf& color) const;
	RGBf &getPixel(int x, int y);

	void setPixel(int x, int y, const RGB4f& color);
	void getPixel(int x, int y, RGB4f& color) const;
	RGB4f &getPixel4(int x, int y);
	template <int nRaysPerDim> void setPacketPixels(int x, int y, const RGB4f *color);

	/*
	 * Output Functions:
	 */
	virtual bool writeToFile(const char *filename) {return false;}

	/*
	 * Input Functions
	 */
	virtual bool loadFromFile(const char *filename) {return false;}
	virtual bool loadFromData(unsigned char *data, bool flip = false, bool exchangeRB = false);

	/*
	 *  Public Variables 
	 */

	int width, height, bpp; // Dimensions	
	unsigned char *data;			// Image Data
protected:
	RGBf tempcolor;
	RGB4f tempcolor4;
	int rowOffset;	// equals width*(Bpp)
};

template <int nRaysPerDim> 
void Image::setPacketPixels(int x, int y, const RGB4f *colorPtr) {
	unsigned char *basePtr = &data[(y*width + x)*bpp]; // first row	

	register const __m128 one4 = _mm_set1_ps(1.0f);
	register const __m128 twofiftyfive = _mm_set1_ps(255.0f);		

	// set rectangular pixel region from ray packets
	for (int j = 0; j < nRaysPerDim; j++, basePtr -= (2*rowOffset)) {
		unsigned char *rowPtr1 = basePtr;
		unsigned char *rowPtr2 = rowPtr1 - rowOffset;

		for (int i = 0; i < nRaysPerDim; i++) {
			__declspec(align(16)) char buffer[16];						

			#ifdef _M_X64
			register __m128i color0 = _mm_packs_epi32(_mm_cvtps_epi32(_mm_mul_ps(_mm_min_ps(one4, colorPtr[0].data4), twofiftyfive)), 
				                                      _mm_cvtps_epi32(_mm_mul_ps(_mm_min_ps(one4, colorPtr[1].data4), twofiftyfive)));
			register __m128i color1 = _mm_packs_epi32(_mm_cvtps_epi32(_mm_mul_ps(_mm_min_ps(one4, colorPtr[2].data4), twofiftyfive)), 
				                                      _mm_cvtps_epi32(_mm_mul_ps(_mm_min_ps(one4, colorPtr[3].data4), twofiftyfive)));
			#else // 32-bit only:
			register __m128i color0 = _mm_packs_epi32(_mm_cvtps_epi32(_mm_mul_ps(_mm_min_ps(one4, _mm_load_ps(colorPtr[0].data)), twofiftyfive)), 
				                                      _mm_cvtps_epi32(_mm_mul_ps(_mm_min_ps(one4, _mm_load_ps(colorPtr[1].data)), twofiftyfive)));
			register __m128i color1 = _mm_packs_epi32(_mm_cvtps_epi32(_mm_mul_ps(_mm_min_ps(one4, _mm_load_ps(colorPtr[2].data)), twofiftyfive)), 
				                                      _mm_cvtps_epi32(_mm_mul_ps(_mm_min_ps(one4, _mm_load_ps(colorPtr[3].data)), twofiftyfive)));						
			#endif	

			color0 = _mm_packus_epi16(color0, color1);
			_mm_store_si128((__m128i *)buffer, color0);

			// set colors for one 2x2 ray:
			*rowPtr1++ = buffer[0];
			*rowPtr1++ = buffer[1];
			*rowPtr1++ = buffer[2];
			colorPtr++;

			*rowPtr1++ = buffer[4];
			*rowPtr1++ = buffer[5];
			*rowPtr1++ = buffer[6];
			colorPtr++;

			*rowPtr2++ = buffer[8];
			*rowPtr2++ = buffer[9];
			*rowPtr2++ = buffer[10];
			colorPtr++;

			*rowPtr2++ = buffer[12];
			*rowPtr2++ = buffer[13];
			*rowPtr2++ = buffer[14];
			colorPtr++;
		}
	}

}

inline void Image::setPixel(int x, int y, const RGBf& color) {
	assert(x < width && x >= 0 && y < height && y >= 0);
	assert(data != 0);

	//int offset = ((height - y -1)*width + x)*bpp;
	int offset = (y*width + x)*bpp;

	data[offset] = (unsigned char)(color.r() * 255);
	data[offset + 1] = (unsigned char)(color.g() * 255);
	data[offset + 2] = (unsigned char)(color.b() * 255);
}

inline void Image::getPixel(int x, int y, RGBf& color) const {
	//assert(x < width && x >= 0 && y < height && y >= 0);
	assert(data != 0);

	color.setR(data[(width*y + x)*bpp] / 255.0f);
	color.setG(data[(width*y + x)*bpp + 1] / 255.0f);
	color.setB(data[(width*y + x)*bpp + 2] / 255.0f);
}

inline RGBf &Image::getPixel(int x, int y) {
	assert(x < width && x >= 0 && y < height && y >= 0);
	assert(data != 0);

	tempcolor.setR(data[(width*y + x)*bpp] / 255.0f);
	tempcolor.setG(data[(width*y + x)*bpp + 1] / 255.0f);
	tempcolor.setB(data[(width*y + x)*bpp + 2] / 255.0f);
	return tempcolor;
}

inline void Image::setPixel(int x, int y, const RGB4f& color) {
	assert(x < width && x >= 0 && y < height && y >= 0);
	assert(data != 0);

	//int offset = ((height - y -1)*width + x)*bpp;
	int offset = (y*width + x)*bpp;

	data[offset] = (unsigned char)(color.r() * 255);
	data[offset + 1] = (unsigned char)(color.g() * 255);
	data[offset + 2] = (unsigned char)(color.b() * 255);
	data[offset + 3] = (unsigned char)(color.a() * 255);
}

inline void Image::getPixel(int x, int y, RGB4f& color) const {
	//assert(x < width && x >= 0 && y < height && y >= 0);
	assert(data != 0);

	color.setR(data[(width*y + x)*bpp] / 255.0f);
	color.setG(data[(width*y + x)*bpp + 1] / 255.0f);
	color.setB(data[(width*y + x)*bpp + 2] / 255.0f);
	color.setA(data[(width*y + x)*bpp + 3] / 255.0f);
}

inline RGB4f &Image::getPixel4(int x, int y) {
	assert(x < width && x >= 0 && y < height && y >= 0);
	assert(data != 0);

	tempcolor4.setR(data[(width*y + x)*bpp] / 255.0f);
	tempcolor4.setG(data[(width*y + x)*bpp + 1] / 255.0f);
	tempcolor4.setB(data[(width*y + x)*bpp + 2] / 255.0f);
	tempcolor4.setA(data[(width*y + x)*bpp + 3] / 255.0f);
	return tempcolor4;
}

};