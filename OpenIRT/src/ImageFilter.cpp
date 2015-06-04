#include "ImageFilter.h"

using namespace irt;

void ImageFilter::boxFilter(Image *source, Image *target, int startX, int startY, int width, int height, KernelType type, float *kernel, Image *selector)
{
	if(!source || !target) return;

	static float kOriginal[] = {0, 0, 0, 0, 1, 0, 0, 0, 0};
	static float kSmoothing[] = {1, 1, 1, 1, 2, 1, 1, 1, 1};
	static float kSharpening[] = {-1, -1, -1, -1, 9, -1, -1, -1, -1};
	static float kRaised[] = {0, 0, -2, 0, 2, 0, 1, 0, 0};
	static float kMotionBlur[] = {0, 0, 1, 0, 0, 0, 1, 0, 0};
	static float kEdgeDetection[] = {-1, -1, -1, -1, 8, -1, -1, -1, -1};

	switch(type)
	{
	case SELECTIVE_SMOOTING :
	case SMOOTHING : kernel = kSmoothing; break;
	case SHARPENING : kernel = kSharpening; break;
	case RAISED : kernel = kRaised; break;
	case MOTION_BLUR : kernel = kMotionBlur; break;
	case EDGE_DETECTION : kernel = kEdgeDetection; break;
	case MANUAL : if(!kernel) return; break;
	default : kernel = kOriginal;
	}

	float denominator = 0.0f;
	float red, green, blue;
	int ired, igreen, iblue, indexOffset;

	int indices[] = {
		-(source->width + 1),  -source->width,     -(source->width - 1), 
		-1,                0,           +1, 
		source->width - 1,      source->width,      source->width + 1
	};
	for(int i=0;i<9;i++)
		denominator += kernel[i];
	if (denominator==0.0f) denominator = 1.0f;
	for(int i=startY;i<startY+height;i++) 
	{
		for(int j=startX;j<startX+width;j++) 
		{
			red = green = blue = 0.0f;
			indexOffset = (i*source->width)+j;
			if(type == SELECTIVE_SMOOTING)
			{
				denominator = 0.0f;

				float newKernel[9];

				for(int k=0;k<9;k++)
				{
					if(indexOffset+indices[k] < 0 || indexOffset+indices[k] >= source->width*source->height) continue;

					newKernel[k] = (1.0f-((selector->data[(indexOffset+indices[k])*3] + selector->data[(indexOffset+indices[k])*3+1] + selector->data[(indexOffset+indices[k])*3+2])/(255.0f*3.0f)))*kernel[k];
					denominator += newKernel[k];
				}
				if (denominator==0.0f) denominator = 1.0f;

				for(int k=0;k<9;k++) 
				{
					if(indexOffset+indices[k] < 0 || indexOffset+indices[k] >= source->width*source->height) continue;

					red += source->data[(indexOffset+indices[k])*3]*newKernel[k];
					green += source->data[(indexOffset+indices[k])*3+1]*newKernel[k];
					blue += source->data[(indexOffset+indices[k])*3+2]*newKernel[k];
				} 
			}
			else
			{
				for(int k=0;k<9;k++) 
				{
					if(indexOffset+indices[k] < 0 || indexOffset+indices[k] >= source->width*source->height) continue;

					red += source->data[(indexOffset+indices[k])*3]*kernel[k];
					green += source->data[(indexOffset+indices[k])*3+1]*kernel[k];
					blue += source->data[(indexOffset+indices[k])*3+2]*kernel[k];
				} 
			}
			ired = (int)(red / denominator);
			igreen = (int)(green / denominator);
			iblue = (int)(blue / denominator);
			if (ired>0xff) ired = 0xff;
			else if (ired<0) ired = 0;
			if (igreen>0xff) igreen = 0xff;
			else if (igreen<0) igreen = 0;
			if (iblue>0xff) iblue = 0xff;
			else if (iblue<0) iblue = 0;            
			target->data[indexOffset*3] = ired;
			target->data[indexOffset*3+1] = igreen;
			target->data[indexOffset*3+2] = iblue;
		}
	}
}