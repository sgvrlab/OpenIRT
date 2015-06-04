// HSaliency.h: interface for the HSaliency class.
//
//////////////////////////////////////////////////////////////////////
//===========================================================================
// This code implements the saliency method described in:
//
// R. Achanta, S. Hemami, F. Estrada and S. S?strunk,
// "Frequency-tuned Salient Region Detection",
// IEEE International Conference on Computer Vision and Pattern Recognition (CVPR), 2009
//===========================================================================
// Copyright (c) 2011 Radhakrishna Achanta [EPFL]
//===========================================================================

#pragma once

#include "Vector3.h"
#include "Image.h"
#include "Saliency.h"

namespace irt
{

class HSaliency
{
public:
	HSaliency(int width = 0, int height = 0) : m_image(0)
	{
		reset(width, height);
	}

	~HSaliency()
	{
		destroy();
	}

	void destroy()
	{
		if(m_image)
		{
			/*
			for(int i=0;i<m_width;i++)
			{
				delete[] m_image[i];
				delete[] m_imageLAB[i];
				delete[] m_smoothImage[i];
				delete[] m_saliency[i];
			}
			*/
			delete[] m_image;
			delete[] m_imageLAB;
			delete[] m_smoothImage;
			delete[] m_saliency;
		}
	}

	void clear()
	{
		/*
		for(int i=0;i<m_width;i++)
		{
			memset(m_image[i], 0, sizeof(Vector3)*m_height);
			//memset(m_saliency[i], 0, sizeof(float)*m_height);
		}
		*/
		memset(m_image, 0, sizeof(Vector3)*m_width*m_height);
		memset(m_imageLAB, 0, sizeof(Vector3)*m_width*m_height);
		memset(m_smoothImage, 0, sizeof(Vector3)*m_width*m_height);
		memset(m_saliency, 0, sizeof(float)*m_width*m_height);
	}

	void reset(int width, int height)
	{
		destroy();
		m_width = width;
		m_height = height;
		/*
		m_image = new Vector3*[width];
		m_imageLAB = new Vector3*[width];
		m_smoothImage = new Vector3*[width];
		m_saliency = new float*[width];
		*/
			/*
		for(int i=0;i<width;i++)
		{
			m_image[i] = new Vector3[height];
			m_imageLAB[i] = new Vector3[height];
			m_smoothImage[i] = new Vector3[height];
			m_saliency[i] = new float[height];
		}
			*/
		m_image = new Vector3[width*height];
		m_imageLAB = new Vector3[width*height];
		m_smoothImage = new Vector3[width*height];
		m_saliency = new float[width*height];
	}

	void set(Image *image)
	{
		float maxValue = -FLT_MAX;
		for(int x=0;x<image->width;x++)
		{
			for(int y=0;y<image->height;y++)
			{
				RGBf color;
				image->getPixel(x, y, color);
				/*
				int tX = x*m_width/image->width;
				int tY = y*m_height/image->height;
				m_image[tX][tY].e[0] += color.e[2];
				m_image[tX][tY].e[1] += color.e[1];
				m_image[tX][tY].e[2] += color.e[0];

				maxValue = max(maxValue, m_image[tX][tY].e[0]);
				maxValue = max(maxValue, m_image[tX][tY].e[1]);
				maxValue = max(maxValue, m_image[tX][tY].e[2]);
				*/
				int offset = (x*m_width/image->width) + (y*m_height/image->height)*m_width;
				m_image[offset].e[0] += color.e[0];
				m_image[offset].e[1] += color.e[1];
				m_image[offset].e[2] += color.e[2];

				maxValue = max(maxValue, m_image[offset].e[0]);
				maxValue = max(maxValue, m_image[offset].e[1]);
				maxValue = max(maxValue, m_image[offset].e[2]);
			}
		}

		/*
		for(int x=0;x<m_width;x++)
		{
			for(int y=0;y<m_height;y++)
			{
				m_image[x][y].e[0] /= maxValue;
				m_image[x][y].e[1] /= maxValue;
				m_image[x][y].e[2] /= maxValue;
			}
		}
		*/
		for(int i=0;i<m_width*m_height;i++)
		{
			m_image[i].e[0] /= maxValue;
			m_image[i].e[1] /= maxValue;
			m_image[i].e[2] /= maxValue;
		}

		// rgb->lab
		double labSum[3] = {0.0, };
		for(int x=0;x<m_width;x++)
		{
			for(int y=0;y<m_height;y++)
			{
				//------------------------
				// sRGB to XYZ conversion
				// (D65 illuminant assumption)
				//------------------------
				/*
				float R = m_image[x][y].e[0];
				float G = m_image[x][y].e[1];
				float B = m_image[x][y].e[2];
				*/
				float R = m_image[x+y*m_width].e[0];
				float G = m_image[x+y*m_width].e[1];
				float B = m_image[x+y*m_width].e[2];

				float r, g, b;

				if(R <= 0.04045f)	r = R/12.92f;
				else				r = pow((R+0.055f)/1.055f,2.4f);
				if(G <= 0.04045f)	g = G/12.92f;
				else				g = pow((G+0.055f)/1.055f,2.4f);
				if(B <= 0.04045f)	b = B/12.92f;
				else				b = pow((B+0.055f)/1.055f,2.4f);

				float X = r*0.4124564f + g*0.3575761f + b*0.1804375f;
				float Y = r*0.2126729f + g*0.7151522f + b*0.0721750f;
				float Z = r*0.0193339f + g*0.1191920f + b*0.9503041f;
				//------------------------
				// XYZ to LAB conversion
				//------------------------
				float epsilon = 0.008856f;	//actual CIE standard
				float kappa   = 903.3f;		//actual CIE standard

				float Xr = 0.950456f;	//reference white
				float Yr = 1.0f;		//reference white
				float Zr = 1.088754f;	//reference white

				float xr = X/Xr;
				float yr = Y/Yr;
				float zr = Z/Zr;

				float fx, fy, fz;
				if(xr > epsilon)	fx = pow(xr, 1.0f/3.0f);
				else				fx = (kappa*xr + 16.0f)/116.0f;
				if(yr > epsilon)	fy = pow(yr, 1.0f/3.0f);
				else				fy = (kappa*yr + 16.0f)/116.0f;
				if(zr > epsilon)	fz = pow(zr, 1.0f/3.0f);
				else				fz = (kappa*zr + 16.0f)/116.0f;

				/*
				m_imageLAB[x][y].e[0] = 116.0f*fy-16.0f;
				m_imageLAB[x][y].e[1] = 500.0f*(fx-fy);
				m_imageLAB[x][y].e[2] = 200.0f*(fy-fz);

				labSum[0] += m_imageLAB[x][y].e[0];
				labSum[1] += m_imageLAB[x][y].e[1];
				labSum[2] += m_imageLAB[x][y].e[2];
				*/
				m_imageLAB[x+y*m_width].e[0] = 116.0f*fy-16.0f;
				m_imageLAB[x+y*m_width].e[1] = 500.0f*(fx-fy);
				m_imageLAB[x+y*m_width].e[2] = 200.0f*(fy-fz);

				/*
				labSum[0] += m_imageLAB[x+y*m_width].e[0];
				labSum[1] += m_imageLAB[x+y*m_width].e[1];
				labSum[2] += m_imageLAB[x+y*m_width].e[2];
				*/
				labSum[0] += R;
				labSum[1] += G;
				labSum[2] += B;
			}
		}

		m_labAvg[0] = (float)(labSum[0]/(m_width*m_height));
		m_labAvg[1] = (float)(labSum[1]/(m_width*m_height));
		m_labAvg[2] = (float)(labSum[2]/(m_width*m_height));
	}

	void gaussianSmooth(int axis, const Vector3 *inputImg, Vector3 *tempim, Vector3 *smoothImg, const std::vector<float>& kernel)
	{
		int center = int(kernel.size())/2;

		int sz = m_width*m_height;
		int rows = m_height;
		int cols = m_width;
	   //--------------------------------------------------------------------------
	   // Blur in the x direction.
	   //---------------------------------------------------------------------------
		{int index(0);
		for( int r = 0; r < rows; r++ )
		{
			for( int c = 0; c < cols; c++ )
			{
				double kernelsum(0);
				double sum(0);
				for( int cc = (-center); cc <= center; cc++ )
				{
					if(((c+cc) >= 0) && ((c+cc) < cols))
					{
						sum += inputImg[r*cols+(c+cc)].e[axis] * kernel[center+cc];
						kernelsum += kernel[center+cc];
					}
				}
				tempim[r*cols+c].e[axis] = (float)(sum/kernelsum);
				index++;
			}
		}}

		//--------------------------------------------------------------------------
		// Blur in the y direction.
		//---------------------------------------------------------------------------
		{int index = 0;
		for( int r = 0; r < rows; r++ )
		{
			for( int c = 0; c < cols; c++ )
			{
				double kernelsum(0);
				double sum(0);
				for( int rr = (-center); rr <= center; rr++ )
				{
					if(((r+rr) >= 0) && ((r+rr) < rows))
					{
					   sum += tempim[(r+rr)*cols+c].e[axis] * kernel[center+rr];
					   kernelsum += kernel[center+rr];
					}
				}
				smoothImg[r*cols+c].e[axis] = (float)(sum/kernelsum);
				index++;
			}
		}}
	}

	void calculate()
	{
		//----------------------------------------------------
		// The kernel can be [1 2 1] or [1 4 6 4 1] as needed.
		// The code below show usage of [1 2 1] kernel.
		//----------------------------------------------------
		std::vector<float> kernel(0);
		kernel.push_back(1.0f);
		kernel.push_back(2.0f);
		kernel.push_back(1.0f);

		/*
		gaussianSmooth(0, m_imageLAB, m_image, m_smoothImage, kernel);
		gaussianSmooth(1, m_imageLAB, m_image, m_smoothImage, kernel);
		gaussianSmooth(2, m_imageLAB, m_image, m_smoothImage, kernel);
		*/
		gaussianSmooth(0, m_image, m_imageLAB, m_smoothImage, kernel);
		gaussianSmooth(1, m_image, m_imageLAB, m_smoothImage, kernel);
		gaussianSmooth(2, m_image, m_imageLAB, m_smoothImage, kernel);

		for(int i=0;i<m_width*m_height;i++)
		{
			m_saliency[i] =	(m_smoothImage[i].e[0]-m_labAvg[0])*(m_smoothImage[i].e[0]-m_labAvg[0]) +
							(m_smoothImage[i].e[1]-m_labAvg[1])*(m_smoothImage[i].e[1]-m_labAvg[1]) +
							(m_smoothImage[i].e[2]-m_labAvg[2])*(m_smoothImage[i].e[2]-m_labAvg[2]);
		}

		normalize(m_saliency, m_saliency);

		/*
		Saliency saliency;
		saliency.GetSaliencyMap(m_imageUINT, m_width, m_height, m_saliency, true);
		*/

		/*
		Image image(m_width, m_height);
		for(int x=0;x<m_width;x++)
			for(int y=0;y<m_height;y++)
				//image.setPixel(x, y, RGBf(m_image[x][y].e[0], m_image[x][y].e[1], m_image[x][y].e[2]));
				//image.setPixel(x, y, RGBf((m_imageUINT[x+y*m_width] >> 16) / 255.0f, (m_imageUINT[x+y*m_width] >> 8) / 255.0f, (m_imageUINT[x+y*m_width] >> 0) / 255.0f));
				//image.setPixel(x, y, RGBf(m_smoothImage[x+y*m_width].e[0], m_smoothImage[x+y*m_width].e[1], m_smoothImage[x+y*m_width].e[2]));
				//image.setPixel(x, y, RGBf(m_imageLAB[x+y*m_width].e[0], m_imageLAB[x+y*m_width].e[1], m_imageLAB[x+y*m_width].e[2]));
				image.setPixel(x, y, RGBf(m_saliency[x+y*m_width]/255.0f, m_saliency[x+y*m_width]/255.0f, m_saliency[x+y*m_width]/255.0f));
		static int ss = 0;
		char fileName[255];
		sprintf(fileName, "saliency%03d.bmp", ss++);
		image.writeToFile(fileName);
		*/
	}

	void normalize(const float *input, float *output, const int& normrange = 255)
	{
		float maxval(-FLT_MAX);
		float minval(FLT_MAX);
		{int i(0);
		for( int y = 0; y < m_height; y++ )
		{
			for( int x = 0; x < m_width; x++ )
			{
				if( maxval < input[i] ) maxval = input[i];
				if( minval > input[i] ) minval = input[i];
				i++;
			}
		}}
		float range = maxval-minval;
		if( 0 == range ) range = 1;
		int i(0);
		for( int y = 0; y < m_height; y++ )
		{
			for( int x = 0; x < m_width; x++ )
			{
				output[i] = ((normrange*(input[i]-minval))/range);
				i++;
			}
		}
	}

	float getSaliency(int x, int y) 
	{
		//return m_saliency[x][y];
		return m_saliency[x+y*m_width];
	}

	public:
//protected:
	int m_width;
	int m_height;
	/*
	Vector3 **m_image;
	std::vector<UINT> m_imageUINT;
	std::vector<double> m_saliency;
	*/
	Vector3 *m_image;
	Vector3 *m_imageLAB;
	Vector3 *m_smoothImage;
	float *m_saliency;
	float m_labAvg[3];
};

};