/********************************************************************
	created:	2011/07/28
	file path:	d:\Projects\Redering\OpenIRT\include
	file base:	Renderer
	file ext:	h
	author:		Tae-Joon Kim (tjkim.kaist@gmail.com)
	
	comment:	Abstract class for various renderers. All derived classes must override the render() method.
*********************************************************************/

#pragma once

#include <stopwatch.h>
#include "controls.h"
#include "Camera.h"
#include "Scene.h"
#include "Image.h"

namespace irt
{
class Renderer
{
public:
	typedef struct RayPixelPosition_t
	{
		int x, y;

		RayPixelPosition_t() : x(-1), y(-1) {}
		RayPixelPosition_t(int x, int y) : x(x), y(y) {}
	} RayPixelPosition;

protected:
	int m_width;
	int m_height;

	Scene *m_scene;
	RendererType::Type m_rendererType;
	RayPixelPosition *m_rayOrder;
	int m_curRayOrder;
	Controller m_controller;
	Controller m_controllerBackup;
	RenderDoneCallBack m_renderDoneCallBack;
	int m_intersectionStream;
public:

	Renderer(void) : m_rayOrder(0), m_renderDoneCallBack(0), m_width(0), m_height(0) {}
	virtual ~Renderer(void) {if(m_rayOrder) delete[] m_rayOrder; m_rayOrder = NULL;}

	void setRendererType(RendererType::Type type) {m_rendererType = type;}
	RendererType::Type getRendererType() {return m_rendererType;}
	void setController(const Controller &controller) {m_controller = controller;}

	virtual void init(Scene *scene) {m_scene = scene;}
	virtual void done() {}
	virtual void restart() {}

	void setRenderDoneCallBack(RenderDoneCallBack function) {m_renderDoneCallBack = function;}

	int getWidth() {return m_width;}
	int getHeight() {return m_height;}
	void setScene(Scene *scene) {m_scene = scene;}

	virtual void sceneChanged() {}
	virtual void materialChanged() {}
	virtual void lightChanged(bool soft = false) {}
	virtual void controllerUpdated() {}
	virtual void resetModel(Model *model) {}

	virtual void getCurrentColorImage(Image *image) {}
	virtual void getCurrentDepthImage(Image *image) {}
	virtual void getCurrentNormalImage(Image *image) {}

	virtual void filter(Image *image) {}

	virtual void clearResult() {}

	virtual void flushImage(Image *image) {}

	virtual void prepareRender() {}
	virtual void render(Camera *camera, Image *image, unsigned int seed = UINT_MAX) = 0;

	bool getUseZCurveOrdering() {return m_controller.useZCurveOrdering;}
	void setUseZCurveOrdering(bool use) {m_controller.useZCurveOrdering = use;}
	Controller *getController() {return &m_controller;}
	Controller *getControllerBackup() {return &m_controllerBackup;}

	virtual void resized(int width, int height)
	{
		if(m_width == width && m_height == height) return;

		if(m_controller.useZCurveOrdering) computeZCurveOrder(width, height);
		//else computeRowByRowOrder(width, height);
		else computeTileRowByRowOrder(width, height, 8, 8);

		m_width = width;
		m_height = height;
	}

	void convert3DByteTo4DFloat(float *dst, unsigned char *src, int numFloat)
	{
		for(int j=0, k=0;j<numFloat;j++)
		{
			if(j % 4 == 3)
				dst[j] = 1.0f;
			else
			{
				dst[j] = src[k++] / 255.0f;
			}
		}
	}

	void convert4DByteTo4DFloat(float *dst, unsigned char *src, int numFloat)
	{
		for(int j=0;j<numFloat;j++)
		{
			dst[j] = src[j] / 255.0f;
		}
	}

	void computeZCurveOrder(int width, int height)
	{
		if(m_rayOrder) delete[] m_rayOrder;
		m_rayOrder = new RayPixelPosition[width*height];
		m_curRayOrder = 0;
		computeZCurveOrder(RayPixelPosition(0, 0), RayPixelPosition(width, height));
	}

	void computeZCurveOrder(const RayPixelPosition &topLeft, const RayPixelPosition &bottomRight)
	{
		RayPixelPosition window(bottomRight.x - topLeft.x, bottomRight.y - topLeft.y);

		if(window.x == 0 || window.y == 0) return;

		if(window.x == 1 && window.y == 1)
		{
			m_rayOrder[m_curRayOrder++] = topLeft;
			return;
		}

		// compute sub box
		int size = 1;
		while(size * 2 < window.x && size * 2 < window.y) size *= 2;

		RayPixelPosition divide(topLeft.x + size, topLeft.y + size);
		computeZCurveOrder(divide, bottomRight);
		computeZCurveOrder(RayPixelPosition(topLeft.x, divide.y) , RayPixelPosition(divide.x, bottomRight.y));
		computeZCurveOrder(RayPixelPosition(divide.x, topLeft.y) , RayPixelPosition(bottomRight.x, divide.y));
		computeZCurveOrder(topLeft, divide);
	}

	void computeRowByRowOrder(int width, int height)
	{
		if(m_rayOrder) delete[] m_rayOrder;
		m_rayOrder = new RayPixelPosition[width*height];
		m_curRayOrder = 0;
		for(int i=0;i<height;i++)
			for(int j=0;j<width;j++)
			{
				m_rayOrder[m_curRayOrder].x = j;
				m_rayOrder[m_curRayOrder++].y = i;
			}
	}

	void computeTileRowByRowOrder(int width, int height, int tileWidth, int tileHeight)
	{
		if(m_rayOrder) delete[] m_rayOrder;
		m_rayOrder = new RayPixelPosition[width*height];
		m_curRayOrder = 0;
		unsigned int tilesX = width / tileWidth;
		unsigned int tilesY = height / tileHeight;
		int numTiles = tilesX * tilesY;	

		for (int curTile = 0; curTile < numTiles; curTile++) 
		{
			unsigned int start_x = (curTile % tilesX) * tileWidth;
			unsigned int start_y = (unsigned int)(curTile / tilesY) * tileHeight;	
		
			for (int y = 0; y < tileHeight; y++)
				for (int x = 0; x < tileWidth; x++)
				{
					m_rayOrder[m_curRayOrder].x = start_x+x;
					m_rayOrder[m_curRayOrder++].y = start_y+y;
				}
		}
	}

};

};