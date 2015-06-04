/********************************************************************
	created:	2011/07/29
	file path:	d:\Projects\Redering\OpenIRT\include
	file base:	OpenIRT
	file ext:	h
	author:		Tae-Joon Kim (tjkim.kaist@gmail.com)
	
	comment:	Wrapper class for interactive ray tracer
*********************************************************************/

#pragma once

#include <map>
#include "controls.h"

namespace irt
{
class Renderer;
class Camera;
class Scene;
class Model;
class Image;
};

class OpenIRT
{
public:

	typedef std::map<std::string, irt::Camera*> CameraMap;

protected:
	irt::Camera *m_currentCamera;
	irt::Scene *m_currentScene;
	irt::Renderer *m_renderer;

	CameraMap m_cameraMap;

	float m_currentFrameTime;
	float m_currentFPS;

	Progress m_progress;

	int m_width, m_height;

	bool m_isInitialized;

	std::map<irt::Scene*, bool> m_sceneList;

	unsigned int m_timer;
public:
	OpenIRT(RendererType::Type rendererType = RendererType::NONE);
	~OpenIRT(void);

	static OpenIRT *getSingletonPtr();

	void setController(const Controller &controller);
	Controller *getController();

	irt::Camera *getCurrentCamera();
	irt::Scene *getCurrentScene();
	irt::Renderer *getRenderer();

	float getCurrentFrameTime();
	float getCurrentFPS();

	void setCurrentFrameTime(float frameTime);
	void setCurrentFPS(float FPS);

	void setCurrentCamera(const char *cameraName);
	void setCurrentCamera(irt::Camera *camera);

	void setCurrentScene(irt::Scene *scene);

	void setProgress(const Progress &progress);
	Progress &getProgress();

	bool loadScene(char *sceneFileName);
	bool loadScene(irt::Scene *scene);

	void exportScene(char *sceneFileName);

	CameraMap &getCameraMap();
	void pushCamera(const char *cameraName, irt::Camera *camera);

	// OpenGL style camera setup
	void pushCamera(const char *cameraName, float *eye, float *center, float *up, float fovy, float aspect, float zNear, float zFar);
	void pushCamera(const char *cameraName, float eyeX, float eyeY, float eyeZ, float centerX, float centerY, float centerZ, float upX, float upY, float upZ, float fovy, float aspect, float zNear, float zFar);

	void clearCameras();

	void resetRenderer(RendererType::Type rendererType, void *renderingContext = NULL, void *renderingDC = NULL);
	void init(int width, int height);
	void init(RendererType::Type rendererType, int width, int height, void *renderingContext = NULL, void *renderingDC = NULL);

	void prepareRender();

	void render(irt::Image *image = NULL, unsigned int seed = UINT_MAX);

	void clearResult();

	void flushImage(irt::Image *image);

	void resized(int width, int height);

	void materialChanged();

	void lightChanged(bool soft = false);

	void controllerUpdated();

	void doneRenderer();

	void setRenderDoneCallBack(RenderDoneCallBack function);
};

