#include "CommonHeaders.h"
#include "OpenIRT.h"
#include "Renderer.h"
#include "Camera.h"
#include "Scene.h"
#include "Image.h"

#include "defines.h"
#include "handler.h"

#include "CPURayTracer.h"
#include "GLDebugging.h"
#include "SimpleRasterizer.h"
#include "CUDARayTracer.h"
#include "CUDAPathTracer.h"
#include "CUDAPhotonMapping.h"
#include "TReX.h"

#include <stopwatch.h>

using namespace irt;

OpenIRT::OpenIRT(RendererType::Type rendererType)
	: m_currentCamera(0), m_currentScene(0), m_renderer(0), m_currentFPS(0), m_isInitialized(0), m_width(0), m_height(0)
{
	if(rendererType != RendererType::NONE) 
		resetRenderer(rendererType);

	m_timer = StopWatch::create();
}

OpenIRT::~OpenIRT(void)
{
	clearCameras();

	for(std::map<irt::Scene*, bool>::iterator it=m_sceneList.begin();it!=m_sceneList.end();it++)
		if(it->second) delete it->first;
	m_sceneList.clear();

	if(m_renderer)
		delete m_renderer;

	StopWatch::destroy(m_timer);
}

OpenIRT *OpenIRT::getSingletonPtr()
{
	static OpenIRT renderer;
	return &renderer;
}

bool OpenIRT::loadScene(char *sceneFileName)
{
	m_currentScene = new Scene();
	m_sceneList[m_currentScene] = true;

	bool ret = m_currentScene->load(sceneFileName);

	if(!m_currentScene->hasSceneStructure())
		m_currentScene->generateSceneStructure();
	if(!m_currentScene->hasEmitters())
		m_currentScene->generateEmitter();
	for(int i=0;i<m_currentScene->getNumEmitters();i++)
	{
		Emitter &emitter = m_currentScene->getEmitter(i);
	}
	return ret;
}

bool OpenIRT::loadScene(irt::Scene *scene)
{
	std::map<irt::Scene*, bool>::iterator it = m_sceneList.find(scene);
	if(it != m_sceneList.end())
		if(it->second) delete it->first;

	m_sceneList[scene] = false;

	setCurrentScene(scene);

	return true;
}

void OpenIRT::exportScene(char *sceneFileName)
{
	if(!m_currentScene) return ;

	m_currentScene->exportScene(sceneFileName);
}

void OpenIRT::pushCamera(const char *cameraName, Camera *camera)
{
	Camera *cam = new Camera;
	*cam = *camera;

	cam->setName(cameraName);
	m_cameraMap[cameraName] = cam;

	m_currentCamera = cam;
}

void OpenIRT::pushCamera(const char *cameraName, float *eye, float *center, float *up, float fovy, float aspect, float zNear, float zFar)
{
	Camera *cam = new Camera(eye[0], eye[1], eye[2], center[0], center[1], center[2], up[0], up[1], up[2], fovy, aspect, zNear, zFar);

	pushCamera(cameraName, cam);

	delete cam;
}

void OpenIRT::pushCamera(const char *cameraName, float eyeX, float eyeY, float eyeZ, float centerX, float centerY, float centerZ, float upX, float upY, float upZ, float fovy, float aspect, float zNear, float zFar)
{
	Camera *cam = new Camera(eyeX, eyeY, eyeZ, centerX, centerY, centerZ, upX, upY, upZ, fovy, aspect, zNear, zFar);

	pushCamera(cameraName, cam);

	delete cam;
}

void OpenIRT::clearCameras()
{
	for(CameraMap::iterator it=m_cameraMap.begin();it!=m_cameraMap.end();it++)
	{
		if(it->second)
			delete it->second;
	}
	m_cameraMap.clear();
}

void OpenIRT::resetRenderer(RendererType::Type rendererType, void *renderingContext, void *renderingDC)
{
	if(m_renderer) doneRenderer();

	switch(rendererType)
	{
	case RendererType::CPU_RAY_TRACER :
		m_renderer = new CPURayTracer();
		break;
	case RendererType::DEBUGGING :
		m_renderer = new GLDebugging();
		((GLDebugging*)m_renderer)->initGL(m_width, m_height, renderingContext, renderingDC);
		break;
	case RendererType::SIMPLE_RASTERIZER :
		m_renderer = new SimpleRasterizer();
		((SimpleRasterizer*)m_renderer)->initGL(m_width, m_height, renderingContext, renderingDC);
		break;
	case RendererType::CUDA_PATH_TRACER :
		m_renderer = new CUDAPathTracer();
		break;
	case RendererType::CUDA_PHOTON_MAPPING :
		m_renderer = new CUDAPhotonMapping();
		break;
	case RendererType::TREX :
		m_renderer = new TReX();
		break;
	case RendererType::CUDA_RAY_TRACER :
		m_renderer = new CUDARayTracer();
		break;
	}
	m_renderer->setRendererType(rendererType);

	m_isInitialized = false;
}

void OpenIRT::init(int width, int height)
{
	m_renderer->init(m_currentScene);

	m_renderer->resized(width, height);

	m_width = width;
	m_height = height;

	m_isInitialized = true;
}

void OpenIRT::init(RendererType::Type rendererType, int width, int height, void *renderingContext, void *renderingDC)
{
	if(rendererType == RendererType::NONE)
	{
		m_isInitialized = false;
		return;
	}

	resetRenderer(rendererType, renderingContext, renderingDC);

	init(width, height);
}

void OpenIRT::doneRenderer()
{
	if(!m_renderer) return;

	m_renderer->done();

	delete m_renderer;
	m_renderer = NULL;
}

void OpenIRT::prepareRender()
{
	if(m_renderer) m_renderer->prepareRender();
}

void OpenIRT::render(Image *image, unsigned int seed)
{
	if(!m_isInitialized)
	{
		printf("Initialize renderer...");

		if(image) init(image->width, image->height);
		else init(m_width, m_height);
		printf("done\n");
	}

	if(image)
	{
		if(m_renderer->getWidth() != image->width || m_renderer->getHeight() != image->height)
			m_renderer->resized(image->width, image->height);
	}

	if(seed == UINT_MAX)
	{
		static unsigned int iter = 0;
		seed = iter++;
	}

	StopWatch::get(m_timer).start();
	m_renderer->render(m_currentCamera, image, seed);
	StopWatch::get(m_timer).stop();


	m_currentFrameTime = StopWatch::get(m_timer).getTime();
	StopWatch::get(m_timer).reset();

	m_currentFPS = 1000.0f / m_currentFrameTime;
}

void OpenIRT::clearResult()
{
	m_renderer->clearResult();
}

void OpenIRT::flushImage(irt::Image *image)
{
	m_renderer->flushImage(image);
}

void OpenIRT::resized(int width, int height)
{
	m_currentCamera->setAspect(width/(float)height);
	m_renderer->resized(width, height);

	m_width = width;
	m_height = height;
}

void OpenIRT::materialChanged()
{
	m_renderer->materialChanged();
}

void OpenIRT::lightChanged(bool soft)
{
	m_renderer->lightChanged(soft);
}

void OpenIRT::controllerUpdated()
{
	m_renderer->controllerUpdated();
}

void OpenIRT::setController(const Controller &controller) 
{
	if(m_renderer) m_renderer->setController(controller);
}

Controller *OpenIRT::getController() 
{
	if(m_renderer) return m_renderer->getController();
	return NULL;
}

Camera *OpenIRT::getCurrentCamera() 
{
	return m_currentCamera;
}

Scene *OpenIRT::getCurrentScene() 
{
	return m_currentScene;
}

float OpenIRT::getCurrentFrameTime()
{
	return m_currentFrameTime;
}

float OpenIRT::getCurrentFPS()
{
	return m_currentFPS;
}

Renderer *OpenIRT::getRenderer()
{
	return m_renderer;
}

void OpenIRT::setCurrentFrameTime(float frameTime)
{
	m_currentFrameTime = frameTime;
}

void OpenIRT::setCurrentFPS(float FPS)
{
	m_currentFPS = FPS;
}

void OpenIRT::setCurrentCamera(const char *cameraName)
{
	*m_currentCamera = *m_cameraMap[cameraName];
}

void OpenIRT::setCurrentCamera(Camera *camera)
{
	m_currentCamera = camera;
}

void OpenIRT::setCurrentScene(irt::Scene *scene)
{
	m_currentScene = scene;

	if(m_renderer) m_renderer->setScene(scene);
}

void OpenIRT::setProgress(const Progress &progress)
{
	m_progress = progress;
}

Progress &OpenIRT::getProgress()
{
	return m_progress;
}

OpenIRT::CameraMap &OpenIRT::getCameraMap()
{
	return m_cameraMap;
}

void OpenIRT::setRenderDoneCallBack(RenderDoneCallBack function)
{
	m_renderer->setRenderDoneCallBack(function);
}