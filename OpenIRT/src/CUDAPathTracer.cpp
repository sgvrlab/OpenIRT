#include "CUDAPathTracer.h"

#ifndef fminf
#define fminf(a,b) (((a) < (b)) ? (a) : (b))
#endif

#ifndef fmaxf
#define fmaxf(a,b) (((a) > (b)) ? (a) : (b))
#endif

#include "CUDA/CUDADataStructures.cuh"
#include <process.h>


extern "C" void loadSceneCUDAPathTracer(CUDA::Scene *scene);
extern "C" void renderCUDAPathTracer(CUDA::Camera *camera, CUDA::Image *image, CUDA::Controller *controller, int frame, int seed);
extern "C" void materialChangedCUDAPathTracer(CUDA::Scene *scene);
extern "C" void lightChangedCUDAPathTracer(CUDA::Scene *scene);
extern "C" void getImageCUDAPathTracer(CUDA::Image *image);
extern "C" void getDepthImageCUDAPathTracer(CUDA::Image *image);
extern "C" void getNormalImageCUDAPathTracer(CUDA::Image *image);
extern "C" void gBufferFilterCUDAPathTracer(CUDA::Image *image);
extern "C" void updateControllerCUDAPathTracer(CUDA::Controller *controller);
extern "C" void unloadSceneCUDAPathTracer();

using namespace irt;

CUDAPathTracer::CUDAPathTracer(void)
{
	m_dstScene = 0;

	extern unsigned __stdcall initCUDAContext(void *arg);
	_beginthreadex(NULL, 0, initCUDAContext, NULL, 0, NULL);
}

CUDAPathTracer::~CUDAPathTracer(void)
{
	done();

	//clearScene();
	unloadSceneCUDAPathTracer();
}
void CUDAPathTracer::init(Scene *scene)
{
	Renderer::init(scene);

	Vector3 rot(-19.8f, -3.6f, 0.0f);

	Matrix mat = rotateX(rot.x() * PI / 180.0f) * rotateY(rot.y() * PI / 180.0f) * rotateZ(rot.z() * PI / 180.0f);

	Vector4 u = (rotateX(rot.x() * PI / 180.0f) * rotateY(rot.y() * PI / 180.0f) * rotateZ(rot.z() * PI / 180.0f)) * Vector4(-1.0f, 0.0f, 0.0f);
	Vector4 v = (rotateX(rot.x() * PI / 180.0f) * rotateY(rot.y() * PI / 180.0f) * rotateZ(rot.z() * PI / 180.0f)) * Vector4(0.0f, 1.0f, 0.0f);
	Vector4 w = (rotateX(rot.x() * PI / 180.0f) * rotateY(rot.y() * PI / 180.0f) * rotateZ(rot.z() * PI / 180.0f)) * Vector4(0.0f, 0.0f, -1.0f);

	if(!scene) return;

	done();

	m_controller.drawBackground = false;

	m_intersectionStream = scene->getIntersectionStream();

	sceneChanged();

	adaptScene(scene, m_dstScene);
	loadSceneCUDAPathTracer((CUDA::Scene*)m_dstScene);
}

void CUDAPathTracer::done()
{
}

void CUDAPathTracer::resized(int width, int height)
{
	Renderer::resized(width, height);
}

void CUDAPathTracer::sceneChanged()
{
}

void CUDAPathTracer::lightChanged(bool soft)
{
	CUDA::Scene *sceneCUDA = (CUDA::Scene *)m_dstScene;
	Scene *sceneCPU = m_scene;

	sceneCUDA->numEmitters = 0;

	int numEmitters = sceneCPU->getNumEmitters();
	for(int i=0;i<numEmitters;i++)
	{
		if(sceneCPU->getEmitter(i).type != Emitter::ENVIRONMENT_LIGHT)
			sceneCUDA->numEmitters++;
	}

	if(sceneCUDA->emitters) delete[] sceneCUDA->emitters;
	sceneCUDA->emitters = new CUDA::Emitter[sceneCUDA->numEmitters];

	for(int i=0, j=0;i<numEmitters;i++)
	{
		Emitter &emitter = sceneCPU->getEmitter(i);
		if(emitter.type == Emitter::ENVIRONMENT_LIGHT)
		{
			memcpy(&sceneCUDA->envColor, &emitter.color_Kd, sizeof(CUDA::Vector3));
		}
		else
		{
			memcpy(&sceneCUDA->emitters[j++], &emitter, sizeof(CUDA::Emitter));
		}
	}

	lightChangedCUDAPathTracer(sceneCUDA);
}

void CUDAPathTracer::getCurrentColorImage(Image *image)
{
	getImageCUDAPathTracer((CUDA::Image*)&image->width);
}

void CUDAPathTracer::getCurrentDepthImage(Image *image)
{
	getDepthImageCUDAPathTracer((CUDA::Image*)&image->width);
}

void CUDAPathTracer::getCurrentNormalImage(Image *image)
{
	getNormalImageCUDAPathTracer((CUDA::Image*)&image->width);
}

void CUDAPathTracer::materialChanged()
{
	Scene *sceneCPU = (Scene*)m_scene;
	CUDA::Scene *sceneCUDA = (CUDA::Scene*)m_dstScene;

	for(int i=0;i<sceneCUDA->numModels;i++)
	{
		Model &modelCPU = *(sceneCPU->getModelList()[i]);
		CUDA::Model &modelCUDA = sceneCUDA->models[i];

		for(int j=0;j<modelCUDA.numMats;j++)
		{
			Material &matCPU = modelCPU.getMaterial(j);
			memcpy(&modelCUDA.mats[j], &matCPU, sizeof(Material) - sizeof(BitmapTexture *)*3);
		}
	}

	materialChangedCUDAPathTracer(sceneCUDA);
}

void CUDAPathTracer::filter(Image *image)
{
	updateControllerCUDAPathTracer((CUDA::Controller *)&m_controller);
	gBufferFilterCUDAPathTracer((CUDA::Image*)&image->width);
}

void CUDAPathTracer::render(Camera *camera, Image *image, unsigned int seed)
{
	CUDA::Camera cameraCUDA;

	memset(image->data, 0, image->width*image->height*image->bpp);
	//adaptDataStructures(m_scene, camera, &sceneCUDA, &cameraCUDA);

	extern int g_frame;

	if(m_materialUpdated)
	{
		applyChangedMaterial();
		m_materialUpdated = false;
	}

	renderCUDAPathTracer((CUDA::Camera*)camera, (CUDA::Image*)&image->width, (CUDA::Controller *)&m_controller, g_frame++, seed);
}