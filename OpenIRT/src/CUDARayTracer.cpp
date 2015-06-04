#include "CUDARayTracer.h"
#include <process.h>

#ifndef fminf
#define fminf(a,b) (((a) < (b)) ? (a) : (b))
#endif

#ifndef fmaxf
#define fmaxf(a,b) (((a) > (b)) ? (a) : (b))
#endif

#include "CUDA/CUDADataStructures.cuh"

using namespace irt;

unsigned __stdcall initCUDAContext(void *arg)
{
	cudaFree(0);
	_endthreadex(0);
	return 0;
}

CUDARayTracer::CUDARayTracer(void)
{
	m_dstScene = 0;
	m_materialUpdated = false;
}

CUDARayTracer::~CUDARayTracer(void)
{
	done();

	clearScene();

	printf("destroy CUDA context.\n");
	cudaDeviceReset();
}

unsigned int CUDARayTracer::makeCUDASceneGraph(void *srcSceneNode, void *dstSceneNode, void *dstScene, int &index)
{
	SceneNode *sceneNodeCPU = (SceneNode*)srcSceneNode;
	CUDA::SceneNode *sceneNodeCUDA = (CUDA::SceneNode*)dstSceneNode;
	CUDA::Scene *sceneCUDA = (CUDA::Scene*)dstScene;

	int curIdx = index++;

	//memcpy(&(sceneNodeCUDA->bbMin), &(sceneNodeCPU->nodeBB), sizeof(AABB));
	sceneNodeCUDA->bbMin.x = sceneNodeCPU->nodeBB.min.x();
	sceneNodeCUDA->bbMin.y = sceneNodeCPU->nodeBB.min.y();
	sceneNodeCUDA->bbMin.z = sceneNodeCPU->nodeBB.min.z();
	sceneNodeCUDA->bbMax.x = sceneNodeCPU->nodeBB.max.x();
	sceneNodeCUDA->bbMax.y = sceneNodeCPU->nodeBB.max.y();
	sceneNodeCUDA->bbMax.z = sceneNodeCPU->nodeBB.max.z();
	sceneNodeCUDA->modelIdx = m_scene->getModelListMap()[sceneNodeCPU->model];

	int numChilds = sceneNodeCPU->childs ? (int)sceneNodeCPU->childs->size() : 0;
	sceneNodeCUDA->numChilds = numChilds;
	for(int i=0;i<numChilds;i++)
	{
		sceneNodeCUDA->childIdx[i] = makeCUDASceneGraph(sceneNodeCPU->childs->at(i), &sceneCUDA->sceneGraph[index], sceneCUDA, index);
	}

	return curIdx;
}

int CUDARayTracer::adaptScene(void *srcScene, void *dstScene)
{
	m_dstScene = dstScene;

	clearScene();

	m_dstScene = dstScene = new CUDA::Scene;

	Scene *sceneCPU = (Scene*)srcScene;
	CUDA::Scene *sceneCUDA = (CUDA::Scene*)dstScene;

	sceneCUDA->numModels = sceneCPU->getNumModels();
	sceneCUDA->numEmitters = sceneCPU->getNumEmitters();

	sceneCUDA->bbMin = *((CUDA::Vector3*)&sceneCPU->getSceneBB().min);
	sceneCUDA->bbMax = *((CUDA::Vector3*)&sceneCPU->getSceneBB().max);

	sceneCUDA->models = new CUDA::Model[sceneCUDA->numModels];
	for(int i=0;i<sceneCUDA->numModels;i++)
	{
		Model &modelCPU = *(sceneCPU->getModelList()[i]);
		CUDA::Model &modelCUDA = sceneCUDA->models[i];

		modelCUDA.numVerts = modelCPU.getNumVertexs();
		modelCUDA.numTris = modelCPU.getNumTriangles();
		modelCUDA.numNodes = modelCPU.getNumNodes();

		modelCUDA.verts = (CUDA::Vertex*)modelCPU.getVertex(0);
		modelCUDA.tris = (CUDA::Triangle*)modelCPU.getTriangle(0);
		modelCUDA.nodes = (CUDA::BVHNode*)modelCPU.getBV(0);

		modelCUDA.numMats = modelCPU.getNumMaterials();
		modelCUDA.mats = new CUDA::Material[modelCUDA.numMats];
		for(int j=0;j<modelCUDA.numMats;j++)
		{
			Material &matCPU = modelCPU.getMaterial(j);
			memcpy(&modelCUDA.mats[j], &matCPU, sizeof(CUDA::Material));

			BitmapTexture *mapCPU = NULL;
			CUDA::Texture *mapCUDA = NULL;

			mapCPU = matCPU.getMapKa();
			mapCUDA = &(modelCUDA.mats[j].map_Ka);
			mapCUDA->data = NULL;
			if(mapCPU)
			{
				mapCUDA->width = (unsigned short)mapCPU->getWidth();
				mapCUDA->height = (unsigned short)mapCPU->getHeight();
				mapCUDA->data = new float[mapCUDA->width*mapCUDA->height*4];
				if(mapCPU->getBpp() == 3) convert3DByteTo4DFloat(mapCUDA->data, mapCPU->getData(), mapCUDA->width*mapCUDA->height*4);
				if(mapCPU->getBpp() == 4) convert4DByteTo4DFloat(mapCUDA->data, mapCPU->getData(), mapCUDA->width*mapCUDA->height*4);
			}

			mapCPU = matCPU.getMapKd();
			mapCUDA = &(modelCUDA.mats[j].map_Kd);
			mapCUDA->data = NULL;
			if(mapCPU)
			{
				mapCUDA->width = (unsigned short)mapCPU->getWidth();
				mapCUDA->height = (unsigned short)mapCPU->getHeight();
				mapCUDA->data = new float[mapCUDA->width*mapCUDA->height*4];
				if(mapCPU->getBpp() == 3) convert3DByteTo4DFloat(mapCUDA->data, mapCPU->getData(), mapCUDA->width*mapCUDA->height*4);
				if(mapCPU->getBpp() == 4) convert4DByteTo4DFloat(mapCUDA->data, mapCPU->getData(), mapCUDA->width*mapCUDA->height*4);
			}

			mapCPU = matCPU.getMapBump();
			mapCUDA = &(modelCUDA.mats[j].map_bump);
			mapCUDA->data = NULL;
			if(mapCPU)
			{
				mapCUDA->width = (unsigned short)mapCPU->getWidth();
				mapCUDA->height = (unsigned short)mapCPU->getHeight();
				mapCUDA->data = new float[mapCUDA->width*mapCUDA->height*4];
				if(mapCPU->getBpp() == 3) convert3DByteTo4DFloat(mapCUDA->data, mapCPU->getData(), mapCUDA->width*mapCUDA->height*4);
				if(mapCPU->getBpp() == 4) convert4DByteTo4DFloat(mapCUDA->data, mapCPU->getData(), mapCUDA->width*mapCUDA->height*4);
			}
		}

		memcpy(modelCUDA.transfMatrix, &(modelCPU.getTransfMatrix()), sizeof(Matrix));
		memcpy(modelCUDA.invTransfMatrix, &(modelCPU.getInvTransfMatrix()), sizeof(Matrix));
	}

	sceneCUDA->numEmitters = 0;
	for(int i=0;i<sceneCPU->getNumEmitters();i++)
	{
		if(sceneCPU->getEmitter(i).type != Emitter::ENVIRONMENT_LIGHT)
			sceneCUDA->numEmitters++;
	}

	sceneCUDA->emitters = new CUDA::Emitter[sceneCUDA->numEmitters];
	for(int i=0, j=0;i<sceneCPU->getNumEmitters();i++)
	{
		Emitter &emitter = sceneCPU->getEmitter(i);
		if(emitter.type == Emitter::ENVIRONMENT_LIGHT)
		{
			//if(emitter.environmentTexName[0] == 0)
			{
				memcpy(&sceneCUDA->envColor, &emitter.color_Kd, sizeof(CUDA::Vector3));
			}
		}
		else
		{
			memcpy(&sceneCUDA->emitters[j++], &emitter, sizeof(CUDA::Emitter));
		}
	}

	if(sceneCPU->getEnvironmentMap().tex[0])
	{
		Scene::EnvironmentMap &mapCPU = sceneCPU->getEnvironmentMap();

		// has environment map
		for(int i=0;i<6;i++)
		{
			CUDA::Image &mapCUDA = sceneCUDA->envMap[i];
			mapCUDA.width = mapCPU.tex[i]->getWidth();
			mapCUDA.height = mapCPU.tex[i]->getHeight();
			mapCUDA.bpp = 4;

			int numTexel = mapCUDA.width*mapCUDA.height*mapCUDA.bpp;

			mapCUDA.data = new float[numTexel];

			if(mapCPU.tex[i]->getBpp() == 3) convert3DByteTo4DFloat(mapCUDA.data, mapCPU.tex[i]->getData(), numTexel);
			if(mapCPU.tex[i]->getBpp() == 4) convert4DByteTo4DFloat(mapCUDA.data, mapCPU.tex[i]->getData(), numTexel);
			/*
			// convert 3D byte color -> 4D float color
			mapCUDA.data = new float[numTexel];
			for(int j=0, k=0;j<numTexel;j++)
			{
				if(j % 4 == 3)
					mapCUDA.data[j] = 1.0f;
				else
				{
					mapCUDA.data[j] = mapCPU.tex[i]->getData()[k++] / 255.0f;
				}

			}
			*/
		}
	}
	else
	{
		sceneCUDA->envMap[0].data = 0;
	}

	int idx = 0;
	makeCUDASceneGraph(&(sceneCPU->getSceneGraph()), &(sceneCUDA->sceneGraph[0]), sceneCUDA, idx);

	return 1;
}

int CUDARayTracer::adaptCamera(void *srcCamera, void *dstCamera)
{
	Camera *cameraCPU = (Camera*)srcCamera;
	CUDA::Camera *cameraCUDA = (CUDA::Camera*)dstCamera;
	memcpy(cameraCUDA, cameraCPU, sizeof(Camera));
	return 1;
}

int CUDARayTracer::adaptDataStructures(void *srcScene, void *srcCamera, void *dstScene, void *dstCamera)
{
	return adaptScene(srcScene, dstScene) && adaptCamera(srcCamera, dstCamera);
}

extern "C" void materialChangedRayTracer(CUDA::Scene *scene);
extern "C" void loadSceneCUDARayTracer(CUDA::Scene *scene);
extern "C" void renderCUDARayTracer(CUDA::Camera *camera, CUDA::Image *image, CUDA::Controller *controller);
extern "C" void clearResultCUDARayTracer(int &frame);

void CUDARayTracer::init(Scene *scene)
{
	Renderer::init(scene);

	if(!scene) return;

	done();

	m_intersectionStream = scene->getIntersectionStream();

	sceneChanged();

	adaptScene(scene, m_dstScene);
	loadSceneCUDARayTracer((CUDA::Scene*)m_dstScene);
}

void CUDARayTracer::done()
{
}

void CUDARayTracer::resized(int width, int height)
{
	Renderer::resized(width, height);
}

void CUDARayTracer::sceneChanged()
{
}

void CUDARayTracer::materialChanged()
{
	//adaptScene(m_scene, (CUDA::Scene*)m_dstScene);
	m_materialUpdated = true;
}

void CUDARayTracer::applyChangedMaterial()
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

	materialChangedRayTracer(sceneCUDA);
}

void CUDARayTracer::render(Camera *camera, Image *image, unsigned int seed)
{
	CUDA::Camera cameraCUDA;

	memset(image->data, 0, image->width*image->height*image->bpp);
	//adaptDataStructures(m_scene, camera, &sceneCUDA, &cameraCUDA);

	if(m_materialUpdated)
	{
		applyChangedMaterial();
		m_materialUpdated = false;
	}

	renderCUDARayTracer((CUDA::Camera*)camera, (CUDA::Image*)&image->width, (CUDA::Controller *)&m_controller);
}

void CUDARayTracer::clearResult()
{
	extern int g_frame;
	clearResultCUDARayTracer(g_frame);
}

void CUDARayTracer::clearScene()
{
	if(!m_dstScene) return;

	CUDA::Scene *sceneCUDA = (CUDA::Scene*)m_dstScene;

	if(sceneCUDA->models)
	{
		for(int i=0;i<sceneCUDA->numModels;i++)
		{
			CUDA::Model &modelCUDA = sceneCUDA->models[i];
			if(modelCUDA.mats)
			{
				for(int j=0;j<modelCUDA.numMats;j++)
				{
					CUDA::Material &matCUDA = modelCUDA.mats[j];
					if(matCUDA.map_Ka.offset)
						delete[] matCUDA.map_Ka.data;
					if(matCUDA.map_Kd.offset)
						delete[] matCUDA.map_Kd.data;
					if(matCUDA.map_bump.offset)
						delete[] matCUDA.map_bump.data;
				}
				delete[] modelCUDA.mats;
			}
		}
		delete[] sceneCUDA->models;
	}

	if(sceneCUDA->emitters)
		delete[] sceneCUDA->emitters;

	if(sceneCUDA->envMap[0].data)
		for(int i=0;i<6;i++)
			delete[] sceneCUDA->envMap[i].data;

	delete m_dstScene;
	m_dstScene = NULL;
}