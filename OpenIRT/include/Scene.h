/********************************************************************
	created:	2009/04/24
	file path:	d:\Projects\Redering\OpenIRT\include
	file base:	Scene
	file ext:	h
	author:		Tae-Joon Kim (tjkim@tclab.kaist.ac.kr)
	
	comment:	Scene class, manages scenes. Scene information is
				included.
*********************************************************************/

#pragma once

#include "Model.h"
#include "Emitter.h"
#include "SceneNode.h"
#include "Photon.h"

namespace irt
{

class Scene
{
public:
	class EnvironmentMap
	{
	public:
		BitmapTexture *tex[6];

		EnvironmentMap();
		~EnvironmentMap();

		bool hasEnvMap() {return tex[0] != NULL;}

		void load(const char *fileNameBase);
		void clear();

		void shade(Vector3 &direction, RGBf &backgroundColor) {
			if(!tex[0]) return;
			// Order: Right, Left, Up, Down, Back, Front	
			static int faceToU[] = { 2, 2, 0, 0, 0, 0 };
			static int faceToV[] = { 1, 1, 2, 2, 1, 1 };
			static float faceToUSign[] = { 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f };
			static float faceToVSign[] = { 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f };

			int idx = direction.indexOfMaxAbsComponent();
			int face = (direction.e[idx] > 0.0)?0:1;
			face += 2*idx;

			//int idx1 = min((idx+1)%3, (idx+2)%3);
			//int idx2 = max((idx+1)%3, (idx+2)%3);
			int idx1 = faceToU[face];
			int idx2 = faceToV[face];

			float u = (faceToUSign[face]*direction.e[idx1] / fabs(direction.e[idx]) + 1.0f) / 2.0f;
			float v = (faceToVSign[face]*direction.e[idx2] / fabs(direction.e[idx]) + 1.0f) / 2.0f;

			tex[face]->getTexValue(backgroundColor, u, 0.999999f - v);
		}
	};

// Member variables
protected:
	ModelList m_modelList;
	ModelListMap m_modelListMap;

	std::vector<Model*> m_modelsToBeDeleted;

	EmitterList m_emitList;

	AABB m_sceneBB;
	SceneNode m_sceneGraph;

	bool m_hasSceneStructure;

	EnvironmentMap m_envMap;

	typedef SceneNode* StackElem;
	__declspec(align(16)) StackElem **m_stacks;

	Model::ModelType m_modelTypeSelector;

	int m_lastIntersectionStream;

	char m_lastModelFileName[MAX_PATH];
	char m_ASVOFileBase[MAX_PATH];

// Member functions
public:
	Scene(void);
	~Scene(void);

	void setModelTypeSelector(Model::ModelType newType) {m_modelTypeSelector = newType;}

	int getNumModels() {return (int)m_modelList.size();}
	int getNumEmitters() {return (int)m_emitList.size();}
	Model *getModel(const char *name);
	ModelList &getModelList() {return m_modelList;}
	ModelListMap &getModelListMap() {return m_modelListMap;}
	Emitter &getEmitter(int i) {return m_emitList[i];}
	const Emitter &getConstEmitter(int i) const {return m_emitList[i];}
	const AABB& getSceneBB() {return m_sceneBB;}
	void setSceneBB(const AABB &bb) {m_sceneBB = bb;}
	SceneNode &getSceneGraph() {return m_sceneGraph;}
	EnvironmentMap &getEnvironmentMap() {return m_envMap;}

	int pushEmitter(const Emitter &emitter);
	void removeEmitter(int pos);

	void loadEnvironmentMap(const char *fileNameBase) {m_envMap.load(fileNameBase);}

	bool hasSceneStructure() {return m_hasSceneStructure;}
	bool hasEmitters() {return m_emitList.size() != 0;}

	/**
	 *	Load scene description file (xml) or single model
	 *
	 *  The file type will be determined by the extension of fileName.
	 *  attach : attach the loaded file to 'attach' (ex) attach simplified model
	 */
	bool load(const char *fileName, Model::ModelType type = Model::NONE, bool clear = true);
	void unload();

	bool loadScene(const char *fileName);

	bool loadModel(Model *model);
	bool loadOOC(const char *fileName);
	bool loadPly(const char *fileName);
	bool loadHCCMesh(const char *fileName);
	bool loadHCCMesh2(const char *fileName);
	bool loadOBJ(const char *fileName);


	bool loadOOCAnimation(const char *fileName);

	void exportScene(const char *fileName);

	void generateEmitter();
	void generateSceneStructure();
	void generatePMTarget(Emitter &emitter);

	int getIntersectionStream();

	void trace(const Ray &ray, RGB4f &color, Material *outMat = 0, int depth = 0, float traveledDist = 0.0f, HitPointInfo *outHit = 0, int stream = 0);
	void shade(const Ray &ray, RGB4f &color, HitPointInfo &hit, bool hasHit = true);
	RayPacketTemplate
	void trace(RayPacketT &ray, RGB4f *color, int depth = 0, int stream = 0);
	bool getIntersection(const Ray &ray, HitPointInfo &hitPointInfo, float tLimit = 0.0f, int stream = 0);
	RayPacketTemplate
	void getIntersection(RayPacketT &rayPacket, int stream = 0);

	int getNumMaxPhotons();
	int tracePhotons(int size, Photon *outPhotons, void (*funcProcessPhoton)(const Photon &photon) = NULL);
	void tracePhotons(int emitterIndex, Photon *outPhotons, int idx, int &numTotalPhotons, void (*funcProcessPhoton)(const Photon &photon));
	void buildPhotonKDTree(Photon *photons, Photon *outPhotons, int left, int right, int depth, int curRoot, Photon::SplitChoice splitChoice, const AABB &bb);
	int buildPhotonKDTree(int size, Photon **photons, AABB &bb);

	char *getASVOFileBase() {if(strlen(m_ASVOFileBase) > 0) return m_ASVOFileBase; return "default";}
protected:
	void loadSceneGraph(void *sceneNodeSrc, void *sceneNodeTarget);
	void loadSceneEmitters(void *sceneNodeSrc);

	void saveSceneGraph(void *sceneNodeSrc, void *sceneNodeTarget);
	void saveSceneEmitters(void *sceneNodeTarget);
};

RayPacketTemplate
void Scene::getIntersection(RayPacketT &rayPacket, int stream)
{
	//Model *model = m_modelList[0];
	//switch(model->getType())
	//{
	//case Model::OOC_FILE : ((Model*)model)->getIntersection(rayPacket, stream); break;
	//case Model::HCCMESH : ((HCCMesh*)model)->getIntersection(rayPacket, stream); break;
	//case Model::HCCMESH2 : ((HCCMesh2*)model)->getIntersection(rayPacket, stream); break;
	//}

	int threadID = omp_get_thread_num();
	StackElem *stack = m_stacks[threadID+MAX_NUM_THREADS*stream];

	unsigned int stackPtr;
	SceneNode *currentNode;
	bool hasHit = false;


	currentNode = &(m_sceneGraph);
	stack[0] = currentNode;
	stackPtr = 1;

	while(true)
	{
		// is current node intersected and also closer than previous hit?
		if (rayPacket.intersectWithBox(&currentNode->nodeBB.min, 0) < nRays)  // yes, at least one ray intersects
		{
			if(currentNode->model)
			{
				Model *model = currentNode->model;
				switch(model->getType())
				{
				case Model::OOC_FILE : ((Model*)model)->getIntersection(rayPacket, stream); break;
				case Model::HCCMESH : ((HCCMesh*)model)->getIntersection(rayPacket, stream); break;
				case Model::HCCMESH2 : ((HCCMesh2*)model)->getIntersection(rayPacket, stream); break;
				}
			}
			if(currentNode->hasChilds())
			{
				for(size_t i=0;i<currentNode->childs->size();i++)
				{
					stack[stackPtr++] = currentNode->childs->at(i);
				}
			}
		}
		// traversal ends when stack empty
		if(--stackPtr == 0) break;

		currentNode = stack[stackPtr];
	}
}

RayPacketTemplate
void Scene::trace(RayPacketT &rayPacket, RGB4f *colors, int depth, int stream) {
	/*
	__declspec(align(16)) RGBf shaded_color[nRays * 4];
	__declspec(align(16)) int reflection_hit[nRays];
	__declspec(align(16)) float specular_reflectance[nRays * 4];
	__declspec(align(16)) rgb specular_color[nRays * 4];
	Sampler &sampler = samplerList[omp_get_thread_num()];	
	*/

	//
	// Intersect ray with scene in parallel:
	//	
	//((HCCMesh*)m_modelList[0])->getIntersection(rayPacket);
	getIntersection(rayPacket, stream);
	
	//bool allowReflection = depth < maxRecursionDepth && g_ReflectionRays;
	//bool allowRefraction = depth < maxRecursionDepth && g_RefractionRays;

	for (int r = 0; r < nRays; r++) {	
		//reflection_hit[r] = 0;
		
		int bit = 1;
		for (int i = 0 ; i < 4; i++, bit <<= 1) {

			// has missed the scene, i.e. no hit ?
			if ((rayPacket.rayHasHit[r] & bit) == 0) {
				// shade from background		
				Vector3 dir = rayPacket.rays[r].getDirection(i);

				RGBf col;

				m_envMap.shade(dir, col);
				colors[r*4 + i].set(col, 1.0f);

				continue;
			}		

			Material currentMaterial;
			if(rayPacket.hitpoints[r].modelPtr[i]) currentMaterial = rayPacket.hitpoints[r].modelPtr[i]->getMaterial(rayPacket.hitpoints[r].m[i]);

			Vector3 hitPosition(rayPacket.hitpoints[r].x[0].e[i], rayPacket.hitpoints[r].x[1].e[i], rayPacket.hitpoints[r].x[2].e[i]);
			Vector3 hitNormal(rayPacket.hitpoints[r].n[0].e[i], rayPacket.hitpoints[r].n[1].e[i], rayPacket.hitpoints[r].n[2].e[i]);
			Vector3 shadowDir;

			//float weight = 2.0f / m_emitList.size();
			float maxCosFactor = 0.0f;
			// for each emitter
			for(size_t l=0;l<m_emitList.size();l++)
			{
				Emitter &emitter = m_emitList[l];

				if(emitter.type == Emitter::ENVIRONMENT_LIGHT) continue;
				// just use diffuse values now...
				const RGBf &emitColor = emitter.color_Kd;

				// compute shadow ray direction

				shadowDir = emitter.pos - hitPosition;
				shadowDir.makeUnitVector();

				float cosFactor = dot(shadowDir, hitNormal);

				if(maxCosFactor < cosFactor)
				{
					maxCosFactor = cosFactor;
				}

				if(cosFactor > 0.0f)
				{
					/*
					int idx = shadowDir.indexOfMaxComponent();
					float tLimit = (emitter.pos.e[idx] - hitPosition.e[idx]) / shadowDir.e[idx];
					HitPointInfo shadowHit;
					shadowHit.t = tLimit;

					Ray shadowRay;
					shadowRay.set(hitPosition, shadowDir);

					if(!getIntersection(shadowRay, shadowHit, stream))
					*/
						colors[r*4 + i] += currentMaterial.getMatKd() * emitColor * cosFactor;// * weight;
				}
			}

			//colors[r*4 + i] = currentMaterial.getMatKd() * maxCosFactor;

			colors[r*4 + i].e[0] = min(1.0f, colors[r*4 + i].e[0]);
			colors[r*4 + i].e[1] = min(1.0f, colors[r*4 + i].e[1]);
			colors[r*4 + i].e[2] = min(1.0f, colors[r*4 + i].e[2]);

			
			// get local shaded color from material
			//currentMaterial->shade(rayPacket.rays[r], rayPacket.hitpoints[r], i, shaded_color[r*4 + i]);
			
			/*
			// add constant ambient term
			if (currentMaterial->isEmitter()) {
				currentMaterial->emittedRadiance( rayPacket.hitpoints[r],  rayPacket.rays[r].getOrigin(i), i,  colors[r*4 + i]);
			}
			else
				colors[r*4 + i] = ambientColor * shaded_color[r*4 + i];	
			*/
			/*
			#ifdef _USE_REFLECTIONS
			specular_reflectance[r*4 + i] = currentMaterial->getReflectance();
			reflection_hit[r] |= (currentMaterial->hasReflection()&g_ReflectionRays)?bit:0;	
			#endif
			*/
		}	

	} // for all rays in packet


	// find number of rays that hit and first hitting ray
	// (do this only now because of fake plane)
	rayPacket.calculateHitStats();

	// if this packet hit something:
	if (rayPacket.numRaysHit) {
		
		/*
		#ifdef _USE_REFLECTIONS
		if (allowReflection) { // we can still trace secondary rays			
			for (int r = rayPacket.firstHitRay; r <= rayPacket.lastHitRay; r++) {
				// at least one ray needs reflection?
				if (reflection_hit[r]) {
					SIMDRay reflectionRay;
					
					// build the combined reflection ray of all the hitpoints
					makeReflectionRay(&reflectionRay, &rayPacket.rays[r], &rayPacket.hitpoints[r], reflection_hit[r]);

					// trace the reflection rays
					trace(reflectionRay, &specular_color[r*4], depth + 1);
				}
			}
		}
		#endif
		*/
		
		/*
		if (g_ShadowRays) {
		}
		for (int r = rayPacket.firstHitRay; r <= rayPacket.lastHitRay; r++) {
			if (!rayPacket.rayHasHit[r])
				continue;

			for (int l = 0; l < lightlist.size(); l++) {
				computeLocalLighting(rayPacket.hitpoints[r], rayPacket.rayHasHit[r], &colors[r*4], &shaded_color[r*4], rgb(lightlist[l].color), lightlist[l].pos);
			} // for each light			

		} // for each ray in packet
		*/

		/*
		#if defined(_USE_REFLECTIONS) || defined(_USE_REFRACTIONS)					
		if (g_ReflectionRays) {
			for (int r = rayPacket.firstHitRay; r < nRays; r++) {
				#define RAYCODE(i, bit) colors[r*4 + i] = specular_reflectance[r*4 + i]*specular_color[r*4 + i] + (1.0f - specular_reflectance[r*4 + i])*colors[r*4 + i]
				FOREACHRAY2X2(reflection_hit[r]);				
				#undef RAYCODE
			}
		}
		#endif
		*/
	} // if (rayPacket.numRaysHit)
}

};