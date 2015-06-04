#include "defines.h"
#include "CommonHeaders.h"

#include "handler.h"

#include "Scene.h"
#include <tinyxml.h>
#include <direct.h>
#include <stopwatch.h>

#include "TextureManager.h"

#define ElemIndex(elem, index) ( elem.pos.e[index] ) 
#include "select.h"

#include "HCCMesh.h"
#include "HCCMesh2.h"
#include "PLYLoader.h"
#include "OBJLoader.h"
#include "BVHBuilder.h"

using namespace irt;

Scene::Scene(void)
	: m_hasSceneStructure(0), m_lastIntersectionStream(0)
{
	m_modelTypeSelector = Model::NONE;

	m_modelListMap[0] = -1;

	m_ASVOFileBase[0] = 0;

	m_stacks = new StackElem *[MAX_NUM_THREADS*MAX_NUM_INTERSECTION_STREAM];

	for(int i=0;i<MAX_NUM_THREADS*MAX_NUM_INTERSECTION_STREAM;i++)
		m_stacks[i] = (StackElem *)_aligned_malloc(150 * sizeof(StackElem), 16);
}

Scene::~Scene(void)
{
	unload();

	if(m_stacks)
	{
		for(int i=0;i<MAX_NUM_THREADS*MAX_NUM_INTERSECTION_STREAM;i++)
			_aligned_free(m_stacks[i]);
		delete[] m_stacks;
	}
}

bool Scene::load(const char *fileName, Model::ModelType type, bool clear)
{
	if(clear) unload();

	// extract file extension
	char ext[MAX_PATH] = {0, };
	for(int i=(int)strlen(fileName)-1;i>=0;i--)
	{
		if(fileName[i] == '.')
		{
			strcpy_s(ext, MAX_PATH, &fileName[i+1]);
			break;
		}
	}

	// load scene files
	if(strcmp(ext, "scene") == 0 || strcmp(ext, "xml") == 0)
	{
		return m_hasSceneStructure = loadScene(fileName);
	}
	if(strcmp(ext, "ply") == 0)
		return loadPly(fileName);
	if(strcmp(ext, "obj") == 0)
		return loadOBJ(fileName);
	if(type == Model::HCCMESH || strcmp(ext, "hccmesh") == 0)
		return loadHCCMesh(fileName);
	if(type == Model::HCCMESH2 || strcmp(ext, "hccmesh2") == 0)
		return loadHCCMesh2(fileName);
	if(strcmp(ext, "ooc") == 0) 
		return loadOOC(fileName);

	printf("This renderer does not support the scene(model) type : %s\n", ext);
	return false;
}

void Scene::unload()
{
	for(size_t i=0;i<m_modelsToBeDeleted.size();i++)
		if(m_modelsToBeDeleted[i]) delete m_modelsToBeDeleted[i];
	m_modelsToBeDeleted.clear();

	m_modelList.clear();
	m_emitList.clear();
}

#define ASSIGN_ATTRIBUTE_INT(dst, node, name) if(node) {if((node)->ToElement()->Attribute(name)) (dst) = atoi((node)->ToElement()->Attribute(name));}
#define ASSIGN_ATTRIBUTE_FLOAT(dst, node, name) if(node) {if((node)->ToElement()->Attribute(name)) (dst) = (float)atof((node)->ToElement()->Attribute(name));}
#define ASSIGN_ATTRIBUTE_STR_CPY(dst, node, name) if(node) strcpy_s((dst), 255, (node)->ToElement()->Attribute(name));

void Scene::loadSceneGraph(void *sceneNodeSrc, void *sceneNodeTarget)
{
	if(!sceneNodeSrc) return;

	TiXmlNode *nodeXml = (TiXmlNode*)sceneNodeSrc;
	SceneNode *nodeScene = (SceneNode*)sceneNodeTarget;

	//Model *model = NULL;
	//Matrix *matrix = NULL;

	std::vector<Model*> modelList;
	std::vector<Matrix*> matrixList;

	for(TiXmlNode *nodeXmlChild=nodeXml->FirstChild();nodeXmlChild;nodeXmlChild=nodeXmlChild->NextSibling())
	{
		if(strcmp(nodeXmlChild->Value(), "_model_file") == 0)
		{
			char fileName[256];
			char fileType[256];
			ASSIGN_ATTRIBUTE_STR_CPY(fileName, nodeXmlChild, "value");
			ASSIGN_ATTRIBUTE_STR_CPY(fileType, nodeXmlChild, "type");
		
			Model::ModelType type = Model::OOC_FILE;

			if(!strcmp(fileType, "HCCMESH") || !strcmp(fileType, "hccmesh") || !strcmp(fileType, "Hccmesh") || !strcmp(fileType, "HCCMesh"))
				type = Model::HCCMESH;

			if(!strcmp(fileType, "HCCMESH2") || !strcmp(fileType, "hccmesh2") || !strcmp(fileType, "Hccmesh2") || !strcmp(fileType, "HCCMesh2"))
				type = Model::HCCMESH2;

			size_t lastModelIndex = m_modelList.size();
			load(fileName, type, false);

			strcpy_s(m_lastModelFileName, MAX_PATH, fileName);

			for(size_t i=lastModelIndex;i<m_modelList.size();i++)
			{
				modelList.push_back(m_modelList[i]);
			}
			//model = m_modelList.back();
			continue;
		}
		if(strcmp(nodeXmlChild->Value(), "_transformation_matrix") == 0)
		{
			int i = 0;
			Matrix *matrix = new Matrix;
			for(TiXmlNode *nodeXmlMatrixElement=nodeXmlChild->FirstChild();nodeXmlMatrixElement;nodeXmlMatrixElement=nodeXmlMatrixElement->NextSibling())
			{
				ASSIGN_ATTRIBUTE_FLOAT(matrix->getRef(i++), nodeXmlMatrixElement, "value");
			}

			matrixList.push_back(matrix);
			continue;
		}
		loadSceneGraph(nodeXmlChild, nodeScene->addChild(nodeXmlChild->Value(), NULL));
	}
	
	//nodeScene->set(nodeXml->Value(), nodeScene->parent, model, matrix);
	//if(matrix)
	//	delete matrix;

	if(modelList.size() == 1)
	{
		nodeScene->set(nodeXml->Value(), nodeScene->parent, modelList[0], matrixList.size() != 0 ? matrixList[0] : NULL);
	}
	else
	{
		nodeScene->set(nodeXml->Value(), nodeScene->parent);
		for(size_t i=0;i<modelList.size();i++)
		{
			nodeScene->addChild(modelList[i]->getName(), modelList[i], matrixList.size() > i ? matrixList[i] : NULL);
		}
	}

	for(size_t i=0;i<matrixList.size();i++)
		delete matrixList[i];
}

void Scene::loadSceneEmitters(void *sceneNodeSrc)
{
	if(!sceneNodeSrc) return;

	TiXmlNode *nodeXml = (TiXmlNode*)sceneNodeSrc;

	for(TiXmlNode *emitterXmlNode=nodeXml->FirstChild();emitterXmlNode;emitterXmlNode=emitterXmlNode->NextSibling())
	{
		Emitter emitter;
		char emitterType[256];
		ASSIGN_ATTRIBUTE_STR_CPY(emitterType, emitterXmlNode, "type");
		emitter.setType(emitterType);
		emitter.setName(emitterXmlNode->Value());
		
		for(TiXmlNode *attributeXmlNode=emitterXmlNode->FirstChild();attributeXmlNode;attributeXmlNode=attributeXmlNode->NextSibling())
		{
			if(strcmp(attributeXmlNode->Value(), "_ambient_color") == 0)
			{
				RGBf color;
				ASSIGN_ATTRIBUTE_FLOAT(color.e[0], attributeXmlNode, "R");
				ASSIGN_ATTRIBUTE_FLOAT(color.e[1], attributeXmlNode, "G");
				ASSIGN_ATTRIBUTE_FLOAT(color.e[2], attributeXmlNode, "B");
				emitter.color_Ka = color;
			}
			if(strcmp(attributeXmlNode->Value(), "_diffuse_color") == 0)
			{
				RGBf color;
				ASSIGN_ATTRIBUTE_FLOAT(color.e[0], attributeXmlNode, "R");
				ASSIGN_ATTRIBUTE_FLOAT(color.e[1], attributeXmlNode, "G");
				ASSIGN_ATTRIBUTE_FLOAT(color.e[2], attributeXmlNode, "B");
				emitter.color_Kd = color;
			}
			if(strcmp(attributeXmlNode->Value(), "_position") == 0)
			{
				Vector3 pos;
				ASSIGN_ATTRIBUTE_FLOAT(pos.e[0], attributeXmlNode, "X");
				ASSIGN_ATTRIBUTE_FLOAT(pos.e[1], attributeXmlNode, "Y");
				ASSIGN_ATTRIBUTE_FLOAT(pos.e[2], attributeXmlNode, "Z");
				emitter.pos = pos;
			}
			if(strcmp(attributeXmlNode->Value(), "_corner") == 0)
			{
				Vector3 vec;
				ASSIGN_ATTRIBUTE_FLOAT(vec.e[0], attributeXmlNode, "X");
				ASSIGN_ATTRIBUTE_FLOAT(vec.e[1], attributeXmlNode, "Y");
				ASSIGN_ATTRIBUTE_FLOAT(vec.e[2], attributeXmlNode, "Z");
				emitter.planar.corner = vec;
			}
			if(strcmp(attributeXmlNode->Value(), "_v1") == 0)
			{
				Vector3 vec;
				ASSIGN_ATTRIBUTE_FLOAT(vec.e[0], attributeXmlNode, "X");
				ASSIGN_ATTRIBUTE_FLOAT(vec.e[1], attributeXmlNode, "Y");
				ASSIGN_ATTRIBUTE_FLOAT(vec.e[2], attributeXmlNode, "Z");
				emitter.planar.v1 = vec;
			}
			if(strcmp(attributeXmlNode->Value(), "_v2") == 0)
			{
				Vector3 vec;
				ASSIGN_ATTRIBUTE_FLOAT(vec.e[0], attributeXmlNode, "X");
				ASSIGN_ATTRIBUTE_FLOAT(vec.e[1], attributeXmlNode, "Y");
				ASSIGN_ATTRIBUTE_FLOAT(vec.e[2], attributeXmlNode, "Z");
				emitter.planar.v2 = vec;
			}
			if(strcmp(attributeXmlNode->Value(), "_normal") == 0)
			{
				Vector3 vec;
				ASSIGN_ATTRIBUTE_FLOAT(vec.e[0], attributeXmlNode, "X");
				ASSIGN_ATTRIBUTE_FLOAT(vec.e[1], attributeXmlNode, "Y");
				ASSIGN_ATTRIBUTE_FLOAT(vec.e[2], attributeXmlNode, "Z");
				emitter.planar.normal = vec;
			}
			if(
				strcmp(attributeXmlNode->Value(), "_spot_corner") == 0 ||
				strcmp(attributeXmlNode->Value(), "_spot_center") == 0
				)
			{
				Vector3 vec;
				ASSIGN_ATTRIBUTE_FLOAT(vec.e[0], attributeXmlNode, "X");
				ASSIGN_ATTRIBUTE_FLOAT(vec.e[1], attributeXmlNode, "Y");
				ASSIGN_ATTRIBUTE_FLOAT(vec.e[2], attributeXmlNode, "Z");
				emitter.spotTarget.corner = vec;
			}
			if(
				strcmp(attributeXmlNode->Value(), "_spot_v1") == 0 ||
				strcmp(attributeXmlNode->Value(), "_spot_radius") == 0
				)
			{
				Vector3 vec;
				ASSIGN_ATTRIBUTE_FLOAT(vec.e[0], attributeXmlNode, "X");
				ASSIGN_ATTRIBUTE_FLOAT(vec.e[1], attributeXmlNode, "Y");
				ASSIGN_ATTRIBUTE_FLOAT(vec.e[2], attributeXmlNode, "Z");
				emitter.spotTarget.v1 = vec;
			}
			if(strcmp(attributeXmlNode->Value(), "_spot_v2") == 0)
			{
				Vector3 vec;
				ASSIGN_ATTRIBUTE_FLOAT(vec.e[0], attributeXmlNode, "X");
				ASSIGN_ATTRIBUTE_FLOAT(vec.e[1], attributeXmlNode, "Y");
				ASSIGN_ATTRIBUTE_FLOAT(vec.e[2], attributeXmlNode, "Z");
				emitter.spotTarget.v2 = vec;
			}
			if(strcmp(attributeXmlNode->Value(), "_number_scattering_photons") == 0)
			{
				int value;
				ASSIGN_ATTRIBUTE_INT(value, attributeXmlNode, "value");
				emitter.numScatteringPhotons = value;
			}
			if(strcmp(attributeXmlNode->Value(), "_intensity") == 0)
			{
				float value;
				ASSIGN_ATTRIBUTE_FLOAT(value, attributeXmlNode, "value");
				emitter.intensity = value;
			}
			if(strcmp(attributeXmlNode->Value(), "_is_cos_light") == 0)
			{
				float value;
				ASSIGN_ATTRIBUTE_FLOAT(value, attributeXmlNode, "value");
				emitter.isCosLight = value != 0.0f;
			}
			if(strcmp(attributeXmlNode->Value(), "_texture_base_file") == 0)
			{
				char fileName[256];
				ASSIGN_ATTRIBUTE_STR_CPY(fileName, attributeXmlNode, "value");
				strcpy_s(emitter.environmentTexName, 255, fileName);

				m_envMap.load(fileName);
			}
		}

		emitter.planar.normal = cross(emitter.planar.v1, emitter.planar.v2);
		emitter.planar.normal.makeUnitVector();

		if(emitter.spotTarget.v1.maxAbsComponent() != 0.0f || emitter.spotTarget.v2.maxAbsComponent() != 0.0f)
		{
			emitter.spotTarget.normal = cross(emitter.spotTarget.v1, emitter.spotTarget.v2);
			emitter.spotTarget.normal.makeUnitVector();
			emitter.targetType = Emitter::LIGHT_TARGET_PARALLELOGRAM;
		}
		else
		{
			// not spotlight
			emitter.spotTarget.corner.set(FLT_MAX);
			switch(emitter.type)
			{
			case Emitter::POINT_LIGHT : emitter.targetType = Emitter::LIGHT_TARGET_SPHERE; break;
			case Emitter::PARALLELOGRAM_LIGHT : emitter.targetType = Emitter::LIGHT_TARGET_HALF_SPHERE; break;
			}
			
		}

		if(emitter.pos.maxAbsComponent() == 0.0f)
		{
			emitter.pos = (emitter.planar.v1 + emitter.planar.v1) / 2.0f + emitter.planar.corner;
		}

		m_emitList.push_back(emitter);
	}
}

void Scene::saveSceneGraph(void *sceneNodeSrc, void *sceneNodeTarget)
{
	SceneNode *nodeScene = (SceneNode*)sceneNodeSrc;
	TiXmlNode *nodeXml = (TiXmlNode*)sceneNodeTarget;

	nodeXml->SetValue(nodeScene->name);

	if(nodeScene->model)
	{
		TiXmlElement modelFile("_model_file");
		modelFile.SetAttribute("value", nodeScene->model->getFileName());
		nodeXml->InsertEndChild(modelFile);
		if(nodeScene->model->getTransfMatrix() != identityMatrix())
		{
			TiXmlElement matrix("_transformation_matrix");
			char value[256];
			for(int i=0;i<16;i++)
			{
				sprintf_s(value, 256, "%f", nodeScene->model->getTransfMatrix().x[i/4][i%4]);
				TiXmlElement e("e");
				e.SetAttribute("value", value);
				matrix.InsertEndChild(e);
			}
			nodeXml->InsertEndChild(matrix);
		}
	}
	if(!nodeScene->hasChilds()) return;
	for(int i=0;i<(int)nodeScene->childs->size();i++)
	{
		TiXmlElement child("");
		saveSceneGraph(nodeScene->childs->at(i), &child);
		nodeXml->InsertEndChild(child);
	}
}

void Scene::saveSceneEmitters(void *sceneNodeTarget)
{
	TiXmlNode *nodeXml = (TiXmlNode*)sceneNodeTarget;

	for(int i=0;i<(int)m_emitList.size();i++)
	{
		Emitter &emitter = m_emitList[i];
		char typeName[256];
		char emitterName[256];
		sprintf_s(emitterName, 256, "%s", emitter.name);
		for(int i=0;i<strlen(emitterName);i++)
			if(emitterName[i] == ' ') emitterName[i] = '_';	// remove blank
		emitter.getTypeName(typeName, 256);
		TiXmlElement lightXml(emitterName);
		lightXml.SetAttribute("type", typeName);

		if(emitter.type == Emitter::ENVIRONMENT_LIGHT)
		{
			TiXmlElement att1("_texture_base_file");
			att1.SetAttribute("value", emitter.environmentTexName);
			lightXml.InsertEndChild(att1);

			char value[256];
			TiXmlElement att2("_diffuse_color");
			sprintf_s(value, 256, "%f", emitter.color_Kd.r());
			att2.SetAttribute("R", value);
			sprintf_s(value, 256, "%f", emitter.color_Kd.g());
			att2.SetAttribute("G", value);
			sprintf_s(value, 256, "%f", emitter.color_Kd.b());
			att2.SetAttribute("B", value);
			lightXml.InsertEndChild(att2);
		}
		else
		{
			char value[256];
			TiXmlElement att1("_ambient_color");
			sprintf_s(value, 256, "%f", emitter.color_Ka.r());
			att1.SetAttribute("R", value);
			sprintf_s(value, 256, "%f", emitter.color_Ka.g());
			att1.SetAttribute("G", value);
			sprintf_s(value, 256, "%f", emitter.color_Ka.b());
			att1.SetAttribute("B", value);
			lightXml.InsertEndChild(att1);

			TiXmlElement att2("_diffuse_color");
			sprintf_s(value, 256, "%f", emitter.color_Kd.r());
			att2.SetAttribute("R", value);
			sprintf_s(value, 256, "%f", emitter.color_Kd.g());
			att2.SetAttribute("G", value);
			sprintf_s(value, 256, "%f", emitter.color_Kd.b());
			att2.SetAttribute("B", value);
			lightXml.InsertEndChild(att2);

			TiXmlElement att3("_position");
			sprintf_s(value, 256, "%f", emitter.pos.x());
			att3.SetAttribute("X", value);
			sprintf_s(value, 256, "%f", emitter.pos.y());
			att3.SetAttribute("Y", value);
			sprintf_s(value, 256, "%f", emitter.pos.z());
			att3.SetAttribute("Z", value);
			lightXml.InsertEndChild(att3);

			TiXmlElement att4("_corner");
			sprintf_s(value, 256, "%f", emitter.planar.corner.x());
			att4.SetAttribute("X", value);
			sprintf_s(value, 256, "%f", emitter.planar.corner.y());
			att4.SetAttribute("Y", value);
			sprintf_s(value, 256, "%f", emitter.planar.corner.z());
			att4.SetAttribute("Z", value);
			lightXml.InsertEndChild(att4);

			TiXmlElement att5("_v1");
			sprintf_s(value, 256, "%f", emitter.planar.v1.x());
			att5.SetAttribute("X", value);
			sprintf_s(value, 256, "%f", emitter.planar.v1.y());
			att5.SetAttribute("Y", value);
			sprintf_s(value, 256, "%f", emitter.planar.v1.z());
			att5.SetAttribute("Z", value);
			lightXml.InsertEndChild(att5);

			TiXmlElement att6("_v2");
			sprintf_s(value, 256, "%f", emitter.planar.v2.x());
			att6.SetAttribute("X", value);
			sprintf_s(value, 256, "%f", emitter.planar.v2.y());
			att6.SetAttribute("Y", value);
			sprintf_s(value, 256, "%f", emitter.planar.v2.z());
			att6.SetAttribute("Z", value);
			lightXml.InsertEndChild(att6);

			TiXmlElement att7("_normal");
			sprintf_s(value, 256, "%f", emitter.planar.normal.x());
			att7.SetAttribute("X", value);
			sprintf_s(value, 256, "%f", emitter.planar.normal.y());
			att7.SetAttribute("Y", value);
			sprintf_s(value, 256, "%f", emitter.planar.normal.z());
			att7.SetAttribute("Z", value);
			lightXml.InsertEndChild(att7);

			if(emitter.spotTarget.corner.x() != FLT_MAX)
			{
				TiXmlElement att8("_spot_corner");
				sprintf_s(value, 256, "%f", emitter.spotTarget.corner.x());
				att8.SetAttribute("X", value);
				sprintf_s(value, 256, "%f", emitter.spotTarget.corner.y());
				att8.SetAttribute("Y", value);
				sprintf_s(value, 256, "%f", emitter.spotTarget.corner.z());
				att8.SetAttribute("Z", value);
				lightXml.InsertEndChild(att8);

				TiXmlElement att9("_spot_v1");
				sprintf_s(value, 256, "%f", emitter.spotTarget.v1.x());
				att9.SetAttribute("X", value);
				sprintf_s(value, 256, "%f", emitter.spotTarget.v1.y());
				att9.SetAttribute("Y", value);
				sprintf_s(value, 256, "%f", emitter.spotTarget.v1.z());
				att9.SetAttribute("Z", value);
				lightXml.InsertEndChild(att9);

				TiXmlElement att10("_spot_v2");
				sprintf_s(value, 256, "%f", emitter.spotTarget.v2.x());
				att10.SetAttribute("X", value);
				sprintf_s(value, 256, "%f", emitter.spotTarget.v2.y());
				att10.SetAttribute("Y", value);
				sprintf_s(value, 256, "%f", emitter.spotTarget.v2.z());
				att10.SetAttribute("Z", value);
				lightXml.InsertEndChild(att10);
			}

			TiXmlElement att11("_number_scattering_photons");
			sprintf_s(value, 256, "%d", emitter.numScatteringPhotons);
			att11.SetAttribute("value", value);
			lightXml.InsertEndChild(att11);

			TiXmlElement att12("_intensity");
			sprintf_s(value, 256, "%f", emitter.intensity);
			att12.SetAttribute("value", value);
			lightXml.InsertEndChild(att12);

			TiXmlElement att13("_is_cos_light");
			sprintf_s(value, 256, "%f", emitter.isCosLight);
			att13.SetAttribute("value", value);
			lightXml.InsertEndChild(att13);
		}
		nodeXml->InsertEndChild(lightXml);
	}
}

bool Scene::loadScene(const char *fileName)
{
	char workingDirectory[MAX_PATH];
	char shortFileName[MAX_PATH];
	strcpy_s(workingDirectory, MAX_PATH-1, fileName);

	// remove last entry of prompt
	for(int i=(int)strlen(workingDirectory)-1;i>=0;i--)
	{
		if(workingDirectory[i] == '/' || workingDirectory[i] == '\\')
		{
			workingDirectory[i] = 0;
			strcpy_s(shortFileName, MAX_PATH, &fileName[i+1]);
			break;
		}
		workingDirectory[i] = 0;
	}

	char oldDir[MAX_PATH];
	_getcwd(oldDir, MAX_PATH-1);
	_chdir(workingDirectory);		

	TiXmlDocument sceneXml;
	if(sceneXml.LoadFile(shortFileName) == false)
	{
		return false;
	}

	TiXmlNode *sceneXmlNode = sceneXml.FirstChild("_scene");

	if(!sceneXmlNode)
		return false;

	loadSceneGraph(sceneXmlNode->FirstChild("_scene_graph"), &m_sceneGraph);
	loadSceneEmitters(sceneXmlNode->FirstChild("_emitters"));

	// load ASVO file base
	TiXmlNode *ASVOXmlNode = sceneXmlNode->FirstChild("_ASVO");
	if(ASVOXmlNode)
	{
		for(TiXmlNode *nodeXmlChild=ASVOXmlNode->FirstChild();nodeXmlChild;nodeXmlChild=nodeXmlChild->NextSibling())
		{
			if(strcmp(nodeXmlChild->Value(), "_file_base") == 0)
			{
				ASSIGN_ATTRIBUTE_STR_CPY(m_ASVOFileBase, nodeXmlChild, "value");
				continue;
			}
		}
	}
	else
	{
		// use default ASVO for a model
		// Assumes that this scene contains only one model
		sprintf_s(m_ASVOFileBase, MAX_PATH, "%s\\%s\\default", workingDirectory, m_lastModelFileName);
	}

	m_sceneBB = m_sceneGraph.nodeBB;
	//m_sceneGraph.updateBB();

	_chdir(oldDir);

	return true;
}

bool Scene::loadModel(Model *model)
{
	m_modelList.push_back(model);
	m_modelListMap[model] = (int)(m_modelList.size()-1);

	return true;
}

bool Scene::loadOOC(const char *fileName)
{
	Model *newModel = new Model;
	m_modelsToBeDeleted.push_back(newModel);

	if(!newModel->load(fileName))
	{
		printf("Load OOC model failed\n");
		delete newModel;
	}

#	if 0
	FILE *fp = fopen("sponza.ply", "w");
	fprintf(fp, "ply\n");
	fprintf(fp, "format ascii 1.0\n");
	fprintf(fp, "element vertex %d\n", newModel->getNumVertexs());
	fprintf(fp, "property float32 x\n");
	fprintf(fp, "property float32 y\n");
	fprintf(fp, "property float32 z\n");
	fprintf(fp, "property float32 nx\n");
	fprintf(fp, "property float32 ny\n");
	fprintf(fp, "property float32 nz\n");
	fprintf(fp, "property float32 s\n");
	fprintf(fp, "property float32 t\n");
	fprintf(fp, "element face %d\n", newModel->getNumTriangles());
	fprintf(fp, "property list uint8 int32 vertex_indices\n");
	fprintf(fp, "end_header\n");

	for(int i=0;i<newModel->getNumVertexs();i++)
	{
		const Vertex &v = *newModel->getVertex(i);
		fprintf(fp, "%f %f %f %f %f %f %f %f\n", v.v.e[0], v.v.e[1], v.v.e[2], v.n.e[0], v.n.e[1], v.n.e[2], v.uv.e[0], v.uv.e[1]);
	}

	for(int i=0;i<newModel->getNumTriangles();i++)
	{
		const Triangle &tri = *newModel->getTriangle(i);
		fprintf(fp, "3 %d %d %d\n", tri.p[0], tri.p[1], tri.p[2]);
	}

	fclose(fp);
#	endif

	loadModel(newModel);

	return true;
}

bool Scene::loadOOCAnimation(const char *fileName)
{
	return true;
}

bool Scene::loadPly(const char *fileName)
{
	PLYLoader *loader = new PLYLoader;

	if(!loader->load(fileName))
	{
		delete loader;
		return false;
	}

	Model *newModel = new Model;
	m_modelsToBeDeleted.push_back(newModel);

	newModel->load(loader->getVertex(), loader->getNumVertexs(), loader->getFaces(), loader->getNumFaces(), Material());

	newModel->setName("model0");

	BVHBuilder::build(newModel);

	loadModel(newModel);

	delete loader;

	return false;
}

bool Scene::loadOBJ(const char *fileName)
{
	OBJLoader *loader = new OBJLoader;

	if(!loader->load(fileName, true))
	{
		delete loader;
		return false;
	}

	for(int i=0;i<loader->getNumSubMeshes();i++)
	{
		Model *newModel = new Model;
		m_modelsToBeDeleted.push_back(newModel);

		const GroupInfo &group = loader->getGroupInfo(i);

		newModel->load(loader->getVertex(i), loader->getNumVertexs(i), loader->getFaces(i), loader->getNumFaces(i), loader->getMaterial(group.materialFileName, group.materialName));

		if(strlen(group.name.c_str()) == 0)
		{
			char name[256];
			sprintf_s(name, 256, "model%d\n", i);
			newModel->setName(name);
		}
		else
			newModel->setName(group.name.c_str());

		BVHBuilder::build(newModel);

		loadModel(newModel);
	}

	delete loader;
	return true;
}

bool Scene::loadHCCMesh(const char *fileName)
{
	Model *newModel = new HCCMesh;
	m_modelsToBeDeleted.push_back(newModel);

	if(!newModel->load(fileName))
	{
		printf("Load OOC model failed\n");
		delete newModel;
	}

	loadModel(newModel);

	return true;
}

bool Scene::loadHCCMesh2(const char *fileName)
{
	Model *newModel = new HCCMesh2;
	m_modelsToBeDeleted.push_back(newModel);

	if(!newModel->load(fileName))
	{
		printf("Load OOC model failed\n");
		delete newModel;
	}

	loadModel(newModel);

	return true;
}

void Scene::exportScene(const char *fileName)
{
	TiXmlDocument xml;
	TiXmlDeclaration decXml("1.0", "utf-8", "");
	xml.InsertEndChild(decXml);
	TiXmlElement sceneXml("_scene");

	TiXmlElement sceneEmittersXml("_emitters");
	saveSceneEmitters(&sceneEmittersXml);
	sceneXml.InsertEndChild(sceneEmittersXml);

	TiXmlElement sceneGraphXml("_scene_graph");
	saveSceneGraph(&m_sceneGraph, &sceneGraphXml);
	sceneXml.InsertEndChild(sceneGraphXml);

	if(strlen(m_ASVOFileBase) != 0)
	{
		TiXmlElement ASVOXml("_ASVO");

		TiXmlElement att("_file_base");
		att.SetAttribute("value", m_ASVOFileBase);
		ASVOXml.InsertEndChild(att);

		sceneXml.InsertEndChild(ASVOXml);
	}

	xml.InsertEndChild(sceneXml);

	xml.SaveFile(fileName);
}

void Scene::generateEmitter()
{
	Emitter emitter;

	emitter.pos = m_sceneBB.max + 10.1f * (m_sceneBB.max - m_sceneBB.min);
	emitter.spotTarget.corner.set(FLT_MAX);

	m_emitList.push_back(emitter);
}

void Scene::generatePMTarget(Emitter &emitter)
{
	emitter.spotTarget.corner = Vector3(m_sceneBB.max.x(), m_sceneBB.min.y(), m_sceneBB.min.z());
	emitter.spotTarget.v1 = Vector3(0.0f, 0.0f, m_sceneBB.max.z() - m_sceneBB.min.z());
	emitter.spotTarget.v2 = Vector3(m_sceneBB.min.x() - m_sceneBB.max.x(), m_sceneBB.max.y() - m_sceneBB.min.y(), 0.0f);
	if(emitter.spotTarget.v1.maxAbsComponent() != 0.0f || emitter.spotTarget.v2.maxAbsComponent() != 0.0f)
	{
		emitter.spotTarget.normal = cross(emitter.spotTarget.v1, emitter.spotTarget.v2);
		emitter.spotTarget.normal.makeUnitVector();
	}
	else
	{
		// not spotlight
		emitter.spotTarget.corner.set(FLT_MAX);
	}
	
	if(emitter.numScatteringPhotons == 0) emitter.numScatteringPhotons = 512000;
	if(emitter.intensity == 0) emitter.intensity = 2000000;
}

void Scene::generateSceneStructure()
{
	m_sceneBB.min.set(FLT_MAX);
	m_sceneBB.max.set(-FLT_MAX);
	
	m_sceneGraph.clear();
	m_sceneGraph.set("_scene_graph", NULL, 0);

	char modelName[256];

	for(size_t i=0;i<m_modelList.size();i++)
	{
		m_sceneBB.update(m_modelList[i]->getModelBB().min);
		m_sceneBB.update(m_modelList[i]->getModelBB().max);

		if(strlen(m_modelList[i]->getName()))
			strcpy_s(modelName, 256, m_modelList[i]->getName());
		else
		{
			sprintf_s(modelName, 256, "model%d", i);
			m_modelList[i]->setName(modelName);
		}
		m_sceneGraph.addChild(modelName, m_modelList[i]);
	}

	if(getNumEmitters() == 0)
	{
		// use single emitter
		generateEmitter();
	}

	m_hasSceneStructure = true;
}

void Scene::trace(const Ray &ray, RGB4f &color, Material *outMat, int depth, float traveledDist, HitPointInfo *outHit, int stream)
{
	HitPointInfo hit;
	bool hasHit = false;
	hit.t = FLT_MAX;

	color = RGBf(0.0f, 0.0f, 0.0f);

	hasHit = getIntersection(ray, hit, 0.0f, stream);

	if(outHit)
		*outHit = hit;

	if(!hasHit)
	{
		RGBf col;
		m_envMap.shade(ray.direction(), col);
		color.set(col, 1.0f);
		return;
	}

	Model *hitModel = hit.modelPtr;

	Material mat;
	if(hitModel) mat = hitModel->getMaterial(hit.m);

	if(mat.getMapKd())
	{
		RGBf texValue;
		mat.getMapKd()->getTexValue(texValue, hit.uv.x(), hit.uv.y());
		mat.setMatKd(mat.getMatKd() * texValue);
	}

	Vector3 hitPosition = ray.origin() + hit.t * ray.direction();
	Vector3 shadowDir;
	hit.x = hitPosition;
	if(outMat) *outMat = mat;

	// for each emitter
	for(size_t i=0;i<m_emitList.size();i++)
	{
		Emitter &emitter = m_emitList[i];

		if(emitter.type == Emitter::ENVIRONMENT_LIGHT) continue;

		// just use diffuse values now...
		const RGBf &emitColor = emitter.color_Kd;

		// compute shadow ray direction

		shadowDir = emitter.pos - hitPosition;
		shadowDir.makeUnitVector();

		float cosFactor = dot(shadowDir, hit.n);

		if(cosFactor > 0.0f)
		{
			/*
			int idx = shadowDir.indexOfMaxComponent();
			float tLimit = (emitter.pos.e[idx] - hitPosition.e[idx]) / shadowDir.e[idx];
			HitPointInfo shadowHit;
			shadowHit.t = tLimit;

			Ray shadowRay;
			shadowRay.set(hitPosition, shadowDir);

			if(!getIntersection(shadowRay, shadowHit, m_intersectionStream))
			*/
				color += mat.getMatKd() * emitColor * cosFactor;
		}
	}
}

void Scene::shade(const Ray &ray, RGB4f &color, HitPointInfo &hit, bool hasHit)
{
	if(!hasHit)
	{
		RGBf col;
		m_envMap.shade(ray.direction(), col);
		color.set(col, 1.0f);
		return;
	}

	Model *hitModel = hit.modelPtr;

	Material mat;
	if(hitModel) mat = hitModel->getMaterial(hit.m);

	if(mat.getMapKd())
	{
		RGBf texValue;
		mat.getMapKd()->getTexValue(texValue, hit.uv.x(), hit.uv.y());
		mat.setMatKd(mat.getMatKd() * texValue);
	}

	Vector3 hitPosition = ray.origin() + hit.t * ray.direction();
	Vector3 shadowDir;

	// for each emitter
	for(size_t i=0;i<m_emitList.size();i++)
	{
		Emitter &emitter = m_emitList[i];
		// just use diffuse values now...
		const RGBf &emitColor = emitter.color_Kd;

		// compute shadow ray direction

		shadowDir = emitter.pos - hitPosition;
		shadowDir.makeUnitVector();

		float cosFactor = dot(shadowDir, hit.n);

		if(cosFactor > 0.0f)
		{
			/*
			int idx = shadowDir.indexOfMaxComponent();
			float tLimit = (emitter.pos.e[idx] - hitPosition.e[idx]) / shadowDir.e[idx];
			HitPointInfo shadowHit;
			shadowHit.t = tLimit;

			Ray shadowRay;
			shadowRay.set(hitPosition, shadowDir);

			if(!getIntersection(shadowRay, shadowHit, m_intersectionStream))
			*/
				color += mat.getMatKd() * emitColor * cosFactor;
		}
	}
}

Scene::EnvironmentMap::EnvironmentMap()
{
	for(int i=0;i<6;i++)
		tex[i] = 0;
}

Scene::EnvironmentMap::~EnvironmentMap()
{
	clear();
}

void Scene::EnvironmentMap::load(const char *fileNameBase)
{
	TextureManager *texMan = TextureManager::getSingletonPtr();

	static const char *fileSuffixes[] = {"_RT.jpg", "_LF.jpg", "_UP.jpg", "_DN.jpg", "_BK.jpg", "_FR.jpg"};

	char texFileName[MAX_PATH];

	for (int i=0;i<6;i++) {
		sprintf_s(texFileName, MAX_PATH, "%s%s", fileNameBase, fileSuffixes[i]);

		FILE *fp;
		fopen_s(&fp, texFileName, "r");
		if(!fp)
			sprintf_s(texFileName, MAX_PATH, "%s", fileNameBase);
		else
			fclose(fp);

		tex[i] = texMan->loadTexture(texFileName);

		if (!tex[i]) return;
	}	
}

void Scene::EnvironmentMap::clear()
{
	TextureManager *texMan = TextureManager::getSingletonPtr();

	for(int i=0;i<6;i++)
		if(tex[i])
			texMan->unloadTexture(tex[i]);
}

#include <typeinfo>
bool Scene::getIntersection(const Ray &ray, HitPointInfo &hitPointInfo, float tLimit, int stream)
{
	int threadID = omp_get_thread_num();
	StackElem *stack = m_stacks[threadID+MAX_NUM_THREADS*stream];

	unsigned int stackPtr;
	SceneNode *currentNode;
	float minT, maxT;
	bool hasHit = false;

	stack[0] = 0;
	stackPtr = 1;

	currentNode = &(m_sceneGraph);

	for(;;)
	{
		if(ray.boxIntersect(currentNode->nodeBB.min, currentNode->nodeBB.max, minT, maxT) && minT < hitPointInfo.t && maxT > 0.000005f)
		{
			if(currentNode->model && 
				(m_modelTypeSelector == Model::NONE ? true : m_modelTypeSelector == currentNode->model->getType()))
			{
				hasHit = currentNode->model->getIntersection(ray, hitPointInfo, tLimit, stream) | hasHit;
				if(tLimit > 0.0f && hasHit)
				{
					if(hitPointInfo.t < tLimit) return true;
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

		if (--stackPtr == 0) break;

		currentNode = stack[stackPtr];
	}
	return hasHit;
}

int Scene::getNumMaxPhotons()
{
	int numTotalPhotons = 0;
	for(int i=0;i<getNumEmitters();i++)
	{
		numTotalPhotons += getEmitter(i).numScatteringPhotons;
	}
	return numTotalPhotons;
}

void Scene::tracePhotons(int emitterIndex, Photon *outPhotons, int idx, int &numTotalPhotons, void (*funcProcessPhoton)(const Photon &photon))
{
	const Emitter &emitter = getEmitter(emitterIndex);
	const Parallelogram &target = emitter.spotTarget;

	const unsigned int numPhotons = emitter.numScatteringPhotons;

	unsigned int seed = tea<16>(idx, emitterIndex);

	Ray ray;
	Vector3 ori = emitter.sample(seed);
	lcg(seed);
	//Vector3 dir = (target.sample(seed) - ori);
	Vector3 dir = emitter.sampleEmitDirection(ori, seed);
	lcg(seed);
	dir.makeUnitVector();

	ray.set(ori, dir);

	HitPointInfo hit;
	hit.t = FLT_MAX;

	bool hasHit = false;

	int skip = seed % 6;
	skip = skip < 3 ? 0 : (skip == 5 ? 2 : 1);
	skip = 0;
	int maxDepth = 5;
	int depth = 0;
	Material material;
	while(++depth < maxDepth)
	{
		if(!getIntersection(ray, hit, 0.0f)) return;
		material = hit.modelPtr->getMaterial(hit);
		material.recalculateRanges();
		if(material.hasDiffuse())
			if(skip-- == 0) break;
		ray.set(hit.t * ray.direction() + ray.origin(), material.sampleDirection(hit.n, ray.direction(), seed));
		ray.set(ray.origin() + ray.direction(), ray.direction());	// offseting
		lcg(seed);
		hit.t = FLT_MAX;
	}

	float cosFac = -dot(ray.direction(), hit.n);
	if(cosFac > 0.0f)
	{
		Photon *photon;
#		pragma omp critical
		{
			photon = &outPhotons[numTotalPhotons++];
		}
		photon->pos = ray.origin() + hit.t * ray.direction();

		//photon->power = (emitter.color_Ka * material.getMatKa() + emitter.color_Kd * material.getMatKd() * cosFac) * (emitter.intensity / numPhotons);
		photon->power = (emitter.color_Ka + emitter.color_Kd * cosFac) * (emitter.intensity / numPhotons);
		photon->setDirection(ray.direction());

		if(funcProcessPhoton) funcProcessPhoton(*photon);
	}
}

int Scene::tracePhotons(int size, Photon *outPhotons, void (*funcProcessPhoton)(const Photon &photon))
{
	if(!g_timerPhotonTracing)
		g_timerPhotonTracing = StopWatch::create();
	StopWatch::get(g_timerPhotonTracing).start();

	int numTotalPhotons = 0;

	for(int i=0;i<getNumEmitters();i++)
	{
		int numPhotons = getEmitter(i).numScatteringPhotons;

		if(numPhotons == 0) continue;

#		pragma omp parallel for shared(numTotalPhotons) schedule(dynamic)
		for(int j=0;j<numPhotons;j++)
		{
			if(j % 10000 == 0) printf("%d photons are processed.\n", j);
			tracePhotons(i, outPhotons, j, numTotalPhotons, funcProcessPhoton);
		}
	}


#	if 0
	int numTotalPhotons = 0;

	int numEmitters = getNumEmitters();
	for(int i=0;i<numEmitters;i++)
	{
		const Emitter &emitter = getEmitter(i);
		const Parallelogram &target = emitter.spotTarget;

		int numPhotons = emitter.numScatteringPhotons;
		printf("[%d] num photons = %d\n", i, numPhotons);

#		pragma omp parallel for schedule(dynamic)
		for(int j=0;j<numPhotons;j++)
		{
			if(j % 10000 == 0) printf("%d photons are processed.\n", j);
			unsigned int seed = tea<16>(j, i);
			//unsigned int seed2 = tea<16>(j+1, 1);

			Ray ray;
			Vector3 ori = emitter.sample(seed);
			//Vector3 dir = (target.sample(seed) - ori);
			Vector3 dir = emitter.sampleEmitDirection(ori, seed);
			dir.makeUnitVector();

			ray.set(ori, dir);

			HitPointInfo hit;
			hit.t = FLT_MAX;

			/*
			if(getIntersection(ray, hit, 0.0f))
			{
				ray.set(hit.t * ray.direction() + ray.origin(), Material::sampleDirection(hit.n, ray.direction(), seed2));
				hit.t = FLT_MAX;
			if(getIntersection(ray, hit, 0.0f))
			{
				Photon &photon = outPhotons[numTotalPhotons++];
				photon.pos = ray.origin() + hit.t * ray.direction();

				Model *model = hit.modelPtr;
				Material mat = model->getMaterial(hit);
				photon.power = emitter.color_Ka * mat.getMatKa() + emitter.color_Kd * mat.getMatKd() * (emitter.intensity / numPhotons);
				photon.setDirection(ray.direction());

				if(size == numTotalPhotons) break;
			}
			}
			*/
			// bounce once when hit diffuse material
			// bounce when hit specular material
			//int skip = 0;
			int skip = seed % 6;
			skip = skip < 3 ? 0 : (skip == 5 ? 2 : 1);
			int maxDepth = 5;
			int depth = 0;
			Material material;
			bool hasPhoton = true;
			while(++depth < maxDepth)
			{
				if(!getIntersection(ray, hit, 0.0f))
				{
					hasPhoton = false;
					break;
				}
				material = hit.modelPtr->getMaterial(hit);
				if(material.hasDiffuse())
					if(skip-- == 0) break;
				ray.set(hit.t * ray.direction() + ray.origin(), material.sampleDirection(hit.n, ray.direction(), seed));
				ray.set(ray.origin() + ray.direction(), ray.direction());	// offseting
				hit.t = FLT_MAX;
			}
			if(!hasPhoton) continue;

			float cosFac = -dot(ray.direction(), hit.n);
			if(cosFac > 0.0f)
			{
				Photon *photon;
#				pragma omp critical
				{
					photon = &outPhotons[numTotalPhotons++];
				}
				photon->pos = ray.origin() + hit.t * ray.direction();

				//photon->power = (emitter.color_Ka * material.getMatKa() + emitter.color_Kd * material.getMatKd() * cosFac) * (emitter.intensity / numPhotons);
				photon->power = (emitter.color_Ka + emitter.color_Kd * cosFac) * (emitter.intensity / numPhotons);
				photon->setDirection(ray.direction());

				if(funcProcessPhoton) funcProcessPhoton(*photon);
			}

		}
	}
#	endif

	StopWatch::get(g_timerPhotonTracing).stop();
	printf("Tracing photons : %f ms\n", StopWatch::get(g_timerPhotonTracing).getTime());

	return numTotalPhotons;
}

int Scene::buildPhotonKDTree(int size, Photon **photons, AABB &bb)
{
	if(!g_timerBuildingPhotonKDTree)
		g_timerBuildingPhotonKDTree = StopWatch::create();
	StopWatch::get(g_timerBuildingPhotonKDTree).start();

	int sizeKDTree = size;
	pow2roundup(sizeKDTree);
	sizeKDTree--;
	printf("sizeKDTree = %d\n", sizeKDTree); 

	if(sizeKDTree <= 0) return sizeKDTree;

	Photon *temp = new Photon[sizeKDTree];

	Photon::SplitChoice splitChoice = Photon::SPLIT_CHOICE_LONGEST_DIM;

	if(splitChoice == Photon::SPLIT_CHOICE_LONGEST_DIM)
	{
		for(int i=0;i<size;i++)
			bb.update((*photons)[i].pos);
	}

	buildPhotonKDTree(*photons, temp, 0, size, 0, 0, splitChoice, bb);
	delete[] *photons;
	*photons = temp;

	StopWatch::get(g_timerBuildingPhotonKDTree).stop();
	printf("Building photon kd-tree : %f ms\n", StopWatch::get(g_timerBuildingPhotonKDTree).getTime());

	return sizeKDTree;
}

void Scene::buildPhotonKDTree(Photon *photons, Photon *outPhotons, int left, int right, int depth, int curRoot, Photon::SplitChoice splitChoice, const AABB &bb)
{
	if(left == right) 
	{
		outPhotons[curRoot].axis = Photon::SPLIT_AXIS_NULL;
		outPhotons[curRoot].power = RGBf(0.0f, 0.0f, 0.0f);
		return;
	}

	if(right - left == 1) 
	{
		photons[left].axis = Photon::SPLIT_AXIS_LEAF;
		outPhotons[curRoot] = photons[left];
		return;
	}

	// Choose axis to split on
	int axis;
	switch(splitChoice) 
	{
	case Photon::SPLIT_CHOICE_ROUND_ROBIN:
		axis = depth%3;
		break;
	case Photon::SPLIT_CHOICE_HIGHEST_VARIANCE:
	{
		Vector3 mean  = Vector3( 0.0f ); 
		Vector3 diff2 = Vector3( 0.0f );
		for(int i = left; i < right; ++i) {
			Vector3 x     = photons[i].pos;
			Vector3 delta = x - mean;
			Vector3 n_inv = Vector3( 1.0f / ( static_cast<float>( i - left ) + 1.0f ) );
			mean = mean + delta * n_inv;
			diff2 += delta*( x - mean );
		}
		Vector3 n_inv = Vector3( 1.0f / ( static_cast<float>(right-left) - 1.0f ) );
		Vector3 variance = diff2 * n_inv;
		axis = variance.indexOfMaxComponent();
	}
		break;
	case Photon::SPLIT_CHOICE_LONGEST_DIM:
		axis = (bb.max - bb.min).indexOfMaxComponent();
		break;
	}

	int median = (left+right) / 2;
	Photon* addr = &(photons[left]);

	switch( axis ) {
	case 0:
		select<Photon, 0>( addr, 0, right-left-1, median-left );
		photons[median].axis = Photon::SPLIT_AXIS_X;
		break;
	case 1:
		select<Photon, 1>( addr, 0, right-left-1, median-left );
		photons[median].axis = Photon::SPLIT_AXIS_Y;
		break;
	case 2:
		select<Photon, 2>( addr, 0, right-left-1, median-left );
		photons[median].axis = Photon::SPLIT_AXIS_Z;
		break;
	}

	AABB leftBB(bb), rightBB(bb);
	if(splitChoice == Photon::SPLIT_CHOICE_LONGEST_DIM) 
	{
		leftBB.max.e[axis] = photons[median].pos.e[axis];
		rightBB.min.e[axis] = photons[median].pos.e[axis];
	}

	outPhotons[curRoot] = photons[median];

	buildPhotonKDTree(photons, outPhotons, left, median, depth+1, 2*curRoot+1, splitChoice, leftBB);
	buildPhotonKDTree(photons, outPhotons, median+1, right, depth+1, 2*curRoot+2, splitChoice, rightBB);
}

Model *Scene::getModel(const char *name)
{
	for(size_t i=0;i<m_modelList.size();i++)
	{
		if(!strcmp(m_modelList[i]->getName(), name)) return m_modelList[i];
	}
	return NULL;
}

int Scene::pushEmitter(const Emitter &emitter) 
{
	int ret = (int)m_emitList.size();
	m_emitList.push_back(emitter);

	if(ret > 0)
	{
		// move environment light to back
		if(m_emitList[ret-1].type == Emitter::ENVIRONMENT_LIGHT)
		{
			Emitter temp = m_emitList[ret-1];
			m_emitList[ret-1] = m_emitList[ret];
			m_emitList[ret] = temp;
			ret--;
		}
	}
	return ret;
}
void Scene::removeEmitter(int pos)
{
	for(int i=pos;i<(int)m_emitList.size()-1;i++)
		m_emitList[i] = m_emitList[i+1];
	m_emitList.pop_back();
}

int Scene::getIntersectionStream()
{
	if(m_lastIntersectionStream == MAX_NUM_INTERSECTION_STREAM)
	{
		printf("Not enough reserved stream, MAX_NUM_INTERSECTION_STREAM = %d\n", MAX_NUM_INTERSECTION_STREAM);
		return -1;
	}
	return m_lastIntersectionStream++;
}
