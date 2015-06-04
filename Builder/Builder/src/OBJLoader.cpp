#include <stdlib.h>
#include <string>
#include <direct.h>
#include <float.h>
#include "OBJLoader.h"

using namespace irt;

OBJLoader::OBJLoader(void)
	: m_hasTextureCoordinates(0), m_hasNormals(0)
{
	m_BBMin.set(FLT_MAX);
	m_BBMax.set(-FLT_MAX);
}

OBJLoader::~OBJLoader(void)
{
	clear();
}

void OBJLoader::clear(void)
{
	for(size_t i=0;i<m_faceList.size();i++)
	{
		for(size_t j=0;j<m_faceList[i].size();j++)
		{
			if(m_faceList[i][j].verts) delete[] m_faceList[i][j].verts;
		}
	}

	m_materialMap.clear();
	m_materialList.clear();
	m_groupList.clear();
	m_vList.clear();
	m_vtList.clear();
	m_vnList.clear();
	m_vpList.clear();
	m_fList.clear();

	m_vOriIndex.clear();
	m_vtOriIndex.clear();
	m_vnOriIndex.clear();
	m_vpOriIndex.clear();

	m_faceList.clear();
	m_vertList.clear();
	m_groupedVertList.clear();

	m_BBMin.set(FLT_MAX);
	m_BBMax.set(-FLT_MAX);
}

OBJLoader::OBJLoader(const char *fileName)
{
	load(fileName);
}

void OBJLoader::getModelBB(Vector3 &min, Vector3 &max)
{
	min = m_BBMin;
	max = m_BBMax;
}

Vector3 *OBJLoader::getV(void)
{
	if(m_vList.size() == 0) return NULL;

	return &m_vList[0];
}

Vector3 *OBJLoader::getVt(void)
{
	if(m_vtList.size() == 0) return NULL;

	return &m_vtList[0];
}

Vector3 *OBJLoader::getVn(void)
{
	if(m_vnList.size() == 0) return NULL;

	return &m_vnList[0];
}

Vector3 *OBJLoader::getVp(void)
{
	if(m_vpList.size() == 0) return NULL;

	return &m_vpList[0];
}

int OBJLoader::getNumV(void)
{
	return (int)m_vList.size();
}

int OBJLoader::getNumVt(void)
{
	return (int)m_vtList.size();
}

int OBJLoader::getNumVn(void)
{
	return (int)m_vnList.size();
}

int OBJLoader::getNumVp(void)
{
	return (int)m_vpList.size();
}

int *OBJLoader::getOriginalVIndex(void)
{
	if(m_vOriIndex.size() == 0) return NULL;

	return &m_vOriIndex[0];
}

int *OBJLoader::getOriginalVtIndex(void)
{
	if(m_vtOriIndex.size() == 0) return NULL;

	return &m_vtOriIndex[0];
}

int *OBJLoader::getOriginalVnIndex(void)
{
	if(m_vnOriIndex.size() == 0) return NULL;

	return &m_vnOriIndex[0];
}

int *OBJLoader::getOriginalVpIndex(void)
{
	if(m_vpOriIndex.size() == 0) return NULL;

	return &m_vpOriIndex[0];
}

Vertex *OBJLoader::getVertex()
{
	if(m_vertList.size() == 0) return NULL;

	return &m_vertList[0];
}

Vertex *OBJLoader::getVertex(int subMesh)
{
	if((int)m_groupedVertList.size() <= subMesh) return NULL;

	if(m_groupedVertList[subMesh].size() == 0) return NULL;

	return &m_groupedVertList[subMesh][0];
}

int OBJLoader::getNumSubMeshes()
{
	return (int)m_fList.size();
}

Face *OBJLoader::getFaces(int subMesh)
{
	if((int)m_faceList.size() <= subMesh) return NULL;

	if(m_faceList[subMesh].size() == 0) return NULL;

	return &m_faceList[subMesh][0];
}

Material OBJLoader::getMaterial(const string &fileName, const string &matName)
{
	if(m_materialList.size() == 0) return Material();

	int idx = m_materialMap[fileName][matName];

	return m_materialList[idx];
}

int OBJLoader::getNumVertexs()
{
	return (int)m_vertList.size();
}

int OBJLoader::getNumVertexs(int subMesh)
{
	if((int)m_groupedVertList.size() <= subMesh) return -1;

	return (int)m_groupedVertList[subMesh].size();
}

int OBJLoader::getNumFaces(int subMesh)
{
	if((int)m_fList.size() <= subMesh) return -1;

	return (int)m_fList[subMesh].size();
}

const GroupInfo & OBJLoader::getGroupInfo(int subMesh)
{
	return m_groupList[subMesh];
}

void OBJLoader::getGroupName(int subMesh, char *groupName)
{
	memcpy_s(groupName, 256, m_groupList[subMesh].name.c_str(), m_groupList[subMesh].name.length()+1);
}

void OBJLoader::getMaterialName(int subMesh, char *fileName, char *materialName)
{
	memcpy_s(fileName, 256, m_groupList[subMesh].materialFileName.c_str(), m_groupList[subMesh].materialFileName.length()+1);
	memcpy_s(materialName, 256, m_groupList[subMesh].materialName.c_str(), m_groupList[subMesh].materialName.length()+1);
}

void OBJLoader::addSubMesh(GroupInfo &group, vector<FaceStr> &subFaceList)
{
	if(subFaceList.size()) 
	{
		m_fList.push_back(subFaceList);
		subFaceList.clear();
		m_groupList.push_back(group);
	}
}

bool OBJLoader::load(const char *fileName, bool localizeVertices)
{
	FILE *fp;
	errno_t err;

	if((err = fopen_s(&fp, fileName, "r")) != 0)
	{
		printf("The file '%s' was not opened!\n", fileName);
		return false;
	}

	char oldDir[MAX_PATH];
	_getcwd(oldDir, MAX_PATH-1);
	_chdir(fileName);		

	// read raw file
	char currentLine[1024];
	char *mainContext;

	vector<FaceStr> *pSubFaceList = new vector<FaceStr>;
	vector<FaceStr> &subFaceList = *pSubFaceList;
	GroupInfo curGroup;

	vector<string> tempStrVector;
	while(fgets(currentLine, 1024, fp))
	{
		trim(currentLine);
		char *type = strtok_s(currentLine, " ", &mainContext);
		
		if(type == NULL || type[0] == 0 || type[0] == '#') continue;

		if(!strncmp("mtllib", type, 1024))
		{
			const char *matFileName = strtok_s(NULL, " ", &mainContext);
			curGroup.materialFileName = matFileName;

			char fullMatFileName[256];
			strncpy_s(fullMatFileName, 256, fileName, strlen(fileName));
			for(int i=(int)strlen(fullMatFileName)-1;i>=0;i--)
			{
				if(fullMatFileName[i] == '/' || fullMatFileName[i] == '\\')
				{
					memcpy_s(&fullMatFileName[i+1], 256, matFileName, strlen(matFileName)+1);
					break;
				}
				else if(i == 0)
				{
					memcpy_s(fullMatFileName, 256, matFileName, strlen(matFileName)+1);
				}
			}
			loadMTLMaterial(fullMatFileName);
		}

		if(!strncmp("v", type, 1024))
		{
			Vector3 v;
			for(int i=0;i<3;i++)
			{
				char *elem = strtok_s(NULL, " ", &mainContext);
				if(!elem) break;
				v.e[i] = (float)atof(elem);
			}

			for(int i=0;i<3;i++) m_BBMin.e[i] = min(m_BBMin.e[i], v.e[i]);
			for(int i=0;i<3;i++) m_BBMax.e[i] = max(m_BBMax.e[i], v.e[i]);

			m_vList.push_back(v);
		}

		if(!strncmp("vt", type, 1024))
		{
			Vector3 v;
			for(int i=0;i<3;i++)
			{
				char *elem = strtok_s(NULL, " ", &mainContext);
				if(!elem) break;
				v.e[i] = (float)atof(elem);
			}
			m_vtList.push_back(v);
		}

		if(!strncmp("vn", type, 1024))
		{
			Vector3 v;
			for(int i=0;i<3;i++)
			{
				char *elem = strtok_s(NULL, " ", &mainContext);
				if(!elem) break;
				v.e[i] = (float)atof(elem);
			}
			m_vnList.push_back(v);
		}

		if(!strncmp("vp", type, 1024))
		{
			Vector3 v;
			for(int i=0;i<3;i++)
			{
				char *elem = strtok_s(NULL, " ", &mainContext);
				if(!elem) break;
				v.e[i] = (float)atof(elem);
			}
			m_vpList.push_back(v);
		}

		if(!strncmp("f", type, 1024))
		{
			FaceStr face;
			tempStrVector.clear();
			for(face.n=0;;face.n++)
			{
				char *elem = strtok_s(NULL, " ", &mainContext);
				if(!elem) break;
				//face.f[face.n] = elem;
				tempStrVector.push_back(elem);
			}

			face.f = new string[face.n];
			for(int i=0;i<face.n;i++)
			{
				face.f[i] = tempStrVector[i];
			}

			subFaceList.push_back(face);
		}

		if(!strncmp("usemtl", type, 1024))
		{
			addSubMesh(curGroup, subFaceList);

			curGroup.materialName = strtok_s(NULL, " ", &mainContext);
		}

		if(!strncmp("o", type, 1024))
		{
			addSubMesh(curGroup, subFaceList);
		}

		if(!strncmp("g", type, 1024))
		{
			addSubMesh(curGroup, subFaceList);

			curGroup.name = strtok_s(NULL, " ", &mainContext);
		}

		if(!strncmp("s", type, 1024))
		{
			addSubMesh(curGroup, subFaceList);
		}
	}

	addSubMesh(curGroup, subFaceList);

	delete pSubFaceList;

	fclose(fp);

	map<string, int> *pVertMap = new map<string, int>;
	map<string, int> &vertMap = *pVertMap;
	char *context;
	char str[1024];
	char *elem;

	m_hasTextureCoordinates = true;
	m_hasNormals = true;

	if(localizeVertices) m_groupedVertList.resize(m_fList.size());
	
	for(size_t i=0;i<m_fList.size();i++)
	{
		vector<Face> *pSubFaceList = new vector<Face>;
		vector<Face> &subFaceList = *pSubFaceList;

		vector<Vertex> *pVertList = &m_vertList;

		if(localizeVertices)
		{
			vertMap.clear();
			pVertList = &m_groupedVertList[i];
		}
		vector<Vertex> &vertList = *pVertList;

		//int idx = 0;
		for(size_t j=0, k=0;j<m_fList[i].size();j++)
		{
			const FaceStr &faceStr = m_fList[i][j];
			Face face;
			face.n = faceStr.n;
			face.verts = new int[face.n];
			for(int p=0;p<faceStr.n;p++)
			{
				Vertex vert;
				int vertIdx;
				int vIdx, vtIdx, vnIdx;
				map<string, int>::iterator it = vertMap.find(faceStr.f[p]);
				if(it == vertMap.end())
				{
					strncpy_s(str, 1024, faceStr.f[p].c_str(), faceStr.f[p].length());
					str[faceStr.f[p].length()] = 0;

					if(strstr(str, "//"))
					{
						m_hasTextureCoordinates = false;
						if(elem = strtok_s(str, " /", &context))
						{
							vIdx = atoi(elem)-1;
							//CONVERT_V4_to_V3(m_vList[vIdx], vert.v);
							vert.v = m_vList[vIdx];
						}
						if(elem = strtok_s(NULL, " /", &context))
						{
							vnIdx = atoi(elem)-1;
							vert.n = m_vnList[vnIdx];
						}
						else
							m_hasNormals = false;
					}
					else
					{
						if(elem = strtok_s(str, " /", &context))
						{
							vIdx = atoi(elem)-1;
							//CONVERT_V4_to_V3(m_vList[vIdx], vert.v);
							vert.v = m_vList[vIdx];
						}
						if(elem = strtok_s(NULL, " /", &context))
						{
							vtIdx = atoi(elem)-1;
							vert.uv.e[0] = m_vtList[vtIdx].e[0];
							vert.uv.e[1] = m_vtList[vtIdx].e[1];
						}
						else
							m_hasTextureCoordinates = false;

						if(elem = strtok_s(NULL, " /", &context))
						{
							vnIdx = atoi(elem)-1;
							vert.n = m_vnList[vnIdx];
						}
						else
							m_hasNormals = false;

					}
					vertIdx = (int)vertList.size();
					vertMap[faceStr.f[p]] = vertIdx;
					vertList.push_back(vert);

					// keep original index of each vertex component
					m_vOriIndex.push_back(vIdx);
					m_vtOriIndex.push_back(vtIdx);
					m_vnOriIndex.push_back(vnIdx);
				}
				else
				{
					vert = vertList[it->second];
					vertIdx = it->second;
				}
				face.verts[p] = vertIdx;
			}

			subFaceList.push_back(face);

			delete[] faceStr.f;
		}
		m_faceList.push_back(subFaceList);
		delete pSubFaceList;
	}

	delete pVertMap;

	_chdir(oldDir);
	return true;
}

int OBJLoader::save(const char *fileName)
{
	char fileNameWithOutExt[MAX_PATH];
	strcpy_s(fileNameWithOutExt, MAX_PATH, fileName);
	for(int i=strlen(fileName)-1;i>=0;i--)
	{
		if(fileName[i] == '.')
		{
			fileNameWithOutExt[i] = 0;
			break;
		}
	}

	bool multiple = m_groupedVertList.size() > 1;

	char subFileName[MAX_PATH];

	for(size_t i=0;i<m_groupedVertList.size();i++)
	{
		vector<Vertex> &vertList = m_groupedVertList[i];
		vector<Face> &faceList = m_faceList[i];
		if(multiple) sprintf_s(subFileName, MAX_PATH, "%s%d.ply", fileNameWithOutExt, i);
		else sprintf_s(subFileName, MAX_PATH, "%s.ply", fileNameWithOutExt);

		FILE *fp;
		fopen_s(&fp, subFileName, "wb");

		fprintf(fp, "ply\n");
		fprintf(fp, "format ascii 1.0\n");
		fprintf(fp, "comment used material = %s\n", m_groupList[i].materialName);
		fprintf(fp, "element vertex %d\n", vertList.size());
		fprintf(fp, "property float32 x\n");
		fprintf(fp, "property float32 y\n");
		fprintf(fp, "property float32 z\n");
		fprintf(fp, "property float32 nx\n");
		fprintf(fp, "property float32 ny\n");
		fprintf(fp, "property float32 nz\n");
		fprintf(fp, "property float32 s\n");
		fprintf(fp, "property float32 t\n");
		fprintf(fp, "property float32 d\n");
		fprintf(fp, "element face %d\n", faceList.size());
		fprintf(fp, "property list uint8 int32 vertex_indices\n");
		fprintf(fp, "end_header\n");

		for(size_t j=0;j<vertList.size();j++)
		{
			Vertex &vert = vertList[j];
			fprintf(fp, "%f %f %f %f %f %f %f %f %f\n", 
				vert.v.e[0], vert.v.e[1], vert.v.e[2], 
				vert.n.e[0], vert.n.e[1], vert.n.e[2], 
				vert.uv.e[0], vert.uv.e[1], 0.0f); 
		}

		for(size_t j=0;j<faceList.size();j++)
		{
			Face &face = faceList[j];
			fprintf(fp, "%d", face.n);
			for(int k=0;k<face.n;k++)
				fprintf(fp, " %d", face.verts[k]);
			fprintf(fp, "\n");
		}

		fclose(fp);
	}

	return (int)m_groupedVertList.size();
}

bool OBJLoader::hasTextureCoordinates(void)
{
	return m_hasTextureCoordinates;
}

bool OBJLoader::hasNormals(void)
{
	return m_hasNormals;
}


string OBJLoader::trim(const string& s)
{

	size_t f,e ;

	if (s.length() == 0)
		return s;

	if (s.c_str()[0] == 10)
		return "";

	f = s.find_first_not_of(" \t\r\n");
	e = s.find_last_not_of(" \t\r\n");

	if (f == string::npos)
		return "";
	return string(s,f,e-f+1);
}

char *OBJLoader::trim(char* s)
{
	size_t f,e ;

	size_t len = strlen(s);

	if (len == 0)
		return s;

	if (s[0] == 10)
		return "";

	char d[] = " \t\r\n";

	for(f=0;f<len;f++)
	{
		bool notMatch = true;
		for(size_t i=0;i<strlen(d);i++)
		{
			if(s[f] == d[i]) notMatch = false;
		}
		if(notMatch) break;
	}

	if(f == len) return "";

	for(e=len-1;e>=0;e--)
	{
		bool notMatch = true;
		for(size_t i=0;i<strlen(d);i++)
		{
			if(s[e] == d[i]) notMatch = false;
		}
		if(notMatch) break;
	}

	if(e < 0) return "";

	char *temp = new char[len];
	memset(temp, 0, len);
	memcpy_s(temp, len, s+f, e-f+1);
	temp[e-f+1] = 0;
	memcpy_s(s, len, temp, len);
	delete[] temp;
	
	return s;
}

void OBJLoader::loadMTLMaterial(const char *fileName)
{
	FILE *fp;
	errno_t err;

	char workingDirectory[MAX_PATH];
	char shortFileName[MAX_PATH];
	strcpy_s(workingDirectory, MAX_PATH-1, fileName);
	// remove last entry of prompt
	for(int i=(int)strlen(workingDirectory)-1;i>=0;i--)
	{
		if(workingDirectory[i] == '/' || workingDirectory[i] == '\\')
		{
			strcpy_s(shortFileName, MAX_PATH, &workingDirectory[i+1]);
			workingDirectory[i] = 0;
			break;
		}
//		workingDirectory[i] = 0;
	}

	if((err = fopen_s(&fp, fileName, "r")) != 0)
	{
		printf("The file '%s' was not opened!\n", fileName);
		return ;
	}

	// parse MTL file
	char currentLine[1024];
	Material mat;
	bool isFirstMat = true;
	string lastMatName;
	while(fgets(currentLine, 1024, fp))
	{
		trim(currentLine);
		if(strstr(currentLine, "newmtl"))
		{
			string curMatName = currentLine+7;
			curMatName = trim(curMatName);

			if(!isFirstMat) 
			{
				m_materialMap[shortFileName][lastMatName] = (int)m_materialList.size();
				m_materialList.push_back(mat);

				//mat.setDefault();
			}
			lastMatName = curMatName;

			memcpy_s(mat.name, 60, curMatName.c_str(), curMatName.size()+1);

			isFirstMat = false;

			continue;
		}

		string curStr = currentLine;
		curStr = trim(curStr);
		const char *curC = curStr.c_str();

		if(strstr(curC, "#")) continue;		// skip comments

		if(strstr(curC, "Ka") == curC)
		{
			sscanf_s(curC, "Ka %f %f %f", 
				&mat.mat_Ka.e[0], 
				&mat.mat_Ka.e[1], 
				&mat.mat_Ka.e[2]);
		}
		if(strstr(curC, "Kd") == curC)
		{
			sscanf_s(curC, "Kd %f %f %f", 
				&mat.mat_Kd.e[0], 
				&mat.mat_Kd.e[1], 
				&mat.mat_Kd.e[2]);
		}
		if(strstr(curC, "Ks") == curC)
		{
			sscanf_s(curC, "Ks %f %f %f", 
				&mat.mat_Ks.e[0], 
				&mat.mat_Ks.e[1], 
				&mat.mat_Ks.e[2]);
		}
		if(strstr(curC, "Tf") == curC)
		{
			sscanf_s(curC, "Tf %f %f %f", 
				&mat.mat_Tf.e[0], 
				&mat.mat_Tf.e[1], 
				&mat.mat_Tf.e[2]);
		}

		if(strstr(curC, "Ns") == curC)
		{
			sscanf_s(curC, "Ns %f", &mat.mat_Ns);
		}
		if(strstr(curC, "d") == curC)
		{
			sscanf_s(curC, "d %f", &mat.mat_d);
		}

		if(strstr(curC, "illum") == curC)
		{
			sscanf_s(curC, "illum %d", &mat.mat_illum);
		}

		if(strstr(curC, "map_Ka") == curC)
		{
			char mapFileName[MAX_PATH] = {0, };
			sscanf_s(curC, "map_Ka %s", mapFileName, MAX_PATH);
			if(strlen(mapFileName) != 0) 
			{
				mat.setMapKa(mapFileName);
				//if(!mat.getMapKa())
				//{
				//	char fullFileName[MAX_PATH];
				//	sprintf_s(fullFileName, MAX_PATH, "%s\\%s", workingDirectory, mapFileName);
				//	mat.setMapKa(fullFileName);
				//}
				//if(!mat.getMapKa())
				//{
				//	char fullFileName[MAX_PATH];
				//	sprintf_s(fullFileName, MAX_PATH, "%s\\texture\\%s", workingDirectory, mapFileName);
				//	mat.setMapKa(fullFileName);
				//}
			}
		}

		if(strstr(curC, "map_Kd") == curC)
		{
			char mapFileName[MAX_PATH] = {0, };
			sscanf_s(curC, "map_Kd %s", mapFileName, MAX_PATH);
			if(strlen(mapFileName) != 0) 
			{
				mat.setMapKd(mapFileName);
				//if(!mat.getMapKd())
				//{
				//	char fullFileName[MAX_PATH];
				//	sprintf_s(fullFileName, MAX_PATH, "%s\\%s", workingDirectory, mapFileName);
				//	mat.setMapKd(fullFileName);
				//}
				//if(!mat.getMapKd())
				//{
				//	char fullFileName[MAX_PATH];
				//	sprintf_s(fullFileName, MAX_PATH, "%s\\texture\\%s", workingDirectory, mapFileName);
				//	mat.setMapKd(fullFileName);
				//}
			}
		}

		if(strstr(curC, "map_bump") == curC)
		{
			char mapFileName[MAX_PATH] = {0, };
			sscanf_s(curC, "map_bump %s", mapFileName, MAX_PATH);
			if(strlen(mapFileName) != 0) 
			{
				mat.setMapBump(mapFileName);
				//if(!mat.getMapBump())
				//{
				//	char fullFileName[MAX_PATH];
				//	sprintf_s(fullFileName, MAX_PATH, "%s\\%s", workingDirectory, mapFileName);
				//	mat.setMapBump(fullFileName);
				//}
				//if(!mat.getMapBump())
				//{
				//	char fullFileName[MAX_PATH];
				//	sprintf_s(fullFileName, MAX_PATH, "%s\\texture\\%s", workingDirectory, mapFileName);
				//	mat.setMapBump(fullFileName);
				//}
			}
		}
	}
	m_materialMap[shortFileName][lastMatName] = (int)m_materialList.size();
	m_materialList.push_back(mat);
	fclose(fp);
}