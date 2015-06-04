/********************************************************************
	created:	2013/09/26
	file path:	d:\Projects\Redering\OpenIRT\include
	file base:	OBJLoader
	file ext:	h
	author:		Tae-Joon Kim (tjkim.kaist@gmail.com)
	
	comment:	OBJLoader class, load WaveFrontOBJ (.obj) and MTL (.mtl) file formats.
*********************************************************************/

#pragma once

#include <vector>
#include <map>

#include "Vertex.h"
#include "Face.h"
#include "Material.h"
#include "GroupInfo.h"

using namespace std;

namespace irt
{

class OBJLoader
{
public:
	OBJLoader(void);
	~OBJLoader(void);

	void clear(void);

	OBJLoader(const char *fileName);

	bool load(const char *fileName, bool localizeVertices = false);

	bool hasTextureCoordinates(void);
	bool hasNormals(void);

	void getModelBB(Vector3 &min, Vector3 &max);

	Vertex *getVertex();
	Vertex *getVertex(int subMesh);
	int getNumSubMeshes();
	int *getIndex(int subMesh);
	Face *getFaces(int subMesh);
	Material getMaterial(const string &fileName, const string &matName);

	int getNumVertexs();
	int getNumVertexs(int subMesh);
	int getNumFaces(int subMesh);
	const GroupInfo &getGroupInfo(int subMesh);
	void getGroupName(int subMesh, char *groupName);
	void getMaterialName(int subMesh, char *fileName, char *materialName);

	Vector3 *getV(void);
	Vector3 *getVt(void);
	Vector3 *getVn(void);
	Vector3 *getVp(void);
	int getNumV(void);
	int getNumVt(void);
	int getNumVn(void);
	int getNumVp(void);

	int *getOriginalVIndex(void);
	int *getOriginalVtIndex(void);
	int *getOriginalVnIndex(void);
	int *getOriginalVpIndex(void);

private:
	bool m_hasTextureCoordinates;
	bool m_hasNormals;

	map<string, map<string, int> > m_materialMap;
	vector<Material> m_materialList;
	vector<GroupInfo> m_groupList;
	vector<Vector3> m_vList;
	vector<Vector3> m_vtList;
	vector<Vector3> m_vnList;
	vector<Vector3> m_vpList;
	vector<vector<FaceStr> > m_fList;

	// Vertices may be duplicated to make them unified (a vertex has position, normal, and texture coordinate together)
	vector<vector<Face> > m_faceList;
	vector<Vertex> m_vertList;
	vector<vector<Vertex> > m_groupedVertList;

	// Original vertex index of a unified vertex before duplication
	vector<int> m_vOriIndex;
	vector<int> m_vtOriIndex;
	vector<int> m_vnOriIndex;
	vector<int> m_vpOriIndex;

	Vector3 m_BBMin, m_BBMax;

	string trim(const string& s);
	char *trim(char* s);
	void loadMTLMaterial(const char *fileName);
	void addSubMesh(GroupInfo &group, vector<FaceStr> &subFaceList);
};

};