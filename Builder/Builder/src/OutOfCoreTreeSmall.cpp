#include "OutOfCoreTree.h"

#define _CRT_SECURE_NO_DEPRECATE
#pragma warning (disable: 4996)
#include <math.h>
#include <direct.h>

#include "CashedBoxFiles.h"
#include "Vertex.h"
#include "Triangle.h"
#include "helpers.h"

#include "VoxelBVH.h"

#include "OptionManager.h"
#include "Progression.h"
#include "Materials.h"
#include "Files.h"

//#define USE_DOE
#ifdef USE_DOE
DOEColoring g_doeColoring;
#endif

bool OutOfCoreTree::buildSmall(const char *outputPath, const char *fileName, bool useModelTransform) {
	TimerValue start, end;
	//OptionManager *opt = OptionManager::getSingletonPtr();

	start.set();

	this->useModelTransform = useModelTransform;

	char curFileName[MAX_PATH];
	strcpy(curFileName, fileName);

	memset(mat, 0, sizeof(float)*16);
	mat[0] = mat[5] = mat[10] = mat[15] = 1.0f;

	if(strncmp(&fileName[strlen(fileName)-4], ".ply", 4) != 0)
	{
		FILE *fp = fopen(fileName, "r");
		char currentLine[500];
		fgets(currentLine, 499, fp);
		if(useModelTransform)
		{
			sscanf(currentLine, "%s %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f", curFileName,
						&mat[0*4+0], &mat[1*4+0], &mat[2*4+0], &mat[3*4+0], 
						&mat[0*4+1], &mat[1*4+1], &mat[2*4+1], &mat[3*4+1], 
						&mat[0*4+2], &mat[1*4+2], &mat[2*4+2], &mat[3*4+2], 
						&mat[0*4+3], &mat[1*4+3], &mat[2*4+3], &mat[3*4+3]
						);
		}
		else
		{
			sscanf(currentLine, "%s", curFileName);
		}

		fclose(fp);
	}

	//const char *baseDirName = opt->getOption("global", "scenePath", "");

	char tempFileName[MAX_PATH];
	int pos = 0;
	for(int i=strlen(curFileName)-1;i>=0;i--)
	{
		if(curFileName[i] == '/' || curFileName[i] == '\\')
		{
			pos = i+1;
			break;
		}
	}
	strcpy(tempFileName, &curFileName[pos]);
	sprintf(outDirName, "%s\\%s.ooc", outputPath, tempFileName);
	mkdir(outDirName);

	initSmall();

	firstVertexPassSmall(curFileName);

	writeModelInfo();

	secondVertexPassSmall(curFileName);

	bridge2to3();

	thirdVertexPassSmall(curFileName, 0);

	finalizeMeshPass();

	buildBVHSmall();

	unlink(getTriangleIdxFileName().c_str());

	end.set();
	
	float elapsedHours;
	int elapsedMinOfHour;
	double elapsedMinOfHourFrac = modf((end - start)/(float)(60*60), &elapsedHours);
	elapsedMinOfHour = elapsedMinOfHourFrac * 60.0;
	
	cout << "OOC tree build ended, time = " << (end - start) << "s (" << (int)elapsedHours << " h, " << elapsedMinOfHour << " min)" << endl;

	return true;
}

void OutOfCoreTree::initSmall() {
	// init bounding box
	grid.clear();
	// count number of vertices
	numVertices = 0;
	numFaces = 0;

	m_bb_min.e[0] = FLT_MAX;
	m_bb_min.e[1] = FLT_MAX;
	m_bb_min.e[2] = FLT_MAX;
	m_bb_max.e[0] = -FLT_MAX;
	m_bb_max.e[1] = -FLT_MAX;
	m_bb_max.e[2] = -FLT_MAX;

	hasVertexColors = false;
	hasVertexNormals = false;
	hasVertexTextures = false;

	m_pVertices = new BufferedOutputs<Vertex>(getVertexFileName(), 100000);
	m_pVertices->clear();

	// init out file for all vertices' colors (temporary)
	m_pColorList = new BufferedOutputs<rgb>(getColorListName(), 10000);
	m_pColorList->clear();

	m_numVertexList.push_back(0);
			
}

void OutOfCoreTree::writeModelInfo() {
	FILE *firstPassFile = fopen(getSceneInfoName().c_str(), "wb");
	fwrite(&numVertices, sizeof(int), 1, firstPassFile);
	fwrite(&numFaces, sizeof(int), 1, firstPassFile);
	fwrite(&grid, sizeof(Grid), 1, firstPassFile);
	fclose(firstPassFile);
}

bool OutOfCoreTree::firstVertexPassSmall(const char *fileName, float *mat) {
	// build reader:
	reader = new PlyReader(fileName);

	hasVertexColors |= reader->hasColor();
	hasVertexNormals |= reader->hasVertNormal();
	hasVertexTextures |= reader->hasVertTexture();

	//float mat[] = {1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1};

	// read all vertices
	while (reader->haveVertex()) {
		// update bounding box
		++numVertices;
		if (numVertices%1000000 == 0) {
			cout << " > " << numVertices << " vertices read" << endl;
		}

		Vertex curVert = reader->readVertex();
		if(mat)
		{
			Vector3 &pv = curVert.v;
			Vector3 pt;
			pt.e[0] = pv.e[0]*mat[0*4+0] + pv.e[1]*mat[0*4+1] + pv.e[2]*mat[0*4+2] + mat[0*4+3];
			pt.e[1] = pv.e[0]*mat[1*4+0] + pv.e[1]*mat[1*4+1] + pv.e[2]*mat[1*4+2] + mat[1*4+3];
			pt.e[2] = pv.e[0]*mat[2*4+0] + pv.e[1]*mat[2*4+1] + pv.e[2]*mat[2*4+2] + mat[2*4+3];

			pv.e[0] = pt.e[0];
			pv.e[1] = pt.e[1];
			pv.e[2] = pt.e[2];

			Vector3 &pn = curVert.n;
			pt.e[0] = pn.e[0]*mat[0*4+0] + pn.e[1]*mat[0*4+1] + pn.e[2]*mat[0*4+2];
			pt.e[1] = pn.e[0]*mat[1*4+0] + pn.e[1]*mat[1*4+1] + pn.e[2]*mat[1*4+2];
			pt.e[2] = pn.e[0]*mat[2*4+0] + pn.e[1]*mat[2*4+1] + pn.e[2]*mat[2*4+2];

			pn.e[0] = pt.e[0];
			pn.e[1] = pt.e[1];
			pn.e[2] = pt.e[2];
			pn.makeUnitVector();
		}
		updateBB(m_bb_min, m_bb_max, curVert.v);
	}

	m_numVertexList.push_back(numVertices);

#	ifdef USE_DOE
	g_doeColoring.zMidForDOE = m_bb_min.e[2];
	g_doeColoring.maxDistForDOE = m_bb_max.e[2] - m_bb_min.e[2];
	g_doeColoring.scaleForDOE = 1.0f/(COLORS_IN_MAP-1);
#	endif

	cout << numVertices << " Vertices read.\n";

	// read all faces
	std::vector<int> vIdxList;
	while (reader->haveFace()) {
		// update bounding box
		++numFaces;
		if (numFaces%1000000 == 0) {
			cout << " > " << numFaces << " tris read " << endl;
		}		

		reader->readFace(vIdxList);
	}

	cout << numFaces << " Tris read.\n";

	delete reader;

	return true;
}

bool OutOfCoreTree::secondVertexPassSmall(const char *fileName, float *mat) {
	reader = new PlyReader(fileName);

	//float mat[] = {1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1};

	// read all vertices:
	// 
	#ifdef _USE_TRI_MATERIALS
	rgb tempColor;
	if (reader->hasColor()) {
		while (reader->haveVertex()) {					
			Vertex curVert = reader->readVertexWithColor(tempColor);

			if(mat)
			{
				Vector3 &pv = curVert.v;
				Vector3 pt;
				pt.e[0] = pv.e[0]*mat[0*4+0] + pv.e[1]*mat[0*4+1] + pv.e[2]*mat[0*4+2] + mat[0*4+3];
				pt.e[1] = pv.e[0]*mat[1*4+0] + pv.e[1]*mat[1*4+1] + pv.e[2]*mat[1*4+2] + mat[1*4+3];
				pt.e[2] = pv.e[0]*mat[2*4+0] + pv.e[1]*mat[2*4+1] + pv.e[2]*mat[2*4+2] + mat[2*4+3];

				pv.e[0] = pt.e[0];
				pv.e[1] = pt.e[1];
				pv.e[2] = pt.e[2];

				Vector3 &pn = curVert.n;
				pt.e[0] = pn.e[0]*mat[0*4+0] + pn.e[1]*mat[0*4+1] + pn.e[2]*mat[0*4+2];
				pt.e[1] = pn.e[0]*mat[1*4+0] + pn.e[1]*mat[1*4+1] + pn.e[2]*mat[1*4+2];
				pt.e[2] = pn.e[0]*mat[2*4+0] + pn.e[1]*mat[2*4+1] + pn.e[2]*mat[2*4+2];

				pn.e[0] = pt.e[0];
				pn.e[1] = pt.e[1];
				pn.e[2] = pt.e[2];
				pn.makeUnitVector();
			}

			// write to file:
			m_pVertices->appendElement(curVert);
			m_pColorList->appendElement(tempColor);
		}			
	}
	else
	#endif

	while (reader->haveVertex()) {	
		Vertex curVert = reader->readVertex();

		#ifdef _USE_TRI_MATERIALS
#		ifdef USE_DOE
		rgb col;
		g_doeColoring.getDOEColor(curVert.v.e[2], col);
		m_pColorList->appendElement(col);
#		else
		// insert dummy color:
		m_pColorList->appendElement(rgb(0.7f, 0.7f, 0.7f));
#		endif
		#endif

		if(mat)
		{
			Vector3 &pv = curVert.v;
			Vector3 pt;
			pt.e[0] = pv.e[0]*mat[0*4+0] + pv.e[1]*mat[0*4+1] + pv.e[2]*mat[0*4+2] + mat[0*4+3];
			pt.e[1] = pv.e[0]*mat[1*4+0] + pv.e[1]*mat[1*4+1] + pv.e[2]*mat[1*4+2] + mat[1*4+3];
			pt.e[2] = pv.e[0]*mat[2*4+0] + pv.e[1]*mat[2*4+1] + pv.e[2]*mat[2*4+2] + mat[2*4+3];

			pv.e[0] = pt.e[0];
			pv.e[1] = pt.e[1];
			pv.e[2] = pt.e[2];

			Vector3 &pn = curVert.n;
			pt.e[0] = pn.e[0]*mat[0*4+0] + pn.e[1]*mat[0*4+1] + pn.e[2]*mat[0*4+2];
			pt.e[1] = pn.e[0]*mat[1*4+0] + pn.e[1]*mat[1*4+1] + pn.e[2]*mat[1*4+2];
			pt.e[2] = pn.e[0]*mat[2*4+0] + pn.e[1]*mat[2*4+1] + pn.e[2]*mat[2*4+2];

			pn.e[0] = pt.e[0];
			pn.e[1] = pt.e[1];
			pn.e[2] = pt.e[2];
			pn.makeUnitVector();
		}

		// write to file:
		m_pVertices->appendElement(curVert);
	}
	delete reader;

	return true;
}

bool OutOfCoreTree::bridge2to3()
{
	// close vertex files...
	delete m_pVertices;
	delete m_pColorList;

	// ...and open again for access
	SYSTEM_INFO systemInfo;
	BY_HANDLE_FILE_INFORMATION fileInfo;
	GetSystemInfo(&systemInfo);

	m_hFileVertex = CreateFile(getVertexFileName().c_str(), GENERIC_WRITE | GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, NULL);
	GetFileInformationByHandle(m_hFileVertex, &fileInfo);
	m_hMappingVertex = CreateFileMapping(m_hFileVertex, NULL, PAGE_READWRITE, fileInfo.nFileSizeHigh, fileInfo.nFileSizeLow, NULL);
	m_pVertexFile = (Vertex *)MapViewOfFile(m_hMappingVertex, FILE_MAP_WRITE, 0, 0, 0);

	m_hFileColor = CreateFile(getColorListName().c_str(), GENERIC_WRITE | GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, NULL);
	GetFileInformationByHandle(m_hFileColor, &fileInfo);
	m_hMappingColor = CreateFileMapping(m_hFileColor, NULL, PAGE_READWRITE, fileInfo.nFileSizeHigh, fileInfo.nFileSizeLow, NULL);
	m_pColorFile = (rgb *)MapViewOfFile(m_hMappingColor, FILE_MAP_WRITE, 0, 0, 0);

	// init out file for all tris	
	m_pTris = new BufferedOutputs<Triangle>(getTriangleFileName(), 100000);
	m_pTris->clear();

	// init out file for all materials
	m_pOutputs_mat = new BufferedOutputs<MaterialDiffuse>(getMaterialListName().c_str(), 1000);
	m_pOutputs_mat->clear();

	// init out file for all tri indices
	m_pOutputs_idx = new BufferedOutputs<unsigned int>(getTriangleIdxFileName().c_str(), 100000);
	m_pOutputs_idx->clear();

	return true;
}

bool OutOfCoreTree::thirdVertexPassSmall(const char *fileName, int curIter, int mtlMatIndex) {
	reader = new PlyReader(fileName);
	//
	// read all triangles:
	//

	std::vector<int> vIdxList;
	Triangle tri;
	Vector3 p[3];
	static unsigned int triCount = 0;		
	typedef stdext::hash_map<__int64, unsigned int> ColorTable;
	typedef ColorTable::iterator ColorTableIterator;

	unsigned int materialIndex = 0;
	ColorTableIterator colorIter;

	if(useFileMat)
	{
		MaterialDiffuse newMat(rgb(0.7f, 0.7f, 0.7f));
		m_pOutputs_mat->appendElement(newMat);
	}

	// move current position to face region
	while (reader->haveVertex()) reader->readVertex();

	Progression prog("Process triangles" , numFaces, 20);
	while (reader->haveFace()) {	
		prog.step();

		// read vertex indices into vector
		reader->readFace(vIdxList);
		for(int i=0;i<vIdxList.size();i++)
		{
			vIdxList[i] += m_numVertexList[curIter];
		}

		// read in vertices
		p[0] = m_pVertexFile[vIdxList[0]].v;
		p[1] = m_pVertexFile[vIdxList[1]].v;
		p[2] = m_pVertexFile[vIdxList[2]].v;
		
		// write triangle to complete list:		
		tri.n = cross(p[1] - p[0], p[2] - p[0]);
		//tri.n = cross(p[2] - p[0], p[1] - p[0]);
		tri.n.makeUnitVector();

		// sungeui start ------------------------
		// Detect degerated cases

		bool isValid = true;
		if (vIdxList[0] == vIdxList[1] ||
			vIdxList[0] == vIdxList[2] ||
			vIdxList[1] == vIdxList[2] ||
			p[0] == p[1] ||
			p[1] == p[2] ||
			p[2] == p[0] ||
			tri.n.e [0] < -1.f || tri.n.e [0] > 1.f ||
			tri.n.e [1] < -1.f || tri.n.e [1] > 1.f ||
			tri.n.e [2] < -1.f || tri.n.e [2] > 1.f) {
			//printf ("Found degenerated triangle\n");
			//continue;
				isValid = false;
		}
		// sungeui end ---------------------------


		tri.d = dot(p[0], tri.n);

		// find best projection plane (YZ, XZ, XY)
		if (fabs(tri.n[0]) > fabs(tri.n[1]) && fabs(tri.n[0]) > fabs(tri.n[2])) {								
			tri.i1 = 1;
			tri.i2 = 2;
		}
		else if (fabs(tri.n[1]) > fabs(tri.n[2])) {								
			tri.i1 = 0;
			tri.i2 = 2;
		}
		else {								
			tri.i1 = 0;
			tri.i2 = 1;
		}

		int firstIdx;
		float u1list[3];
		u1list[0] = fabs(p[1].e[tri.i1] - p[0].e[tri.i1]);
		u1list[1] = fabs(p[2].e[tri.i1] - p[1].e[tri.i1]);
		u1list[2] = fabs(p[0].e[tri.i1] - p[2].e[tri.i1]);

		if (u1list[0] >= u1list[1] && u1list[0] >= u1list[2])
			firstIdx = 0;
		else if (u1list[1] >= u1list[2])
			firstIdx = 1;
		else
			firstIdx = 2;

		int secondIdx = (firstIdx + 1) % 3;
		int thirdIdx = (firstIdx + 2) % 3;

		// apply coordinate order to tri structure:
		tri.p[0] = vIdxList[firstIdx];
		tri.p[1] = vIdxList[secondIdx];
		tri.p[2] = vIdxList[thirdIdx];		

		// handle materials if necessary:
		if(useFileMat)
		{
			tri.material = (unsigned short)curIter;
		}
		else if(mtlMatIndex >= 0)
		{
			tri.material = mtlMatIndex;
			// for base color
			static bool baseColorAdded = false;
			if(!baseColorAdded)
			{
				MaterialDiffuse newMat(rgb(0.7f, 0.7f, 0.7f));
				m_pOutputs_mat->appendElement(newMat);
				baseColorAdded = true;
			}
		}
		else
		{
			#ifdef _USE_TRI_MATERIALS
			rgb color = m_pColorFile[vIdxList[firstIdx]]; 
#			ifdef USE_DOE
			unsigned int hash = (((unsigned int)(color.r() * 256)) & 0xFF) + 256*(((unsigned int)(color.g() * 256)) & 0xFF) + 256*256*(((unsigned int)(color.b() * 256)) & 0xFF);
#			else
			unsigned int hash = (unsigned int)(color.r() + 256*color.g() + 256*256*color.b());
#			endif

			if ((colorIter = m_usedColors.find(hash)) != m_usedColors.end()) {
				tri.material = colorIter->second;
			}
			else {
				MaterialDiffuse newMat(color);					
				m_pOutputs_mat->appendElement(newMat);				
				tri.material = materialIndex;
				m_usedColors[hash] = tri.material;
				materialIndex++;
			}
			#endif
		}

		if(!hasVertexNormals && isValid)
		{
			// calculate vertex normals
			Vertex &v0 = m_pVertexFile[vIdxList[0]];
			Vertex &v1 = m_pVertexFile[vIdxList[1]];
			Vertex &v2 = m_pVertexFile[vIdxList[2]];
			float tArea = triangleArea(v0.v, v1.v, v2.v);
			if(tArea > 0)
			{
				v0.n += tArea*tri.n;
				v1.n += tArea*tri.n;
				v2.n += tArea*tri.n;
			}
		}
		m_pTris->appendElement(tri);
		m_pOutputs_idx->appendElement(triCount);

		triCount++;
	}

	delete reader;

	return true;
}

void OutOfCoreTree::finalizeMeshPass()
{
	if(!hasVertexNormals)
	{
		// initialize vertex normals
		for(int i=0;i<numVertices;i++)
		{
			Vertex &v = m_pVertexFile[i];
			v.n.makeUnitVector();
		}
	}

	m_pOutputs_mat->flush();
	m_pOutputs_idx->flush();
	delete m_pOutputs_mat;
	delete m_pOutputs_idx;
	delete m_pTris;		

	UnmapViewOfFile(m_pVertexFile);
	UnmapViewOfFile(m_pColorFile);
	CloseHandle (m_hMappingVertex);
	CloseHandle (m_hMappingColor);
	CloseHandle (m_hFileVertex);
	CloseHandle (m_hFileColor);
	unlink(getColorListName().c_str());

	reader = 0;
}

bool OutOfCoreTree::buildBVHSmall() {
	TimerValue start, end;
	unsigned int boxesBuilt = 0;
	cout << "Building BVH" << endl;

	start.set();

	Vector3 min_limit(FLT_MAX, FLT_MAX, FLT_MAX), max_limit(-FLT_MAX, -FLT_MAX, -FLT_MAX);

	//
	// for all voxels: build individual BVH
	// (in parallel if configured and compiler supports it)
	//
	LogManager *log = LogManager::getSingletonPtr();

	BVH *tree;				
	Triangle *triangleFile;
	unsigned int *triangleIndexFile;
	Vertex *vertexFile;

	// open the vertex file in ooc mode:
	//
	HANDLE hFileTriangle, hFileTriangleIndex, hFileVertex;
	HANDLE hMappingTriangle, hMappingTriangleIndex, hMappingVertex;
	SYSTEM_INFO systemInfo;
	BY_HANDLE_FILE_INFORMATION fileInfo;
	GetSystemInfo(&systemInfo);

	hFileTriangle = CreateFile(getTriangleFileName().c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, NULL);
	GetFileInformationByHandle(hFileTriangle, &fileInfo);
	hMappingTriangle = CreateFileMapping(hFileTriangle, NULL, PAGE_READONLY, fileInfo.nFileSizeHigh, fileInfo.nFileSizeLow, NULL);
	triangleFile = (Triangle *)MapViewOfFile(hMappingTriangle, FILE_MAP_READ, 0, 0, 0);
	if(!triangleFile)
	{
		printf("Mapping falied! [triangleFile]\n");
		exit(-1);
	}

	hFileTriangleIndex = CreateFile(getTriangleIdxFileName().c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, NULL);
	GetFileInformationByHandle(hFileTriangleIndex, &fileInfo);
	hMappingTriangleIndex = CreateFileMapping(hFileTriangleIndex, NULL, PAGE_READONLY, fileInfo.nFileSizeHigh, fileInfo.nFileSizeLow, NULL);
	triangleIndexFile = (unsigned int *)MapViewOfFile(hMappingTriangleIndex, FILE_MAP_READ, 0, 0, 0);
	if(!triangleIndexFile)
	{
		printf("Mapping falied! [triangleIndexFile]\n");
		exit(-1);
	}

	hFileVertex = CreateFile(getVertexFileName().c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, NULL);
	DWORD err = GetLastError();
	GetFileInformationByHandle(hFileVertex, &fileInfo);
	hMappingVertex = CreateFileMapping(hFileVertex, NULL, PAGE_READONLY, fileInfo.nFileSizeHigh, fileInfo.nFileSizeLow, NULL);
	vertexFile = (Vertex *)MapViewOfFile(hMappingVertex, FILE_MAP_READ, 0, 0, 0);
	if(!vertexFile)
	{
		printf("Mapping falied! [vertexFile]\n");
		exit(-1);
	}

	tree = new BVH(triangleFile, triangleIndexFile, numFaces, vertexFile, m_bb_min, m_bb_max);
	tree->buildTreeSAH();		
	//tree->saveToFile(getBVHFileName().c_str());

	char fileNameHeader[MAX_PATH], fileNameTree[MAX_PATH], fileNameIndex[MAX_PATH];
	sprintf(fileNameHeader, "%s", getBVHFileName().c_str());
	sprintf(fileNameTree, "%s.node", getBVHFileName().c_str());
	sprintf(fileNameIndex, "%s.idx", getBVHFileName().c_str());	

	tree->saveToFile(fileNameHeader, fileNameTree, fileNameIndex);
	tree->printTree(false);


	UnmapViewOfFile(triangleFile);
	UnmapViewOfFile(triangleIndexFile);
	UnmapViewOfFile(vertexFile);
	CloseHandle (hMappingTriangle);
	CloseHandle (hMappingTriangleIndex);
	CloseHandle (hMappingVertex);
	CloseHandle (hFileTriangle);
	CloseHandle (hFileTriangleIndex);
	CloseHandle (hFileVertex);
	delete tree;

	end.set();

	cout << "BVH build ended, time = " << (end - start) << "s" << endl;
	
	return true;
}

string Trim(const string& s)
{

	unsigned int f,e ;

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

bool OutOfCoreTree::buildSmallMulti(const char *outputPath, const char *fileListName, bool useModelTransform, bool useFileMat, const char *mtlFileName)
{
	TimerValue start, end;
	//OptionManager *opt = OptionManager::getSingletonPtr();

	start.set();

	this->useFileMat = useFileMat;

	//const char *baseDirName = opt->getOption("global", "scenePath", "");
	char tempFileName[MAX_PATH];
	int pos = 0;
	for(int i=strlen(fileListName)-1;i>=0;i--)
	{
		if(fileListName[i] == '/' || fileListName[i] == '\\')
		{
			pos = i+1;
			break;
		}
	}
	strcpy(tempFileName, &fileListName[pos]);
	sprintf(outDirName, "%s\\%s.ooc", outputPath, tempFileName);
	mkdir(outDirName);

	vector<char *> fileList;
	vector<float *> matList;
	vector<char *> mtlList;
	vector<char *> textureFileList;

	if(mtlFileName)
	{
		// get material and texture file name list
		FILE *fpMtl = fopen(mtlFileName, "r");
		char currentLine[500];
		while(fgets(currentLine, 499, fpMtl))
		{
			if(strstr(currentLine, "newmtl"))
			{
				string curMatName = currentLine+7;
				curMatName = Trim(curMatName);

				char *mtlName = new char[strlen(curMatName.c_str())+1];
				strcpy(mtlName, curMatName.c_str());
				mtlList.push_back(mtlName);
			}

			if(strstr(currentLine, "map"))
			{
				char *fileName = strtok(currentLine, " \r\n");
				fileName = strtok(NULL, " \r\n");
				char *textureFileName = new char[strlen(fileName)+1];
				strcpy(textureFileName, fileName);
				textureFileList.push_back(textureFileName);
			}
		}
		fclose(fpMtl);
	}

	char currentLine[500];

	FILE *fpList = fopen(fileListName, "r");

	while (fgets(currentLine, 499, fpList)) 
	{
		char *fileName = new char[strlen(currentLine)+1];
		float *mat = new float[16];
		memset(mat, 0, sizeof(float)*16);
		mat[0*4+0] = mat[1*4+1] = mat[2*4+2] = mat[3*4+3] = 1.0f;
		// not yet finish the model transform... may be later
		if(useModelTransform)
		{
			sscanf(currentLine, "%s %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f", fileName,
						&mat[0*4+0], &mat[1*4+0], &mat[2*4+0], &mat[3*4+0], 
						&mat[0*4+1], &mat[1*4+1], &mat[2*4+1], &mat[3*4+1], 
						&mat[0*4+2], &mat[1*4+2], &mat[2*4+2], &mat[3*4+2], 
						&mat[0*4+3], &mat[1*4+3], &mat[2*4+3], &mat[3*4+3]
						);
			matList.push_back(mat);
		}
		else
		{
			sscanf(currentLine, "%s", fileName);
		}
		fileList.push_back(fileName);
	}

	fclose(fpList);

	initSmall();

	for(int i=0;i<fileList.size();i++)
	{
		firstVertexPassSmall(fileList[i], useModelTransform ? matList[i] : 0);
	}

	writeModelInfo();

	for(int i=0;i<fileList.size();i++)
	{
		secondVertexPassSmall(fileList[i], useModelTransform ? matList[i] : 0);
	}

	bridge2to3();

	for(int i=0;i<fileList.size();i++)
	{
		// read mtl name from ply comment
		char mtlName[MAX_PATH];
		int matIndex = -1;
		if(mtlList.size() > 0)
		{
			matIndex = 0;
			FILE *fp = fopen(fileList[i], "r");
			while (fgets(currentLine, 499, fp)) 
			{
				if(strstr(currentLine, "end_header")) break;
				if(strstr(currentLine, "used material"))
				{
					sscanf(currentLine, "comment used material = %s", mtlName);
					for(int j=0;j<mtlList.size();j++)
					{
						if(!strcmp(mtlList[j], mtlName)) 
						{
							matIndex = j;
							break;
						}
					}
					break;
				}
			}
			fclose(fp);
		}

		thirdVertexPassSmall(fileList[i], i, matIndex);
	}

	finalizeMeshPass();

	buildBVHSmall();

	unlink(getTriangleIdxFileName().c_str());

	// copy material file
	if(mtlFileName)
	{
		char fileName[256];
		sprintf(fileName, "%s\\material.mtl", outDirName);
		copyFile(mtlFileName, fileName);
	}

	// copy textures
	char textureFolderName[256];
	if(textureFileList.size() > 0)
	{
		sprintf(textureFolderName, "%s\\texture", outDirName);
		mkdir(textureFolderName);
	}
	for(size_t i=0;i<textureFileList.size();i++)
	{
		char srcFileName[256];
		char dstFileName[256];

		strncpy_s(srcFileName, 256, mtlFileName, strlen(mtlFileName));
		for(int j=(int)strlen(srcFileName)-1;j>=0;j--)
		{
			if(srcFileName[j] == '/' || srcFileName[j] == '\\')
			{
				memcpy_s(&srcFileName[j+1], 256, textureFileList[i], strlen(textureFileList[i])+1);
				break;
			}
			else if(j == 0)
			{
				memcpy_s(srcFileName, 256, textureFileList[i], strlen(textureFileList[i])+1);
			}
		}

		sprintf(dstFileName, "%s\\%s", textureFolderName, textureFileList[i]);
		copyFile(srcFileName, dstFileName);
	}


	for(size_t i=0;i<fileList.size();i++) delete[] fileList[i];
	for(size_t i=0;i<matList.size();i++) delete[] matList[i];
	for(size_t i=0;i<mtlList.size();i++) delete[] mtlList[i];
	for(size_t i=0;i<textureFileList.size();i++) delete[] textureFileList[i];
	
	end.set();
	
	float elapsedHours;
	int elapsedMinOfHour;
	double elapsedMinOfHourFrac = modf((end - start)/(float)(60*60), &elapsedHours);
	elapsedMinOfHour = elapsedMinOfHourFrac * 60.0;
	
	cout << "OOC tree build ended, time = " << (end - start) << "s (" << (int)elapsedHours << " h, " << elapsedMinOfHour << " min)" << endl;

	return true;
}