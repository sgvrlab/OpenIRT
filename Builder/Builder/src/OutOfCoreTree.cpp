//#include "kDTree.h"
#include "App.h"
#include "OutOfCoreTree.h"

#define _CRT_SECURE_NO_DEPRECATE
#pragma warning (disable: 4996)
#include <math.h>
#include <direct.h>

#include "CashedBoxFiles.h"
#include "Vertex.h"
#include "Triangle.h"
#include "helpers.h"

#if HIERARCHY_TYPE == TYPE_BVH
#include "VoxelBVH.h"
//#include "BVHCompression.h"
//#include "BVHRefine.h"
#endif
#include "OptionManager.h"
#include "Progression.h"
#include "Materials.h"

//#define USE_CUSTOM_MATRIX

OutOfCoreTree::OutOfCoreTree() {
	reader = 0;
	fileName[0] = 0;
	numFaces = 0;
	numVertices = 0;
	useFileMat = false;
}

OutOfCoreTree::~OutOfCoreTree() {

	if (reader)
		delete reader;
}

#if HIERARCHY_TYPE == TYPE_BVH 
bool OutOfCoreTree::build(const char *outputPath, const char *fileName, bool useModelTransform) {
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

	// build reader:
	reader = new PlyReader(curFileName);

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

	//
	// first pass:
	// find number of vertices/tris and bounding-box:
	//
	firstVertexPass();

	cout << "Model dimensions: " << grid.getExtent() << endl;
	cout << "BB: min: " << grid.p_min << ", max: " << grid.p_max << endl;

	// calculate number of voxels
	calculateVoxelResolution();
	
	cout << "Using " << grid.getSize() << " Voxels." << endl;

	//
	// second pass: sort vertices & triangles to voxels
	//
	secondVertexPass();

	//
	// for each voxel: build BVH:
	//
	buildVoxelBVH();

	//
	// build high-level BVH from voxels:
	//
	buildHighLevelBVH();

	delete voxelHashTable;

	end.set();
	
	float elapsedHours;
	int elapsedMinOfHour;
	double elapsedMinOfHourFrac = modf((end - start)/(float)(60*60), &elapsedHours);
	elapsedMinOfHour = elapsedMinOfHourFrac * 60.0;
	
	cout << "OOC tree build ended, time = " << (end - start) << "s (" << (int)elapsedHours << " h, " << elapsedMinOfHour << " min)" << endl;


	return true;
}
#endif

bool OutOfCoreTree::firstVertexPass() {

	if (fileExists(getSceneInfoName())) {
		cout << "Skipping 1st pass, already generated...\n";
		
		FILE *firstPassFile = fopen(getSceneInfoName().c_str(), "rb");
		fread(&numVertices, sizeof(int), 1, firstPassFile);
		fread(&numFaces, sizeof(int), 1, firstPassFile);
		fread(&grid, sizeof(Grid), 1, firstPassFile);
		fclose(firstPassFile);

		cout << numVertices << " Vertices read.\n";
		cout << numFaces << " Tris read.\n";
	}
	else {
	
		// init reader
		reader->restart();
		// init bounding box
		grid.clear();
		// count number of vertices
		numVertices = 0;
		numFaces = 0;

		//
		///
		//

		/*
		FILE *fp = fopen("david_1_50.ply", "w");
		fprintf(fp, "ply\n");
		fprintf(fp, "format ascii 1.0\n");
		fprintf(fp, "element vertex 4129614\n");
		fprintf(fp, "property float x\n");
		fprintf(fp, "property float y\n");
		fprintf(fp, "property float z\n");
		fprintf(fp, "element face 8254150\n");
		fprintf(fp, "property list uint8 int32 vertex_indices\n");
		fprintf(fp, "end_header\n");
		while(reader->haveVertex())
		{
			Vector3 v = reader->readVertex().v;
			fprintf(fp, "%f %f %f\n", v.e[0]/50.0f, v.e[1]/50.0f, v.e[2]/50.0f);
		}
		std::vector<int> IdxList;
		while(reader->haveFace())
		{
			reader->readFace(IdxList);
			fprintf(fp, "3 %d %d %d\n", IdxList[0], IdxList[1], IdxList[2]);
		}
		fclose(fp);
		exit(1);
		*/

		//
		///
		//

		// read all vertices
		while (reader->haveVertex()) {
			// update bounding box
			++numVertices;
			if (numVertices%1000000 == 0) {
				cout << " > " << numVertices << " vertices read" << endl;
			}		

			if(useModelTransform)
			{
				Vertex curVert = reader->readVertex();
				Vector3 &pv = curVert.v;
				Vector3 pt;
				pt.e[0] = pv.e[0]*mat[0*4+0] + pv.e[1]*mat[0*4+1] + pv.e[2]*mat[0*4+2] + mat[0*4+3];
				pt.e[1] = pv.e[0]*mat[1*4+0] + pv.e[1]*mat[1*4+1] + pv.e[2]*mat[1*4+2] + mat[1*4+3];
				pt.e[2] = pv.e[0]*mat[2*4+0] + pv.e[1]*mat[2*4+1] + pv.e[2]*mat[2*4+2] + mat[2*4+3];

				grid.addPoint(pt);
			}
			else
				grid.addPoint(reader->readVertex().v);
		}

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

		FILE *firstPassFile = fopen(getSceneInfoName().c_str(), "wb");
		fwrite(&numVertices, sizeof(int), 1, firstPassFile);
		fwrite(&numFaces, sizeof(int), 1, firstPassFile);
		fwrite(&grid, sizeof(Grid), 1, firstPassFile);
		fclose(firstPassFile);
	}

	return numVertices > 0 && numFaces > 0;
}

void OutOfCoreTree::calculateVoxelResolution() {
	OptionManager *opt = OptionManager::getSingletonPtr();
	float maxExtent = grid.getMaxExtent();
	unsigned int targetVoxelSize = 100000;
	unsigned int nrCountDivisions;
	float avgVoxelFaces = numFaces / (float)targetVoxelSize;
	
	nrCountDivisions = (unsigned int)ceil(2.0 * pow(avgVoxelFaces, 1.0f/3.0f));
//	if (nrCountDivisions == 1)
//		nrCountDivisions = 2;

	grid.setResolution(nrCountDivisions);
}


bool OutOfCoreTree::secondVertexPass() {
	VoxelHashTableIterator it;
	voxelHashTable = new VoxelHashTable;
	MaterialDiffuse *plyMat = new MaterialDiffuse(rgb(0.7f,0.7f,0.7f));
	numUsedVoxels = 0;

	bool hasVertexColors = false;	// initialize
	bool hasVertexNormals = false;
	bool hasVertexTextures = false;

	// Test if this pass was already completed in an earlier run:
	// then just read in the already generated data.
	if (fileExists(getVertexFileName()) && fileExists(getTriangleFileName())) {
		cout << "Skipping 2nd pass, already generated...\n";
		 // go through all voxels:
		for (unsigned int boxNr = 0; boxNr < grid.getSize(); boxNr++) {			
			if (fileExists(getVTriangleFileName(boxNr))) {
				// # tris = filesize / triangle size
				unsigned int voxelTris = fileSize(getVTriangleFileName(boxNr)) / sizeof(Triangle);
				voxelHashTable->insert(std::pair<int,int>(boxNr,voxelTris));						
				numUsedVoxels++;
			}	
			else
			{
				printf("hello \n");
			}
		}		
	}
	else { // start construction:
		reader->restart();
		if (fileExists(getVertexFileName()) && fileExists(getVertexIndexFileName()) && fileExists(getColorListName())) {
			// skip first half of second pass...
			reader->skipVertices();
		}
		else {

			// sungeui's note: there is no guarantee that this single vertex file supports
			//					cache-coherent access patters.
			//					Need to use small vertex file for each voxel.
			// init out file for all vertices	
			//BufferedOutputs<Vector3> *pVertices = new BufferedOutputs<Vector3>(getVertexFileName(), 100000);
			BufferedOutputs<Vertex> *pVertices = new BufferedOutputs<Vertex>(getVertexFileName(), 100000);
			pVertices->clear();

			// init out file for all vertices' voxel IDs (temporary)	
			BufferedOutputs<unsigned int> *pVertexIndexMap = new BufferedOutputs<unsigned int>(getVertexIndexFileName(), 100000);
			pVertexIndexMap->clear();

			// init out file for all vertices' colors (temporary)
			BufferedOutputs<rgba> *pColorList = new BufferedOutputs<rgba>(getColorListName(), 10000);
			pColorList->clear();
					
			// init reader
			
			Progression prog1("read verts" , numVertices, 20);

			hasVertexColors = reader->hasColor();
			hasVertexNormals = reader->hasVertNormal();
			hasVertexTextures = reader->hasVertTexture();

//			float mat[] = {1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1};

			// read all vertices:
			// 
			#ifdef _USE_TRI_MATERIALS
			rgba tempColor;
			int tempMat, tempFile;
			if (reader->hasColor()) {
				while (reader->haveVertex()) {					
					//Vertex curVert = reader->readVertexWithColor(tempColor);
					Vertex curVert = reader->readVertexWithColorMatFile(tempColor, tempMat, tempFile);

					if(useModelTransform)
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
					tempColor.alpha = (float)tempFile;
					
					// get voxel for point:
					unsigned int idx = grid.getCellIndex(curVert.v);

					// write to file:
					//pVertices->appendElement(curVert);
					pVertices->appendElement(curVert);
					pVertexIndexMap->appendElement(idx);
					pColorList->appendElement(tempColor);
					prog1.step();
				}			
			}
			else
			#endif

			while (reader->haveVertex()) {	
				Vertex curVert = reader->readVertex();
				if(useModelTransform)
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

				#ifdef _USE_TRI_MATERIALS
				// insert dummy color:
				pColorList->appendElement(rgba(0.7f, 0.7f, 0.7f, 1.0f));
				#endif

				// get voxel for point:
				unsigned int idx = grid.getCellIndex(curVert.v);

				// write to file:
				pVertices->appendElement(curVert);
				//pVertices->appendElement(curVert);
				pVertexIndexMap->appendElement(idx);
				prog1.step();
			}

			// close vertex files...
			delete pVertexIndexMap;
			delete pVertices;
			delete pColorList;
		}

		// ...and open again for cached access
		CashedBoxSingleFile<unsigned int> *pVertexIndexCache = new CashedBoxSingleFile<unsigned int>(getVertexIndexFileName(), grid.getSize(), 10, false);
		CashedBoxSingleFile<unsigned int> &voxelIndexCache = *pVertexIndexCache;
		//CashedBoxSingleFile<Vector3> *pVertexCache = new CashedBoxSingleFile<Vector3>(getVertexFileName(), grid.getSize(), 10, false);
		//CashedBoxSingleFile<Vector3> &vertexCache = *pVertexCache;
		CashedBoxSingleFile<Vertex> *pVertexCache = new CashedBoxSingleFile<Vertex>(getVertexFileName(), grid.getSize(), 10, false);
		CashedBoxSingleFile<Vertex> &vertexCache = *pVertexCache;
		CashedBoxSingleFile<rgba> *pColorCache = new CashedBoxSingleFile<rgba>(getColorListName(), grid.getSize(), 10, false);
		CashedBoxSingleFile<rgba> &colorCache = *pColorCache;

		// init out file for all tris	
		BufferedOutputs<Triangle> *pTris = new BufferedOutputs<Triangle>(getTriangleFileName(), 100000);
		pTris->clear();

		// init out file for all materials
		//BufferedOutputs<MaterialDiffuse> *outputs_mat = new BufferedOutputs<MaterialDiffuse>(getMaterialListName().c_str(), 1000);
		m_outputs_mat = new BufferedOutputs<MaterialDiffuse>(getMaterialListName().c_str(), 1000);
		m_outputs_mat->clear();
		
		// init out files for single voxels	
		BufferedOutputs<Triangle> *outputs = new BufferedOutputs<Triangle>(getVTriangleFileName().c_str(), grid.getSize(), 5000);
		outputs->clear();

		// init out file for all tri indices
		BufferedOutputs<unsigned int> *outputs_idx = new BufferedOutputs<unsigned int>(getTriangleIdxFileName().c_str(), grid.getSize(), 5000);
		outputs_idx->clear();
	
		//
		// read all triangles:
		//

		std::vector<int> vIdxList;
		Triangle tri;
		Vector3 p[3];
		unsigned int voxel[3];
		unsigned int triCount = 0;		
		typedef stdext::hash_map<__int64, unsigned int> ColorTable;
		typedef ColorTable::iterator ColorTableIterator;

		#ifdef NORMAL_INCLUDED
		if(!hasVertexNormals)
		{
			// initialize vertex normals
			for(int i=0;i<vertexCache.nrElements;i++)
			{
				Vertex &v = vertexCache.getRef(i);
				v.n.e[0] = 0;
				v.n.e[1] = 0;
				v.n.e[2] = 0;
			}
		}
		#endif
		// ColorTable usedColors;
		//unsigned int materialIndex = 0;
		m_materialIndex = 0;
		ColorTableIterator colorIter;
		Progression prog("make voxels" , numFaces, 20);
		while (reader->haveFace()) {	
			prog.step();

			// read vertex indices into vector
			reader->readFace(vIdxList);

			// read in vertices
			p[0] = vertexCache[vIdxList[0]].v;
			p[1] = vertexCache[vIdxList[1]].v;
			p[2] = vertexCache[vIdxList[2]].v;
			
			// write triangle to complete list:		
			tri.n = cross(p[1] - p[0], p[2] - p[0]);
			//tri.n = cross(p[2] - p[0], p[1] - p[0]);
			tri.n.makeUnitVector();
 
			// sungeui start ------------------------
			// Detect degerated cases

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
			#ifdef _USE_TRI_MATERIALS
			rgba color = colorCache[vIdxList[firstIdx]]; 
			__int64 hash = (__int64)(color.r() + 256*color.g() + 256*256*color.b()) + ((__int64)color.alpha)*256*256*256;

			if ((colorIter = m_usedColors.find(hash)) != m_usedColors.end()) {
				tri.material = colorIter->second;
			}
			else {
				MaterialDiffuse newMat(color);					
				m_outputs_mat->appendElement(newMat);				
				tri.material = m_materialIndex;
				m_usedColors[hash] = tri.material;
				m_materialIndex++;
			}

			#endif

			#ifdef NORMAL_INCLUDED
			if(!hasVertexNormals)
			{
				// calculate vertex normals
				Vertex &v0 = vertexCache.getRef(vIdxList[0]);
				Vertex &v1 = vertexCache.getRef(vIdxList[1]);
				Vertex &v2 = vertexCache.getRef(vIdxList[2]);
				float tArea = triangleArea(v0.v, v1.v, v2.v);
				if(tArea > 0)
				{
					v0.n += tArea*tri.n;
					v1.n += tArea*tri.n;
					v2.n += tArea*tri.n;
				}
			}
			#endif
			pTris->appendElement(tri);

			// insert triangle into each voxel it belongs in:		
			voxel[0] = voxelIndexCache[vIdxList[0]];
			voxel[1] = voxelIndexCache[vIdxList[1]];
			voxel[2] = voxelIndexCache[vIdxList[2]];

			for (int i = 0; i < 3; i++) {
				/// TODO: triangles spanning multiple voxels!

				/*
				// don't insert triangle multiple times
				if (i == 1 && voxel[1] == voxel[0])
					continue;
				else if (i == 2 && (voxel[2] == voxel[0] || voxel[2] == voxel[1]))
					continue;
				*/

				// assign the triangle to the Voxel [0]
				if (i != 0)
					continue;

				outputs->appendElement(voxel[i], tri);
				outputs_idx->appendElement(voxel[i], triCount);

				// add to hash table counter
				it = voxelHashTable->find(voxel[i]);
				if (it != voxelHashTable->end()) {
					(*it).second = (*it).second + 1;
				}
				else {
					voxelHashTable->insert(std::pair<int,int>(voxel[i],1));		
					numUsedVoxels++;
				}
			}

			triCount++;
		}

		outputs->flush();
		outputs_idx->flush();
		m_outputs_mat->flush();
		delete outputs;
		delete outputs_idx;
		delete m_outputs_mat;
		delete pTris;		

		#ifdef NORMAL_INCLUDED
		if(!hasVertexNormals)
		{
			// initialize vertex normals
			for(int i=0;i<vertexCache.nrElements;i++)
			{
				Vertex &v = vertexCache.getRef(i);
				v.n.makeUnitVector();
			}
			pVertexCache->flush();
		}
		#endif

		// delete temporary cache file for vertex->voxel cache
		delete pVertexIndexCache;
		delete pVertexCache;
		delete pColorCache;
		unlink(getVertexIndexFileName().c_str());
		unlink(getColorListName().c_str());
	}
		
	cout << "Voxels used: " << numUsedVoxels << " of " << grid.getSize() << endl;
	unsigned int sumTris = 0;
	for (it = voxelHashTable->begin(); it != voxelHashTable->end(); it++) {
		cout << " > " << it->first << " -> " << it->second << " tris" << endl;
		sumTris += it->second;
	}
	cout << "Tri references: " << sumTris << " (original: " << numFaces << ")" << endl;
	
	return true;
}

#if HIERARCHY_TYPE == TYPE_BVH 
bool OutOfCoreTree::buildVoxelBVH() {
	TimerValue start, end;
	unsigned int numBoxes = grid.getSize();
	unsigned int boxesBuilt = 0;
	cout << "Building BVH for " << numUsedVoxels << " voxels..." << endl;
	cout << "Overall BB: [" << grid.p_min << "] - [" << grid.p_max << "]" << endl;

	start.set();
	voxelMins = new Vector3[numUsedVoxels];
	voxelMaxs = new Vector3[numUsedVoxels];
	Vector3 *voxelMinsTemp = new Vector3[numBoxes];
	Vector3 *voxelMaxsTemp = new Vector3[numBoxes];

	Vector3 min_limit(FLT_MAX, FLT_MAX, FLT_MAX), max_limit(-FLT_MAX, -FLT_MAX, -FLT_MAX);

	for (int i=0;i<numBoxes;i++)
	{
		if(i < numUsedVoxels)
		{
			voxelMins[i] = min_limit;
			voxelMaxs[i] = max_limit;
		}
		voxelMinsTemp[i] = min_limit;
		voxelMaxsTemp[i] = max_limit;
	}

#ifdef NUM_OMP_THREADS
	omp_set_num_threads(NUM_OMP_THREADS);
#endif

	//
	// for all voxels: build individual BVH
	// (in parallel if configured and compiler supports it)
	//
#ifdef _USE_OPENMP
	#pragma omp parallel for schedule(dynamic,1) shared(boxesBuilt, numBoxes)
#endif
	for (int boxNr = 0; boxNr < numBoxes; boxNr++) {		
		
		LogManager *log = LogManager::getSingletonPtr();
		BVH *tree;				
		VoxelHashTableIterator it;		
		//std::hash_map<unsigned int, Vector3> vertexMap;
		stdext::hash_map<unsigned int, Vertex> vertexMap;
		Triangle *triangleCache;
		unsigned int *triangleIndexCache;
		FILE *triangleFile, *triangleIdxFile;
		char output[500];

		// test if voxel is in use:
		it = voxelHashTable->find(boxNr);
		if (it == voxelHashTable->end())
			continue;

#ifdef _USE_OPENMP
#pragma omp critical
		{
			boxesBuilt++;
		}		
#else
		boxesBuilt++;
#endif

		// open the vertex file in ooc mode:
		//
		//CashedBoxSingleFile<Vector3> *pVertexCache = new CashedBoxSingleFile<Vector3>(getVertexFileName(), grid.getSize(), 10, false);
		//CashedBoxSingleFile<Vector3> &vertexCache = *pVertexCache;
		CashedBoxSingleFile<Vertex> *pVertexCache = new CashedBoxSingleFile<Vertex>(getVertexFileName(), grid.getSize(), 10, false);
		CashedBoxSingleFile<Vertex> &vertexCache = *pVertexCache;

		unsigned int numTris = it->second;		
		
		sprintf(output, " - Processor %d voxel %d / %d (%u tris)", omp_get_thread_num(), boxesBuilt, numUsedVoxels, numTris);
		log->logMessage(LOG_INFO, output);		

		// skip this if tree already exists
		FILE *test = fopen(getBVHFileName(boxNr).c_str(), "rb");
		if (test != 0) {
			cout << "   skipping, already built." << endl;
			fclose(test);
			continue;
		}

		// read in file with triangle indices for this voxel
		//cout << "   > caching triangles..." << endl;
		if ((triangleFile = fopen(getVTriangleFileName(boxNr).c_str(), "rb")) == NULL) {
			cout << "ERROR: could not open file " << getVTriangleFileName(boxNr) << " !" << endl;
			continue;
		}

		if ((triangleIdxFile = fopen(getTriangleIdxFileName(boxNr).c_str(), "rb")) == NULL) {
			cout << "ERROR: could not open file " << getTriangleIdxFileName(boxNr) << " !" << endl;
			continue;
		}		

		triangleCache = new Triangle[numTris];
		triangleIndexCache = new unsigned int[numTris];
		size_t ret = fread(triangleCache, sizeof(Triangle), numTris, triangleFile);
		if (ret != numTris) {
			cout << "ERROR: could only read " << ret << " of " << numTris << " triangles from file " << getVTriangleFileName(boxNr) << " !" << endl;			
			continue;
		}
		ret = fread(triangleIndexCache, sizeof(unsigned int), numTris, triangleIdxFile);
		if (ret != numTris) {
			cout << "ERROR: could only read " << ret << " of " << numTris << " triangles from file " << getVTriangleFileName(boxNr) << " !" << endl;			
			continue;
		}

		fclose(triangleIdxFile);
		fclose(triangleFile);

		Vector3 bb_min, bb_max;
		bb_min.e[0] = FLT_MAX;
		bb_min.e[1] = FLT_MAX;
		bb_min.e[2] = FLT_MAX;
		bb_max.e[0] = -FLT_MAX;
		bb_max.e[1] = -FLT_MAX;
		bb_max.e[2] = -FLT_MAX;
		// read in vertices from vertex file and store in hash table for sparse
		// storage. that way, we can still access the vertices via absolute index.
		//cout << "   > caching vertices..." << endl;
		for (unsigned int i = 0; i < numTris; i++) {			
			vertexMap[triangleCache[i].p[0]] = vertexCache[triangleCache[i].p[0]];
			vertexMap[triangleCache[i].p[1]] = vertexCache[triangleCache[i].p[1]];
			vertexMap[triangleCache[i].p[2]] = vertexCache[triangleCache[i].p[2]];
			updateBB(bb_min, bb_max, vertexCache[triangleCache[i].p[0]].v);
			updateBB(bb_min, bb_max, vertexCache[triangleCache[i].p[1]].v);
			updateBB(bb_min, bb_max, vertexCache[triangleCache[i].p[2]].v);
		}

		voxelMinsTemp[boxNr] = bb_min;
		voxelMaxsTemp[boxNr] = bb_max;

		//cout << "   > building tree..." << endl;

		//
		// build tree:
		//
		//grid.getCellBoundingBox(boxNr, bb_min, bb_max);
		//cout << "Voxel BB: [" << bb_min << "] - [" << bb_max << "]" << endl;

		tree = new BVH(triangleCache, triangleIndexCache, numTris, &vertexMap, bb_min, bb_max, boxNr, getBVHTestFileName(boxNr).c_str());
		tree->buildTreeSAH();		
		tree->saveToFile(getBVHFileName(boxNr).c_str());
		tree->printTree(false);
		
		delete pVertexCache;
		delete triangleCache;
		delete triangleIndexCache;
		delete tree;
	}

	end.set();

	// Pack the voxel mins and maxes
	for (int boxTemp = 0, box = 0; boxTemp < numBoxes; boxTemp++)
	{
		if(voxelMinsTemp[boxTemp].e[0] != FLT_MAX && voxelMaxsTemp[boxTemp].e[0] != -FLT_MAX)
		{
			voxelMins[box] = voxelMinsTemp[boxTemp];
			voxelMaxs[box] = voxelMaxsTemp[boxTemp];
			box++;
		}
	}
	delete voxelMinsTemp;
	delete voxelMaxsTemp;

	cout << "BVH build ended, time = " << (end - start) << "s" << endl;
	
	return true;
}

bool OutOfCoreTree::buildHighLevelBVH() {
	TimerValue start, end;
	Voxel *voxellist;
	VoxelBVH *tree;
	VoxelHashTableIterator it;

	cout << "Building high-level BVH for " << numUsedVoxels << " voxels..." << endl;
	cout << "Overall BB: [" << grid.p_min << "] - [" << grid.p_max << "]" << endl;

	start.set();

	//
	// build array of all voxels and number of triangles
	// in them.
	//

	voxellist = new Voxel[numUsedVoxels];
	unsigned int curVoxel = 0;
	int boxNr;
	for (boxNr = 0; boxNr < grid.getSize(); boxNr++) {
		// test if voxel is in use:
		it = voxelHashTable->find(boxNr);
		if (it == voxelHashTable->end())
			continue;

		// enter voxel information:
		voxellist[curVoxel].index = boxNr;
		voxellist[curVoxel].numTris = it->second;
		grid.getCellBoundingBox(boxNr, voxellist[curVoxel].min, voxellist[curVoxel].max);
		curVoxel++;
	}	

	// build tree from voxels
	tree = new VoxelBVH(voxellist, numUsedVoxels, grid.p_min, grid.p_max, voxelMins, voxelMaxs);
	tree->buildTree();
	//tree->printTree(true);
	tree->printTree(false);

	cout << "Building final tree by including voxel BVH..." << endl;

	// make real tree from the high-level structure by including 
	// all the individual BVH from the voxels:
	tree->saveToFile(getBVHFileName().c_str());

	end.set();
	cout << "High-Level tree build ended, time = " << (end - start) << "s" << endl;

	delete tree;
	delete voxellist;
	delete voxelMins;
	delete voxelMaxs;
	
	/*
/// DEBUG: output BVH:
	std::hash_map<unsigned int, Vertex> vertexMap;
	Triangle *triangleCache = 0;
	unsigned int *idxCache = 0; 
	BVH *testTree = new BVH(triangleCache, idxCache, 0, &vertexMap, grid.p_min, grid.p_max);
	testTree->loadFromFiles(getBVHFileName().c_str());
	testTree->printTree(true);
	delete testTree;
/// DEBUG END
	*/
	
	// delete individual voxel BVH files
	cout << "Removing temporary files..." << endl;
	for (boxNr = 0; boxNr < grid.getSize(); boxNr++) {
		unlink(getVTriangleFileName(boxNr).c_str());
		unlink(getBVHFileName(boxNr).c_str());
		unlink(getTriangleIdxFileName(boxNr).c_str());
	}

	return true;
}
#endif

