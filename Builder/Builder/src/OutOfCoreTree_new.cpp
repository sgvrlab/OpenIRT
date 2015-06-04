//#include "kDTree.h"
#include "OutOfCoreTree.h"

#define _CRT_SECURE_NO_DEPRECATE
#pragma warning (disable: 4996)
#include <math.h>
#include <direct.h>

#include "CashedBoxFiles.h"
#include "Vertex.h"
#include "Triangle.h"
#include "helpers.h"

#if HIERARCHY_TYPE == TYPE_KD_TREE
#include "VoxelkDTree.h"
#endif
#if HIERARCHY_TYPE == TYPE_BVH
#include "VoxelBVH.h"
//#include "BVHCompression.h"
//#include "BVHRefine.h"
#endif
#include "OptionManager.h"
#include "Progression.h"
#include "Materials.h"

#include "TriBoxIntersect.hpp"


OutOfCoreTree::OutOfCoreTree() {
	reader = 0;
	fileName[0] = 0;
	numFaces = 0;
	numVertices = 0;
}

OutOfCoreTree::~OutOfCoreTree() {

	if (reader)
		delete reader;
}

#if HIERARCHY_TYPE == TYPE_KD_TREE
bool OutOfCoreTree::build(const char *fileName) {
	TimerValue start, end;
	OptionManager *opt = OptionManager::getSingletonPtr();

	start.set();

	// build reader:
	reader = new PlyReader(fileName);

	const char *baseDirName = opt->getOption("global", "scenePath", "");
	sprintf(outDirName, "%s%s.ooc", baseDirName, fileName);
	mkdir(outDirName);
	
	//
	// first pass:
	// find number of vertices/tris and bounding-box:
	//
	firstVertexPass();

	cout << "Model dimensions: " << grid.getExtent() << endl;
	cout << "BB: min: " << grid.p_min << ", max: " << grid.p_max << endl;

	// calculate number of voxels
	calculateVoxelResolution(grid);

	numVoxels = grid.getSize();
	
	cout << "Using " << numVoxels << " Voxels." << endl;

	//
	// second pass: sort vertices & triangles to voxels
	//
	secondVertexPass();

	//
	// for each voxel: build kD-tree:
	//
	buildVoxelkDTrees();

	//
	// build high-level kD-tree from voxels:
	//
	buildHighLevelkdTree();

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
#if HIERARCHY_TYPE == TYPE_BVH 
bool OutOfCoreTree::build(const char *fileName) {
	TimerValue start, end;
	OptionManager *opt = OptionManager::getSingletonPtr();

	start.set();

	// build reader:
	reader = new PlyReader(fileName);

	const char *baseDirName = opt->getOption("global", "scenePath", "");
	sprintf(outDirName, "%s%s.ooc", baseDirName, fileName);
	mkdir(outDirName);

	//
	// first pass:
	// find number of vertices/tris and bounding-box:
	//
	firstVertexPass();

	cout << "Model dimensions: " << grid.getExtent() << endl;
	cout << "BB: min: " << grid.p_min << ", max: " << grid.p_max << endl;

	// calculate number of voxels
	calculateVoxelResolution(grid, numFaces, opt->getOptionAsInt("raytracing", "targetTrianglesPerOOCVoxel", 100000));

	numVoxels = grid.getSize();
	
	cout << "Using " << numVoxels << " Voxels." << endl;

	//
	// second pass: sort vertices & triangles to voxels
	//
	secondVertexPass();

	//
	// Chunking pass, divide voxels which have too many triangles
	//
	thirdVertexPass();

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

		// read all vertices
		while (reader->haveVertex()) {
			++numVertices;
			if (numVertices%1000000 == 0) {
				cout << " > " << numVertices << " vertices read" << endl;
			}		

			// update bounding box
			grid.addPoint(reader->readVertex().v);
		}

		cout << numVertices << " Vertices read.\n";

		// read all faces
		std::vector<int> vIdxList;
		while (reader->haveFace()) {
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

void OutOfCoreTree::calculateVoxelResolution(Grid &grid, unsigned int numFaces, unsigned int targetVoxelSize) {
	OptionManager *opt = OptionManager::getSingletonPtr();
	float maxExtent = grid.getMaxExtent();
	unsigned int nrCountDivisions;
	float avgVoxelFaces = numFaces / (float)targetVoxelSize;
	
	nrCountDivisions = (unsigned int)ceil(2.0 * pow(avgVoxelFaces, 1.0f/3.0f));
	if (nrCountDivisions == 1)
		nrCountDivisions = 2;

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
		for (unsigned int boxNr = 0; boxNr < numVoxels; boxNr++) {			
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
			BufferedOutputs<Point<int> > *pVertexIndexMap = new BufferedOutputs<Point<int> >(getVertexIndexFileName(), 100000);
			pVertexIndexMap->clear();

			// init out file for all vertices' colors (temporary)
			BufferedOutputs<rgb> *pColorList = new BufferedOutputs<rgb>(getColorListName(), 10000);
			pColorList->clear();
					
			// init reader
			
			Progression prog1("read verts" , numVertices, 20);

			hasVertexColors = reader->hasColor();
			hasVertexNormals = reader->hasVertNormal();
			hasVertexTextures = reader->hasVertTexture();

			// read all vertices:
			// 
			#ifdef _USE_TRI_MATERIALS
			rgb tempColor;
			if (reader->hasColor()) {
				while (reader->haveVertex()) {					
					Vertex curVert = reader->readVertexWithColor(tempColor);			


					// get voxel for point:
					Point<int> idx = grid.getCellIndices_pt(curVert.v);

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

				#ifdef _USE_TRI_MATERIALS
				// insert dummy color:
				pColorList->appendElement(rgb(0.7f, 0.7f, 0.7f));
				#endif

				// get voxel for point:
				Point<int> idx = grid.getCellIndices_pt(curVert.v);

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
		//CashedBoxSingleFile<Point<int> > *pVertexIndexCache = new CashedBoxSingleFile<Point<int> >(getVertexIndexFileName(), numVoxels, 10, false);
		//CashedBoxSingleFile<Point<int> > &voxelIndexCache = *pVertexIndexCache;
		//CashedBoxSingleFile<Vector3> *pVertexCache = new CashedBoxSingleFile<Vector3>(getVertexFileName(), numVoxels, 10, false);
		//CashedBoxSingleFile<Vector3> &vertexCache = *pVertexCache;
		CashedBoxSingleFile<Vertex> *pVertexCache = new CashedBoxSingleFile<Vertex>(getVertexFileName(), numVoxels, 10, false);
		CashedBoxSingleFile<Vertex> &vertexCache = *pVertexCache;
		CashedBoxSingleFile<rgb> *pColorCache = new CashedBoxSingleFile<rgb>(getColorListName(), numVoxels, 10, false);
		CashedBoxSingleFile<rgb> &colorCache = *pColorCache;

		// init out file for all tris	
		BufferedOutputs<Triangle> *pTris = new BufferedOutputs<Triangle>(getTriangleFileName(), 100000);
		pTris->clear();

		// init out file for all materials
		//BufferedOutputs<MaterialDiffuse> *outputs_mat = new BufferedOutputs<MaterialDiffuse>(getMaterialListName().c_str(), 1000);
		m_outputs_mat = new BufferedOutputs<MaterialDiffuse>(getMaterialListName().c_str(), 1000);
		m_outputs_mat->clear();
		
		// init out files for single voxels	
		BufferedOutputs<Triangle> *outputs = new BufferedOutputs<Triangle>(getVTriangleFileName().c_str(), numVoxels, 5000);
		outputs->clear();

		// init out file for all tri indices
		BufferedOutputs<unsigned int> *outputs_idx = new BufferedOutputs<unsigned int>(getTriangleIdxFileName().c_str(), numVoxels, 5000);
		outputs_idx->clear();
	
		//
		// read all triangles:
		//

		std::vector<int> vIdxList;
		Triangle tri;
		Vector3 p[3];
		unsigned int voxel[3];
		//Point<int> voxel[3];
		unsigned int triCount = 0;		
		typedef stdext::hash_map<unsigned int, unsigned int> ColorTable;
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
			rgb color = colorCache[vIdxList[firstIdx]]; 
			unsigned int hash = (unsigned int)(color.r() + 256*color.g() + 256*256*color.b());

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

			/*
			Point<int> minV(
				min(min(voxel[0].coords[0], voxel[1].coords[0]), voxel[2].coords[0]),
				min(min(voxel[0].coords[1], voxel[1].coords[1]), voxel[2].coords[1]),
				min(min(voxel[0].coords[2], voxel[1].coords[2]), voxel[2].coords[2]));
			Point<int> maxV(
				max(max(voxel[0].coords[0], voxel[1].coords[0]), voxel[2].coords[0]),
				max(max(voxel[0].coords[1], voxel[1].coords[1]), voxel[2].coords[1]),
				max(max(voxel[0].coords[2], voxel[1].coords[2]), voxel[2].coords[2]));

			Vector3 bbMin, bbMax;
			Point<int> cellIdxs;
			int cellIdx;
			for(int x=minV.coords[0];x<=maxV.coords[0];x++)
				for(int y=minV.coords[1];y<=maxV.coords[1];y++)
					for(int z=minV.coords[2];z<=maxV.coords[2];z++)
					{
						cellIdxs.setPoint(x, y, z);
						grid.getCellBoundingBox(cellIdxs, bbMin, bbMax);
						if(triBoxIntersect(p[0], p[1], p[2], bbMin, bbMax))
						{
							cellIdx = grid.getCellIndex(cellIdxs);
							outputs->appendElement(cellIdx, tri);
							outputs_idx->appendElement(cellIdx, triCount);

							// add to hash table counter
							it = voxelHashTable->find(cellIdx);
							if (it != voxelHashTable->end()) {
								(*it).second = (*it).second + 1;
							}
							else {
								voxelHashTable->insert(std::pair<int,int>(cellIdx,1));		
								numUsedVoxels++;
							}
						}
					}
			*/


			for (int i = 0; i < 3; i++) {
				/// TODO: triangles spanning multiple voxels!

				#if defined(_USE_VOXEL_OUTPUT)
				// don't insert triangle multiple times
				if (i == 1 && voxel[1] == voxel[0])
					continue;
				else if (i == 2 && (voxel[2] == voxel[0] || voxel[2] == voxel[1]))
					continue;
				#else
				// assign the triangle to the Voxel [0]
				if (i != 0)
					continue;
				#endif

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
		
	cout << "Voxels used: " << numUsedVoxels << " of " << numVoxels << endl;
	unsigned int sumTris = 0;
	for (it = voxelHashTable->begin(); it != voxelHashTable->end(); it++) {
		cout << " > " << it->first << " -> " << it->second << " tris" << endl;
		sumTris += it->second;
	}
	cout << "Tri references: " << sumTris << " (original: " << numFaces << ")" << endl;
	
	return true;
}

bool OutOfCoreTree::thirdVertexPass()
{
	typedef struct Elem_t
	{
		AdaptiveVoxel *aVoxel;
		unsigned int globalCellIndex;
		unsigned int localCellIndex;
		unsigned int numTris;
		Elem_t(AdaptiveVoxel *v, unsigned int gc, unsigned int lc, unsigned int n) 
		{
			aVoxel = v; globalCellIndex = gc; localCellIndex = lc; numTris = n;
		}
	} Elem;
	VoxelHashTableIterator it;		

	std::list<Elem> bigVoxels;
	OptionManager *opt = OptionManager::getSingletonPtr();
	unsigned int targetVoxelSize = opt->getOptionAsInt("raytracing", "targetTrianglesPerOOCVoxel", 100000);
	//targetVoxelSize = 5000;

	AdaptiveVoxel aVoxel;
	aVoxel.setGrid(&grid);
	aVoxel.fileIndex = -2;

	FILE *fpDist = fopen("voxel_dist.txt", "w");
	cout << "Voxel distribution... (number of triangles per each voxel)" << endl;
	fprintf(fpDist, "Voxel distribution... (number of triangles per each voxel)\n");

	for(int i=0;i<numVoxels;i++)
	{
		AdaptiveVoxel &aV = aVoxel.children[i];
		aV.cellIndex = i;
		Point<int> cellIdx = grid.getCellIndices(i);
		aV.parent = &aVoxel;
		grid.getCellBoundingBox(i, aV.getGrid()->p_min, aV.getGrid()->p_max);

		if(cellIdx[0] > 0) aV.neighbor[0] = grid.getCellIndex_(Point<int>(cellIdx[0] - 1, cellIdx[1], cellIdx[2]));
		if(cellIdx[1] > 0) aV.neighbor[1] = grid.getCellIndex_(Point<int>(cellIdx[0], cellIdx[1] - 1, cellIdx[2]));
		if(cellIdx[2] > 0) aV.neighbor[2] = grid.getCellIndex_(Point<int>(cellIdx[0], cellIdx[1], cellIdx[2] - 1));
		if(cellIdx[0] < grid.resolution[0] - 1) aV.neighbor[3] = grid.getCellIndex_(Point<int>(cellIdx[0] + 1, cellIdx[1], cellIdx[2]));
		if(cellIdx[1] < grid.resolution[1] - 1) aV.neighbor[4] = grid.getCellIndex_(Point<int>(cellIdx[0], cellIdx[1] + 1, cellIdx[2]));
		if(cellIdx[2] < grid.resolution[2] - 1) aV.neighbor[5] = grid.getCellIndex_(Point<int>(cellIdx[0], cellIdx[1], cellIdx[2] + 1));

		// test if voxel is in use:
		it = voxelHashTable->find(i);
		if (it == voxelHashTable->end())
			continue;

		aV.fileIndex = i;

		unsigned int numTris = it->second;		

		printf("[%05d] %d\n", i, numTris);
		fprintf(fpDist, "[%05d] %d\n", i, numTris);

		if(targetVoxelSize < numTris)
		{
			bigVoxels.push_back(Elem(&aVoxel, i, i, numTris));
		}
	}

	while(!bigVoxels.empty())
	{
		Elem cur = bigVoxels.front();
		bigVoxels.pop_front();

		Grid subGrid;
		Vector3 minBB, maxBB;

		printf("Divide cell %d...\n", cur.globalCellIndex);
		fprintf(fpDist, "Divide cell %d...\n", cur.globalCellIndex);

		CashedBoxSingleFile<Triangle> *pTriangleCache = new CashedBoxSingleFile<Triangle>(getVTriangleFileName(cur.globalCellIndex), cur.aVoxel->getGrid()->getSize(), 10, false);
		CashedBoxSingleFile<Triangle> &triangleCache = *pTriangleCache;
		CashedBoxSingleFile<unsigned int> *pTriangleindexCache = new CashedBoxSingleFile<unsigned int>(getTriangleIdxFileName(cur.globalCellIndex), cur.aVoxel->getGrid()->getSize(), 10, false);
		CashedBoxSingleFile<unsigned int> &triangleIndexCache = *pTriangleindexCache;
		CashedBoxSingleFile<Vertex> *pVertexCache = new CashedBoxSingleFile<Vertex>(getVertexFileName(), cur.aVoxel->getGrid()->getSize(), 10, false);
		CashedBoxSingleFile<Vertex> &vertexCache = *pVertexCache;

		// set bounding box of sub grid
		subGrid.clear();

		cur.aVoxel->getGrid()->getCellBoundingBox(cur.localCellIndex, minBB, maxBB);
		subGrid.addPoint(minBB);
		subGrid.addPoint(maxBB);

		calculateVoxelResolution(subGrid, cur.numTris, targetVoxelSize);

		// init out files for single voxels	
		BufferedOutputs<Triangle> *outputs = new BufferedOutputs<Triangle>(getVTriangleFileName().c_str(), subGrid.getSize(), 5000, numVoxels);
		outputs->clear();

		// init out file for all tri indices
		BufferedOutputs<unsigned int> *outputs_idx = new BufferedOutputs<unsigned int>(getTriangleIdxFileName().c_str(), subGrid.getSize(), 5000, numVoxels);
		outputs_idx->clear();

		Vector3 p[3];
		for(unsigned int i=0;i<cur.numTris;i++)
		{
			Triangle &tri = triangleCache[i];
			Point<int> voxel[3];
			p[0] = vertexCache[tri.p[0]].v;
			p[1] = vertexCache[tri.p[1]].v;
			p[2] = vertexCache[tri.p[2]].v;

			// insert triangle into each voxel it belongs in:		
			/*
			voxel[0] = subGrid.getCellIndex(vertexCache[tri.p[0]].v) + numVoxels;
			voxel[1] = subGrid.getCellIndex(vertexCache[tri.p[1]].v) + numVoxels;
			voxel[2] = subGrid.getCellIndex(vertexCache[tri.p[2]].v) + numVoxels;

			Point<int> idx = grid.getCellIndices_pt(curVert.v);
			*/
			voxel[0] = subGrid.getCellIndices_pt(p[0]);
			voxel[1] = subGrid.getCellIndices_pt(p[1]);
			voxel[2] = subGrid.getCellIndices_pt(p[2]);

			Point<int> minV(
				min(min(voxel[0].coords[0], voxel[1].coords[0]), voxel[2].coords[0]),
				min(min(voxel[0].coords[1], voxel[1].coords[1]), voxel[2].coords[1]),
				min(min(voxel[0].coords[2], voxel[1].coords[2]), voxel[2].coords[2]));
			Point<int> maxV(
				max(max(voxel[0].coords[0], voxel[1].coords[0]), voxel[2].coords[0]),
				max(max(voxel[0].coords[1], voxel[1].coords[1]), voxel[2].coords[1]),
				max(max(voxel[0].coords[2], voxel[1].coords[2]), voxel[2].coords[2]));

			Vector3 bbMin, bbMax;
			Point<int> cellIdxs;
			int cellIdx;
			for(int x=minV.coords[0];x<=maxV.coords[0];x++)
				for(int y=minV.coords[1];y<=maxV.coords[1];y++)
					for(int z=minV.coords[2];z<=maxV.coords[2];z++)
					{
						cellIdxs.setPoint(x, y, z);
						subGrid.getCellBoundingBox(cellIdxs, bbMin, bbMax);
						if(triBoxIntersect(p[0], p[1], p[2], bbMin, bbMax))
						{
							cellIdx = subGrid.getCellIndex(cellIdxs) + numVoxels;
							outputs->appendElement(cellIdx, tri);
							outputs_idx->appendElement(cellIdx, triangleIndexCache[i]);

							// add to hash table counter
							it = voxelHashTable->find(cellIdx);
							if (it != voxelHashTable->end()) {
								(*it).second = (*it).second + 1;
							}
							else {
								voxelHashTable->insert(std::pair<int,int>(cellIdx,1));		
								numUsedVoxels++;
							}
						}
					}

			/*
			for (int j = 0; j < 3; j++) {
				/// TODO: triangles spanning multiple voxels!

				#if defined(_USE_VOXEL_OUTPUT)
				// don't insert triangle multiple times
				if (j == 1 && voxel[1] == voxel[0])
					continue;
				else if (j == 2 && (voxel[2] == voxel[0] || voxel[2] == voxel[1]))
					continue;
				#else
				// assign the triangle to the Voxel [0]
				if (j != 0)
					continue;
				#endif

				outputs->appendElement(voxel[j], tri);
				outputs_idx->appendElement(voxel[j], triangleIndexCache[i]);

				// add to hash table counter
				it = voxelHashTable->find(voxel[j]);
				if (it != voxelHashTable->end()) {
					(*it).second = (*it).second + 1;
				}
				else {
					voxelHashTable->insert(std::pair<int,int>(voxel[j],1));		
					numUsedVoxels++;
				}
			}
			*/
		}

		outputs->flush();
		outputs_idx->flush();
		delete pTriangleCache;
		delete pTriangleindexCache;
		delete pVertexCache;
		delete outputs;
		delete outputs_idx;

		AdaptiveVoxel &curAVoxel = cur.aVoxel->children[cur.localCellIndex];
		curAVoxel.fileIndex = -2;
		curAVoxel.setGrid(&subGrid);

		int numSubSubGrid = 0;
		for(int i=0;i<subGrid.getSize();i++)
		{
			AdaptiveVoxel &aV = curAVoxel.children[i];
			aV.cellIndex = i;
			Point<int> cellIdx = subGrid.getCellIndices(i);
			aV.parent = &curAVoxel;
			subGrid.getCellBoundingBox(i, aV.getGrid()->p_min, aV.getGrid()->p_max);
			
			if(cellIdx[0] > 0) aV.neighbor[0] = subGrid.getCellIndex_(Point<int>(cellIdx[0] - 1, cellIdx[1], cellIdx[2]));
			if(cellIdx[1] > 0) aV.neighbor[1] = subGrid.getCellIndex_(Point<int>(cellIdx[0], cellIdx[1] - 1, cellIdx[2]));
			if(cellIdx[2] > 0) aV.neighbor[2] = subGrid.getCellIndex_(Point<int>(cellIdx[0], cellIdx[1], cellIdx[2] - 1));
			if(cellIdx[0] < subGrid.resolution[0] - 1) aV.neighbor[3] = subGrid.getCellIndex_(Point<int>(cellIdx[0] + 1, cellIdx[1], cellIdx[2]));
			if(cellIdx[1] < subGrid.resolution[1] - 1) aV.neighbor[4] = subGrid.getCellIndex_(Point<int>(cellIdx[0], cellIdx[1] + 1, cellIdx[2]));
			if(cellIdx[2] < subGrid.resolution[2] - 1) aV.neighbor[5] = subGrid.getCellIndex_(Point<int>(cellIdx[0], cellIdx[1], cellIdx[2] + 1));

			int globalIndex = numVoxels + i;

			// test if voxel is in use:
			it = voxelHashTable->find(globalIndex);
			if (it == voxelHashTable->end())
				continue;

			aV.fileIndex = globalIndex;

			unsigned int numTris = it->second;		

			printf("[%05d] %d\n", globalIndex, numTris);
			fprintf(fpDist, "[%05d] %d\n", globalIndex, numTris);

			numSubSubGrid++;

			if(targetVoxelSize < numTris)
			{
				bigVoxels.push_back(Elem(&curAVoxel, globalIndex, i, numTris));
			}
		}

		if(numSubSubGrid == 1)
		{
			// cannot further divide
			bigVoxels.pop_back();
		}
		//else
		{
			// remove current big voxel
			it = voxelHashTable->find(cur.globalCellIndex);
			if(it != voxelHashTable->end()) voxelHashTable->erase(it);
			unlink(getVTriangleFileName(cur.globalCellIndex).c_str());
			unlink(getTriangleIdxFileName(cur.globalCellIndex).c_str());
			numUsedVoxels--;
		}

		numVoxels += subGrid.getSize();
	}

	fclose(fpDist);

	FILE *fpAVs = fopen(getAVoxelFileName().c_str(), "wb");
	aVoxel.saveToFile(fpAVs);
	fclose(fpAVs);

	FILE *sceneFile = fopen(getSceneInfoName().c_str(), "ab");
	fwrite(&numVoxels, sizeof(unsigned int), 1, sceneFile);
	fclose(sceneFile);

	return true;
}

#if HIERARCHY_TYPE == TYPE_KD_TREE
bool OutOfCoreTree::buildVoxelkDTrees() {
	TimerValue start, end;
	unsigned int boxesBuilt = 0;
	cout << "Building kD trees for " << numUsedVoxels << " voxels..." << endl;
	cout << "Overall BB: [" << grid.p_min << "] - [" << grid.p_max << "]" << endl;

	start.set();

#ifdef NUM_OMP_THREADS
	omp_set_num_threads(NUM_OMP_THREADS);
#endif

	//
	// for all voxels: build individual kD-trees
	// (in parallel if configured and compiler supports it)
	//
#ifdef _USE_OPENMP
	#pragma omp parallel for schedule(dynamic,1) shared(boxesBuilt, numVoxels)
#endif
	for (int boxNr = 0; boxNr < numVoxels; boxNr++) {		
		
		LogManager *log = LogManager::getSingletonPtr();
		kDTree *tree;				
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
		//CashedBoxSingleFile<Vector3> *pVertexCache = new CashedBoxSingleFile<Vector3>(getVertexFileName(), numVoxels, 10, false);
		//CashedBoxSingleFile<Vector3> &vertexCache = *pVertexCache;
		CashedBoxSingleFile<Vertex> *pVertexCache = new CashedBoxSingleFile<Vertex>(getVertexFileName(), numVoxels, 10, false);
		CashedBoxSingleFile<Vertex> &vertexCache = *pVertexCache;

		unsigned int numTris = it->second;		
		
		sprintf(output, " - Processor %d voxel %d / %d (%u tris)", omp_get_thread_num(), boxesBuilt, numUsedVoxels, numTris);
		log->logMessage(LOG_INFO, output);		

		// skip this if tree already exists
		FILE *test = fopen(getkDTreeFileName(boxNr).c_str(), "rb");
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

		// read in vertices from vertex file and store in hash table for sparse
		// storage. that way, we can still access the vertices via absolute index.
		//cout << "   > caching vertices..." << endl;
		for (unsigned int i = 0; i < numTris; i++) {			
			vertexMap[triangleCache[i].p[0]] = vertexCache[triangleCache[i].p[0]];
			vertexMap[triangleCache[i].p[1]] = vertexCache[triangleCache[i].p[1]];
			vertexMap[triangleCache[i].p[2]] = vertexCache[triangleCache[i].p[2]];			
		}
		 
		//cout << "   > building tree..." << endl;

		//
		// build tree:
		//
		Vector3 bb_min, bb_max;
		grid.getCellBoundingBox(boxNr, bb_min, bb_max);
		//cout << "Voxel BB: [" << bb_min << "] - [" << bb_max << "]" << endl;

		tree = new kDTree(triangleCache, triangleIndexCache, numTris, &vertexMap, bb_min, bb_max, boxNr);
		tree->buildTree();		
		tree->saveToFile(getkDTreeFileName(boxNr).c_str());
		tree->printTree(false);
		
		delete pVertexCache;
		delete triangleCache;
		delete triangleIndexCache;
		delete tree;
	}

	end.set();

	cout << "kD tree build ended, time = " << (end - start) << "s" << endl;
	
	return true;
}

bool OutOfCoreTree::buildHighLevelkdTree() {
	TimerValue start, end;
	Voxel *voxellist;
	VoxelkDTree *tree;
	VoxelHashTableIterator it;

	cout << "Building high-level kD tree for " << numUsedVoxels << " voxels..." << endl;
	cout << "Overall BB: [" << grid.p_min << "] - [" << grid.p_max << "]" << endl;

	start.set();

	//
	// build array of all voxels and number of triangles
	// in them.
	//

	voxellist = new Voxel[numUsedVoxels];
	unsigned int curVoxel = 0;
	int boxNr;
	for (boxNr = 0; boxNr < numVoxels; boxNr++) {
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
	tree = new VoxelkDTree(voxellist, numUsedVoxels, grid.p_min, grid.p_max);
	tree->buildTree();
	//tree->printTree(true);
	tree->printTree(false);

	cout << "Building final tree by including voxel kD trees..." << endl;

	// make real tree from the high-level structure by including 
	// all the individual kD trees from the voxels:
	tree->saveToFile(getkDTreeFileName().c_str());

	end.set();
	cout << "High-Level tree build ended, time = " << (end - start) << "s" << endl;

	delete tree;
	delete voxellist;
	
	/*
/// DEBUG: output kD tree:
	std::hash_map<unsigned int, Vertex> vertexMap;
	Triangle *triangleCache = 0;
	unsigned int *idxCache = 0; 
	kDTree *testTree = new kDTree(triangleCache, idxCache, 0, &vertexMap, grid.p_min, grid.p_max);
	testTree->loadFromFiles(getkDTreeFileName().c_str());
	testTree->printTree(true);
	delete testTree;
/// DEBUG END
	*/
	
	// delete individual voxel kD tree files
	cout << "Removing temporary files..." << endl;
	for (boxNr = 0; boxNr < numVoxels; boxNr++) {
		unlink(getVTriangleFileName(boxNr).c_str());
		unlink(getkDTreeFileName(boxNr).c_str());
		unlink(getTriangleIdxFileName(boxNr).c_str());
	}

	return true;
}
#endif

#if HIERARCHY_TYPE == TYPE_BVH 
bool OutOfCoreTree::distributeVertexs() {
	TimerValue start, end;
	unsigned int boxesBuilt = 0;
	cout << "Distributing vertexs into " << numUsedVoxels << " voxels..." << endl;

	start.set();

	BufferedOutputs<Vertex> *outputs = new BufferedOutputs<Vertex>(getVVertexFileName().c_str(), numVoxels, 5000);
	outputs->clear();

	for (int boxNr = 0; boxNr < numVoxels; boxNr++) {
		LogManager *log = LogManager::getSingletonPtr();
		VoxelHashTableIterator it;		
		stdext::hash_map<unsigned int, unsigned int> vertexHash;
		stdext::hash_map<unsigned int, Vertex> vertexMap;
		Triangle *triangleCache;
		FILE *triangleFile;

		// test if voxel is in use:
		it = voxelHashTable->find(boxNr);
		if (it == voxelHashTable->end())
			continue;

		boxesBuilt++;

		// open the vertex file in ooc mode:
		CashedBoxSingleFile<Vertex> *pVertexCache = new CashedBoxSingleFile<Vertex>(getVertexFileName(), numVoxels, 10, false);
		CashedBoxSingleFile<Vertex> &vertexCache = *pVertexCache;

		unsigned int numTris = it->second;		
		
		// read in file with triangle indices for this voxel
		if ((triangleFile = fopen(getVTriangleFileName(boxNr).c_str(), "rb")) == NULL) {
			cout << "ERROR: could not open file " << getVTriangleFileName(boxNr) << " !" << endl;
			continue;
		}

		triangleCache = new Triangle[numTris];
		size_t ret = fread(triangleCache, sizeof(Triangle), numTris, triangleFile);
		if (ret != numTris) {
			cout << "ERROR: could only read " << ret << " of " << numTris << " triangles from file " << getVTriangleFileName(boxNr) << " !" << endl;			
			continue;
		}

		fclose(triangleFile);

		stdext::hash_map<unsigned int, unsigned int>::iterator it2;

		for (unsigned int i = 0; i < numTris; i++) {			
			for(int j=0;j<3;j++)
			{
				// add to hash table counter
				it2 = vertexHash.find(triangleCache[i].p[j]);
				if (it2 == vertexHash.end()) 
				{
					vertexHash.insert(std::pair<unsigned int,unsigned int>(triangleCache[i].p[j], (unsigned int)vertexHash.size()));
					outputs->appendElement(boxNr, vertexCache[triangleCache[i].p[j]]);
					// for compatibility with previous version
					vertexMap[vertexHash.size() - 1] = vertexCache[triangleCache[i].p[j]];
					triangleCache[i].p[j] = vertexHash.size() - 1;
				}
				else
				{
					triangleCache[i].p[j] = it2->second;
				}
			}
		}
	
		delete triangleCache;
		delete pVertexCache;
	}

	outputs->flush();
	delete outputs;

	end.set();

	cout << "Distributing vertexs ended, time = " << (end - start) << "s" << endl;
	
	return true;
}

bool OutOfCoreTree::buildVoxelBVH() {
	TimerValue start, end;
	unsigned int boxesBuilt = 0;
	cout << "Building BVH for " << numUsedVoxels << " voxels..." << endl;
	cout << "Overall BB: [" << grid.p_min << "] - [" << grid.p_max << "]" << endl;

	start.set();
	voxelMins = new Vector3[numUsedVoxels];
	voxelMaxs = new Vector3[numUsedVoxels];
	Vector3 *voxelMinsTemp = new Vector3[numVoxels];
	Vector3 *voxelMaxsTemp = new Vector3[numVoxels];

	Vector3 min_limit(FLT_MAX, FLT_MAX, FLT_MAX), max_limit(-FLT_MAX, -FLT_MAX, -FLT_MAX);

	for (int i=0;i<numVoxels;i++)
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
	unsigned int numBoxes = numVoxels;
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

#ifdef _USE_VOXEL_OUTPUT
		stdext::hash_map<unsigned int, unsigned int> vertexHash;
		BufferedOutputs<Vertex> *vert_output = new BufferedOutputs<Vertex>(getVVertexFileName(boxNr).c_str(), 100000);
		vert_output->clear();
#endif

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
		//CashedBoxSingleFile<Vector3> *pVertexCache = new CashedBoxSingleFile<Vector3>(getVertexFileName(), numBoxes, 10, false);
		//CashedBoxSingleFile<Vector3> &vertexCache = *pVertexCache;
		CashedBoxSingleFile<Vertex> *pVertexCache = new CashedBoxSingleFile<Vertex>(getVertexFileName(), numBoxes, 10, false);
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

#ifdef _USE_VOXEL_OUTPUT
		stdext::hash_map<unsigned int, unsigned int>::iterator it2;

		for (unsigned int i = 0; i < numTris; i++) {			
			for(int j=0;j<3;j++)
			{
				// add to hash table counter
				it2 = vertexHash.find(triangleCache[i].p[j]);
				if (it2 == vertexHash.end()) 
				{
					vertexHash.insert(std::pair<unsigned int,unsigned int>(triangleCache[i].p[j], (unsigned int)vertexHash.size()));
					vert_output->appendElement(vertexCache[triangleCache[i].p[j]]);
					
					// for compatibility with previous version
					vertexMap[vertexHash.size() - 1] = vertexCache[triangleCache[i].p[j]];

					updateBB(bb_min, bb_max, vertexCache[triangleCache[i].p[j]].v);
					triangleCache[i].p[j] = vertexHash.size() - 1;
				}
				else
				{
					triangleCache[i].p[j] = it2->second;
				}
			}
		}

		// apply changes
		if ((triangleFile = fopen(getVTriangleFileName(boxNr).c_str(), "wb")) == NULL) {
			cout << "ERROR: could not open file " << getVTriangleFileName(boxNr) << " !" << endl;
			continue;
		}

		ret = fwrite(triangleCache, sizeof(Triangle), numTris, triangleFile);

		fclose(triangleFile);
#else
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
#endif

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

#ifdef _USE_VOXEL_OUTPUT
		vert_output->flush();
		delete vert_output;
#endif
	}

	end.set();

	// Pack the voxel mins and maxes
	for (int boxTemp = 0, box = 0; boxTemp < numVoxels; boxTemp++)
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

bool OutOfCoreTree::buildHighLevelBVH(bool useVoxelBBList) {
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
	for (boxNr = 0; boxNr < numVoxels; boxNr++) {
		// test if voxel is in use:
		it = voxelHashTable->find(boxNr);
		if (it == voxelHashTable->end())
			continue;

		// enter voxel information:
		voxellist[curVoxel].index = boxNr;
		voxellist[curVoxel].numTris = it->second;
		if(useVoxelBBList)
		{
			voxellist[curVoxel].min = voxelBBList[curVoxel].p_min;
			voxellist[curVoxel].max = voxelBBList[curVoxel].p_max;
		}
		else
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
	
#ifndef _USE_VOXEL_OUTPUT
	// delete individual voxel BVH files
	cout << "Removing temporary files..." << endl;
	for (boxNr = 0; boxNr < numVoxels; boxNr++) {
		unlink(getVTriangleFileName(boxNr).c_str());
		unlink(getBVHFileName(boxNr).c_str());
		unlink(getTriangleIdxFileName(boxNr).c_str());
	}
#endif

	return true;
}
#endif

bool OutOfCoreTree::temp(const char *fileName)
{
	OptionManager *opt = OptionManager::getSingletonPtr();
	const char *baseDirName = opt->getOption("global", "scenePath", "");

	FILE *fpSrc = fopen("log_22.txt", "r");
	FILE *fpDst = fopen("log_23.txt", "w");

	char line[4096];

	typedef stdext::hash_map<int, int> Hash;
	typedef Hash::iterator HashIt;

	Hash hash;
	int sumAll = 0;
	int sumBig = 0;
	int numAll = 0;
	int numBig = 0;

	while(fgets(line, 4096, fpSrc))
	{
		if(strstr(line, "[") == line)
		{
			int idx, num;
			sscanf(line, "[%d] %d", &idx, &num);
			hash[idx] = num;
			sumAll += num;
			numAll++;
			fputs(line, fpDst);
		}

		if(strstr(line, "Divide") == line)
		{
			int idx;
			sscanf(line, "Divide cell %d...", &idx);
			sumBig += hash[idx];
			numBig++;
		}
	}

	fclose(fpSrc);
	fclose(fpDst);
	return true;
}