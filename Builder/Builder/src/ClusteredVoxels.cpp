#include "OutOfCoreTree.h"
#include "Progression.h"
#include "CashedBoxFiles.h"
#include <algorithm>

void OutOfCoreTree::buildClusteredVoxels(const char *fileListName, float *mat)
{
	TimerValue start, end;
	OptionManager *opt = OptionManager::getSingletonPtr();

	start.set();

	const char *baseDirName = opt->getOption("global", "scenePath", "");
	sprintf(outDirName, "%s%s.ooc", baseDirName, fileListName);
	mkdir(outDirName);

	vector<char *> fileList;

	char currentLine[500];

	FILE *fpList = fopen(fileListName, "r");

	while (fgets(currentLine, 499, fpList)) 
	{
		char *fileName = new char[strlen(currentLine)+1];
		sscanf(currentLine, "%s", fileName);
		fileList.push_back(fileName);
	}

	fclose(fpList);

	//
	// first pass:
	// find number of vertices/tris and bounding-box:
	//
	firstVertexPassClusteredVoxels(fileList, mat);

	cout << "Model dimensions: " << grid.getExtent() << endl;
	cout << "BB: min: " << grid.p_min << ", max: " << grid.p_max << endl;

	numVoxels = fileList.size();
	
	cout << "Using " << numVoxels << " Voxels." << endl;

	//
	// second pass: sort vertices & triangles to voxels
	//
	secondVertexPassClusteredVoxels(fileList, mat);

	thirdVertexPassClusteredVoxels();

	//
	// for each voxel: build BVH:
	//
	buildVoxelBVH();

	//
	// build high-level BVH from voxels:
	//
	buildHighLevelBVH(true);

	buildHighLevelBVHClusteredVoxels();

	delete voxelHashTable;

	end.set();
	
	float elapsedHours;
	int elapsedMinOfHour;
	double elapsedMinOfHourFrac = modf((end - start)/(float)(60*60), &elapsedHours);
	elapsedMinOfHour = elapsedMinOfHourFrac * 60.0;
	
	cout << "OOC tree build ended, time = " << (end - start) << "s (" << (int)elapsedHours << " h, " << elapsedMinOfHour << " min)" << endl;

	for(int i=0;i<fileList.size();i++)
		delete[] fileList[i];
}

void OutOfCoreTree::firstVertexPassClusteredVoxels(const std::vector<char *> &fileList, float *mat)
{
	// init bounding box
	grid.clear();
	grid.size = fileList.size();
	// count number of vertices
	numVertices = 0;
	numFaces = 0;

	voxelBBList.resize(fileList.size());

	for(int fl=0;fl<fileList.size();fl++)
	{
		// init reader
		reader = new PlyReader(fileList[fl]);
		reader->restart();

		// read all vertices
		while (reader->haveVertex()) {
			++numVertices;
			if (numVertices%1000000 == 0) {
				cout << " > " << numVertices << " vertices read" << endl;
			}		

			// update bounding box
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
			}
			grid.addPoint(curVert.v);
			voxelBBList[fl].addPoint(curVert.v);
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

		delete reader;
	}

	FILE *firstPassFile = fopen(getSceneInfoName().c_str(), "wb");
	fwrite(&numVertices, sizeof(int), 1, firstPassFile);
	fwrite(&numFaces, sizeof(int), 1, firstPassFile);
	fwrite(&grid, sizeof(Grid), 1, firstPassFile);
	fclose(firstPassFile);
}

void OutOfCoreTree::secondVertexPassClusteredVoxels(const std::vector<char *> &fileList, float *mat)
{
	VoxelHashTableIterator it;
	voxelHashTable = new VoxelHashTable;
	MaterialDiffuse *plyMat = new MaterialDiffuse(rgb(0.7f,0.7f,0.7f));
	numUsedVoxels = 0;

	bool hasVertexColors = false;	// initialize
	bool hasVertexNormals = false;
	bool hasVertexTextures = false;

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
	BufferedOutputs<rgb> *pColorList = new BufferedOutputs<rgb>(getColorListName(), 10000);
	pColorList->clear();

	Progression prog("make voxels" , fileList.size(), 20);
	for(int fl=0;fl<fileList.size();fl++)
	{
		reader = new PlyReader(fileList[fl]);
		reader->restart();
			
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
					pt.e[0] = pv.e[0]*mat[0*4+0] + pv.e[1]*mat[0*4+1] + pv.e[2]*mat[0*4+2];
					pt.e[1] = pv.e[0]*mat[1*4+0] + pv.e[1]*mat[1*4+1] + pv.e[2]*mat[1*4+2];
					pt.e[2] = pv.e[0]*mat[2*4+0] + pv.e[1]*mat[2*4+1] + pv.e[2]*mat[2*4+2];
					pt.makeUnitVector();
					pn.e[0] = pt.e[0];
					pn.e[1] = pt.e[1];
					pn.e[2] = pt.e[2];
				}


				// write to file:
				pVertices->appendElement(curVert);
				pVertexIndexMap->appendElement(fl);
				pColorList->appendElement(tempColor);
			}			
		}
		else
		#endif

		while (reader->haveVertex()) {	
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
				pt.e[0] = pv.e[0]*mat[0*4+0] + pv.e[1]*mat[0*4+1] + pv.e[2]*mat[0*4+2];
				pt.e[1] = pv.e[0]*mat[1*4+0] + pv.e[1]*mat[1*4+1] + pv.e[2]*mat[1*4+2];
				pt.e[2] = pv.e[0]*mat[2*4+0] + pv.e[1]*mat[2*4+1] + pv.e[2]*mat[2*4+2];
				pt.makeUnitVector();
				pn.e[0] = pt.e[0];
				pn.e[1] = pt.e[1];
				pn.e[2] = pt.e[2];
			}

			#ifdef _USE_TRI_MATERIALS
			// insert dummy color:
			pColorList->appendElement(rgb(0.7f, 0.7f, 0.7f));
			#endif

			// write to file:
			pVertices->appendElement(curVert);
			//pVertices->appendElement(curVert);
			pVertexIndexMap->appendElement(fl);
		}

		delete reader;
	}
	// close vertex files...
	delete pVertexIndexMap;
	delete pVertices;
	delete pColorList;

	// ...and open again for cached access
	CashedBoxSingleFile<unsigned int> *pVertexIndexCache = new CashedBoxSingleFile<unsigned int>(getVertexIndexFileName(), numVoxels, 10, false);
	CashedBoxSingleFile<unsigned int> &voxelIndexCache = *pVertexIndexCache;
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
	
	int vertexIdxOffset = 0;
	int beforeVertexIdxOffset;
	for(int fl=0;fl<fileList.size();fl++)
	{
		reader = new PlyReader(fileList[fl]);
		reader->restart();
		// skip vertices
		Vertex dummy;
		beforeVertexIdxOffset = vertexIdxOffset;
		while (reader->haveVertex())
		{
			dummy = reader->readVertex();
			vertexIdxOffset++;
		}

		//
		// read all triangles:
		//

		std::vector<int> vIdxList;
		Triangle tri;
		Vector3 p[3];
		unsigned int voxel[3];
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
		while (reader->haveFace()) {	

			// read vertex indices into vector
			reader->readFace(vIdxList);
			vIdxList[0] += beforeVertexIdxOffset;
			vIdxList[1] += beforeVertexIdxOffset;
			vIdxList[2] += beforeVertexIdxOffset;

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

			if(voxel[0] != voxel[1] || voxel[1] != voxel[2] || voxel[2] != voxel[0])
			{
				printf("wrong vertex assign\n");
				exit(-1);
			}
			outputs->appendElement(voxel[0], tri);
			outputs_idx->appendElement(voxel[0], triCount);

			// add to hash table counter
			it = voxelHashTable->find(voxel[0]);
			if (it != voxelHashTable->end()) {
				(*it).second = (*it).second + 1;
			}
			else {
				voxelHashTable->insert(std::pair<int,int>(voxel[0],1));		
				numUsedVoxels++;
			}

			triCount++;
		}
		prog.step();
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
		
	cout << "Voxels used: " << numUsedVoxels << " of " << numVoxels << endl;
	unsigned int sumTris = 0;
	for (it = voxelHashTable->begin(); it != voxelHashTable->end(); it++) {
		cout << " > " << it->first << " -> " << it->second << " tris" << endl;
		sumTris += it->second;
	}
	cout << "Tri references: " << sumTris << " (original: " << numFaces << ")" << endl;
}

void OutOfCoreTree::thirdVertexPassClusteredVoxels()
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
		aV.parent = &aVoxel;
		aV.getGrid()->p_min = voxelBBList[i].p_min;
		aV.getGrid()->p_max = voxelBBList[i].p_max;
		// test if voxel is in use:
		it = voxelHashTable->find(i);
		if (it == voxelHashTable->end())
			continue;

		aV.fileIndex = i;
	}

	fclose(fpDist);

	FILE *fpAVs = fopen(getAVoxelFileName().c_str(), "wb");
	aVoxel.saveToFile(fpAVs);
	fclose(fpAVs);

	FILE *sceneFile = fopen(getSceneInfoName().c_str(), "ab");
	fwrite(&numVoxels, sizeof(unsigned int), 1, sceneFile);
	fclose(sceneFile);
}

#include "BVH.h"
void OutOfCoreTree::buildHighLevelBVHClusteredVoxels()
{
	unsigned int *indexlists = new unsigned int(numVoxels);

	Vector3 sceneMinBB(FLT_MAX, FLT_MAX, FLT_MAX);
	Vector3 sceneMaxBB(-FLT_MAX, -FLT_MAX, -FLT_MAX);

	for(int i=0;i<numVoxels;i++) 
	{
		indexlists[i] = i;
		updateBB(sceneMinBB, sceneMaxBB, voxelBBList[i].p_min);
		updateBB(sceneMinBB, sceneMaxBB, voxelBBList[i].p_max);
	}

	// use largest axis for first subdivision
	treeClusteredVoxels = new BSPArrayTreeNode[numVoxels*2 - 1];
		
	BSPArrayTreeNodePtr root = treeClusteredVoxels;
	root->max = sceneMaxBB;
	root->min = sceneMinBB;
		
	maxDepthClusteredVoxels = 0;

	subDivideSAHBlusteredVoxels(indexlists, 0, numVoxels-1, 0, 1 , 0);

	printf("\n\nMAX DEPTH : %d\n",maxDepthClusteredVoxels);
		
	delete indexlists;

	char fileName[MAX_PATH];
	sprintf(fileName, "%s/clusterBVH", outDirName);
	FILE *fp = fopen(fileName, "wb");
	fwrite(treeClusteredVoxels, sizeof(BSPArrayTreeNode), numVoxels*2 - 1, fp);
	fclose(fp);
}

// surface area of a voxel:
__inline float surfaceArea(float dim1, float dim2, float dim3) {
	return 2.0f * ((dim1 * dim2) + (dim2 * dim3) + (dim1 * dim3));
}

__inline Vector3 getMidPoint(const Box &box)
{
	return 0.5f*(box.p_min + box.p_max);
}

void OutOfCoreTree::subDivideSAHBlusteredVoxels(unsigned int *triIDs, unsigned int left, unsigned int right, unsigned int myIndex, unsigned int nextIndex, int depth)
{
	typedef BVH::FeelEvent FeelEvent;
	static BSPTreeInfo treeStats(numVoxels, 1, treeClusteredVoxels->min, treeClusteredVoxels->max);

	if( depth > maxDepthClusteredVoxels )
	{
		maxDepthClusteredVoxels = depth;
	}

	BSPArrayTreeNodePtr lChild = (BSPArrayTreeNodePtr)((char *)treeClusteredVoxels + ((unsigned int)(((nextIndex) * BVHNODE_BYTES) << 0) & ~3) );
	BSPArrayTreeNodePtr rChild = (BSPArrayTreeNodePtr)((char *)treeClusteredVoxels + ((unsigned int)(((nextIndex + 1) * BVHNODE_BYTES) << 0) & ~3) );
	BSPArrayTreeNodePtr node = (BSPArrayTreeNodePtr)((char *)treeClusteredVoxels + ((unsigned int)(myIndex * BVHNODE_BYTES) & ~3) );

	if(left == 0 && right < 1)
	{
		// contain only single node - root
		int nTris = right - left + 1;
		node->indexCount = MAKECHILDCOUNT(1);;
		node->indexOffset = 0;

		treeStats.numLeafs++;
		treeStats.sumDepth += depth;
		treeStats.sumTris += nTris;
		return;
	}

	int bestAxis;
	int k, i, nE, temp, nL, nR, nT;
	float bestCoor, bestVal, val;
	int curL, curR;
	float SA, *SL, *SR;
	FeelEvent *events;
	Vector3 p1, p2;
		
	nT = right - left + 1;
	events =new FeelEvent[ nT + 2 ];
	nE = nT;
	SL = new float[ nE + 2 ];
	SR = new float[ nE + 2 ];
	
	bestAxis = -1;
	bestVal = FLT_MAX;

	SA = surfaceArea( node->max[0] - node->min[0] , node->max[1] - node->min[1] , node->max[2] - node->min[2] );

	/*bool isSAzero = ( fabs( SA - 0.0f ) == 0.0f );
	if( isSAzero )
	{
		node->indexCount = MAKECHILDCOUNT(nT);
		node->indexOffset = curIndex;
		treeStats.numLeafs++;
		treeStats.sumDepth += depth;
		treeStats.sumTris += nT;
		for(int i=left;i<=right;i++)
			indexlists[curIndex++] = GETTRIIDX(triIDs->at(i));

		delete [] events;
		delete [] SL;
		delete [] SR;
		return true;
	}
	*/

	for(k = 0 ; k <= 2 ; k++ )		// for all AXIS  0 : X   1 : Y    2 : Z
	{
		for(i=0;i<nE;i++)
		{
			Box &box = voxelBBList[triIDs[left + i]];
			events[ i ].coor = getMidPoint(box).e[k];
			events[ i ].triIndex = left + i;
		}

		std::sort( &events[0] , &events[nE] );

		
		p1.e[0] = FLT_MAX;		p1.e[1] = FLT_MAX;		p1.e[2] = FLT_MAX;
		p2.e[0] = -FLT_MAX;		p2.e[1] = -FLT_MAX;		p2.e[2] = -FLT_MAX;
		for(i=0;i<nE;i++)
		{
			Box &box = voxelBBList[triIDs[events[i].triIndex]];
			
			updateBB( p1 , p2 , box.p_min);
			updateBB( p1 , p2 , box.p_max);
			
			SL[ i ] = surfaceArea( p2[0] - p1[0] , p2[1] - p1[1] , p2[2] - p1[2] );
		}

		p1.e[0] = FLT_MAX;		p1.e[1] = FLT_MAX;		p1.e[2] = FLT_MAX;
		p2.e[0] = -FLT_MAX;		p2.e[1] = -FLT_MAX;		p2.e[2] = -FLT_MAX;
		for(i=nE-1;i>=0;i--)
		{
			Box &box = voxelBBList[triIDs[events[i].triIndex]];

			updateBB( p1 , p2 , box.p_min);
			updateBB( p1 , p2 , box.p_max);

			SR[ i ] = surfaceArea( p2[0] - p1[0] , p2[1] - p1[1] , p2[2] - p1[2] );
		}

		for(i=0;i<=nE-1;i++)
		{
			val = SL[ i ]/ SA * ( (float)(i) ) + SR[ i ] / SA * ( (float)( nT - i ) );

			/*
			if( nT == 294 )
			{
				printf("L : %d  R : %d    SL/SA : %lf   SR/SA : %lf   val : %lf \n",i,nT-i,SL[i]/SA,SR[i]/SA,val);
				getch();
			}
			*/

			

			if( val < bestVal )
			{
				bestVal = val;
				bestAxis = k;
				bestCoor = events[i].coor;
			}
		}
	}

	if(bestAxis == -1)
	{
		// cannot find best axis
		// just median split

		// find biggest axis:
		Vector3 diff = node->max - node->min;
		bestAxis = diff.indexOfMaxComponent();
		bestCoor = .5 * diff[bestAxis] + node->min[bestAxis];
	}

	delete [] events;
	delete [] SL;
	delete [] SR;

	curL = left;
	curR = right;

	for(i=0;i<nT;i++)
	{
		Box &box = voxelBBList[triIDs[curL]];
		val = getMidPoint(box).e[bestAxis];
		if( val < bestCoor )
		{
			curL++;
		}
		else
		{
			temp = triIDs[curL];
			triIDs[curL] = triIDs[curR];
			triIDs[curR] = temp;
			curR--;
		}
	}

	nL = curL - left;
	nR = nT - nL;

	/*
	if( nL == 0 || nR == 0 )
	{
		node->indexCount = MAKECHILDCOUNT(nT);
		node->indexOffset = curIndex;
		treeStats.numLeafs++;
		treeStats.sumDepth += depth;
		treeStats.sumTris += nT;
		for(int i=left;i<=right;i++)
		{
			indexlists[curIndex++] = GETTRIIDX(triIDs->at(i));
			progBuildTree->step();
		}
		return true;
	}
	*/

	if( nL == 0 || nR == 0 )
	{
		nL = (nT+1) / 2;
		nR = nT - nL;
	}

	node->children = ((nextIndex) * BVHNODE_BYTES >> 3 ) | bestAxis;
	#ifndef _USE_CONTI_NODE
	node->children2 = ((nextIndex+1) * BVHNODE_BYTES >> 3 );
	#else
	node->children2 = 0;
	#endif

	treeStats.numNodes += 2;

	float BB_min_limit[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
	float BB_max_limit[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};

	lChild->min.set(BB_min_limit);
	lChild->max.set(BB_max_limit);
	for(int i=left;i<=(left+nL-1);i++)
	{
		Box &box = voxelBBList[triIDs[i]];
		updateBB( lChild->min , lChild->max , box.p_min);
		updateBB( lChild->min , lChild->max , box.p_max);
	}

	if( nL <= 1 )
	{
		lChild->indexCount = MAKECHILDCOUNT(nL);
		lChild->indexOffset = triIDs[left];

		treeStats.numLeafs++;
		treeStats.sumDepth += depth;
		treeStats.sumTris += nL;
	}
	else
	{
		subDivideSAHBlusteredVoxels( triIDs , left , left + nL -1 , nextIndex , nextIndex + 2 , depth + 1 );
	}

	rChild->min.set(BB_min_limit);
	rChild->max.set(BB_max_limit);
	for(i=left+nL;i<=right;i++)
	{
		Box &box = voxelBBList[triIDs[i]];
		updateBB( rChild->min , rChild->max , box.p_min);
		updateBB( rChild->min , rChild->max , box.p_max);
	}

	if( nR <= 1 )
	{
		rChild->indexCount = MAKECHILDCOUNT(nR);
		rChild->indexOffset = triIDs[left+nL];

		if(left+nL != right)
		{
			printf("left+nL != right\n");
			exit(-1);
		}

		treeStats.numLeafs++;
		treeStats.sumDepth += depth;
		treeStats.sumTris += nR;
	}
	else
	{
		subDivideSAHBlusteredVoxels( triIDs , left + nL , right , nextIndex + 1 , treeStats.numNodes , depth+1);
	}
}
