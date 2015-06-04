#include "stdafx.h"
#include "common.h"

// Quantization
const int g_MaxQuanIdx = (2 << ERR_BITs) - 1;

bool ModelInstance::loadTreeFromFiles(const char* filename) {	
	LogManager *log = LogManager::getSingletonPtr();	
	OptionManager *opt = OptionManager::getSingletonPtr();
	char filenameNodes[MAX_PATH], fileNameIndices[MAX_PATH];
	char output[1000];
	char header[100];
	char bspfilestring[50];
	size_t ret;
	#ifdef _USE_RACBVH
	sprintf(filenameNodes, "%s.cmp", filename);
	#else
	sprintf(filenameNodes, "%s.node", filename);
	#endif
	sprintf(fileNameIndices, "%s.idx", filename);

	FILE *fp = fopen(filename, "rb");
	FILE *fpNodes = fopen(filenameNodes, "rb");
	FILE *fpIndices = fopen(fileNameIndices, "rb");

	if (fp == NULL) {
		sprintf(output, "Could not open BSP tree file '%s'!", filename);
		log->logMessage(LOG_WARNING, output);
		return false;
	}

	if (fpNodes == NULL) {
		sprintf(output, "Could not open BSP tree file '%s'!", filenameNodes);
		log->logMessage(LOG_WARNING, output);
		fclose(fp);
		return false;
	}

	if (fpIndices == NULL) {
		sprintf(output, "Could not open BSP tree file '%s'!", fileNameIndices);
		log->logMessage(LOG_WARNING, output);
		fclose(fp);
		fclose(fpNodes);
		return false;
	}

	sprintf(output, "Loading BSP tree from files ('%s')...", filename);
	log->logMessage(LOG_INFO, output);

	ret = fread(header, 1, BSP_FILEIDSTRINGLEN + 1, fp);
	if (ret != (BSP_FILEIDSTRINGLEN + 1)) {
		sprintf(output, "Could not read header from BSP tree file '%s', aborting. (empty file?)", filename);
		log->logMessage(LOG_ERROR, output);
		return false;
	}

	// test header format:
	strcpy(bspfilestring, BSP_FILEIDSTRING);
	unsigned int i;
	for (i = 0; i < BSP_FILEIDSTRINGLEN; i++) {
		if (header[i] != bspfilestring[i]) {
			printf(output, "Invalid BSP tree header, aborting. (expected:'%c', found:'%c')", bspfilestring[i], header[i]);
			log->logMessage(LOG_ERROR, output);
			return false;		
		}
	}

	// test file version:
	if (header[BSP_FILEIDSTRINGLEN] != BSP_FILEVERSION) {
		printf(output, "Wrong BSP tree file version (expected:%d, found:%d)", BSP_FILEVERSION, header[BSP_FILEIDSTRINGLEN]);
			log->logMessage(LOG_ERROR, output);
		return false;		
	}

	// format correct, read in full BSP tree info structure:

	// write count of nodes and tri indices:
	ret = fread(&treeStats, sizeof(BSPTreeInfo), 1, fp);
	if (ret != 1) {
		sprintf(output, "Could not read tree info header!");
		log->logMessage(LOG_ERROR, output);
		return false;
	}

	#ifdef USE_LOD
		// sungeui start ----------------------
		printf ("Size of BSP = %d\n", sizeof(BSPTreeInfo));
		printf ("Min %f, ErrRatio %f, # of Bits = %d\n", treeStats.m_MinErr, treeStats.m_ErrRatio, 
			treeStats.m_NumAllowedBit);
		// generate table of quantized errors
		if (treeStats.m_NumAllowedBit != ERR_BITs) {
			printf ("Error: precomputed # of quantization bit is not matched with runtime one.\n");
			exit (-1);
		}

		g_QuanErrs = new float[g_MaxQuanIdx];

		for (i = 0; i < g_MaxQuanIdx;i++)
		{
			g_QuanErrs [i] = treeStats.m_MinErr*pow (treeStats.m_ErrRatio, float (i));
			//printf ("Err = %f\n", g_QuanErrs[i]);
		}
		// sungeui end ------------------------
	#endif

	#ifdef _USE_OOC

	#ifdef _USE_OOC_DIRECTMM
		tree = (BSPArrayTreeNode *) allocateFullMemoryMap(filenameNodes);
		indexlist = (unsigned int *) allocateFullMemoryMap(fileNameIndices);
	#else
		tree = new OOC_BSPNODE_FILECLASS<BSPArrayTreeNode>(filenameNodes, 										 
							1024*1024*opt->getOptionAsInt("ooc", "maxCacheMemTreeNodesMB", 512),
							1024*opt->getOptionAsInt("ooc", "cacheEntrySizeTreeNodesKB", 1024 * 4));
		#if defined(_USE_RACBVH) && defined(_USE_RACM) && defined(_USE_TRI_3_TYPE_ENCODING)
		tree->g_pMesh = pRACM;
		#endif

		indexlist = new OOC_BSPIDX_FILECLASS<unsigned int>(fileNameIndices, 
							1024*1024*opt->getOptionAsInt("ooc", "maxCacheMemTreeListsMB", 256),
							1024*opt->getOptionAsInt("ooc", "cacheEntrySizeTreeListsKB", 1024 * 4));
	#endif

	#else
		sprintf(output, "Allocating memory...");
		log->logMessage(LOG_INFO, output);

		tree = new BSPArrayTreeNode[treeStats.numNodes];	
		indexlist = new unsigned int[treeStats.sumTris];

		// read tree node array:
		sprintf(output, "  ... reading %d tree nodes ...", treeStats.numNodes);
		log->logMessage(LOG_INFO, output);
		ret = fread(tree, sizeof(BSPArrayTreeNode), treeStats.numNodes, fpNodes);

		if (ret != treeStats.numNodes) {
			sprintf(output, "Could only read %u nodes, expecting %u!", ret, treeStats.numNodes);
			log->logMessage(LOG_ERROR, output);
			return false;
		}

		// read tri index array
		sprintf(output, "  ... reading %d tri indices ...", treeStats.sumTris);
		log->logMessage(LOG_INFO, output);
		ret = fread(indexlist, sizeof(int), treeStats.sumTris, fpIndices);

		if (ret != treeStats.sumTris) {
			sprintf(output, "Could only read %u indices, expecting %u!", ret, treeStats.sumTris);
			log->logMessage(LOG_ERROR, output);
			return false;
		}

		sprintf(output, "  done!");
		log->logMessage(LOG_INFO, output);
	#endif

	fclose(fp);
	fclose(fpNodes);
	fclose(fpIndices);

	bb[0] = treeStats.min + translate_world;
	bb[1] = treeStats.max + translate_world;
	boundingBoxGenerated = true;

	return true;
}	

#if !defined(_USE_OOC) || defined(_USE_OOC_DIRECTMM)
bool ModelInstance::loadTreeFromFile(const char* filename) {	
	LogManager *log = LogManager::getSingletonPtr();	
	char output[1000];
	char header[100];
	char bspfilestring[50];
	size_t ret;
	FILE *fp = fopen(filename, "rb");

	if (fp == NULL) {
		sprintf(output, "Could not open BSP tree file '%s'!", filename);
		log->logMessage(LOG_WARNING, output);
		return false;
	}

	sprintf(output, "Loading BSP tree from file '%s'...", filename);
	log->logMessage(LOG_INFO, output);

	ret = fread(header, 1, BSP_FILEIDSTRINGLEN + 1, fp);
	if (ret != (BSP_FILEIDSTRINGLEN + 1)) {
		sprintf(output, "Could not read header from BSP tree file '%s', aborting. (empty file?)", filename);
		log->logMessage(LOG_ERROR, output);
		return false;
	}

	// test header format:
	strcpy(bspfilestring, BSP_FILEIDSTRING);
	for (unsigned int i = 0; i < BSP_FILEIDSTRINGLEN; i++)
		if (header[i] != bspfilestring[i]) {
			printf(output, "Invalid BSP tree header, aborting. (expected:'%c', found:'%c')", bspfilestring[i], header[i]);
			log->logMessage(LOG_ERROR, output);
			return false;		
		} 

	// test file version:
	if (header[BSP_FILEIDSTRINGLEN] != BSP_FILEVERSION) {
		printf(output, "Wrong BSP tree file version (expected:%d, found:%d)", BSP_FILEVERSION, header[BSP_FILEIDSTRINGLEN]);
		log->logMessage(LOG_ERROR, output);
		return false;		
	}

	// format correct, read in full BSP tree info structure:

	// write count of nodes and tri indices:
	ret = fread(&treeStats, sizeof(BSPTreeInfo), 1, fp);
	if (ret != 1) {
		sprintf(output, "Could not read tree info header!");
		log->logMessage(LOG_ERROR, output);
		return false;
	}

	
	sprintf(output, "Allocating memory...");
	log->logMessage(LOG_INFO, output);

	tree = new BSPArrayTreeNode[treeStats.numNodes];	
	indexlist = new unsigned int[treeStats.sumTris];

	// write tree node array:
	sprintf(output, "  ... reading %d tree nodes ...", treeStats.numNodes);
	log->logMessage(LOG_INFO, output);
	ret = fread(tree, sizeof(BSPArrayTreeNode), treeStats.numNodes, fp);

	if (ret != treeStats.numNodes) {
		sprintf(output, "Could only read %u nodes, expecting %u!", ret, treeStats.numNodes);
		log->logMessage(LOG_ERROR, output);
		return false;
	}

	// write tri index array
	sprintf(output, "  ... reading %d tri indices ...", treeStats.sumTris);
	log->logMessage(LOG_INFO, output);
	ret = fread(indexlist, sizeof(int), treeStats.sumTris, fp);

	if (ret != treeStats.sumTris) {
		sprintf(output, "Could only read %u indices, expecting %u!", ret, treeStats.sumTris);
		log->logMessage(LOG_ERROR, output);
		return false;
	}

	sprintf(output, "  done!");
	log->logMessage(LOG_INFO, output);
	fclose(fp);

	bb[0] = treeStats.min + translate_world;
	bb[1] = treeStats.max + translate_world;
	boundingBoxGenerated = true;

	return true;
}	

bool ModelInstance::saveTreeToFile(const char* filename) {	
	LogManager *log = LogManager::getSingletonPtr();	
	char output[255];
	size_t  ret;
	FILE *fp = fopen(filename, "wb");

	if (fp == NULL) {
		sprintf(output, "Could not open BSP tree file '%s'!", filename);
		log->logMessage(LOG_ERROR, output);
		return false;
	}

	sprintf(output, "Saving BSP tree to file '%s'...", filename);
	log->logMessage(LOG_INFO, output);

	// write header and version:
	fwrite(BSP_FILEIDSTRING, 1, BSP_FILEIDSTRINGLEN, fp);
	fputc(BSP_FILEVERSION, fp);

	// write stats:
	fwrite(&treeStats, sizeof(BSPTreeInfo), 1, fp);

	// write tree node array:
	sprintf(output, "  ... writing %d tree nodes ...", treeStats.numNodes);
	log->logMessage(LOG_INFO, output);
	ret = fwrite(tree, sizeof(BSPArrayTreeNode), treeStats.numNodes, fp);

	if (ret != treeStats.numNodes) {
		sprintf(output, "Could only write %u nodes, expecting %u!", ret, treeStats.numNodes);
		log->logMessage(LOG_ERROR, output);
		return false;
	}

	// write tri index array
	sprintf(output, "  ... writing %d tri indices ...", treeStats.sumTris);
	log->logMessage(LOG_INFO, output);
	ret = fwrite(indexlist, sizeof(unsigned int), treeStats.sumTris, fp);

	if (ret != treeStats.sumTris) {
		sprintf(output, "Could only write %u indices, expecting %u!", ret, treeStats.sumTris);
		log->logMessage(LOG_ERROR, output);
		return false;
	}

	sprintf(output, "  done!");
	log->logMessage(LOG_INFO, output);
	fclose(fp);

	return true;
}
#endif // !_USE_OOC

void ModelInstance::instanceFrom(ModelInstance *other) {
	if(!useRACM)
	{
		trilist = other->trilist;
		vertices = other->vertices;
	}
	if(useRACM)
	{
		#ifdef _USE_RACM
		pRACM = other->pRACM;
		#endif
	}
	lodlist = other->lodlist;
	tree = other->tree;
	indexlist = other->indexlist;
	materiallist = other->materiallist;
	g_QuanErrs = other->g_QuanErrs;
	//vertexNormals = other->vertexNormals;

	nVerts = other->nVerts;
	nTris = other->nTris;
	nLODs = other->nLODs;

	modelFileName = other->modelFileName;

	i1 = other->i1;
	i2 = other->i2;
	coordToU = other->coordToU;
	coordToV = other->coordToV;

	memcpy(&treeStats, &other->treeStats, sizeof(BSPTreeInfo));

	isInstanced = true;
	if (other->boundingBoxGenerated) {
		bb[0] = (other->bb[0] - other->translate_world) + translate_world;
		bb[1] = (other->bb[1] - other->translate_world) + translate_world;

		boundingBoxGenerated = other->boundingBoxGenerated;
	}		
}

void ModelInstance::animate(float timeDifference) {
	if (currentAnimationFrame > animationFrames) {
		bb[0] -= translate_world;
		bb[0] += translate_world_start;
		bb[1] -= translate_world;
		bb[1] += translate_world_start;
		translate_world = translate_world_start;
		timeDifference = timeDifference + (currentAnimationFrame - animationFrames);
		currentAnimationFrame = (currentAnimationFrame - animationFrames);
	}
	
	Vector3 translation = translate_animate * timeDifference;
	bb[0] += translation;
	bb[1] += translation;
	translate_world += translation;	
	currentAnimationFrame += timeDifference;	
}

void ModelInstance::transformObj(const Matrix& tranMat)
{
	// step1. current Transform 갱신


	// !!Note : Original bb 를 가지고 있어야, Error 를 줄일 수 있다.^^;;
	// step2. object 의 bb 와 current Transform 를 이용해서 새로운 bb 를 구함
}