#ifndef OUTOFCORETREE_H
#define OUTOFCORETREE_H

#include <hash_map>
#include <vector>
#include "MeshReader.h"
#include "Grid.h"
#include "common.h"

#include "helpers.h"
#include "OptionManager.h"
#include "Logger.h"
#include "Vector3.h"
#include "Ray.h"
#include "Vertex.h"
#include "Triangle.h"
#include "BufferedOutputs.h"
#include "Materials.h"

#include <xmmintrin.h>


typedef stdext::hash_map<unsigned int, unsigned int> ColorTable;
typedef ColorTable::iterator ColorTableIterator;


#include "bvhnodedefine.h"
class OutOfCoreTree
{
public:
	OutOfCoreTree();
	~OutOfCoreTree();

	/**
	* Main method for building the tree and OoC representation
	**/
	bool build(const char *fileName);


	// sungeui

	// Main caller to compute simplification representation
	bool ComputeSimpRep (void);
	bool ComputeSimpRep (unsigned int NodeIdx, Vector3 & BBMin, Vector3 & BBMaxi,
									float MinVolume);

	// taejoon
	// functions for small models
	bool buildSmall(const char *fileName);
	void initSmall();
	void writeModelInfo();
	bool firstVertexPassSmall(const char *fileName, float *mat = NULL);
	bool secondVertexPassSmall(const char *fileName, float *mat = NULL);
	bool bridge2to3();
	bool thirdVertexPassSmall(const char *fileName, int curIter, int matIndex = -1);
	void finalizeMeshPass();
	bool buildBVHSmall();

	bool buildSmallMulti(const char *fileListName, bool useModelTransform, const char *mtlFileName = NULL);
	bool hasVertexColors;
	bool hasVertexNormals;
	bool hasVertexTextures;
	BufferedOutputs<Vertex> *m_pVertices;
	BufferedOutputs<rgb> *m_pColorList;
	HANDLE m_hFileVertex, m_hFileColor;
	HANDLE m_hMappingVertex, m_hMappingColor;
	Vertex *m_pVertexFile;
	rgb *m_pColorFile;
	BufferedOutputs<Triangle> *m_pTris;
	BufferedOutputs<MaterialDiffuse> *m_pOutputs_mat;
	BufferedOutputs<unsigned int> *m_pOutputs_idx;
	std::vector<unsigned int> m_numVertexList;

	bool temp(const char *fileName);

	std::vector<Box> voxelBBList;
	BSPArrayTreeNode *treeClusteredVoxels;
	int maxDepthClusteredVoxels;
	void buildClusteredVoxels(const char *fileListName, float *mat = NULL);
	void firstVertexPassClusteredVoxels(const std::vector<char *> &fileList, float *mat = NULL);
	void secondVertexPassClusteredVoxels(const std::vector<char *> &fileList, float *mat = NULL);
	void thirdVertexPassClusteredVoxels();
	void buildHighLevelBVHClusteredVoxels();
	void subDivideSAHBlusteredVoxels(unsigned int *triIDs, unsigned int left, unsigned int right, unsigned int myIndex, unsigned int nextIndex, int depth);

protected:	

	/**
	 * First vertex pass, counts vertices & faces, determines
	 * bounding box
	 **/
	bool firstVertexPass();

	/**
	 * Calculate resolution of voxel grid from scene information
	 **/
	void calculateVoxelResolution(Grid &grid, unsigned int numFaces, unsigned int targetVoxelSize);

	/*	*
	 * Sort vertices and triangles into voxels
	 **/
	bool secondVertexPass();

	/*	*
	 * Chunking pass, divide voxels which have too many triangles
	 **/
	bool thirdVertexPass();

	/**
	 * Build kD trees for each voxel.
	 **/
	bool buildVoxelkDTrees();

	/**
	 * Build high-level kD tree from the voxel
	 * structure, then merge individual voxel kD trees
	 * into the high-level tree.
	 **/
	bool buildHighLevelkdTree();

	/**
	 * Distribute vertex into each voxel.
	 **/
	bool distributeVertexs();

	/**
	 * Build BVH for each voxel.
	 **/
	bool buildVoxelBVH();

	/**
	 * Build high-level BVH from the voxel
	 * structure, then merge individual voxel BVH
	 * into the high-level tree.
	 **/
	bool buildHighLevelBVH(bool useVoxelBBList = false);

	/**
	 * Get various filenames
	 **/
	std::string getVertexFileName() {
		return std::string(outDirName) + "/vertex.ooc";
	}
	std::string getVertexIndexFileName() {
		return std::string(outDirName) + "/vertexindex.temp";
	}
	std::string getTriangleFileName() {
		return std::string(outDirName) + "/tris.ooc";
	}
	#if HIERARCHY_TYPE == TYPE_KD_TREE
	std::string getkDTreeFileName() {
		return std::string(outDirName) + "/kdtree";
	}
	std::string getkDTreeFileName(unsigned int voxelNr) {
		return std::string(outDirName) + "/kdtree_" + toString(voxelNr,5,'0') + ".ooc";
	}
	#endif
	#if HIERARCHY_TYPE == TYPE_BVH
	std::string getkDTreeFileName() {
		return std::string(outDirName) + "/BVH";
	}
	std::string getkDTreeFileName(unsigned int voxelNr) {
		return std::string(outDirName) + "/BVH_" + toString(voxelNr,5,'0') + ".ooc";
	}
	std::string getBVHFileName() {
		return std::string(outDirName) + "/BVH";
	}
	std::string getBVHFileName(unsigned int voxelNr) {
		return std::string(outDirName) + "/BVH_" + toString(voxelNr,5,'0') + ".ooc";
	}
	std::string getBVHTestFileName(unsigned int voxelNr) {
		return std::string(outDirName) + "/BVH_" + toString(voxelNr,5,'0') + ".sts";
	}
	#endif
	std::string getVTriangleFileName() {
		return std::string(outDirName) + "/tri_";
	}

	std::string getVTriangleFileName(unsigned int voxelNr) {
		return std::string(outDirName) + "/tri_" + toString(voxelNr,5,'0') + ".ooc";
	}

	std::string getTriangleIdxFileName() {
		return std::string(outDirName) + "/triidx_";
	}

	std::string getTriangleIdxFileName(unsigned int voxelNr) {
		return std::string(outDirName) + "/triidx_" + toString(voxelNr,5,'0') + ".ooc";
	}

	std::string getVVertexFileName() {
		return std::string(outDirName) + "/vert_";
	}

	std::string getVVertexFileName(unsigned int voxelNr) {
		return std::string(outDirName) + "/vert_" + toString(voxelNr,5,'0') + ".ooc";
	}

	std::string getSceneInfoName() {
		return std::string(outDirName) + "/scene";
	}

	std::string getColorListName() {
		return std::string(outDirName) + "/colors.tmp";
	}

	std::string getMaterialListName() {
		return std::string(outDirName) + "/materials.ooc";
	}

	std::string getLogFileName() {
		return std::string(outDirName) + "/RLOD";
	}

	std::string getAVoxelFileName() {
		return std::string(outDirName) + "/AdaptiveVoxels";
	}

	FORCEINLINE void setBB(Vector3 &min, Vector3 &max, Vector3 &init)
	{
		min.e[0] = init.e[0];
		min.e[1] = init.e[1];
		min.e[2] = init.e[2];
		max.e[0] = init.e[0];
		max.e[1] = init.e[1];
		max.e[2] = init.e[2];
	}

	FORCEINLINE void updateBB(Vector3 &min, Vector3 &max, Vector3 &vec)
	{
		min.e[0] = ( min.e[0] < vec.e[0] ) ? min.e[0] : vec.e[0];
		min.e[1] = ( min.e[1] < vec.e[1] ) ? min.e[1] : vec.e[1];
		min.e[2] = ( min.e[2] < vec.e[2] ) ? min.e[2] : vec.e[2];

		max.e[0] = ( max.e[0] > vec.e[0] ) ? max.e[0] : vec.e[0];
		max.e[1] = ( max.e[1] > vec.e[1] ) ? max.e[1] : vec.e[1];
		max.e[2] = ( max.e[2] > vec.e[2] ) ? max.e[2] : vec.e[2];
	}
	
	//
	// Variables:
	//

	PlyReader *reader;
	Grid grid;
	char fileName[255];
	char outDirName[255];

	unsigned int numFaces, numVertices;
	unsigned int numUsedVoxels;
	unsigned int numVoxels;

	typedef stdext::hash_map<int, int> VoxelHashTable;
	typedef VoxelHashTable::iterator VoxelHashTableIterator;

	VoxelHashTable *voxelHashTable;

	/*
	// sungeui

	BSPNodeList	   m_Tree;       // array containing all nodes after construction
	BSPIndexList   m_Indexlists; // array containing all triangle index lists after construction	
	*/

	// taejoon
	Vector3 *voxelMins;
	Vector3 *voxelMaxs;
	Vector3 m_bb_min, m_bb_max;

	ColorTable m_usedColors;
	unsigned int m_materialIndex;
	BufferedOutputs<MaterialDiffuse> * m_outputs_mat;		// this will be also used during simplification

private:
};



#endif