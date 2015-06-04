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


typedef stdext::hash_map<__int64, unsigned int> ColorTable;
typedef ColorTable::iterator ColorTableIterator;


class OutOfCoreTree
{
public:
	OutOfCoreTree();
	~OutOfCoreTree();

	/**
	* Main method for building the tree and OoC representation
	**/
	bool build(const char *outputPath, const char *fileName, bool useModelTransform = false);


	// sungeui

	// Main caller to compute simplification representation
	bool ComputeSimpRep (void);
	bool ComputeSimpRep (unsigned int NodeIdx, Vector3 & BBMin, Vector3 & BBMaxi,
									float MinVolume);

	// taejoon
	// functions for small models
	bool buildSmall(const char *outputPath, const char *fileName, bool useModelTransform = false);
	void initSmall();
	void writeModelInfo();
	bool firstVertexPassSmall(const char *fileName, float *mat = NULL);
	bool secondVertexPassSmall(const char *fileName, float *mat = NULL);
	bool bridge2to3();
	bool thirdVertexPassSmall(const char *fileName, int curIter, int matIndex = -1);
	void finalizeMeshPass();
	bool buildBVHSmall();

	bool buildSmallMulti(const char *outputPath, const char *fileListName, bool useModelTransform, bool useFileMat, const char *mtlFileName = NULL);
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

protected:	

	/**
	 * First vertex pass, counts vertices & faces, determines
	 * bounding box
	 **/
	bool firstVertexPass();

	/**
	 * Calculate resolution of voxel grid from scene information
	 **/
	void calculateVoxelResolution();

	/*	*
	 * Sort vertices and triangles into voxels
	 **/
	bool secondVertexPass();

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
	 * Build BVH for each voxel.
	 **/
	bool buildVoxelBVH();

	/**
	 * Build high-level BVH from the voxel
	 * structure, then merge individual voxel BVH
	 * into the high-level tree.
	 **/
	bool buildHighLevelBVH();

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
	std::string getVTriangleFileName(unsigned int voxelNr) {
		return std::string(outDirName) + "/tri_" + toString(voxelNr,5,'0') + ".ooc";
	}
	std::string getVTriangleFileName() {
		return std::string(outDirName) + "/tri_";
	}
	std::string getTriangleIdxFileName() {
		return std::string(outDirName) + "/triidx_";
	}

	std::string getTriangleIdxFileName(unsigned int voxelNr) {
		return std::string(outDirName) + "/triidx_" + toString(voxelNr,5,'0') + ".ooc";
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
	bool useModelTransform;
	bool useFileMat;
	float mat[16];

private:
};

class DOEColoring
{
public:
	// For DOE coloring
	#define COLORS_IN_MAP 4
	float maxDistForDOE;
	float zMidForDOE;
	float scaleForDOE;

	unsigned char bluemap(float dist)
	{
		static int blues[COLORS_IN_MAP] = {14, 127, 227, 211};

		// rainbow-ish
		// blues[0]=42;
		// blues[1]=46;
		// blues[2]=124;
		// blues[3]=187;

		// softer
		//blues[0]=14;
		//blues[1]=127;
		//blues[2]=227;
		//blues[3]=211;

		int index = 0;
		float perc=dist/maxDistForDOE;
		if (perc > 1)
			perc = 1;
		while (perc>scaleForDOE)
		{
			perc-=scaleForDOE;
			index++;
		}

		float val = perc/scaleForDOE*blues[index+1]+(1-perc/scaleForDOE)*blues[index];
		return (unsigned char) floor(val);
	}

	unsigned char greenmap(float dist)
	{
		static int greens[COLORS_IN_MAP] = {198, 227, 149, 41};

		// rainbow-ish
		// greens[0]=42;
		// greens[1]=169;
		// greens[2]=187;
		// greens[3]=178;

		// softer colors
		//greens[0]=198;
		//greens[1]=227;
		//greens[2]=149;
		//greens[3]=41;

		int index = 0;
		float perc=dist/maxDistForDOE;
		if (perc > 1)
			perc = 1;
		while (perc>scaleForDOE)
		{
			perc-=scaleForDOE;
			index++;
		}

		float val = perc/scaleForDOE*greens[index+1]+(1-perc/scaleForDOE)*greens[index];
		return (unsigned char) floor(val);
	}

	unsigned char redmap(float dist)
	{
		static int reds[COLORS_IN_MAP] = {211, 39, 39, 49};

		// rainbow-ish
		// reds[0]=211;
		// reds[1]=230;
		// reds[2]=35;
		// reds[3]=46;

		// softer colors
		//reds[0]=211;
		//reds[1]=39;
		//reds[2]=39;
		//reds[3]=49;

		int index = 0;
		float perc=dist/maxDistForDOE;
		if (perc > 1)
			perc = 1;
		while (perc>scaleForDOE)
		{
			perc-=scaleForDOE;
			index++;
		}

		float val = perc/scaleForDOE*reds[index+1]+(1-perc/scaleForDOE)*reds[index];
		return (unsigned char) floor(val);
	}

	void getDOEColor(float z, rgb &color)
	{
		float dist = fabs(z-zMidForDOE);

		color.data[0] = redmap(dist)/255.0f;
		color.data[1] = greenmap(dist)/255.0f;
		color.data[2] = bluemap(dist)/255.0f;
	}
};

#endif