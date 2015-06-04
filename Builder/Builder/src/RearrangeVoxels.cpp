#include <windows.h>
#include <string.h>
#include <stdio.h>
#include "OptionManager.h"
#include "BVHNodeDefine.h"
#include "Triangle.h"
#include "Vertex.h"
#include <io.h>

#include "RearrangeVoxels.h"

#include "Progression.h"
#include "helpers.h"
#include "BVH.h"
#include "Grid.h"

int RearrangeVoxels::Do(const char* filepath)
{
	char fileNameScene[MAX_PATH];
	sprintf(fileNameScene, "%s/scene", filepath);
	FILE *fpScene;
	fpScene = fopen(fileNameScene, "rb");
	int numVertices, numFaces;
	Grid grid;
	unsigned int numVoxels;
	fread(&numVertices, sizeof(int), 1, fpScene);
	fread(&numFaces, sizeof(int), 1, fpScene);
	fread(&grid, sizeof(Grid), 1, fpScene);
	fread(&numVoxels, sizeof(unsigned int), 1, fpScene);
	fclose(fpScene);


	stdext::hash_map<int, int> fileIdxMap;

	unsigned int numUsedBox = 0;
	Progression prog("Rearrange voxels", numVoxels, 100);
	for(int boxNr=0;boxNr<numVoxels;boxNr++)
	{
		prog.step();
		char fileNameTri[MAX_PATH];
		char fileNameTriIdx[MAX_PATH];
		char fileNameVert[MAX_PATH];
		char fileNameTree[MAX_PATH];

		char fileNameTriNew[MAX_PATH];
		char fileNameTriIdxNew[MAX_PATH];
		char fileNameVertNew[MAX_PATH];
		char fileNameTreeNew[MAX_PATH];

		sprintf(fileNameTri, "%s/tri_%05d.ooc", filepath, boxNr);
		sprintf(fileNameTriIdx, "%s/triidx_%05d.ooc", filepath, boxNr);
		sprintf(fileNameVert, "%s/vert_%05d.ooc", filepath, boxNr);
		sprintf(fileNameTree, "%s/BVH_%05d.ooc", filepath, boxNr);

		FILE *fp = fopen(fileNameTri, "rb");
		if(fp == NULL) continue;
		fclose(fp);

		sprintf(fileNameTriNew, "%s/tri_%05d.ooc", filepath, numUsedBox);
		sprintf(fileNameTriIdxNew, "%s/triidx_%05d.ooc", filepath, numUsedBox);
		sprintf(fileNameVertNew, "%s/vert_%05d.ooc", filepath, numUsedBox);
		sprintf(fileNameTreeNew, "%s/BVH_%05d.ooc", filepath, numUsedBox);

		rename(fileNameTri, fileNameTriNew);
		rename(fileNameTriIdx, fileNameTriIdxNew);
		rename(fileNameVert, fileNameVertNew);
		rename(fileNameTree, fileNameTreeNew);

		fileIdxMap[boxNr] = numUsedBox;

		numUsedBox++;
	}
	cout << endl;

	fpScene = fopen(fileNameScene, "wb");
	fwrite(&numVertices, sizeof(int), 1, fpScene);
	fwrite(&numFaces, sizeof(int), 1, fpScene);
	fwrite(&grid, sizeof(Grid), 1, fpScene);
	fwrite(&numUsedBox, sizeof(unsigned int), 1, fpScene);
	fclose(fpScene);

	char fileNameAVoxels[MAX_PATH];
	sprintf(fileNameAVoxels, "%s/AdaptiveVoxels", filepath);
	AdaptiveVoxel aVoxel;
	FILE *fpAVs = fopen(fileNameAVoxels, "rb");
	aVoxel.loadFromFile(fpAVs);
	fclose(fpAVs);

	aVoxel.arrangeFileIdx(fileIdxMap);

	fpAVs = fopen(fileNameAVoxels, "wb");
	aVoxel.saveToFile(fpAVs);
	fclose(fpAVs);
	return 1;
}