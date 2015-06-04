#include <windows.h>
#include <string.h>
#include <stdio.h>
#include "OptionManager.h"
#include "BVHNodeDefine.h"
#include <io.h>

#include "ExtractBB.h"

#include "Progression.h"
#include "helpers.h"

#include "Grid.h"

int ExtractBB::Do(const char* filepath)
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

	char fileNameBB[MAX_PATH];
	sprintf(fileNameBB, "%s/BVs", filepath);
	FILE *fpBB;
	fpBB = fopen(fileNameBB, "wb");

	for(int boxNr=0;boxNr<numVoxels;boxNr++)
	{
		char fileNameTree[MAX_PATH];

		sprintf(fileNameTree, "%s/BVH_%05d.ooc", filepath, boxNr);
		FILE *fpTree;

		fpTree = fopen(fileNameTree, "rb");

		BSPArrayTreeNode node;
		fread(&node, sizeof(BSPArrayTreeNode), 1, fpTree);
		fwrite(&node, sizeof(BSPArrayTreeNode), 1, fpBB);

		fclose(fpTree);
	}

	fclose(fpBB);

	return 1;
}