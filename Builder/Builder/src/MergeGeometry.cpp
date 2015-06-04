#include <windows.h>
#include <string.h>
#include <stdio.h>
#include "OptionManager.h"
#include "BVHNodeDefine.h"
#include "Triangle.h"
#include "Vertex.h"
#include <io.h>

#include "MergeGeometry.h"

#include "Progression.h"
#include "helpers.h"
#include "BVH.h"
#include "Grid.h"

int MergeGeometry::Do(const char* filepath)
{
	char fileNameScene[MAX_PATH];
	sprintf(fileNameScene, "%s/scene", filepath);
	FILE *fpScene;
	fpScene = fopen(fileNameScene, "rb");
	int numVertices, numFaces;
	Grid grid;
	unsigned int numUsedBoxes;
	fread(&numVertices, sizeof(int), 1, fpScene);
	fread(&numFaces, sizeof(int), 1, fpScene);
	fread(&grid, sizeof(Grid), 1, fpScene);
	fread(&numUsedBoxes, sizeof(unsigned int), 1, fpScene);
	fclose(fpScene);

	__int64 fileSizeVoxels = 0;
	Progression prog("Merge geometry", numUsedBoxes, 100);
	for(int boxNr=0;boxNr<numUsedBoxes;boxNr++)
	{
		char fileNameTri[MAX_PATH];
		char fileNameTriIdx[MAX_PATH];
		char fileNameVert[MAX_PATH];
		char fileNameTree[MAX_PATH];

		char fileNameVoxel[MAX_PATH];

		sprintf(fileNameTri, "%s/tri_%05d.ooc", filepath, boxNr);
		sprintf(fileNameTriIdx, "%s/triidx_%05d.ooc", filepath, boxNr);
		sprintf(fileNameVert, "%s/vert_%05d.ooc", filepath, boxNr);
		sprintf(fileNameTree, "%s/BVH_%05d.ooc", filepath, boxNr);

		sprintf(fileNameVoxel, "%s/voxel_%05d.ooc", filepath, boxNr);

		FILE *fpTri = fopen(fileNameTri, "rb");
		FILE *fpVert = fopen(fileNameVert, "rb");
		FILE *fpTree = fopen(fileNameTree, "rb");

		FILE *fpVoxel = fopen(fileNameVoxel, "wb");

		int sizeTri = filelength(fileno(fpTri));
		int sizeVert = filelength(fileno(fpVert));
		int sizeTree = filelength(fileno(fpTree));
		int numTri = sizeTri/sizeof(Triangle);
		int numVert = sizeVert/sizeof(Vertex);
		int numTree = sizeTree/sizeof(BSPArrayTreeNode);

		Triangle *tri = new Triangle[numTri];
		Vertex *vert = new Vertex[numVert];
		BSPArrayTreeNode *tree = new BSPArrayTreeNode[numTree];

		if(fread(tri, sizeof(Triangle), numTri, fpTri) != numTri)
			cout << "file read error! [triangle]" << endl;
		if(fread(vert, sizeof(Vertex), numVert, fpVert) != numVert)
			cout << "file read error! [vertex]" << endl;
		if(fread(tree, sizeof(BSPArrayTreeNode), numTree, fpTree) != numTree)
			cout << "file read error! [tree]" << endl;

		// write header
		fwrite(&numTri, sizeof(int), 1, fpVoxel);
		fwrite(&numVert, sizeof(int), 1, fpVoxel);
		fwrite(&numTree, sizeof(int), 1, fpVoxel);

		// write geometry
		if(fwrite(tri, sizeof(Triangle), numTri, fpVoxel) != numTri)
			cout << "file write error! [triangle]" << endl;
		if(fwrite(vert, sizeof(Vertex), numVert, fpVoxel) != numVert)
			cout << "file write error! [vertex]" << endl;
		if(fwrite(tree, sizeof(BSPArrayTreeNode), numTree, fpVoxel) != numTree)
			cout << "file write error! [tree]" << endl;

		delete[] tri;
		delete[] vert;
		delete[] tree;

		fclose(fpTri);
		fclose(fpVert);
		fclose(fpTree);
		fclose(fpVoxel);

		unlink(fileNameTri);
		unlink(fileNameTriIdx);
		unlink(fileNameVert);
		unlink(fileNameTree);

		fileSizeVoxels += 3*sizeof(int) + sizeTri + sizeVert + sizeTree;
		prog.step();
	}
	cout << endl;

	fpScene = fopen(fileNameScene, "wb");
	fwrite(&numVertices, sizeof(int), 1, fpScene);
	fwrite(&numFaces, sizeof(int), 1, fpScene);
	fwrite(&grid, sizeof(Grid), 1, fpScene);
	fwrite(&numUsedBoxes, sizeof(unsigned int), 1, fpScene);
	fwrite(&fileSizeVoxels, sizeof(__int64), 1, fpScene);
	fclose(fpScene);
	return 1;
}