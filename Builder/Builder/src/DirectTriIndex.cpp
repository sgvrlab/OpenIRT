#include <windows.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "OptionManager.h"
#include "BVHNodeDefine.h"
#include "Triangle.h"
#include <io.h>

#include "DirectTriIndex.h"

#include "Progression.h"
#include "helpers.h"

int DirectTriIndex::Do(const char* filepath)
{
	TimerValue start, end;
	start.set();

	char fileNameTri[MAX_PATH];
	char fileNameTemp[MAX_PATH];
	char fileNameIdx[MAX_PATH];
	sprintf(fileNameTri, "%s/tris.ooc", filepath);
	sprintf(fileNameTemp, "%s/tris_new.ooc", filepath);
	sprintf(fileNameIdx, "%s/BVH.idx", filepath);
	FILE *fpTri, *fpIdx, *fpTemp;
	fpTri = fopen(fileNameTri, "rb");
	fpTemp = fopen(fileNameTemp, "wb");
	fpIdx = fopen(fileNameIdx, "rb");

	__int64 numTris = _filelengthi64(fileno(fpTri))/sizeof(Triangle);

	Progression prog("Reorder triangles", numTris, 100);

	unsigned int idx;
	while(0 != fread(&idx, sizeof(unsigned int), 1, fpIdx))
	{
		Triangle tri;
		_fseeki64(fpTri, ((__int64)idx)*sizeof(Triangle), SEEK_SET);
		fread(&tri, sizeof(Triangle), 1, fpTri);
		fwrite(&tri, sizeof(Triangle), 1, fpTemp);
		prog.step();
	}

	fclose(fpTri);
	fclose(fpTemp);
	fclose(fpIdx);
	remove(fileNameTri);
	rename(fileNameTemp, fileNameTri);

	/*
	char *buf = new char[512*1024*1024];

	for(int i=0;i<651;i++)
	{
		char fileNameSrc[MAX_PATH];
		char fileNameDst[MAX_PATH];
		FILE *fpSrc, *fpDst;
		int size = 0;

		sprintf(fileNameSrc, "%s/BVH_%05d.ooc", filepath, i);
		sprintf(fileNameDst, "E:/tanker_multi_1M.ply.ooc/BVH_%05d.ooc", i);
		fpSrc = fopen(fileNameSrc, "rb");
		fpDst = fopen(fileNameDst, "wb");
		size = filelength(fileno(fpSrc));
		fread(buf, size, 1, fpSrc);
		fwrite(buf, size, 1, fpDst);
		fclose(fpSrc);
		fclose(fpDst);

		sprintf(fileNameSrc, "%s/tri_%05d.ooc", filepath, i);
		sprintf(fileNameDst, "E:/tanker_multi_1M.ply.ooc/tri_%05d.ooc", i);
		fpSrc = fopen(fileNameSrc, "rb");
		fpDst = fopen(fileNameDst, "wb");
		size = filelength(fileno(fpSrc));
		fread(buf, size, 1, fpSrc);
		fwrite(buf, size, 1, fpDst);
		fclose(fpSrc);
		fclose(fpDst);

		sprintf(fileNameSrc, "%s/vert_%05d.ooc", filepath, i);
		sprintf(fileNameDst, "E:/tanker_multi_1M.ply.ooc/vert_%05d.ooc", i);
		fpSrc = fopen(fileNameSrc, "rb");
		fpDst = fopen(fileNameDst, "wb");
		size = filelength(fileno(fpSrc));
		fread(buf, size, 1, fpSrc);
		fwrite(buf, size, 1, fpDst);
		fclose(fpSrc);
		fclose(fpDst);
	}

	delete[] buf;
	*/



	end.set();
	
	float elapsedHours;
	int elapsedMinOfHour;
	double elapsedMinOfHourFrac = modf((end - start)/(float)(60*60), &elapsedHours);
	elapsedMinOfHour = elapsedMinOfHourFrac * 60.0;
	
	cout << "Reordering triangles ended, time = " << (end - start) << "s (" << (int)elapsedHours << " h, " << elapsedMinOfHour << " min)" << endl;

	return 1;
}

#include "BVH.h"
#include "Grid.h"
int DirectTriIndex::DoMulti(const char* filepath)
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

	Progression prog("Rearrange triangles", numVoxels, 100);
	for(int boxNr=0;boxNr<numVoxels;boxNr++)
	{
		prog.step();
		char fileNameTri[MAX_PATH];
		char fileNameTemp[MAX_PATH];
		char fileNameIdx[MAX_PATH];
		char fileNameTree[MAX_PATH];
		char header[200];
		BSPTreeInfo treeStats;
		size_t ret;

		sprintf(fileNameTri, "%s/tri_%05d.ooc", filepath, boxNr);
		sprintf(fileNameTemp, "%s/tri_new_%05d.ooc", filepath, boxNr);
		sprintf(fileNameIdx, "%s/BVH_%05d.ooc", filepath, boxNr);
		sprintf(fileNameTree, "%s/BVH_%05d.node", filepath, boxNr);
		FILE *fpTri, *fpIdx, *fpTemp, *fpTree;

		fpIdx = fopen(fileNameIdx, "rb");

		if(fpIdx == NULL) continue;

		// read header
		ret = fread(header, 1, BSP_FILEIDSTRINGLEN + 1, fpIdx);
		// read count of nodes and tri indices:
		ret = fread(&treeStats, sizeof(BSPTreeInfo), 1, fpIdx);

		// skip tri node array
		//fseek(fpIdx, sizeof(BSPArrayTreeNode)*treeStats.numNodes, SEEK_CUR);
		fpTree = fopen(fileNameTree, "wb");
		for(int i=0;i<treeStats.numNodes;i++)
		{
			BSPArrayTreeNode node;
			fread(&node, sizeof(BSPArrayTreeNode), 1, fpIdx);
			fwrite(&node, sizeof(BSPArrayTreeNode), 1, fpTree);
		}
		fclose(fpTree);

		fpTri = fopen(fileNameTri, "rb");
		fpTemp = fopen(fileNameTemp, "wb");

		unsigned int idx;
		while(0 != fread(&idx, sizeof(unsigned int), 1, fpIdx))
		{
			Triangle tri;
			_fseeki64(fpTri, ((__int64)idx)*sizeof(Triangle), SEEK_SET);
			fread(&tri, sizeof(Triangle), 1, fpTri);
			fwrite(&tri, sizeof(Triangle), 1, fpTemp);
		}

		fclose(fpTri);
		fclose(fpTemp);
		fclose(fpIdx);
		remove(fileNameTri);
		rename(fileNameTemp, fileNameTri);

		// ***** check here
		remove(fileNameIdx);
		rename(fileNameTree, fileNameIdx);
	}
	cout << endl;
	return 1;
}