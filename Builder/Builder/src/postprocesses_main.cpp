// compress_main.cpp : Defines the entry point for the console application.
//
#include <iostream>
#include <stdio.h>

#include "DirectTriIndex.h"
#include "RearrangeVoxels.h"
#include "ExtractBB.h"
#include "MergeGeometry.h"
#include "HCCMesh.h"
#include "GeometryConverter.h"
#include "Voxelize.h"
#include "MTLGenerator.h"
#include "OptionManager.h"

#define REMOVE_INDEX	0x1
#define HCCMESH			0x2
#define MTL				0x4
#define ASVO			0x8
#define GPU				0x10

using namespace std;

int main(int argc, char ** argv)
{
	char fileName[255];
	char filePath[255];
	char outputPath[255];

	printf ("Usage: %s file \"REMOVE_INDEX | HCCMESH | MTL | ASVO | GPU\" ASVO_options\n", argv [0]);

	sprintf(outputPath, ".");

	if (argc >= 2) {
		strcpy(fileName, argv[1]);

	}
	else
		sprintf(fileName, "test.ply");

	OptionManager *opt = OptionManager::getSingletonPtr();
	//const char *baseDirName = opt->getOption("global", "scenePath", "");

	char tempFileName[MAX_PATH];
	int pos = 0;
	for(int i=strlen(fileName)-1;i>=0;i--)
	{
		if(fileName[i] == '/' || fileName[i] == '\\')
		{
			pos = i+1;
			break;
		}
	}
	strcpy(tempFileName, &fileName[pos]);

	sprintf(filePath, "%s\\%s.ooc", outputPath, tempFileName);
	printf("filePath = %s\n", filePath);

	unsigned int flag = REMOVE_INDEX | GPU;

	if(argc >= 3)
	{
		flag = atoi(argv[2]);

		if(flag == 0)
		{
			if(strstr(argv[2], "REMOVE_INDEX")) flag |= REMOVE_INDEX;
			if(strstr(argv[2], "HCCMESH")) flag |= HCCMESH;
			if(strstr(argv[2], "MTL")) flag |= MTL;
			if(strstr(argv[2], "ASVO")) flag |= ASVO;
			if(strstr(argv[2], "GPU")) flag |= GPU;
		}

		printf("Process ");
		if((flag & REMOVE_INDEX) == REMOVE_INDEX) printf("Removing indices, ");
		if((flag & HCCMESH) == HCCMESH) printf("HCCMesh generation, ");
		if((flag & MTL) == MTL) printf("MTL generation, ");
		if((flag & ASVO) == ASVO) printf("ASVO generation, ");
		if((flag & GPU) == GPU) printf("GPU conversion, ");
		printf("\n");
	}
	
	if((flag & REMOVE_INDEX) == REMOVE_INDEX)
	{
		cout << "Remove unnecessary indirected triangle indices." << endl;
		DirectTriIndex *dti = new DirectTriIndex();
		dti->Do(filePath);
		//dti->DoMulti(filePath);
		delete dti;
	}

	/*
	cout << "Make HCCMesh2 representation." << endl;
	HCCMesh *hm = new HCCMesh();
	hm->convertHCCMesh2(fileName);
	delete hm;
	*/

	if((flag & HCCMESH) == HCCMESH)
	{
		cout << "Make HCCMesh representation." << endl;
		HCCMesh *hm = new HCCMesh();
		hm->generateTemplates();
		hm->convertHCCMesh(filePath);
		delete hm;
	}

	if((flag & MTL) == MTL)
	{
		cout << "Generate MTL file." << endl;
		MTLGenerator *mtl = new MTLGenerator();
		mtl->Do(filePath);
		delete mtl;
	}

	if((flag & ASVO) == ASVO)
	{
		cout << "Generating ASVO..." << endl;
		Voxelize *voxelize = new Voxelize();
		if(argc >= 5)
		{
			if(argc >= 6)
				voxelize->Do(filePath, atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
			else
				voxelize->Do(filePath, atoi(argv[3]), atoi(argv[4]));
		}
		else
			voxelize->Do(filePath);
		delete voxelize;
	}

	if((flag & GPU) == GPU)
	{
		cout << "Convert triangle and vertex files into files that suitable for GPU." << endl;
		GeometryConverter *gc = new GeometryConverter();
		//gc->convertCluster(filePath);
		gc->convert(filePath);
		//gc->convert(filePath, GeometryConverter::NEW_OOC, GeometryConverter::SIMP);
		delete gc;
	}
	/*
	cout << "Rearrange voxels into sequential order." << endl;
	RearrangeVoxels *rav = new RearrangeVoxels();
	rav->Do(filePath);
	delete rav;

	cout << "Extract bounding box of each voxel." << endl;
	ExtractBB *ebb = new ExtractBB();
	ebb->Do(filePath);
	delete ebb;

	cout << "Merge geometry data into single voxel." << endl;
	MergeGeometry *mg = new MergeGeometry();
	mg->Do(filePath);
	delete mg;
	*/

	return 0;
}


