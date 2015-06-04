#include <iostream>
#include <stdio.h>
//#include "kDTree.h"
#include "OutOfCoreTree.h"
#include "OoCTreeAccess.h"
#include "OBJLoader.h"

using namespace std;

int main(int argc, char **argv) {
	char outputPath[255];
	char fileName[255];
	char matFileName[MAX_PATH];
	//OutOfCoreTree *tree;
	OoCTreeAccess *tree;

	printf ("Usage: %s [-ifmx] file [mtl file]\n", argv [0]);
	printf ("  -i\t\tbuild in-core-mode\n");
	printf ("  -f\t\tinput file is a file list\n");
	printf ("  -m\t\tapply 4x4 transform matrix to each model (requires file list)\n");
	printf ("  -x\t\assign a material index to each model\n");

	sprintf(outputPath, ".");

	if (argc >= 2)
		strcpy(fileName, argv[1]);
	else
		sprintf(fileName, "test.ply");

	bool useFileList = false;
	bool useModelTransform = false;
	bool useMTL = false;
	bool useIncore = false;
	bool useScaler = false;
	bool useFileMat = false;
	if (argc >= 3)
	{
		if(strstr(argv[1], "i") != NULL) useIncore = true;
		if(strstr(argv[1], "f") != NULL) useFileList = true;
		if(strstr(argv[1], "m") != NULL) useModelTransform = true;
		if(strstr(argv[1], "s") != NULL) useScaler = true;
		if(strstr(argv[1], "x") != NULL) useFileMat = true;
		strcpy(fileName, argv[2]);
	}
	if (argc >= 4)
	{
		useMTL = true;
		strcpy_s(matFileName, MAX_PATH, argv[3]);
	}

	if(useModelTransform && !useFileList)
	{
		printf("Model transform need file list!\n");
		exit(-1);
	}

	char *ext = NULL;
	char shortFileName[MAX_PATH];
	int posExt = 0;
	int posShort = 0;
	strcpy_s(shortFileName, MAX_PATH, fileName);
	for(int i=strlen(fileName)-1;i>=0;i--)
	{
		if(fileName[i] == '.')
			if(!posExt) posExt = i;
		if(fileName[i] == '/' || fileName[i] == '\\')
			if(!posShort) posShort = i;
	}
	ext = &fileName[posExt+1];
	strncpy_s(shortFileName, MAX_PATH, &fileName[posShort+1], strlen(fileName)-posShort-1);
	bool loadOBJFile = !strcmp(ext, "obj") || !strcmp(ext, "OBJ");

	std::vector<std::string> genFileList;
	if(loadOBJFile)
	{
		irt::OBJLoader *objLoader = new irt::OBJLoader;
		objLoader->load(fileName, true);

		int count = objLoader->save(fileName);

		char fileNameWithOutExt[MAX_PATH];
		char subFileName[MAX_PATH];
		strcpy_s(fileNameWithOutExt, MAX_PATH, fileName);
		for(int i=strlen(fileName)-1;i>=0;i--)
		{
			if(fileName[i] == '.')
			{
				fileNameWithOutExt[i] = 0;
				break;
			}
		}

		FILE *fp;
		fopen_s(&fp, shortFileName, "w");
		for(int i=0;i<count;i++)
		{
			sprintf_s(subFileName, MAX_PATH, "%s%d.ply", fileNameWithOutExt, i);

			fprintf(fp, "%s\n", subFileName);

			genFileList.push_back(subFileName);
		}

		fclose(fp);

		genFileList.push_back(shortFileName);

		delete objLoader;

		useFileList = true;
		useIncore = true;
		strcpy_s(fileName, MAX_PATH, shortFileName);

		if(!useMTL)	sprintf_s(matFileName, MAX_PATH, "%s.mtl", fileNameWithOutExt);

		fopen_s(&fp, matFileName, "rb");
		if(fp)
		{
			useMTL = true;
			fclose(fp);
		}
	}

	cout << "OoC Tree Construction." << endl;

	//tree = new OutOfCoreTree();
	tree = new OoCTreeAccess();

	if(useIncore)
	{
		if(useFileList)
		{
			tree->buildSmallMulti(outputPath, fileName, useModelTransform, useFileMat, useMTL ? matFileName : 0);
		}
		else
		{
			tree->buildSmall(outputPath, fileName, useModelTransform);
		}
	}
	else
	{
		tree->build(outputPath, fileName, useModelTransform);
	}
	
	delete tree;

	for(size_t i=0;i<genFileList.size();i++)
	{
		remove(genFileList[i].c_str());
	}
}
