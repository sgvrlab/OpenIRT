// compress_main.cpp : 
//
#include <iostream>
#include <stdio.h>

#include "BVHCompression.h"
#include "OptionManager.h"

using namespace std;

int main(int argc, char ** argv)
{
	char fileName[255];
	char filePath[255];

	char Output [255];
	printf ("Usage: %s file.ply\n", argv [0]);

	cout << "OoC BVH compression." << endl;

	enum TYPE {RACBVH, QBVH, gzipBVH};
	TYPE type = RACBVH;
	if (argc >= 2) {
		strcpy(fileName, argv[1]);

	}
	else
		sprintf(fileName, "test.ply");
	if (argc >= 3)
	{
		if(strcmp(argv[2], "RACBVH") == 0)
			type = RACBVH;
		if(strcmp(argv[2], "QBVH") == 0)
			type = QBVH;
		if(strcmp(argv[2], "gzipBVH") == 0)
			type = gzipBVH;
	}

	BVHCompression *comp = new BVHCompression(1);

	OptionManager *opt = OptionManager::getSingletonPtr();

	const char *baseDirName = opt->getOption("global", "scenePath", "");
	sprintf(filePath, "%s%s.ooc", baseDirName, fileName);
	//comp->test(filePath);
	switch(type)
	{
	case RACBVH : comp->compress(filePath); break;
	case QBVH : comp->compressQBVH(filePath); break;
	case gzipBVH : comp->compressgzipBVH(filePath); break;
	}
	delete comp;

	return 0;
}

