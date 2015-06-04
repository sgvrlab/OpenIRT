// compress_main.cpp : Defines the entry point for the console application.
//
#include <iostream>
#include <stdio.h>

#include "DirectTriIndex.h"
#include "OptionManager.h"

using namespace std;

int main(int argc, char ** argv)
{
	char fileName[255];
	char filePath[255];

	printf ("Usage: %s file.ply\n", argv [0]);

	cout << "Remove unnecessary indirected triangle indices." << endl;

	if (argc >= 2) {
		strcpy(fileName, argv[1]);

	}
	else
		sprintf(fileName, "test.ply");

	DirectTriIndex *dti = new DirectTriIndex();

	OptionManager *opt = OptionManager::getSingletonPtr();

	const char *baseDirName = opt->getOption("global", "scenePath", "");
	sprintf(filePath, "%s%s.ooc", baseDirName, fileName);
	dti->Do(filePath);
	//dti->DoMulti(filePath);
	delete dti;

	return 0;
}

