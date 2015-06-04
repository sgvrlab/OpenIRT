#include <iostream>
#include <stdio.h>

#include "VertexConverter.h"
#include "OptionManager.h"

using namespace std;

int main(int argc, char ** argv)
{
	char filePath[255];

	if (argc < 3) {
		printf ("Usage: %s file.ply [VNCT]\n", argv [0]);
		exit(-1);
	}

	cout << "Convert vertex" << endl;

	VertexConverter *converter = new VertexConverter();

	OptionManager *opt = OptionManager::getSingletonPtr();

	const char *baseDirName = opt->getOption("global", "scenePath", "");
	sprintf(filePath, "%s%s.ooc", baseDirName, argv[1]);
	converter->Do(filePath, argv[2]);
	delete converter;

	return 0;
}

