#include <iostream>
#include <stdio.h>
//#include "kDTree.h"
#include "OutOfCoreTree.h"
#include "OoCTreeAccess.h"

using namespace std;

int main(int argc, char **argv) {
	char fileName[255];
	//OutOfCoreTree *tree;
	OoCTreeAccess *tree;

	cout << "Layout construction." << endl;

	if (argc == 2)
		strcpy(fileName, argv[1]);
	else
		sprintf(fileName, "test.ply");


	//tree = new OutOfCoreTree();
	tree = new OoCTreeAccess();

	//tree->build(fileName);

	// Prepare data (material for buildig R-LODs)
	tree->PrepareDataForRLODs (fileName);

	//tree->PrintError ();
	///*

	tree->Check (false);
	tree->ComputeSimpRep ();
	//tree->Check (true);
	tree->QuantizeErr ();
	tree->Check (true);
	//*/

	delete tree;


}