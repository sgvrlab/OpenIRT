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

	char Output [255];
	printf ("Usage: %s file.ply layout_type(Oblivious(default), Aware, VEB, BFS, DFS)\n", argv [0]);

	cout << "OoC Tree Construction." << endl;

	int Type = -1;
	if (argc >= 2) {
		strcpy(fileName, argv[1]);

	}
	else
		sprintf(fileName, "test.ply");

	if (argc == 3) {

		if (strcmp (argv [2], "BFS") == 0) {
			Type = BFS_LAYOUT;
			printf ("Breath-first layout selected.\n");
		}
		else if (strcmp (argv [2], "DFS") == 0) {
			Type = DFS_LAYOUT;
			printf ("Depth-first layout selected.\n");
		}
		else if (strcmp (argv [2], "VEB") == 0) {
			Type = VEB_LAYOUT;
			printf ("van Emde Boas layout selected.\n");
		}
		else if (strcmp (argv [2], "VEB_CONTI") == 0) {
			Type = VEB_LAYOUT_CONTI;
			printf ("van Emde Boas layout (storing two childs contiguous) selected.\n");
		}
		else if (strcmp (argv [2], "Aware") == 0) {
			Type = CA_LAYOUT;
			printf ("Cache aware (4KB) layout selected.\n");
		}
		else if (strcmp (argv [2], "CO") == 0) {
			Type = CO_LAYOUT;
			printf ("Cache-oblivious layout selected!\n");
		}
		else if (strcmp (argv [2], "CO_PRESERVE") == 0) {
			Type = CO_LAYOUT_PRESERVE_ORDER;
			printf ("Cache-oblivious (preserve tree order) layout selected!\n");
		}
		else if (strcmp (argv [2], "CO_CONTI") == 0) {
			Type = CO_LAYOUT_CONTI;
			printf ("Cache-oblivious layout (left and right childs stored contiguous) selected!\n");
		}
		else {
			printf (Output);
			exit (-1);
		}
	}
	else {
		Type = CO_LAYOUT;
		printf ("Cache-oblivious layout selected!\n");

	}

	tree = new OoCTreeAccess();

	// Prepare data (material for buildig R-LODs)
	tree->PrepareDataForRLODs (fileName, false);

	/*
	tree->Check (false);
	exit (-1);
	*/
	//tree->Check (false);

	tree->ComputeLayout (Type);
	
#ifdef USE_LOD
	tree->Check (true);
#else
	tree->Check (false);
#endif


	delete tree;
}