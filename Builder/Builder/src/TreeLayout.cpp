// TreeLayout.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"
#include <windows.h>
#include <stdio.h>
#include <iostream>
#include <vector>

#include "kDTree.h"
#include "OOCFile64.h"
#include "OOCFile.h"

#undef GETNODE
#undef GETLEFTCHILD
#undef GETRIGHTCHILD
#undef BSPNEXTNODE
#undef MAKEIDX_PTR

#define BSPNEXTNODE 1
#ifdef KDTREENODE_16BYTES
#define GETLEFTCHILD(node) ((node)->children >> 2)
#define GETRIGHTCHILD(node) ((node)->children2 >> 2)
#else
#define GETLEFTCHILD(node) ((node)->children >> 2)
#define GETRIGHTCHILD(node) ((node)->children >> 2) + BSPNEXTNODE
#endif

#define GETCHILDCOUNT(node) ((node)->indexCount >> 2)
#define GETNODE(offset) ((BSPArrayTreeNodePtr)&((*tree)[offset]))
#define GETIDXOFFSET(node) ((node)->indexOffset)
#define MAKEIDX_PTR(offset) ((unsigned int *)&((*indexlists)[offset]))

OOCFile64<BSPArrayTreeNode> *tree;
OOCFile<unsigned int> *indexlists;
unsigned int curIndex = 0;

HANDLE outFile;
HANDLE outFileIndices;

typedef struct {
	unsigned int node;
	unsigned int parent;
	LARGE_INTEGER filePos;
	bool leftChild;
} SubTreeReference;

void printTree(BSPArrayTreeNode *node, OOCFile64<BSPArrayTreeNode> *tree, int depth = 0) {
	char indent[100];
	char *axis = "XYZL";
	int axisNr;
	int numTris = 0;
	BSPArrayTreeNodePtr child_left, child_right;

	indent[0] = 0;
	if (depth > 0) {
		int i;
		for (i = 0; i < (depth-1)*2; i++) {
			indent[i]   = ' ';			
		}
		indent[i]   = '|';
		indent[i+1] = '-';		
		indent[depth*2] = 0;
	}	

	// find axis (encoded in the lower two bits)	
	axisNr = AXIS(node);
	child_left = GETNODE(GETLEFTCHILD(node));
	child_right = GETNODE(GETRIGHTCHILD(node));

	if (ISLEAF(node)) { // leaf		
		numTris = GETCHILDCOUNT(node);
		printf("%sLeaf %d Tris\n", indent, numTris);	
	}
	else {
		printf("%sNode %c (%.2f) Children: %u/%u\n", indent, axis[axisNr], node->splitcoord, GETLEFTCHILD(node), GETRIGHTCHILD(node));
	}	

	if (ISNOLEAF(node)) {
		printTree(child_left, tree, depth + 1);		
		printTree(child_right, tree, depth + 1);
	}
}

void printNode(BSPArrayTreeNode *node, OOCFile64<BSPArrayTreeNode> *tree) {
	BSPArrayTreeNodePtr child_l, child_r;

	char *axis = "XYZL";
	int axisNr;
	int numTris = 0;

	// find axis (encoded in the lower two bits)	
	axisNr = AXIS(node);
	child_l = GETNODE(GETLEFTCHILD(node));
	child_r = GETNODE(GETRIGHTCHILD(node));

	if (ISLEAF(node)) { // leaf		
		numTris = GETCHILDCOUNT(node);
		printf("Leaf %d Tris\n", numTris);		
	}
	else {
		printf("Node %c (%.2f) Children: %u/%u\n", axis[axisNr], node->splitcoord, GETLEFTCHILD(node), GETRIGHTCHILD(node));		
	}	
}

// height of tree:
unsigned int findHeight(BSPArrayTreeNode *node, OOCFile64<BSPArrayTreeNode> *tree) {
	unsigned int child_left = GETLEFTCHILD(node);
	unsigned int child_right = GETRIGHTCHILD(node);

	if (ISLEAF(node)) {		
		return 0;
	}
	else {		
		unsigned int height1 = findHeight(GETNODE(child_left), tree);
		unsigned int height2 = findHeight(GETNODE(child_right), tree);
		return 1 + max(height1, height2); 
	}
}

int writeSubTree(unsigned int divisionHeight, unsigned int myOffset, int treeOffset, OOCFile64<BSPArrayTreeNode> *tree, HANDLE outFile, std::vector<SubTreeReference> &nodeList, unsigned int depth = 0) {
	DWORD written;	
	unsigned int numNodes = 1;
	BSPArrayTreeNode node = *GETNODE(myOffset);
	int axis = AXIS(&node);
	unsigned int child_left = GETLEFTCHILD(&node);
	unsigned int child_right = GETRIGHTCHILD(&node);
	
	//printNode(&node, tree); 

	// if leaf, just write directly:
	if (ISLEAF(&node)) {
		node.indexOffset = curIndex;
		curIndex += GETCHILDCOUNT(&node);

		// write node:		
		WriteFile(outFile, &node, sizeof(BSPArrayTreeNode), &written, NULL);	

		// write indices:
		unsigned int offset = GETIDXOFFSET(&node);
		std::vector<unsigned int> idxList;
			
		for (unsigned int i = 0; i < GETCHILDCOUNT(&node); i++)
			idxList.push_back(*MAKEIDX_PTR(offset + i));

		WriteFile(outFileIndices, &idxList[0], sizeof(unsigned int)*GETCHILDCOUNT(&node), &written, NULL);	
	}
	else { 		
		LARGE_INTEGER myFileOffset, temp;
		myFileOffset.HighPart = 0;
		myFileOffset.LowPart = SetFilePointer(outFile, 0, &myFileOffset.HighPart, FILE_CURRENT);

		if (depth == divisionHeight) {
			SubTreeReference r;
			r.parent = myOffset;
			r.leftChild = true;
			r.node = child_left;
			r.filePos.QuadPart = myFileOffset.QuadPart;
			
			// insert children into list:
			nodeList.push_back(r);

			r.leftChild = false;
			r.node = child_right;
			nodeList.push_back(r);
		}

		
		// change offset:
		node.children = ((myFileOffset.QuadPart >> 1) + (sizeof(BSPArrayTreeNode) >> 1)) | axis;

		WriteFile(outFile, &node, sizeof(BSPArrayTreeNode), &written, NULL);

		if (depth != divisionHeight)  {
			// recurse:
			numNodes += writeSubTree(divisionHeight, child_left, treeOffset, tree, outFile, nodeList, depth + 1);

			#ifdef KDTREENODE_16BYTES
			temp.HighPart = 0;
			temp.LowPart = SetFilePointer(outFile, 0, &temp.HighPart, FILE_CURRENT);
			node.children2 = (temp.QuadPart >> 1);				
			#endif

			numNodes += writeSubTree(divisionHeight, child_right, treeOffset, tree, outFile, nodeList, depth + 1);
		}	
		else {
			#ifdef KDTREENODE_16BYTES			
			node.children2 = 0; 
			#endif
		}

		// jump back to own offset
		SetFilePointer(outFile, myFileOffset.LowPart, &myFileOffset.HighPart, FILE_BEGIN);
		WriteFile(outFile, &node, sizeof(BSPArrayTreeNode), &written, NULL);
		// jump to next free position in file:
		SetFilePointer(outFile, 0, NULL, FILE_END);
	}
	
	return numNodes;
}

// conversion:
void convertToVEB(unsigned int myOffset, unsigned int &treeOffset, OOCFile64<BSPArrayTreeNode> *tree, HANDLE outFile, int maxHeight = -1) {
	std::vector<SubTreeReference> nodeList;
	BSPArrayTreeNodePtr thisNode = GETNODE(myOffset);	
	unsigned int height;
	
	// find height to subdivide the tree
	if (maxHeight == -1)
		height = findHeight(thisNode, tree);
	else 
		height = maxHeight;
	unsigned int divisionHeight = (unsigned int)floor((float)height / 2.0f);

	//cout << height << " | " << divisionHeight << endl;

	if (height <= 1) {
		treeOffset += writeSubTree(height, myOffset, treeOffset - myOffset, tree, outFile, nodeList) * (sizeof(BSPArrayTreeNode) >> 2);
	}
	else 	
		convertToVEB(myOffset, treeOffset, tree, outFile, divisionHeight);

	

	// write out all nodes with depth <= divisionHeight
	//treeOffset += writeSubTree(divisionHeight, myOffset, treeOffset - myOffset, tree, outFile, nodeList) * (sizeof(BSPArrayTreeNode) >> 2);

	// recurse on all children of written nodes:
	for(unsigned int i = 0; i < nodeList.size(); i++) {
		// go back to parent node and insert offset for child:
				
		LARGE_INTEGER  temp;
		BSPArrayTreeNode node;
		
		DWORD written;
		temp.HighPart = 0;
		temp.LowPart = SetFilePointer(outFile, 0, &temp.HighPart, FILE_CURRENT);
		SetFilePointer(outFile, nodeList[i].filePos.LowPart, &nodeList[i].filePos.HighPart, FILE_BEGIN);

		ReadFile(outFile, &node, sizeof(BSPArrayTreeNode), &written, NULL);

		if (nodeList[i].leftChild)
			node.children = (temp.QuadPart >> 1) | AXIS(&node);
		else
			node.children2 = (temp.QuadPart >> 1);	
		
		SetFilePointer(outFile, nodeList[i].filePos.LowPart, &nodeList[i].filePos.HighPart, FILE_BEGIN);
		WriteFile(outFile, &node, sizeof(BSPArrayTreeNode), &written, NULL);
		// jump to next free position in file:
		SetFilePointer(outFile, 0, NULL, FILE_END);

		convertToVEB(nodeList[i].node, treeOffset, tree, outFile);
	}
}

int main(int argc, char* argv[])
{
	char inFile[MAX_PATH];
	char inFileNodes[MAX_PATH];
	char inFileIndices[MAX_PATH];
	char outFileName[MAX_PATH];
	char outFileNameIndices[MAX_PATH];
	char header[100];
	char bspfilestring[50];
	BSPTreeInfo treeStats;

	if (argc < 2) {
		cerr << "Usage: " << argv[0] << " <kdtreefilename>" << endl;
		return 1;
	}
		
	strcpy(inFile, argv[1]);
	sprintf(inFileNodes, "%s.node", inFile);
	sprintf(inFileIndices, "%s.idx", inFile);
	sprintf(outFileName, "%s.veb", inFileNodes);
	sprintf(outFileNameIndices, "%s.veb", inFileIndices);
	FILE *fp = fopen(inFile, "rb");
	outFile = CreateFile(outFileName, GENERIC_WRITE|GENERIC_READ, FILE_SHARE_READ, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);	
	outFileIndices = CreateFile(outFileNameIndices, GENERIC_WRITE|GENERIC_READ, FILE_SHARE_READ, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);	

	if (fp == NULL) {
		cerr << "Could not open BSP tree file '" << inFile << "'!" << endl;		
		return false;
	}

	if (outFile == NULL) {
		cerr << "Could not open ouput tree file '" << outFileName << "'!" << endl;		
		return false;
	}

	size_t ret = fread(header, 1, BSP_FILEIDSTRINGLEN + 1, fp);
	if (ret != (BSP_FILEIDSTRINGLEN + 1)) {
		cerr << "Could not read header from BSP tree file '" << inFile << "', aborting. (empty file?)";		
		return false;
	}

	// test header format:
	strcpy(bspfilestring, BSP_FILEIDSTRING);
	for (unsigned int i = 0; i < BSP_FILEIDSTRINGLEN; i++) {
		if (header[i] != bspfilestring[i]) {
			cerr << "Invalid BSP tree header, aborting. (expected:'" << bspfilestring[i] << "', found:'" << header[i] <<"')" << endl;
			return false;		
		}
	}

	// test file version:
	if (header[BSP_FILEIDSTRINGLEN] != BSP_FILEVERSION) {
		cerr << "Wrong BSP tree file version (expected:" << BSP_FILEVERSION << ", found:" << header[BSP_FILEIDSTRINGLEN] << ")" << endl;
		return false;		
	}

	// format correct, read in full BSP tree info structure:

	// write count of nodes and tri indices:
	ret = fread(&treeStats, sizeof(BSPTreeInfo), 1, fp);
	if (ret != 1) {
		cerr << "Could not read tree info header!" << endl;		
		return false;
	}

	tree = new OOCFile64<BSPArrayTreeNode>(inFileNodes, 										 
										   1024*1024*1024,
										   1024*1024);
	indexlists = new OOCFile<unsigned int>(inFileIndices, 										 
										   1024*1024*512,
										   1024*1024);

	cout << "Converting " << inFile << " to VEB layout in " << outFileName << "..." << endl;
	
	unsigned int myOffset = 0, treeOffset = 0;
	convertToVEB(myOffset, treeOffset, tree, outFile);
	//printTree(GETNODE(0), tree);
	CloseHandle(outFile);
	CloseHandle(outFileIndices);
	
	delete tree;
	delete indexlists;

	//getchar();
	//tree = new OOCFile64<BSPArrayTreeNode>(outFileName, 1024*1024*1024, 1024*1024);
	//printTree(GETNODE(0), tree);	

	return 0;
}

