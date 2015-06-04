#include "stdafx.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <GL/GL.h>
#include <GL/glut.h>
#include <algorithm>

#include "common.h"
#include "SIMDBSPTreeDefines.h"
#include "SIMDBSPTree.h"

extern bool g_Verbose;


// cost constants for the SAH subdivision algorithm
#define BSP_COST_TRAVERSAL 0.3f
#define BSP_COST_INTERSECTION 1.0f

// easy macros for working with the compressed BSP tree
// structure that is a bit hard to understand otherwise
// (and access may vary on whether OOC mode is on or not)
#ifdef FOUR_BYTE_FOR_KD_NODE	
	#define AXIS(node) ((node)->children2 & 3)
	#define ISLEAF(node) (((node)->children2 & 3) == 3)
	#define ISNOLEAF(node) (((node)->children2 & 3) != 3)
#else
	#define AXIS(node) ((node)->children & 3)
	#define ISLEAF(node) (((node)->children & 3) == 3)
	#define ISNOLEAF(node) (((node)->children & 3) != 3)
#endif

// sungeui start ----------------------------------

// Metric
CLODMetric LODMetric;
float g_MaxAllowModifier;
int g_NumTraversed = 0;
int g_NumIntersected = 0;

// sungeui end --------------------------------------

void SIMDBSPTree::buildHighLevelTree() {
	LogManager *log = LogManager::getSingletonPtr();
	BSPArrayTreeNode root;		

	objTreeStats.numTris = nSubObjects;

	// Build list with all indices and list of intervals:	
	leftlist[0] = new TriangleIndexList(nSubObjects);
	leftlist[1] = new TriangleIndexList(nSubObjects);
	int i;
	for (i = 0; i < MAXBSPSIZE; i++)
		rightlist[i] = new TriangleIndexList();

	// open temporary file for the indices
	FILE *indexFP = fopen("hltempindex.tmp", "wb+");
	// .. and tree nodes
	FILE *nodeFP = fopen("hltempnodes.tmp", "wb+");

	if (!indexFP) {		
		log->logMessage(LOG_ERROR, "Unable to write index file hltempindex.tmp!");
		return;
	}

	if (!nodeFP) {		
		log->logMessage(LOG_ERROR, "Unable to write node file hltempnodes.tmp!");
		return;
	}

	objTreeStats.numNodes = 1;	

	// prepare min/max lists
	// for the triangle coordinates...
	minvals = new Vector3[nSubObjects];
	maxvals = new Vector3[nSubObjects];

	for (i = 0; i < nSubObjects; i++) {		
		(*leftlist[0])[i] = i;
	
		for (int j = 0; j < 3; j++) {		
			minvals[i].e[j] = objectList[i].bb[0][j];
			maxvals[i].e[j] = objectList[i].bb[1][j];
		}
	}

	// Start subdividing, if we have got triangles
	if (nSubObjects > 1 && objTreeStats.maxDepth > 0) {	
		int idx = 0;
		int useAxis = 0;

		// use largest axis for first subdivision
		Vector3 dim = objectBB[1] - objectBB[0];

		fseek(nodeFP, sizeof(BSPArrayTreeNode), SEEK_SET);
		
		OBJSubdivide(0, leftlist[0], 0, objectBB[0], objectBB[1], indexFP, nodeFP);		
	}
	else { // no triangles or depth limit == 0: make root
		objTreeStats.numLeafs = 1;
		
		unsigned int count = (unsigned int)leftlist[0]->size();

		root.indexCount = MAKECHILDCOUNT(count);
		root.indexOffset = 0;		
		fwrite(&root, sizeof(BSPArrayTreeNode), 1, nodeFP);

		// write vector to file:
		fwrite(& (*(leftlist[0]))[0], sizeof(int), count, indexFP);

		objTreeStats.sumDepth = 0;
		objTreeStats.sumTris = count;		
		objTreeStats.maxLeafDepth = 0;		
		objTreeStats.maxTriCountPerLeaf = count;
	}

	delete leftlist[0];
	delete leftlist[1];

	for (int k = 0; k < MAXBSPSIZE; k++)
		delete rightlist[k];
	
	delete[] minvals;
	delete[] maxvals;
	minvals = 0;
	maxvals = 0;
		
	// temporary node file:	
	objectTree = new BSPArrayTreeNode[objTreeStats.numNodes];
	fseek(nodeFP, 0, SEEK_SET);
	fread((void *)objectTree, sizeof(BSPArrayTreeNode), objTreeStats.numNodes, nodeFP);
	fclose(nodeFP);

	// temporary index list file:	
	objectIndexList = new unsigned int[objTreeStats.sumTris];
	fseek(indexFP, 0, SEEK_SET);
	fread((void *)objectIndexList, sizeof(int), objTreeStats.sumTris, indexFP);
	fclose(indexFP);

	// remove temporary files (may be huge)
	remove("hltempindex.tmp");
	remove("hltempnodes.tmp");
}

// Tree construction in core is only available when not in OoC mode
#ifndef _USE_OOC
void SIMDBSPTree::buildTree(unsigned int subObjectId)
{				
	BSPArrayTreeNode root;

	timeBuildStart.set();

	subdivisionMode = BSP_SUBDIVISIONMODE_NORMAL;

	objectList[subObjectId].treeStats.numTris = objectList[subObjectId].nTris;

	// Build list with all indices and list of intervals:	
	leftlist[0] = new TriangleIndexList(objectList[subObjectId].nTris);
	leftlist[1] = new TriangleIndexList(objectList[subObjectId].nTris);

	for (int i = 0; i < MAXBSPSIZE; i++)
		rightlist[i] = new TriangleIndexList();

	// open temporary file for the indices
	FILE *indexFP = fopen("tempindex.tmp", "wb+");
	// .. and tree nodes
	FILE *nodeFP = fopen("tempnodes.tmp", "wb+");

	if (!indexFP) {
		LogManager *log = LogManager::getSingletonPtr();
		log->logMessage(LOG_ERROR, "Unable to write index file tempindex.tmp!");
		return;
	}

	if (!nodeFP) {
		LogManager *log = LogManager::getSingletonPtr();
		log->logMessage(LOG_ERROR, "Unable to write node file tempnodes.tmp!");
		return;
	}
		
	objectList[subObjectId].treeStats.numNodes = 1;	
	
	// subdivision mode == SAH ? then prepare min/max lists
	// for the triangle coordinates..
	if (subdivisionMode == BSP_SUBDIVISIONMODE_NORMAL) {
		minvals = new Vector3[objectList[subObjectId].nTris];
		maxvals = new Vector3[objectList[subObjectId].nTris];

		for (int i = 0; i < objectList[subObjectId].treeStats.numTris; i++) {
			const Triangle &tri = GETTRI(&objectList[subObjectId],i);			
			(*leftlist[0])[i] = i;

			for (int j = 0; j < 3; j++) {
				minvals[i].e[j] = min(GETVERTEX(&objectList[subObjectId],tri.p[0]).e[j], min(GETVERTEX(&objectList[subObjectId],tri.p[1]).e[j], GETVERTEX(&objectList[subObjectId],tri.p[2]).e[j]));
				maxvals[i].e[j] = max(GETVERTEX(&objectList[subObjectId],tri.p[0]).e[j], max(GETVERTEX(&objectList[subObjectId],tri.p[1]).e[j], GETVERTEX(&objectList[subObjectId],tri.p[2]).e[j]));
			}
		}
	}
	else { // all other subdivision modes:
		for (int i = 0; i < objectList[subObjectId].treeStats.numTris; i++) {	
			(*leftlist[0])[i] = i;
		}
	}

	// Start subdividing, if we have got triangles
	if (objectList[subObjectId].treeStats.numTris > 0 && objectList[subObjectId].treeStats.maxDepth > 0) {	
		int idx = 0;
		int useAxis = 0;

		// use largest axis for first subdivision
		Vector3 dim = objectList[subObjectId].treeStats.max - objectList[subObjectId].treeStats.min;

		fseek(nodeFP, sizeof(BSPArrayTreeNode), SEEK_SET);
		
		if (subdivisionMode == BSP_SUBDIVISIONMODE_SIMPLE) // spatial median
			Subdivide(0, leftlist[0], 0, dim.indexOfMaxComponent(), objectList[subObjectId].treeStats.min, objectList[subObjectId].treeStats.max, indexFP, nodeFP, subObjectId);
		else if (subdivisionMode == BSP_SUBDIVISIONMODE_NORMAL) // Surface area heuristic
			SubdivideSAH(0, leftlist[0], 0, objectList[subObjectId].treeStats.min, objectList[subObjectId].treeStats.max, indexFP, nodeFP, subObjectId);		
	}
	else { // no triangles or depth limit == 0: make root
		objectList[subObjectId].treeStats.numLeafs = 1;

		unsigned int count = (unsigned int)leftlist[0]->size();

		root.indexCount = MAKECHILDCOUNT(count);
		root.indexOffset = 0;
		fwrite(&root, sizeof(BSPArrayTreeNode), 1, nodeFP);
		
		// write vector to file:
		fwrite(& (*(leftlist[0]))[0], sizeof(int), count, indexFP);
		
		objectList[subObjectId].treeStats.sumDepth = 0;
		objectList[subObjectId].treeStats.sumTris  += count;		
		objectList[subObjectId].treeStats.maxLeafDepth = 0;		
		objectList[subObjectId].treeStats.maxTriCountPerLeaf = count;
	}

	delete leftlist[0];
	delete leftlist[1];

	for (int k = 0; k < MAXBSPSIZE; k++)
		delete rightlist[k];

	if (subdivisionMode == BSP_SUBDIVISIONMODE_NORMAL) {
		delete[] minvals;
		delete[] maxvals;
		minvals = 0;
		maxvals = 0;
	}

	LogManager *log = LogManager::getSingletonPtr();
	log->logMessage(LOG_DEBUG, "Reading in array tree representation from file...");	

	// temporary node file:
	log->logMessage(LOG_DEBUG, " - read in node list.");	
	objectList[subObjectId].tree = new BSPArrayTreeNode[objectList[subObjectId].treeStats.numNodes];
	fseek(nodeFP, 0, SEEK_SET);
	fread((void *)objectList[subObjectId].tree, sizeof(BSPArrayTreeNode), objectList[subObjectId].treeStats.numNodes, nodeFP);
	fclose(nodeFP);

	// temporary index list file:
	log->logMessage(LOG_DEBUG, " - read in index list.");
	objectList[subObjectId].indexlist = new unsigned int[objectList[subObjectId].treeStats.sumTris];
	fseek(indexFP, 0, SEEK_SET);
	fread((void *)objectList[subObjectId].indexlist, sizeof(int), objectList[subObjectId].treeStats.sumTris, indexFP);
	fclose(indexFP);

	// remove temporary files (may be huge)
	remove("tempindex.tmp");
	remove("tempnodes.tmp");

	timeBuildEnd.set();
	objectList[subObjectId].treeStats.timeBuild = timeBuildEnd - timeBuildStart;
}


bool SIMDBSPTree::Subdivide(long myOffset, TriangleIndexList *trilist, int depth, int axis, Vector3 &min, Vector3 &max, FILE *indexFP, FILE *nodeFP, unsigned int subObjectId)
{
	static unsigned int curIndex = 0, curNodeIndex = 0;
	int i, k;
	unsigned int triCount, newCount[2];
	BSPArrayTreeNode node;
	TriangleIndexList *newlists[2];
	Vector3 newmin = min, 
			newmax = max;
	Vector3 dim;
	long curOffset;

	// Subdivide:

	// mark current file offset
	curOffset = ftell(nodeFP);

	// jump back to own offset
	fseek(nodeFP, myOffset, SEEK_SET);
	
	// split coordinate is in middle of previous span
	node.splitcoord = (min.e[axis] + max.e[axis]) / 2.0f;
	
	// set pointer to children and split plane in lower bits 
	
	#ifdef KDTREENODE_16BYTES
		node.lodIndex = 0;
		#ifdef FOUR_BYTE_FOR_KD_NODE	
			node.children = (curOffset >> 4);
			node.children2 = axis;
		#else
			node.children = (curOffset >> 1) | axis;
			node.children2 = (curOffset + sizeof(BSPArrayTreeNode)) >> 1;
		#endif
	#else
	node.children = (curOffset >> 1) | axis;
	#endif

	// write real data to file
	fwrite(&node, sizeof(BSPArrayTreeNode), 1, nodeFP);

	// jump forward to previous address + size of 2 children -> new current position
	fseek(nodeFP, curOffset + 2*sizeof(BSPArrayTreeNode), SEEK_SET);
	objectList[subObjectId].treeStats.numNodes += 2;

	// create new triangle lists
	newlists[0] = leftlist[(depth+1)%2];
	newlists[1] = rightlist[depth];

	newlists[0]->clear();
	newlists[1]->clear();

	// assert 4 byte alignment, or else we mess up our axis marking
	assert(((unsigned int)newlists[0] & 3) == 0);
	assert(((unsigned int)newlists[1] & 3) == 0);

	// go through list, sort triangles into children
	triCount = (unsigned int)trilist->size();
	
	for (TriangleIndexListIterator j = trilist->begin(); j != trilist->end(); ++j) {
		const Triangle &t = GETTRI(&objectList[subObjectId],*j);

		for (k = 0; k < 3; k++) {
			if (GETVERTEX(&objectList[subObjectId],t.p[k]).e[axis] <= node.splitcoord) {
				newlists[0]->push_back(*j);
				break;
			}
		}

		for (k = 0; k < 3; k++) {
			if (GETVERTEX(&objectList[subObjectId],t.p[k]).e[axis] >= node.splitcoord) {
				newlists[1]->push_back(*j);
				break;
			}
		}
	}

	trilist->clear();

	newCount[0] = (unsigned int)newlists[0]->size();
	newCount[1] = (unsigned int)newlists[1]->size();

	depth++;

	for (i = 0; i < 2; i++) { 
		long thisChildFileOffset = curOffset + i*sizeof(BSPArrayTreeNode);

		// should we subdivide child further ?
		if ((newCount[i] > (unsigned int)objectList[subObjectId].treeStats.maxListLength) && (depth < objectList[subObjectId].treeStats.maxDepth) && (newCount[0] + newCount[1] < 2*triCount)) {
			// build new min/max bounding box:
			newmin.e[axis] = min.e[axis] + 0.5f * i * (max.e[axis] - min.e[axis]);
			newmax.e[axis] = min.e[axis] + 0.5f * (i+1) * (max.e[axis] - min.e[axis]);
			dim = newmax - newmin;

			// recursively subdivide further along largest axis
			Subdivide(thisChildFileOffset, newlists[i], depth, dim.indexOfMaxComponent(), newmin, newmax, indexFP, nodeFP, subObjectId);	
			newlists[i]->clear();
		}
		else { // make this child a leaf	
			BSPArrayTreeNode newLeaf;
			unsigned int count = (unsigned int)newlists[i]->size();
			newLeaf.indexCount = MAKECHILDCOUNT(count);
			newLeaf.indexOffset = curIndex;
			curIndex += count;			

			// write final node information to file:
			long tempPos = ftell(nodeFP);
			fseek(nodeFP, thisChildFileOffset, SEEK_SET);
			fwrite(&newLeaf, sizeof(BSPArrayTreeNode), 1, nodeFP);
			fseek(nodeFP, tempPos, SEEK_SET);

			// write vector to file:
			fwrite(& (*(newlists[i]))[0], sizeof(int), count, indexFP);

			// statistical information:
			objectList[subObjectId].treeStats.numLeafs++;
			objectList[subObjectId].treeStats.sumDepth += depth;
			objectList[subObjectId].treeStats.sumTris  += count;
			if (depth > objectList[subObjectId].treeStats.maxLeafDepth)
				objectList[subObjectId].treeStats.maxLeafDepth = depth;
			if (count > objectList[subObjectId].treeStats.maxTriCountPerLeaf)
				objectList[subObjectId].treeStats.maxTriCountPerLeaf = count;
		}
	}

	return true;
}

bool SIMDBSPTree::SubdivideSAH(long myOffset, TriangleIndexList *trilist, int depth,  Vector3 &min, Vector3 &max, FILE *indexFP, FILE *nodeFP, unsigned int subObjectId)
{
	static float *mins, *maxs, *collinear;
	static unsigned int curIndex = 0;	

	BSPArrayTreeNode node;
	unsigned int triCount = (unsigned int)trilist->size(), newCount[2];	
	TriangleIndexList *newlists[2];
	Vector3 newmin = min, newmax = max;
	Vector3 dim;
	int i, k;
	int bestAxis = -1;
	float bestCost = FLT_MAX, bestSplitCoord, currentSplitCoord;
	unsigned int numLeft, numRight;
	unsigned int curMin = 0, curMax = 0, curCol = 0;
	float bestAreaLeft = -1.0f, bestAreaRight = -1.0f, wholeArea = -1.0f;	
	int bestNumLeft = triCount, bestNumRight = triCount;
	long curOffset;
	
	bool debug1 = false; // && depth > 30; 
	bool debug2 = false;  //&& triCount == 23;	

	// allocate space for interval values:	
	mins = new float[triCount+1];
	maxs = new float[triCount+1];	
	collinear = new float[triCount+1];

	// for each axis:
	for (int curAxis = 0; curAxis < 3; curAxis++) {		

		// early termination: bb of axis has dimension 0
		if ((max[curAxis] - min[curAxis]) < EPSILON)
			continue;

		// Build sorted list of min and max vals for this axis:
		//
		i = k = 0;
		for (TriangleIndexListIterator j = trilist->begin(); j != trilist->end(); ++j) {
			if (minvals[*j].e[curAxis] == maxvals[*j].e[curAxis]) {
				collinear[k++] = minvals[*j].e[curAxis];
				continue;
			}

			mins[i] = minvals[*j].e[curAxis];
			maxs[i] = maxvals[*j].e[curAxis];
			
			i++;
		}

		// put guard values at end of array, needed later so we don't go
		// beyond end of array...
		mins[i] = FLT_MAX;
		maxs[i] = FLT_MAX;
		collinear[k] = FLT_MAX;

		// sort arrays:
		std::sort(mins, mins+i);
		std::sort(maxs, maxs+i);
		std::sort(collinear, collinear+k);
		
		int numMinMaxs = i + 1;
		int numCols = k + 1;

		unsigned int subtractRight = 0, addLeft = 0;
		unsigned int curMin = 0, curMax = 0, curCol = 0;
		wholeArea = -1.0f;
		currentSplitCoord = -FLT_MAX;
		
		numLeft = 0;
		numRight = triCount;

		// test for subdivide to create an empty cell:
		float emptySpanBegin = min(mins[0],collinear[0]) - min[curAxis];
		float emptySpanEnd = max[curAxis] - max(maxs[i-1], collinear[k-1]);
		float threshold = objectList[subObjectId].treeStats.emptySubdivideRatio * (max[curAxis] - min[curAxis]);

		// empty area to the left?
		if (emptySpanBegin > threshold) {
			bestSplitCoord = currentSplitCoord = min(mins[0],collinear[0]) - 0.01*emptySpanBegin;
			bestCost = 0;
			bestNumLeft = numLeft = 0;
			bestNumRight = numRight = triCount;
			bestAxis = curAxis;

			if (debug1) {		
				cout << "Found free space left: " << emptySpanBegin << " (" << (emptySpanBegin / (max[curAxis] - min[curAxis])) << ")  at Coord " << currentSplitCoord << endl;
				cout << min << " - " << max << " (" << min(mins[0],collinear[0]) << endl;
			}

			break;
		}
		else if (emptySpanEnd > threshold) { // empty area to the right?
			bestSplitCoord = currentSplitCoord =  max(maxs[i-1], collinear[k-1]) + 0.01*emptySpanEnd;
			bestCost = 0;
			bestNumLeft = numLeft = triCount;
			bestNumRight = numRight = 0;
			bestAxis = curAxis;

			if (debug1) {
				cout << "Found free space right: " << emptySpanEnd << " (" << (emptySpanEnd / (max[curAxis] - min[curAxis])) << ")  at Coord " << currentSplitCoord << endl;
				cout << min << " - " << max << " (" << max(maxs[i-1], collinear[k-1]) << endl;
			}

			break;
		}
		else {

			//
			// test all possible split planes according to surface area heuristic:
			//
			
			wholeArea = surfaceArea(max.e[0] - min.e[0], max.e[1] - min.e[1], max.e[2] - min.e[2]);
			
			if (debug1) {				
				cout << "Find Split curAxis=" << curAxis << " " << min << " - " << max << endl;
				cout << " nTris=" << triCount << " Startcoord = " << currentSplitCoord << endl;
			}

			if (debug2) {
				int l;			
				cout << "Mins: ";
				for (l = 0; l < numMinMaxs; l++)
					cout << mins[l] << " ";
				cout << endl;

				cout << "Maxs: ";
				for (l = 0; l < numMinMaxs; l++)
					cout << maxs[l] << " ";
				cout << endl;

				cout << "Cols: ";
				for (l = 0; l < numCols; l++)
					cout << collinear[l] << " ";
				cout << endl;
			}

			while (mins[curMin] != FLT_MAX || maxs[curMax] != FLT_MAX || collinear[curCol] != FLT_MAX) {
				float newCoord;

				numRight -= subtractRight;
				numLeft  += addLeft;
				addLeft = 0;
				subtractRight = 0;
				
				do {
					if (collinear[curCol] <= mins[curMin] && collinear[curCol] <= maxs[curMax]) {
						newCoord = collinear[curCol++];

						if (newCoord <= (min.e[curAxis] + max.e[curAxis]) / 2.0f) {
							numLeft++;
							numRight--;
						}
						else {
							addLeft++;
							subtractRight++;
						}
					}
					// find next split coord, either from min or max interval values
					else if (mins[curMin] <= maxs[curMax]) { // take from mins:
						newCoord = mins[curMin++];
						
						// since this is a minimal value of some triangle, we now have one more
						// triangle on the left side:
						addLeft++;
					}
					else { // take from maxs:
						newCoord = maxs[curMax++];

						// since this is a maximal value of some triangle, we have one less
						// triangle on the right side at the next interval:
						numRight--;
					}
				} while (mins[curMin] == newCoord || maxs[curMax] == newCoord || collinear[curCol] == newCoord);

				if (debug2)
					cout << " [" << i << "] : " << newCoord << endl;

				// don't test if the new split coord is the same as the old one, waste of time..
				if (newCoord == currentSplitCoord || numLeft == 0 || numRight == 0) 
					continue;

				// set new split coord to test
				currentSplitCoord = newCoord;				

				// calculate area on each side of split plane:
				float areaLeft = surfaceArea(currentSplitCoord - min.e[curAxis],  max.e[(curAxis+1)%3] - min.e[(curAxis+1)%3], max.e[(curAxis+2)%3] - min.e[(curAxis+2)%3]) / wholeArea;
				float areaRight = surfaceArea(max.e[curAxis] - currentSplitCoord,  max.e[(curAxis+1)%3] - min.e[(curAxis+1)%3], max.e[(curAxis+2)%3] - min.e[(curAxis+2)%3]) / wholeArea;

				//
				// calculate cost for this split according to SAH:
				//
				float currentCost = BSP_COST_TRAVERSAL + BSP_COST_INTERSECTION * (areaLeft * numLeft + areaRight * numRight);

				if (debug2)
					cout << "  - accepted! cost=" << currentCost << " L:" << numLeft << " R:" << numRight << endl;

				// better than previous minimum?
				if (currentCost < bestCost && numLeft != triCount && numRight != triCount) {
					bestCost = currentCost;
					bestSplitCoord = currentSplitCoord;
					bestNumLeft = numLeft;
					bestNumRight = numRight;
					bestAreaLeft = areaLeft;
					bestAreaRight = areaRight;
					bestAxis = curAxis;
				}
			}
		}

		if (debug1)
			cout << " axis=" << curAxis << " best: cost=" << bestCost << " Split:" << bestSplitCoord << " (L:" << bestNumLeft << " R:" << bestNumRight << ")" << endl;
	}

	// free memory for interval lists
	delete mins;
	delete maxs;
	delete collinear;

	if (debug1)
		cout << "Best: axis=" << bestAxis << " cost=" << bestCost << " Split:" << bestSplitCoord << " (L:" << bestNumLeft << " R:" << bestNumRight << ")" << endl;

	// this subdivision didn't work out... all triangles are still in one of the sides:													
	if ((bestNumLeft == triCount && bestNumRight > 0) || (bestNumLeft > 0 && bestNumRight == triCount)) {

		//if (debug1)
		//cout << " ----> could not find split coord for any axis! (N:" << triCount << " Axis:" << axis << " L:" << bestNumLeft << " R:" << bestNumRight << ")\n";
		//getchar();

		//renderSplit(bestAxis, bestSplitCoord, min, max, newlists);

		return false;
	}


	// determine if cost for splitting would be greater than just keeping this:
	if (depth != 0 && bestCost > (BSP_COST_INTERSECTION * triCount)) {	
		return false;
	}

	// mark current file offset
	curOffset = ftell(nodeFP);

	// jump back to own offset
	fseek(nodeFP, myOffset, SEEK_SET);

	//cout << "Writing BSP tree node to offset " << myOffset << ", Children = " << curOffset << endl;
	//getchar();

	// set split coordinate
	node.splitcoord = bestSplitCoord;	

	#ifdef KDTREENODE_16BYTES
		node.lodIndex = 0;

		// set pointer to children and split plane in lower bits 
		#ifdef FOUR_BYTE_FOR_KD_NODE	
			node.children = (curOffset >> 4);
			node.children2 = bestAxis;
			#else
		node.children = (curOffset >> 1) | bestAxis;
			node.children2 = (curOffset + sizeof(BSPArrayTreeNode)) >> 1;
		#endif
	#else
	// set pointer to children and split plane in lower bits 
	node.children = (curOffset >> 1) | bestAxis;
	#endif

	// write real data to file
	fwrite(&node, sizeof(BSPArrayTreeNode), 1, nodeFP);
	
	// jump forward to previous address + size of 2 children -> new current position
	fseek(nodeFP, curOffset + 2*sizeof(BSPArrayTreeNode), SEEK_SET);

	// we officially have 2 more tree nodes
	objectList[subObjectId].treeStats.numNodes += 2;

	// create new triangle lists
	newlists[0] = leftlist[(depth+1)%2];
	newlists[1] = rightlist[depth];
	newlists[0]->clear();
	newlists[1]->clear();
	
	// 
	// do the real assignment of triangles to child nodes:
	// 

	int smallerSide = (bestAreaLeft <= bestAreaRight)?0:1;
	for (TriangleIndexListIterator j = trilist->begin(); j != trilist->end(); ++j) {			
		
		// special case: triangle is coplanar to split plane.
		//  - if split plane is also a side of the AABB, put the tri into respective side
		//  - otherwise: put to smaller side (in terms of area)
		if (minvals[*j].e[bestAxis] == maxvals[*j].e[bestAxis]) {
			float v = minvals[*j].e[bestAxis];
			if (v == min.e[bestAxis]) {
				newlists[0]->push_back(*j);			
				continue;
			} 
			else if (v == max.e[bestAxis]) {
				newlists[1]->push_back(*j);
				continue;
			}
			else if (v == node.splitcoord) {
				newlists[smallerSide]->push_back(*j);
				continue;
			}
		}

		// non-collinear triangle, put in respective child or
		// both if overlapping
		if (minvals[*j].e[bestAxis] >= node.splitcoord) 
			newlists[1]->push_back(*j);
		else if (maxvals[*j].e[bestAxis] <= node.splitcoord) 
			newlists[0]->push_back(*j);
		else {
			newlists[0]->push_back(*j);
			newlists[1]->push_back(*j);
		}

	}

	newCount[0] = (unsigned int)newlists[0]->size();
	newCount[1] = (unsigned int)newlists[1]->size();	

	//if (newCount[0] != bestNumLeft || newCount[1] != bestNumRight) {
	/*if (debug2) {
		cout << "Counts inequal: Split:" << bestSplitCoord << " (N:" << triCount << " L:" << newCount[0] << " R:" << newCount[1] << ")" << endl;
		cout << "  should be : (L:" << bestNumLeft << " R:" << bestNumRight << ")" << endl;		


		renderSplit(axis, bestSplitCoord, min, max, newlists);
	}*/
	
	
	depth++;

	for (i = 0; i < 2; i++) { 
		bool was_subdivided = false;
		long thisChildFileOffset = curOffset + i*sizeof(BSPArrayTreeNode);

		// should we subdivide child further ?
		if (newCount[i] > (unsigned int)objectList[subObjectId].treeStats.maxListLength && depth < objectList[subObjectId].treeStats.maxDepth && (newCount[0] + newCount[1] < 2*triCount)) {
			if (debug1)
				cout << " ----> subdivide child " << i << endl;
			// build new min/max bounding box (if i=0, then [min...splitcoord], otherwise [splitcoord...max]:
			newmin.e[bestAxis] = (i==0)?min.e[bestAxis]:node.splitcoord;
			newmax.e[bestAxis] = (i==0)?node.splitcoord:max.e[bestAxis];
			dim = newmax - newmin;

			// recursively subdivide further along largest axis
			if (SubdivideSAH(thisChildFileOffset, newlists[i], depth, newmin, newmax, indexFP, nodeFP, subObjectId))
				was_subdivided = true;
		}

		if (!was_subdivided) { // make this child a leaf
			BSPArrayTreeNode newLeaf;
			unsigned int count = (unsigned int)newlists[i]->size();
			newLeaf.indexCount = MAKECHILDCOUNT(count);
			newLeaf.indexOffset = curIndex;
			curIndex += count;			

			if (debug1)
				cout << " ----> make leaf " << i << " (" << count << " tris)" << endl;

			//cout << "Writing leaf to " << thisChildFileOffset << endl;

			// write final node information to file:
			long tempPos = ftell(nodeFP);
			fseek(nodeFP, thisChildFileOffset, SEEK_SET);
			fwrite(&newLeaf, sizeof(BSPArrayTreeNode), 1, nodeFP);
			fseek(nodeFP, tempPos, SEEK_SET);
			
			// write index vector to file:
			fwrite(& (*(newlists[i]))[0], sizeof(int), count, indexFP);

			// statistical information:
			objectList[subObjectId].treeStats.numLeafs++;
			objectList[subObjectId].treeStats.sumDepth += depth;
			objectList[subObjectId].treeStats.sumTris  += count;
			if (depth > objectList[subObjectId].treeStats.maxLeafDepth)
				objectList[subObjectId].treeStats.maxLeafDepth = depth;
			if (count > objectList[subObjectId].treeStats.maxTriCountPerLeaf)
				objectList[subObjectId].treeStats.maxTriCountPerLeaf = count;
		}
	}

	return true;
}


#endif // !_USE_OOC

bool SIMDBSPTree::OBJSubdivide(long myOffset, TriangleIndexList *trilist, int depth,  Vector3 &min, Vector3 &max, FILE *indexFP, FILE *nodeFP)
{
	static float *mins, *maxs, *collinear;
	static unsigned int curIndex = 0;	

	BSPArrayTreeNode node;
	unsigned int triCount = (unsigned int)trilist->size(), newCount[2];	
	TriangleIndexList *newlists[2];
	Vector3 newmin = min, newmax = max;
	Vector3 dim;
	int i, k;
	int bestAxis = -1;
	float bestCost = FLT_MAX, bestSplitCoord, currentSplitCoord;
	unsigned int numLeft, numRight;
	unsigned int curMin = 0, curMax = 0, curCol = 0;
	float bestAreaLeft = -1.0f, bestAreaRight = -1.0f, wholeArea = -1.0f;	
	int bestNumLeft = triCount, bestNumRight = triCount;
	long curOffset;

	bool debug1 = false; // && depth > 30; 
	bool debug2 = false;  //&& triCount == 23;	

	// allocate space for interval values:	
	mins = new float[triCount+1];
	maxs = new float[triCount+1];	
	collinear = new float[triCount+1];

	// for each axis:
	for (int curAxis = 0; curAxis < 3; curAxis++) {		

		// early termination: bb of axis has dimension 0
		if ((max[curAxis] - min[curAxis]) < EPSILON)
			continue;

		// Build sorted list of min and max vals for this axis:
		//
		i = k = 0;
		for (TriangleIndexListIterator j = trilist->begin(); j != trilist->end(); ++j) {
			if (minvals[*j].e[curAxis] == maxvals[*j].e[curAxis]) {
				collinear[k++] = minvals[*j].e[curAxis];
				continue;
			}

			mins[i] = minvals[*j].e[curAxis];
			maxs[i] = maxvals[*j].e[curAxis];

			i++;
		}

		// put guard values at end of array, needed later so we don't go
		// beyond end of array...
		mins[i] = FLT_MAX;
		maxs[i] = FLT_MAX;
		collinear[k] = FLT_MAX;

		// sort arrays:
		std::sort(mins, mins+i);
		std::sort(maxs, maxs+i);
		std::sort(collinear, collinear+k);

		int numMinMaxs = i + 1;
		int numCols = k + 1;

		unsigned int subtractRight = 0, addLeft = 0;
		unsigned int curMin = 0, curMax = 0, curCol = 0;
		wholeArea = -1.0f;
		currentSplitCoord = -FLT_MAX;

		numLeft = 0;
		numRight = triCount;

		// test for subdivide to create an empty cell:
		float emptySpanBegin = min(mins[0],collinear[0]) - min[curAxis];
		float emptySpanEnd = max[curAxis] - max(maxs[i-1], collinear[k-1]);
		float threshold = objTreeStats.emptySubdivideRatio * (max[curAxis] - min[curAxis]);

		// empty area to the left?
		if (emptySpanBegin > threshold) {
			bestSplitCoord = currentSplitCoord = min(mins[0],collinear[0]) - 0.01*emptySpanBegin;
			bestCost = 0;
			bestNumLeft = numLeft = 0;
			bestNumRight = numRight = triCount;
			bestAxis = curAxis;

			if (debug1) {		
				cout << "Found free space left: " << emptySpanBegin << " (" << (emptySpanBegin / (max[curAxis] - min[curAxis])) << ")  at Coord " << currentSplitCoord << endl;
				cout << min << " - " << max << " (" << min(mins[0],collinear[0]) << endl;
			}

			break;
		}
		else if (emptySpanEnd > threshold) { // empty area to the right?
			bestSplitCoord = currentSplitCoord =  max(maxs[i-1], collinear[k-1]) + 0.01*emptySpanEnd;
			bestCost = 0;
			bestNumLeft = numLeft = triCount;
			bestNumRight = numRight = 0;
			bestAxis = curAxis;

			if (debug1) {
				cout << "Found free space right: " << emptySpanEnd << " (" << (emptySpanEnd / (max[curAxis] - min[curAxis])) << ")  at Coord " << currentSplitCoord << endl;
				cout << min << " - " << max << " (" << max(maxs[i-1], collinear[k-1]) << endl;
			}

			break;
		}
		else {

			//
			// test all possible split planes according to surface area heuristic:
			//

			wholeArea = surfaceArea(max.e[0] - min.e[0], max.e[1] - min.e[1], max.e[2] - min.e[2]);

			if (debug1) {				
				cout << "Find Split curAxis=" << curAxis << " " << min << " - " << max << endl;
				cout << " nTris=" << triCount << " Startcoord = " << currentSplitCoord << endl;
			}

			if (debug2) {
				int l;			
				cout << "Mins: ";
				for (l = 0; l < numMinMaxs; l++)
					cout << mins[l] << " ";
				cout << endl;

				cout << "Maxs: ";
				for (l = 0; l < numMinMaxs; l++)
					cout << maxs[l] << " ";
				cout << endl;

				cout << "Cols: ";
				for (l = 0; l < numCols; l++)
					cout << collinear[l] << " ";
				cout << endl;
			}

			while (mins[curMin] != FLT_MAX || maxs[curMax] != FLT_MAX || collinear[curCol] != FLT_MAX) {
				float newCoord;

				numRight -= subtractRight;
				numLeft  += addLeft;
				addLeft = 0;
				subtractRight = 0;

				do {
					if (collinear[curCol] <= mins[curMin] && collinear[curCol] <= maxs[curMax]) {
						newCoord = collinear[curCol++];

						if (newCoord <= (min.e[curAxis] + max.e[curAxis]) / 2.0f) {
							numLeft++;
							numRight--;
						}
						else {
							addLeft++;
							subtractRight++;
						}
					}
					// find next split coord, either from min or max interval values
					else if (mins[curMin] <= maxs[curMax]) { // take from mins:
						newCoord = mins[curMin++];

						// since this is a minimal value of some triangle, we now have one more
						// triangle on the left side:
						addLeft++;
					}
					else { // take from maxs:
						newCoord = maxs[curMax++];

						// since this is a maximal value of some triangle, we have one less
						// triangle on the right side at the next interval:
						numRight--;
					}
				} while (mins[curMin] == newCoord || maxs[curMax] == newCoord || collinear[curCol] == newCoord);

				if (debug2)
					cout << " [" << i << "] : " << newCoord << endl;

				// don't test if the new split coord is the same as the old one, waste of time..
				if (newCoord == currentSplitCoord || numLeft == 0 || numRight == 0) 
					continue;

				// set new split coord to test
				currentSplitCoord = newCoord;				

				// calculate area on each side of split plane:
				float areaLeft = surfaceArea(currentSplitCoord - min.e[curAxis],  max.e[(curAxis+1)%3] - min.e[(curAxis+1)%3], max.e[(curAxis+2)%3] - min.e[(curAxis+2)%3]) / wholeArea;
				float areaRight = surfaceArea(max.e[curAxis] - currentSplitCoord,  max.e[(curAxis+1)%3] - min.e[(curAxis+1)%3], max.e[(curAxis+2)%3] - min.e[(curAxis+2)%3]) / wholeArea;

				//
				// calculate cost for this split according to SAH:
				//
				float currentCost = BSP_COST_TRAVERSAL + BSP_COST_INTERSECTION * (areaLeft * numLeft + areaRight * numRight);

				if (debug2)
					cout << "  - accepted! cost=" << currentCost << " L:" << numLeft << " R:" << numRight << endl;

				// better than previous minimum?
				if (currentCost < bestCost && numLeft != triCount && numRight != triCount) {
					bestCost = currentCost;
					bestSplitCoord = currentSplitCoord;
					bestNumLeft = numLeft;
					bestNumRight = numRight;
					bestAreaLeft = areaLeft;
					bestAreaRight = areaRight;
					bestAxis = curAxis;
				}
			}
		}

		if (debug1)
			cout << " axis=" << curAxis << " best: cost=" << bestCost << " Split:" << bestSplitCoord << " (L:" << bestNumLeft << " R:" << bestNumRight << ")" << endl;
	}

	// free memory for interval lists
	delete mins;
	delete maxs;
	delete collinear;

	if (debug1)
		cout << "Best: axis=" << bestAxis << " cost=" << bestCost << " Split:" << bestSplitCoord << " (L:" << bestNumLeft << " R:" << bestNumRight << ")" << endl;

	// this subdivision didn't work out... all triangles are still in one of the sides:													
	if ((bestNumLeft == triCount && bestNumRight > 0) || (bestNumLeft > 0 && bestNumRight == triCount)) {
		return false;
	}

	// mark current file offset
	curOffset = ftell(nodeFP);

	// jump back to own offset
	fseek(nodeFP, myOffset, SEEK_SET);


	// set split coordinate
	node.splitcoord = bestSplitCoord;	

	#ifdef KDTREENODE_16BYTES
	node.lodIndex = 0;
	// set pointer to children and split plane in lower bits 
	# ifdef FOUR_BYTE_FOR_KD_NODE	
		node.children = (curOffset >> 4);
		node.children2 = bestAxis;
	# else
		node.children = (curOffset >> 1) | bestAxis;
		node.children2 = (curOffset + sizeof(BSPArrayTreeNode)) >> 1;
	# endif
	#else
	// set pointer to children and split plane in lower bits 
	node.children = (curOffset >> 1) | bestAxis;
	#endif

	// write real data to file
	fwrite(&node, sizeof(BSPArrayTreeNode), 1, nodeFP);

	// jump forward to previous address + size of 2 children -> new current position
	fseek(nodeFP, curOffset + 2*sizeof(BSPArrayTreeNode), SEEK_SET);

	// we officially have 2 more tree nodes
	objTreeStats.numNodes += 2;

	// create new triangle lists
	newlists[0] = leftlist[(depth+1)%2];
	newlists[1] = rightlist[depth];
	newlists[0]->clear();
	newlists[1]->clear();

	// 
	// do the real assignment of triangles to child nodes:
	// 

	int smallerSide = (bestAreaLeft <= bestAreaRight)?0:1;
	for (TriangleIndexListIterator j = trilist->begin(); j != trilist->end(); ++j) {			

		// special case: triangle is coplanar to split plane.
		//  - if split plane is also a side of the AABB, put the tri into respective side
		//  - otherwise: put to smaller side (in terms of area)
		if (minvals[*j].e[bestAxis] == maxvals[*j].e[bestAxis]) {
			float v = minvals[*j].e[bestAxis];
			if (v == min.e[bestAxis]) {
				newlists[0]->push_back(*j);			
				continue;
			} 
			else if (v == max.e[bestAxis]) {
				newlists[1]->push_back(*j);
				continue;
			}
			else if (v == node.splitcoord) {
				newlists[smallerSide]->push_back(*j);
				continue;
			}
		}

		// non-collinear triangle, put in respective child or
		// both if overlapping
		if (minvals[*j].e[bestAxis] >= node.splitcoord) 
			newlists[1]->push_back(*j);
		else if (maxvals[*j].e[bestAxis] <= node.splitcoord) 
			newlists[0]->push_back(*j);
		else {
			newlists[0]->push_back(*j);
			newlists[1]->push_back(*j);
		}
	}

	newCount[0] = (unsigned int)newlists[0]->size();
	newCount[1] = (unsigned int)newlists[1]->size();	

	//if (newCount[0] != bestNumLeft || newCount[1] != bestNumRight) {
	/*if (debug2) {
	cout << "Counts inequal: Split:" << bestSplitCoord << " (N:" << triCount << " L:" << newCount[0] << " R:" << newCount[1] << ")" << endl;
	cout << "  should be : (L:" << bestNumLeft << " R:" << bestNumRight << ")" << endl;		


	renderSplit(axis, bestSplitCoord, min, max, newlists);
	}*/


	depth++;

	for (i = 0; i < 2; i++) { 
		bool was_subdivided = false;
		long thisChildFileOffset = curOffset + i*sizeof(BSPArrayTreeNode);

		
		// should we subdivide child further ?
		if (newCount[i] >= (unsigned int)objTreeStats.maxListLength && depth < objTreeStats.maxDepth && (newCount[0] + newCount[1] < 2*triCount)) {
			if (debug1)
				cout << " ----> subdivide child " << i << endl;
			
			// build new min/max bounding box (if i=0, then [min...splitcoord], otherwise [splitcoord...max]:
			newmin.e[bestAxis] = (i==0)?min.e[bestAxis]:node.splitcoord;
			newmax.e[bestAxis] = (i==0)?node.splitcoord:max.e[bestAxis];
			dim = newmax - newmin;

			// recursively subdivide further along largest axis
			if (OBJSubdivide(thisChildFileOffset, newlists[i], depth, newmin, newmax, indexFP, nodeFP))
				was_subdivided = true;
		}

		//if (newCount[i] == 1)
		//	was_subdivided = OBJSubdivide(thisChildFileOffset, newlists[i], depth, newmin, newmax, indexFP, nodeFP, true);

		if (!was_subdivided) { // make this child a leaf
			BSPArrayTreeNode newLeaf;
			unsigned int count = (unsigned int)newlists[i]->size();
			newLeaf.indexCount = MAKECHILDCOUNT(count);
			newLeaf.indexOffset = curIndex;
			curIndex += count;			

			if (debug1)
				cout << " ----> make leaf " << i << " (" << count << " tris)" << endl;

			//cout << "Writing leaf to " << thisChildFileOffset << endl;

			// write final node information to file:
			long tempPos = ftell(nodeFP);
			fseek(nodeFP, thisChildFileOffset, SEEK_SET);
			fwrite(&newLeaf, sizeof(BSPArrayTreeNode), 1, nodeFP);
			fseek(nodeFP, tempPos, SEEK_SET);

			// write index vector to file:
			for (int ls = 0; ls < newlists[i]->size(); ls++) {
				int foo = newlists[i]->at(ls);
				fwrite(&foo, sizeof(int), 1, indexFP);
			}

			// statistical information:
			objTreeStats.numLeafs++;
			objTreeStats.sumDepth += depth;
			objTreeStats.sumTris  += count;
			if (depth > objTreeStats.maxLeafDepth)
				objTreeStats.maxLeafDepth = depth;
			if (count > objTreeStats.maxTriCountPerLeaf)
				objTreeStats.maxTriCountPerLeaf = count;
		}
	}

	return true;
}


int SIMDBSPTree::BeamTreeIntersect(Beam &beam, float *traveledDistance, unsigned int startNodeOffset)
{	
	BSPArrayTreeNodePtr currentNode;
	unsigned int currentOffset = INT_MAX;
	unsigned int currentEntryPoint, lastEntryPoint;
	int currentAxis;	
	Vector3 nodeBB[2]; // current node's bounding box

	unsigned int subObjectId = 0;
	ModelInstance *subObject = &objectList[0];
	nodeBB[0] = objectList[subObjectId].treeStats.min;
	nodeBB[1] = objectList[subObjectId].treeStats.max;
		    	
	int stackPtr;
	StackElem *stack = stacks[omp_get_thread_num()];

	// Test if the whole BSP tree is missed by the input ray and 
	// min and max (t values for the ray for the scene's bounding box)
	if (beam.intersectWithBox(nodeBB[0], nodeBB[1]) == false) {	
		return 0;
	}

	int r;
	for (r = 0; r < beam.numRealRays; r++) {
		beam.rayHasHit[r] = 15;
	}
	return beam.numRealRays;

	stack[0].node = 0;
	stackPtr = 1;
	
	currentNode = GETNODE(objectList[subObjectId].tree,startNodeOffset);	
	currentEntryPoint = 0;
	lastEntryPoint = 0;

	cout << "BeamTreeIntersect of Beam:" << beam << endl;

	// traverse BSP tree:
	if (beam.directionsMatch()) {
		while (currentOffset != 0) {		
			currentAxis = AXIS(currentNode);

			cout << " from offset: " << currentOffset << " cEP:" << currentEntryPoint << " lEP:" << lastEntryPoint << endl;

			// while we are not a leaf..
			while ( ISNOLEAF(currentNode) ) {
				int childSelect;

				cout << "  - intersect node (" << nodeBB[0] << " - " << nodeBB[1] << ") axis " << currentAxis << " split " << currentNode->splitcoord << endl;

				// does the beam not intersect the splitting plane?
				if (!beam.intersectWithkdNode(currentAxis, currentNode->splitcoord, nodeBB[0], nodeBB[1], childSelect)) {
					cout << "   take one node only: " << childSelect << endl;
					// then traverse one node only:
					currentOffset = GETCHILDNUM(currentNode, childSelect);	
					currentNode = GETNODE(subObject->tree,currentOffset);												
				}
				else { // beam intersects splitting plane:
					// store other node in stack and continue with near one
					// (intersect both)				

					childSelect = beam.cornerRays.rayChildOffsets[currentAxis];				

					// store far child
					stack[stackPtr].node = GETCHILDNUM(currentNode, childSelect);

					// set bounding-box for far node in stack:
					/// TODO: make more efficient
					stack[stackPtr].node_min = nodeBB[0];
					stack[stackPtr].node_min[currentAxis] = currentNode->splitcoord;
					stack[stackPtr].node_max = nodeBB[1];

					cout << "   take both nodes, farBB=(" << stack[stackPtr].node_min << " - " << stack[stackPtr].node_max << ")" << endl;

					stackPtr++;

					// current node = near node
					currentOffset = GETCHILDNUM(currentNode, childSelect ^ 1);
					currentNode = GETNODE(subObject->tree,currentOffset);
				}

				// update node bounding box
				nodeBB[childSelect ^ 1][currentAxis] = currentNode->splitcoord;

				currentAxis = AXIS(currentNode);
			}

			cout << "  - found leaf (" << nodeBB[0] << " - " << nodeBB[1] << ") children " << GETCHILDCOUNT(currentNode) << endl;

			// store leaf node as candidate if not empty	
			if (GETCHILDCOUNT(currentNode))
				currentEntryPoint = (currentEntryPoint != 0)?lastEntryPoint:currentOffset;

			stackPtr--;
			lastEntryPoint = currentOffset = stack[stackPtr].node;
			currentNode = GETNODE(subObject->tree,currentOffset);
			nodeBB[0] = stack[stackPtr].node_min;
			nodeBB[1] = stack[stackPtr].node_max;		 
		}

	}

	cout << "#####EP found: " << currentEntryPoint << endl;
	getchar();

	// low-level traversal
	for (r = 0; r < beam.numRealRays; r++) {
		beam.rayHasHit[r] = RayTreeIntersectOffset(beam.realRays[r], &beam.rayHitpoints[r], traveledDistance, currentEntryPoint);		
	}

	return beam.numRaysHit;
}

int SIMDBSPTree::RayTreeIntersectOffset(SIMDRay &rays, SIMDHitPointPtr hits, float *traveledDistance, unsigned int startOffset)
{	

	#define TRAVERSE_OFFSET
	#include "asm_traversetree.h"
	#undef TRAVERSE_OFFSET

	// return intersection results
	return hitValue;
}

int SIMDBSPTree::RayTreeIntersect(SIMDRay &rays, SIMDHitPointPtr hits, float *traveledDistance)
{	
	#include "asm_traversetree.h"

	// return intersection results
	return hitValue;
}

int SIMDBSPTree::isVisible(SIMDRay &rays, float *tmax, float *target, int initialMask, float *ErrBndForDirLight, int *hitLODIdx) {

#define VISIBILITY_ONLY
#include "asm_traversetree.h"
#undef VISIBILITY_ONLY

	return _mm_movemask_ps(_mm_cmpge_ps(intersect_t.v4, _mm_load_ps(tmax)));
	//return hitValue ^ 15;
}



FORCEINLINE int SIMDBSPTree::BeamObjIntersect(Beam &beam, ModelInstance *subObject, BSPArrayTreeNodePtr objList, float *tmaxs, float *tmins ) {

	int numNewRays = 0;
	for (int r = 0; r < beam.numRealRays; r++) {
		int hitValue = beam.rayHasHit[r];
		if (hitValue == ALL_RAYS) // skip if already all hit
			continue;

		// otherwise fake RayObjIntersect call for now
		SIMDRay &rays = beam.realRays[r];
		SIMDHitPointPtr obj = &beam.rayHitpoints[r];		

		#define BEAM_INTERSECT
		#include "asm_intersect.h"
		#undef BEAM_INTERSECT

		// new hit points found:		
		if (newHitMask > 0) {
			for (int rnum = 0; rnum < 4; rnum++) {
				int bitval = 1 << rnum;

				// ray already hit..
				if ((newHitMask & bitval) != bitval)
					continue;

				numNewRays++;
				const Triangle &pTri = GETTRI(subObject,obj->triIdx[rnum]);

				// Fill hitpoint structure:
				//						

				// interpolate vertex normals..
				#ifdef _USE_VERTEX_NORMALS
				obj->n[0][rnum] = pTri.normals[0][0] + obj->alpha[rnum] * pTri.normals[1][0] + obj->beta[rnum] * pTri.normals[2][0];
				obj->n[1][rnum] = pTri.normals[0][1] + obj->alpha[rnum] * pTri.normals[1][1] + obj->beta[rnum] * pTri.normals[2][1];
				obj->n[2][rnum] = pTri.normals[0][2] + obj->alpha[rnum] * pTri.normals[1][2] + obj->beta[rnum] * pTri.normals[2][2];
				#else
				obj->n[0][rnum] = pTri.n[0];
				obj->n[1][rnum] = pTri.n[1];
				obj->n[2][rnum] = pTri.n[2];
				#endif

				#ifdef _USE_TEXTURING
				// interpolate tex coords..
				obj->uv[rnum] = pTri.uv[0] + obj->alpha[rnum] * pTri.uv[1] + obj->beta[rnum] * pTri.uv[2];
				#endif

				obj->t[rnum] = tmaxs[rnum];

				obj->objectPtr[rnum] = subObject;

				#ifdef _USE_TRI_MATERIALS
				obj->m[rnum] = pTri.material;
				#else
				obj->m[rnum] = 0;
				#endif
			}

			// Calculate hitpoints:
			// x = rays.origin + t*rays.direction;
			rays.pointAtParameters(obj->t, obj->x);
			/*
			__asm {
				push edi;
				push esi;

				mov esi, rays;
				mov edi, obj;

				movaps xmm3, [esi]SIMDRay.direction[0];		// Get ray directions
				movaps xmm4, [esi]SIMDRay.direction[16];
				movaps xmm5, [esi]SIMDRay.direction[32];

				movaps xmm7, [edi]SIMDHitpoint.t;			// load parameter t

				movaps xmm0, [esi]SIMDRay.origin[0];		// load origins
				movaps xmm1, [esi]SIMDRay.origin[16];
				movaps xmm2, [esi]SIMDRay.origin[32];

				mulps xmm3, xmm7; // direction.x * t
				mulps xmm4, xmm7; // direction.y * t
				mulps xmm5, xmm7; // direction.z * t

				addps xmm3, xmm0; // origins.x + direction.x * t
				addps xmm4, xmm1; // origins.y + direction.y * t
				addps xmm5, xmm2; // origins.z + direction.z * t

				movaps [edi]SIMDHitpoint.x[0], xmm3;	// save to hitpoint
				movaps [edi]SIMDHitpoint.x[16], xmm4;
				movaps [edi]SIMDHitpoint.x[32], xmm5;

				pop esi;
				pop edi;
			}*/
		}

		beam.rayHasHit[r] =  newHitMask | hitValue;
	}

	beam.numRaysHit += numNewRays;
	return beam.numRaysHit;
}

FORCEINLINE int SIMDBSPTree::RayObjIntersect(const SIMDRay &rays, ModelInstance *subObject, BSPArrayTreeNodePtr objList, SIMDHitPointPtr obj, float *tmaxs, float *tmins, int hitValue)   
{	
	
	#undef VISIBILITY_ONLY
	#include "asm_intersect.h"

	// new hit points found:
	if (newHitMask > 0) {
		for (int rnum = 0; rnum < 4; rnum++) {
			int bitval = 1 << rnum;

			// ray already hit..
			if ((newHitMask & bitval) != bitval)
				continue;
			
			const Triangle &pTri = GETTRI(subObject,obj->triIdx[rnum]);

			// Fill hitpoint structure:
			//						

			// interpolate vertex normals..
			#ifdef _USE_VERTEX_NORMALS
			obj->n[0][rnum] = pTri.normals[0][0] + obj->alpha[rnum] * pTri.normals[1][0] + obj->beta[rnum] * pTri.normals[2][0];
			obj->n[1][rnum] = pTri.normals[0][1] + obj->alpha[rnum] * pTri.normals[1][1] + obj->beta[rnum] * pTri.normals[2][1];
			obj->n[2][rnum] = pTri.normals[0][2] + obj->alpha[rnum] * pTri.normals[1][2] + obj->beta[rnum] * pTri.normals[2][2];
			#else
			
			obj->n[0][rnum] = pTri.n[0];
			obj->n[1][rnum] = pTri.n[1];
			obj->n[2][rnum] = pTri.n[2];
			#endif

			#ifdef _USE_TEXTURING
			// interpolate tex coords..
			obj->uv[rnum] = pTri.uv[0] + obj->alpha[rnum] * pTri.uv[1] + obj->beta[rnum] * pTri.uv[2];
			#endif

			obj->t[rnum] = tmaxs[rnum];

			obj->objectPtr[rnum] = subObject; 

			#ifdef _USE_TRI_MATERIALS
			obj->m[rnum] = pTri.material;
			#else
			obj->m[rnum] = 0;
			#endif
		}

		// Calculate hitpoints:
		// x = rays.origin + t*rays.direction;
		rays.pointAtParameters(obj->t, obj->x);
		/*
		__asm {
			push edi;
			push esi;

			mov esi, rays;
			mov edi, obj;
									
			movaps xmm3, [esi]SIMDRay.direction[0];		// Get ray directions
			movaps xmm4, [esi]SIMDRay.direction[16];
			movaps xmm5, [esi]SIMDRay.direction[32];

			movaps xmm7, [edi]SIMDHitpoint.t;			// load parameter t

			movaps xmm0, [esi]SIMDRay.origin[0];		// load origins
			movaps xmm1, [esi]SIMDRay.origin[16];
			movaps xmm2, [esi]SIMDRay.origin[32];

			mulps xmm3, xmm7; // direction.x * t
			mulps xmm4, xmm7; // direction.y * t
			mulps xmm5, xmm7; // direction.z * t

			addps xmm3, xmm0; // origins.x + direction.x * t
			addps xmm4, xmm1; // origins.y + direction.y * t
			addps xmm5, xmm2; // origins.z + direction.z * t
			
			movaps [edi]SIMDHitpoint.x[0], xmm3;	// save to hitpoint
			movaps [edi]SIMDHitpoint.x[16], xmm4;
			movaps [edi]SIMDHitpoint.x[32], xmm5;
			
			pop esi;
			pop edi;
		}*/
	}

	return newHitMask | hitValue;
}

FORCEINLINE int SIMDBSPTree::RayObjIntersectTarget(const SIMDRay &rays, ModelInstance *subObject, BSPArrayTreeNodePtr objList, float *tmaxs,  float *tmins, float *intersect_t, int hitValue)  
{
	
	#define VISIBILITY_ONLY
	#include "asm_intersect.h"
	#undef VISIBILITY_ONLY

	return hitValue | newHitMask;	
}

int g_tempMax = 0;
int g_MinMaxDim [2];		// used for min and max of ray allowances

#if HIERARCHY_TYPE == TYPE_KD_TREE
int SIMDBSPTree::RayTreeIntersect(Ray &ray, HitPointPtr hit, float traveledDist)
{	
	ModelInstance *object = &objectList[0];
	BSPArrayTreeNodePtr testNode = GETNODE(object->tree,0);
	BSPArrayTreeNodePtr leftChild = GETNODE(object->tree,GETLEFTCHILD(testNode));
	BSPArrayTreeNodePtr rightChild = GETNODE(object->tree,GETRIGHTCHILD(testNode));

	int QuanIdx;
	float ErrBnd;

	BSPArrayTreeNodePtr currentNode;
	int currentAxis, stackPtr;
	float dist, min = EPSILON, max;
	const Vector3 &origin		= ray.data[0],
				  &invdirection = ray.data[2];

	int threadID = omp_get_thread_num();
	StackElem *stack = stacks[threadID];

	#ifdef _SIMD_SHOW_STATISTICS
	_debug_TreeIntersectCount++;
	#endif	
	
	initializeObjectIntersection(threadID);
	while (ModelInstance *subObject = getNextObjectIntersection(ray, threadID, &min, &max)) {	
		
		unsigned int currentOffset = INT_MAX;
		min = EPSILON;
		hit->m_HitLODIdx = 0;
		stack[0].node = 0;
		stackPtr = 1;	
		currentNode = GETNODE(subObject->tree,0);

		/*
		if (g_Verbose)
			cout << ray << endl;
		*/

		// traverse BSP tree:
		while (currentOffset != 0) {		
			currentAxis = AXIS(currentNode);

			/*
			if (g_Verbose)
				cout << "Start traverse at " << currentOffset << endl;
			*/

			// while we are not a leaf..
			while ( ISNOLEAF(currentNode) ) {

				#ifdef _SIMD_SHOW_STATISTICS
				_debug_NodeIntersections++;
				#endif	

				// sungeui start -----------------------------
				g_NumTraversed++;

				#ifdef USE_LOD

				if ( HAS_LOD(currentNode->lodIndex) ) {

					// Note: later, I need to put this info. in kd-node to remove unnecessary
					//		 data access
					QuanIdx = GET_ERR_QUANTIZATION_IDX(currentNode->lodIndex);
					assert (QuanIdx >= 0 && QuanIdx < (1<< ERR_BITs));

					 ErrBnd = GETLODERROR(subObject, QuanIdx);

					/*
					if (g_Verbose)
						cout << "Intersect with LOD, " << ErrBnd << " / " << (g_MaxAllowModifier * (min + traveledDist)) << endl;
					*/

					//if (LOD.m_ErrBnd < g_MaxAllowModifier * min ) { // we need to put it in kd node to reduce M.access
					if (ErrBnd < g_MaxAllowModifier * (min + traveledDist)) { // we need to put it in kd node to reduce M.access

						//g_tempMax = fabs (max - min) *10;
						if (RayLODIntersect(ray, subObject, currentNode, hit, max, min)) {						
							return true; // was hit
						}

						goto LOD_END; // traverse other node that are not descendent on the current node					
					}
					

				}
				#endif

				/*
				if (g_Verbose)
					cout << "origin:" << origin[currentAxis] << " invDir: " << invdirection[currentAxis] << endl;
				*/

				// calculate distance to splitting plane
				dist = (currentNode->splitcoord - origin[currentAxis]) * invdirection[currentAxis];						
				
				/*
				if (g_Verbose)
					cout << "Axis " << currentAxis << " split=" << currentNode->splitcoord << "dist="<< dist << " min="<< min << " max=" << max << endl;
				*/
				int childSelect = ray.posneg[currentAxis];

				//const float AdaptiveAllowance = fabs (max - min) * 0.0;
				if (dist < min-BSP_EPSILON)  {
					currentNode = GETNODE(subObject->tree,GETCHILDNUM(currentNode, childSelect));

					/*
					if (g_Verbose)
						cout << " take far child only!" << endl
					*/
				}
				else if (dist > max+BSP_EPSILON) {
					currentNode = GETNODE(subObject->tree,GETCHILDNUM(currentNode, childSelect ^ 1));

					/*
					if (g_Verbose)
						cout << " take near child only!" << endl;
					*/
				}
				else {
					/*
					if (g_Verbose)
						cout << " take both children." << endl;
					*/
					stack[stackPtr].node = GETCHILDNUM(currentNode, childSelect);
					stack[stackPtr].min.v[0] = dist;
					stack[stackPtr].max.v[0] = max;	

					// sungeui start
					stack[stackPtr].m_MinDim [0] = currentAxis;
					stack[stackPtr].m_MaxDim [0] = g_MinMaxDim [1];
					// sungeui end


					stackPtr++;				
					currentNode = GETNODE(subObject->tree,GETCHILDNUM(currentNode, childSelect ^ 1));
					max = dist;
				}

				currentAxis = AXIS(currentNode);
			}

			// intersect with current node's members		
			if (GETCHILDCOUNT(currentNode) && RayObjIntersect(ray, subObject, currentNode, hit, max))
			{
				return true; // was hit
			}
		
			#ifdef USE_LOD
			LOD_END:	
			#endif

			stackPtr--;
			currentOffset = stack[stackPtr].node;
			currentNode = GETNODE(subObject->tree,currentOffset);
			min = stack[stackPtr].min.v[0];
			max = stack[stackPtr].max.v[0];

			g_MinMaxDim [0] = stack[stackPtr].m_MinDim [0];
			g_MinMaxDim [1] = stack[stackPtr].m_MaxDim [0];

		} 
	}

	// ray did not intersect with tree
	return false;
}
#endif

#if HIERARCHY_TYPE == TYPE_BVH
int SIMDBSPTree::RayTreeIntersect(Ray &ray, HitPointPtr hit, float traveledDist)
{	
	int QuanIdx;
	float ErrBnd;

	BSPArrayTreeNodePtr currentNode;
	int currentAxis, stackPtr;
	float dist, min = EPSILON, max;
	const Vector3 &origin		= ray.data[0],
				  &invdirection = ray.data[2];

	int threadID = omp_get_thread_num();
	StackElem *stack = stacks[threadID];

	#ifdef _SIMD_SHOW_STATISTICS
	_debug_TreeIntersectCount++;
	#endif	
	
	initializeObjectIntersection(threadID);
	while (ModelInstance *subObject = getNextObjectIntersection(ray, threadID, &min, &max)) {	
		
		unsigned int currentOffset = INT_MAX;
		min = EPSILON;
		hit->m_HitLODIdx = 0;
		stack[0].node = 0;
		stackPtr = 1;	
		currentNode = GETNODE(subObject->tree,0);

		/*
		if (g_Verbose)
			cout << ray << endl;
		*/

		// traverse BSP tree:
		while (currentOffset != 0) {		
			currentAxis = AXIS(currentNode);

			/*
			if (g_Verbose)
				cout << "Start traverse at " << currentOffset << endl;
			*/

			// while we are not a leaf..
			while ( ISNOLEAF(currentNode) ) {

				#ifdef _SIMD_SHOW_STATISTICS
				_debug_NodeIntersections++;
				#endif	

				// sungeui start -----------------------------
				g_NumTraversed++;

				#ifdef USE_LOD

				if ( HAS_LOD(currentNode->lodIndex) ) {

					// Note: later, I need to put this info. in kd-node to remove unnecessary
					//		 data access
					QuanIdx = GET_ERR_QUANTIZATION_IDX(currentNode->lodIndex);
					assert (QuanIdx >= 0 && QuanIdx < (1<< ERR_BITs));

					 ErrBnd = GETLODERROR(subObject, QuanIdx);

					/*
					if (g_Verbose)
						cout << "Intersect with LOD, " << ErrBnd << " / " << (g_MaxAllowModifier * (min + traveledDist)) << endl;
					*/

					//if (LOD.m_ErrBnd < g_MaxAllowModifier * min ) { // we need to put it in kd node to reduce M.access
					if (ErrBnd < g_MaxAllowModifier * (min + traveledDist)) { // we need to put it in kd node to reduce M.access

						//g_tempMax = fabs (max - min) *10;
						if (RayLODIntersect(ray, subObject, currentNode, hit, max, min)) {						
							return true; // was hit
						}

						goto LOD_END; // traverse other node that are not descendent on the current node					
					}
					

				}
				#endif

				/*
				if (g_Verbose)
					cout << "origin:" << origin[currentAxis] << " invDir: " << invdirection[currentAxis] << endl;
				*/

				// calculate distance to splitting plane
				dist = (currentNode->splitcoord - origin[currentAxis]) * invdirection[currentAxis];						
				
				/*
				if (g_Verbose)
					cout << "Axis " << currentAxis << " split=" << currentNode->splitcoord << "dist="<< dist << " min="<< min << " max=" << max << endl;
				*/
				int childSelect = ray.posneg[currentAxis];

				//const float AdaptiveAllowance = fabs (max - min) * 0.0;
				if (dist < min-BSP_EPSILON)  {
					currentNode = GETNODE(subObject->tree,GETCHILDNUM(currentNode, childSelect));

					/*
					if (g_Verbose)
						cout << " take far child only!" << endl
					*/
				}
				else if (dist > max+BSP_EPSILON) {
					currentNode = GETNODE(subObject->tree,GETCHILDNUM(currentNode, childSelect ^ 1));

					/*
					if (g_Verbose)
						cout << " take near child only!" << endl;
					*/
				}
				else {
					/*
					if (g_Verbose)
						cout << " take both children." << endl;
					*/
					stack[stackPtr].node = GETCHILDNUM(currentNode, childSelect);
					stack[stackPtr].min.v[0] = dist;
					stack[stackPtr].max.v[0] = max;	

					// sungeui start
					stack[stackPtr].m_MinDim [0] = currentAxis;
					stack[stackPtr].m_MaxDim [0] = g_MinMaxDim [1];
					// sungeui end


					stackPtr++;				
					currentNode = GETNODE(subObject->tree,GETCHILDNUM(currentNode, childSelect ^ 1));
					max = dist;
				}

				currentAxis = AXIS(currentNode);
			}

			// intersect with current node's members		
			if (GETCHILDCOUNT(currentNode) && RayObjIntersect(ray, subObject, currentNode, hit, max))
				return true; // was hit
		
			#ifdef USE_LOD
			LOD_END:	
			#endif

			stackPtr--;
			currentOffset = stack[stackPtr].node;
			currentNode = GETNODE(subObject->tree,currentOffset);
			min = stack[stackPtr].min.v[0];
			max = stack[stackPtr].max.v[0];

			g_MinMaxDim [0] = stack[stackPtr].m_MinDim [0];
			g_MinMaxDim [1] = stack[stackPtr].m_MaxDim [0];

		} 
	}

	// ray did not intersect with tree
	return false;
#if 0
	int threadID = omp_get_thread_num();
	StackElem *stack = stacks[threadID];
	AccelerationNode *currentNode, *parentNode;
  int CurrentDepth = 0;
	int stackPtr;
	float min, max;	
	bool hasHit = false, HasHitRootBV = false;
	hit->t = FLT_MAX;	

	STATS_INTERSECT_TREE();

	stack[0].node = 0;
  stack[0].m_Depth = 0;
	stackPtr = 1;	
	parentNode = currentNode = objectList->tree;
  CurrentDepth = 0;

	// traverse LazyBVH tree:
	while (1) {
		STATS_INTERSECT_NODE();	
		// is current node intersected and also closer than previous hit?
		if (ray.RayBoxIntersect(currentNode->_bbox.bb, min, max) && min < hit->t && max > 0.0f) {
			//currentNode->setFrame( m_frame );
			HasHitRootBV = true;
			STATS_ACTIVE_RAYS(1,1);

			TRAVERSAL_START:
			currentNode->setFrame( m_Frame );

			// is inner node?
			if (!currentNode->isLeaf()) {	
				stack[stackPtr].parent = parentNode = currentNode;
        stack[stackPtr].m_Depth = CurrentDepth + 1;
				currentNode->storeOrderedChildren(ray, &currentNode, &stack[stackPtr].node);
        CurrentDepth++;
				stackPtr++;

				continue;
			}
			else {

				// is this an unbuilt node?				
				if (!currentNode->isBuilt()) { 					
					if (!currentNode->tryLock()) {
						
						// we can't swap if the stack is empty
						if (stack[stackPtr-1].node == NULL) {
							// no other node to swap: wait until built by other thread
							currentNode->wait();							
							goto TRAVERSAL_START;
						}

						// swap stack:
						stack[stackPtr-1].swapSingle(&currentNode, &parentNode);
						
						goto TRAVERSAL_START;
					}
#ifdef SORT_PREALLOCATE
					currentNode->makeSplit( objectList, parentNode, &deadList, centroids, histograms );
#else
					currentNode->makeSplit( objectList, parentNode, &deadList );
#endif
					g_Stat.AddBuild (currentNode->numTris);
					goto TRAVERSAL_START;
				}

				#ifdef _USE_MESHES
				// is leaf node:
				// intersect with current node's members	
				if (currentNode->isMesh())
					hasHit = RayMeshIntersection(ray, objectList, *currentNode->getMesh(), *hit) || hasHit;					
				else
				#endif
					hasHit = RayTriIntersect(ray, objectList, currentNode->triID, hit, min(max, hit->t)) || hasHit;	// < TODO change this
			}
		}

		// fetch next node from stack
		currentNode = stack[--stackPtr].node;	
		parentNode = stack[stackPtr].parent;
    CurrentDepth = stack[stackPtr].m_Depth;

		// traversal ends when stack empty
		if (currentNode == NULL)
			break;
	}

  if (HasHitRootBV)
    g_NumIntersectedRays++;

	// return hit status
	return hasHit;
#endif
}
#endif

int SIMDBSPTree::isVisibleWithLODs(const Vector3 &light_origin, const Vector3 &hit_p, float errBnd, int hitLODIdx)
{		

	//return 1;
	int threadID = omp_get_thread_num();
	StackElem *stack = stacks[threadID];

	BSPArrayTreeNodePtr currentNode;
	int currentAxis;	
	float dist, min, max, target_t, intersect_t = 0.0f;	
	int stackPtr;	
	const Vector3 &origin = hit_p;
	//const float ErrBndForDirLight = errBnd*30.0f;	// hack for ST mathew model
	const float ErrBndForDirLight = errBnd;
	//const float ErrBndForDirLight = 10000000000000000000000000000000000000.f;

	#ifdef _SIMD_SHOW_STATISTICS
	_debug_TreeIntersectCount++;
	#endif

	//
	// make shadow ray:
	//
	
	/*
	Vector3 dir = origin - light_origin;
	// calculate t value to reach the ray's target
	int idx = dir.indexOfMaxComponent();
	target_t = dir[idx];
	dir.makeUnitVector();
	Ray ray(light_origin, dir);
	*/
	
	
	Vector3 dir = light_origin - origin;
	// calculate t value to reach the ray's target
	int idx = dir.indexOfMaxComponent();
	target_t = dir[idx];
	dir.makeUnitVector();
	Ray ray(origin, dir);
	
	
	target_t = (target_t * ray.data[2][idx]) - BSP_EPSILON;	

	// test if the whole BSP tree is missed by the input ray
	//if (!RayBoxIntersect(ray, objectList[subObjectId].bb[0], objectList[subObjectId].bb[1], &min, &max)) {			
	//	return true;
	//}

	initializeObjectIntersection(threadID);
	while (ModelInstance *subObject = getNextObjectIntersection(ray, threadID, &min, &max)) {	

		unsigned int currentOffset = INT_MAX;

		//min = max(min, BSP_EPSILON);
		stack[0].node = 0;
		stackPtr = 1;	
		currentNode = GETNODE(subObject->tree,0);

		// traverse BSP tree:
		while (currentOffset != 0) {
			currentAxis = AXIS(currentNode);

			#ifdef _SIMD_SHOW_STATISTICS
			_debug_NodeIntersections++;
			#endif	

			// while we are not a leaf..
			while ( ISNOLEAF(currentNode) ) {

				#ifdef USE_LOD
				if ( HAS_LOD(currentNode->lodIndex) ) {
					
					// if we meet the same LOD for the primary ray, we should skip all the nodes
					// below the node. Otherwise, we can get self-intersection.
					unsigned int idxList = GET_REAL_IDX(currentNode->lodIndex);	
					int QuanIdx = GET_ERR_QUANTIZATION_IDX(currentNode->lodIndex);
					assert (QuanIdx >= 0 && QuanIdx < (1<< ERR_BITs));
					float ErrBnd = GETLODERROR(subObject,QuanIdx);				
					
					if (ErrBnd <= ErrBndForDirLight || idxList == hitLODIdx) { // we need to put it in kd node to reduce M.access
						intersect_t = RayLODIntersectTarget(ray, subObject, idxList, max, min);
						
						if (intersect_t > ErrBndForDirLight)
							return (intersect_t >= target_t);

						goto LOD_END;
					}				
				}
				#endif

				// calculate distance to splitting plane
				dist = (currentNode->splitcoord - ray.data[0][currentAxis]) * ray.data[2][currentAxis];						

				int childSelect = ray.posneg[currentAxis];

				if (dist < min-BSP_EPSILON)  {
					currentNode = GETNODE(subObject->tree,GETCHILDNUM(currentNode, childSelect));
				}
				else if (dist > max+BSP_EPSILON) {
					currentNode = GETNODE(subObject->tree,GETCHILDNUM(currentNode, childSelect ^ 1));
				}
				else {
					stack[stackPtr].node = GETCHILDNUM(currentNode, childSelect);
					stack[stackPtr].min.v[0] = dist;
					stack[stackPtr].max.v[0] = max;	
					stackPtr++;				
					currentNode = GETNODE(subObject->tree,GETCHILDNUM(currentNode, childSelect ^ 1));
					max = dist;
				}

				currentAxis = AXIS(currentNode);
			}

			// intersect with current node's members				
			float intersect_t = RayObjIntersectTarget(ray, subObject, currentNode, max);

			if ( intersect_t > (ErrBndForDirLight + INTERSECT_EPSILON)) {
				return (intersect_t >= target_t);
			}
		
			#ifdef USE_LOD
			LOD_END:	
			#endif

			stackPtr--;
			currentOffset = stack[stackPtr].node;
			currentNode = GETNODE(subObject->tree,currentOffset);
			min = stack[stackPtr].min.v[0];
			max = stack[stackPtr].max.v[0];
		} 
	}

	return true;
}


int SIMDBSPTree::isVisible(const Vector3 &light_origin, const Vector3 &hit_p, float traveledDist)
{	
	//return true;
	int threadID = omp_get_thread_num();
	StackElem *stack = stacks[threadID];

	BSPArrayTreeNodePtr currentNode;
	int currentAxis;	
	float dist, min, max, target_t, intersect_t = 0.0f;	
	int stackPtr;	
	const Vector3 &origin = hit_p;	
	//traveledDist = 10000000000000000000000000000.f;

	#ifdef _SIMD_SHOW_STATISTICS
	_debug_TreeIntersectCount++;
	#endif

	//
	// make shadow ray:
	//
	
	/*
	Vector3 dir = origin - light_origin;
	// calculate t value to reach the ray's target
	int idx = dir.indexOfMaxComponent();
	target_t = dir[idx];
	dir.makeUnitVector();
	Ray ray(light_origin, dir);
	*/
	
	
	Vector3 dir = light_origin - origin;
	// calculate t value to reach the ray's target
	int idx = dir.indexOfMaxComponent();
	target_t = dir[idx];
	dir.makeUnitVector();
	Ray ray(origin, dir);
	
	
	target_t = (target_t * ray.data[2][idx]) - BSP_EPSILON;	

	// test if the whole BSP tree is missed by the input ray
	//if (!RayBoxIntersect(ray, objectList[subObjectId].bb[0], objectList[subObjectId].bb[1], &min, &max)) {			
	//	return true;
	//}

	initializeObjectIntersection(threadID);
	while (ModelInstance *subObject = getNextObjectIntersection(ray, threadID, &min, &max)) {

		unsigned int currentOffset = INT_MAX;
		
		if (min < BSP_EPSILON)
			min = BSP_EPSILON;
		
		stack[0].node = 0;
		stackPtr = 1;	
		currentNode = GETNODE(subObject->tree,0);

		// traverse BSP tree:
		while (currentOffset != 0) {
			currentAxis = AXIS(currentNode);

			#ifdef _SIMD_SHOW_STATISTICS
			_debug_NodeIntersections++;
			#endif	

			// while we are not a leaf..
			while ( ISNOLEAF(currentNode) ) {

				#ifdef USE_LOD
				if ( HAS_LOD(currentNode->lodIndex) ) {
					
					// if we meet the same LOD for the primary ray, we should skip all the nodes
					// below the node. Otherwise, we can get self-intersection.
					unsigned int idxList = GET_REAL_IDX(currentNode->lodIndex);	
					int QuanIdx = GET_ERR_QUANTIZATION_IDX(currentNode->lodIndex);
					assert (QuanIdx >= 0 && QuanIdx < (1<< ERR_BITs));
					float ErrBnd = GETLODERROR(subObject,QuanIdx);	

					if (ErrBnd <= (g_MaxAllowModifier * (min + traveledDist))) { // we need to put it in kd node to reduce M.access						
						intersect_t = RayLODIntersectTarget(ray, subObject, idxList, max, min);
									
						if (intersect_t > 0.000001)
							return (intersect_t >= target_t);

						goto LOD_END;
					}											
				}
				#endif

				// calculate distance to splitting plane
				dist = (currentNode->splitcoord - ray.data[0][currentAxis]) * ray.data[2][currentAxis];						

				int childSelect = ray.posneg[currentAxis];

				if (dist < min-BSP_EPSILON)  {
					currentNode = GETNODE(subObject->tree,GETCHILDNUM(currentNode, childSelect));
				}
				else if (dist > max+BSP_EPSILON) {
					currentNode = GETNODE(subObject->tree,GETCHILDNUM(currentNode, childSelect ^ 1));
				}
				else {
					stack[stackPtr].node = GETCHILDNUM(currentNode, childSelect);
					stack[stackPtr].min.v[0] = dist;
					stack[stackPtr].max.v[0] = max;	
					stackPtr++;				
					currentNode = GETNODE(subObject->tree,GETCHILDNUM(currentNode, childSelect ^ 1));
					max = dist;
				}

				currentAxis = AXIS(currentNode);
			}

			// intersect with current node's members				
			float intersect_t = RayObjIntersectTarget(ray, subObject, currentNode, max);

			if ( intersect_t > INTERSECT_EPSILON) {
				return (intersect_t >= target_t);
			}
		
			#ifdef USE_LOD
			LOD_END:	
			#endif

			stackPtr--;
			currentOffset = stack[stackPtr].node;
			currentNode = GETNODE(subObject->tree,currentOffset);
			min = stack[stackPtr].min.v[0];
			max = stack[stackPtr].max.v[0];
		} 
	}

	return true;
	/*
	int threadID = omp_get_thread_num();
	StackElem *stack = stacks[threadID];	

	BSPArrayTreeNodePtr currentNode;
	unsigned int nearChild, farChild;
	int currentAxis;	
	float dist, min, max, target_t, intersect_t;	
	int stackPtr;

	
	Vector3 dir = target - origin;
	dir.makeUnitVector();
	Ray ray(origin, dir);

	// calculate t value to reach the ray's target
	int idx = dir.indexOfMaxComponent();
	target_t = (target.e[idx] - origin.e[idx]) / dir.e[idx];
	target_t -= BSP_EPSILON;

	#ifdef _SIMD_SHOW_STATISTICS
	_debug_TreeIntersectCount++;	
	#endif

	initializeObjectIntersection(threadID);
	while (ModelInstance *subObject = getNextObjectIntersection(ray, threadID, &min, &max)) {	

		unsigned int currentOffset = INT_MAX;		

		// test if the whole BSP tree is missed by the input ray
		//if (!RayBoxIntersect(ray, objectList[subObjectId].bb[0], objectList[subObjectId].bb[1], &min, &max)) {			
		//	return true;
		//}

		stack[0].node = 0;
		stackPtr = 1;	
		currentNode = GETNODE(subObject->tree,0);		

		// traverse BSP tree:
		while (currentOffset != 0) {
			currentAxis = AXIS(currentNode);

			#ifdef _SIMD_SHOW_STATISTICS
			_debug_NodeIntersections++;
			#endif	

			// while we are not a leaf..
			while ( ISNOLEAF(currentNode) ) {

				#ifdef USE_LOD
				if ( HAS_LOD(currentNode->lodIndex) ) {
					
					// if we meet the same LOD for the primary ray, we should skip all the nodes
					// below the node. Otherwise, we can get self-intersection.					
					int QuanIdx = GET_ERR_QUANTIZATION_IDX(currentNode->lodIndex);
					assert (QuanIdx >= 0 && QuanIdx < (1<< ERR_BITs));
					float ErrBnd = GETLODERROR(subObject,QuanIdx);				
					
					if (ErrBnd <= g_MaxAllowModifier * (min + traveledDist)) { // we need to put it in kd node to reduce M.access
						unsigned int idxList = GET_REAL_IDX(currentNode->lodIndex);	
						intersect_t = RayLODIntersectTarget(ray, subObject, idxList, max, min);
									
						if (intersect_t > 0.0f)
							retu rn (intersect_t >= target_t);

						goto LOD_END;
					}						
				}
				#endif
				
				// calculate distance to splitting plane
				dist = (currentNode->splitcoord - ray.data[0][currentAxis]) * ray.data[2][currentAxis];						

				int childSelect = ray.posneg[currentAxis];

				if (dist < min-BSP_EPSILON)  {
					currentNode = GETNODE(subObject->tree,GETCHILDNUM(currentNode, childSelect));
				}
				else if (dist > max+BSP_EPSILON) {
					currentNode = GETNODE(subObject->tree,GETCHILDNUM(currentNode, childSelect ^ 1));
				}
				else {
					stack[stackPtr].node = GETCHILDNUM(currentNode, childSelect);
					stack[stackPtr].min.v[0] = dist;
					stack[stackPtr].max.v[0] = max;
					stackPtr++;				
					currentNode = GETNODE(subObject->tree,GETCHILDNUM(currentNode, childSelect ^ 1));
					max = dist;
				}

				currentAxis = AXIS(currentNode);
			}

			// intersect with current node's members		
			intersect_t = RayObjIntersectTarget(ray, subObject, currentNode, max);

			if ( intersect_t > 0.0f ) {
				return (intersect_t >= target_t);
			}

#ifdef USE_LOD
LOD_END:
#endif

			stackPtr--;
			currentOffset = stack[stackPtr].node;
			currentNode = GETNODE(subObject->tree,currentOffset);
			min = stack[stackPtr].min.v[0];
			max = stack[stackPtr].max.v[0];
		} 
	}

	return true;*/
}

FORCEINLINE int SIMDBSPTree::RayBoxIntersect(const SIMDRay& rays, Vector3 *bb, SIMDVec4 &min, SIMDVec4 &max)  {
	__m128 curMin = _mm_set1_ps(-FLT_MAX);
	__m128 curMax = _mm_set1_ps(FLT_MAX);

	/**
	* SSE Part 1: intersect all 4 rays against the scene bounding box.
	* Essentially, this tests the distances of each ray to the min and
	* max plane of the axis. If we get a minimal value that is larger than
	* the maximal, then the ray will cut the plane behind the viewer or
	* behind the scene, so we can drop out of the intersection calculation
	* immediately, since the ray will not intersect the scene.
	* 
	* The intersection will continue if at least one of the rays hits the
	* bounding box.
	*/
	for (int axis = 0; axis < 3; axis++) {
		__m128 origins = _mm_load_ps(rays.origin[axis]);
		__m128 directions = _mm_load_ps(rays.invdirection[axis]);

		__m128 bbMin = _mm_load1_ps(&bb[rays.rayChildOffsets[axis] ^ 1].e[axis]);
		__m128 bbMax = _mm_load1_ps(&bb[rays.rayChildOffsets[axis]].e[axis]);
		
		// float t0 = (pp[r.posneg[3]].e[0] - r.data[0].e[0]) * r.data[2].e[0];
		__m128 t0 = _mm_mul_ps(_mm_sub_ps(bbMin, origins), directions);
		
		// float t1 = (pp[r.posneg[0]].e[0] - r.data[0].e[0]) * r.data[2].e[0];
		__m128 t1 = _mm_mul_ps(_mm_sub_ps(bbMax, origins), directions);
				
		//
		// if (t0 > interval_min) interval_min = t0;
		//
		curMin = _mm_max_ps(t0, curMin);

		//
		// if (t1 < interval_max) interval_max = t1;
		//
		curMax = _mm_min_ps(t1, curMax);

		// test for early termination (if none of the rays hit the bb)
		//
		// if (min[rnum] > max[rnum]) 
		//		continue;
		//
		if (_mm_movemask_ps(_mm_cmpgt_ps(curMin, curMax)) == ALL_RAYS)
			return 0;		
	}

	// ray hit the box, store min/max values
	min.v4 = _mm_max_ps(curMin, _mm_set1_ps(INTERSECT_EPSILON));
	max.v4 = curMax;

	return ALL_RAYS;
}


FORCEINLINE bool SIMDBSPTree::RayBoxIntersect(const Ray& r, Vector3 &min, Vector3 &max, float *returnMin, float *returnMax)  {

	float interval_min = -FLT_MAX;
	float interval_max = FLT_MAX;
	Vector3 pp[2];

	pp[0] = min;
	pp[1] = max;

	float t0 = (pp[r.posneg[3]].e[0] - r.data[0].e[0]) * r.data[2].e[0];
	float t1 = (pp[r.posneg[0]].e[0] - r.data[0].e[0]) * r.data[2].e[0];
	if (t0 > interval_min) {interval_min = t0; g_MinMaxDim [0] = 0;}
	if (t1 < interval_max) {interval_max = t1; g_MinMaxDim [1] = 0;}
	if (interval_min > interval_max) return false;

	t0 = (pp[r.posneg[4]].e[1] - r.data[0].e[1]) * r.data[2].e[1];
	t1 = (pp[r.posneg[1]].e[1] - r.data[0].e[1]) * r.data[2].e[1];
	if (t0 > interval_min) {interval_min = t0; g_MinMaxDim [0] = 1;}
	if (t1 < interval_max) {interval_max = t1; g_MinMaxDim [1] = 1;}
	if (interval_min > interval_max) return false;

	t0 = (pp[r.posneg[5]].e[2] - r.data[0].e[2]) * r.data[2].e[2];
	t1 = (pp[r.posneg[2]].e[2] - r.data[0].e[2]) * r.data[2].e[2];
	if (t0 > interval_min) {interval_min = t0; g_MinMaxDim [0] = 2;}
	if (t1 < interval_max) {interval_max = t1; g_MinMaxDim [1] = 2;}

	*returnMin = interval_min;
	*returnMax = interval_max;
	return (interval_min <= interval_max);
}

FORCEINLINE int SIMDBSPTree::RayObjIntersect(const Ray &ray, ModelInstance *subObject, BSPArrayTreeNodePtr objList, HitPointPtr obj, float tmax)   
{
	float point[2];
	float vdot, vdot2;
	float alpha, beta;
	float t, u0, v0, u1, v1, u2, v2;
	int foundtri = -1;
	int count = GETCHILDCOUNT(objList);
	unsigned int idxList = GETIDXOFFSET(objList);	

	#ifdef _SIMD_SHOW_STATISTICS
	_debug_LeafIntersectCount++;
	#endif


	for (int i=0; i<count; i++,idxList++) {	
		//assert(idxList < treeStats.sumTris);		
		unsigned int triID = *MAKEIDX_PTR(subObject->indexlist,idxList);		
		const Triangle &tri = GETTRI(subObject,triID);
		assert(tri.i1 <= 2);
		assert(tri.i2 <= 2);

		#ifdef _SIMD_SHOW_STATISTICS
		_debug_LeafTriIntersectCount++;
		#endif

		// is ray parallel to plane or a backface ?
		vdot = dot(ray.direction(), tri.n);		
		if (vdot > -EPSILON)
			continue;

		// find parameter t of ray -> intersection point
		vdot2 = dot(ray.origin(),tri.n);
		t = (tri.d - vdot2) / vdot;

		// if either too near or further away than a previous hit, we stop
		if (t < -INTERSECT_EPSILON || t > (tmax + INTERSECT_EPSILON))
			continue;

		// intersection point with plane
		point[0] = ray.data[0].e[tri.i1] + ray.data[1].e[tri.i1] * t;
		point[1] = ray.data[0].e[tri.i2] + ray.data[1].e[tri.i2] * t;

		// begin barycentric intersection algorithm 
		const Vector3 &tri_p0 = GETVERTEX(subObject,tri.p[0]); 
		float p0_1 = tri_p0.e[tri.i1], p0_2 = tri_p0.e[tri.i2]; 
		u0 = point[0] - p0_1; 
		v0 = point[1] - p0_2; 
		const Vector3 &tri_p1 = GETVERTEX(subObject,tri.p[1]); 
		u1 = tri_p1[tri.i1] - p0_1; 
		v1 = tri_p1[tri.i2] - p0_2; 
		const Vector3 &tri_p2 = GETVERTEX(subObject,tri.p[2]); 
		u2 = tri_p2[tri.i1] - p0_1; 
		v2 = tri_p2[tri.i2] - p0_2;

				
		beta = (v0 * u1 - u0 * v1) / (v2 * u1 - u2 * v1);
		//if (beta < 0 || beta > 1)
		if (beta < -TRI_INTERSECT_EPSILON || beta > 1 + TRI_INTERSECT_EPSILON)
			continue;
		alpha = (u0 - beta * u2) / u1;	
		
		// not in triangle ?
		//if (alpha < 0 || (alpha + beta) > 1.0f)
		if (alpha < -TRI_INTERSECT_EPSILON || (alpha + beta) > 1.0f + TRI_INTERSECT_EPSILON)
			continue;

		// we have a hit:
		tmax = t;			 // new t value
		foundtri = triID;  // save index
		obj->alpha = alpha;  // .. and barycentric coords
		obj->beta  = beta;
	}

	// A triangle was found during intersection :
	if (foundtri >= 0) {

		// catch degenerate cases:
		if (tmax < 0.0f)
			return false;

		const Triangle &pTri = GETTRI(subObject,foundtri);		

		// Fill hitpoint structure:
		//
		#ifdef _USE_TRI_MATERIALS
		obj->m = pTri.material;
		#else
		obj->m = defaultMaterial;
		#endif
		obj->t = tmax;
		obj->triIdx = foundtri;

		obj->objectPtr = subObject;
		obj->m_TraveledDist = tmax;

		#ifdef _USE_VERTEX_NORMALS
		// interpolate vertex normals..
		obj->n = pTri.normals[0] + obj->alpha * pTri.normals[1] + obj->beta * pTri.normals[2];
		#else
		obj->n = pTri.n;
		#endif

		#ifdef _USE_TEXTURING
		// interpolate tex coords..
		obj->uv = pTri.uv[0] + obj->alpha * pTri.uv[1] + obj->beta * pTri.uv[2];
		#endif
	
		// hitpoint:
		obj->x = ray.pointAtParameter(tmax);		

		return true;
	}
	return false;	
}

#ifdef USE_LOD
FORCEINLINE int SIMDBSPTree::RayLODIntersect(const SIMDRay &rays, ModelInstance *subObject, BSPArrayTreeNodePtr currentNode, SIMDHitpoint *hits, SIMDVec4 &tmax, SIMDVec4 &tmin, int hitValue) {
	
	unsigned int idxList = GET_REAL_IDX(currentNode->lodIndex);	
	const LODNode & LOD = GET_LOD(subObject,idxList);
	SIMDVec4 vdot, vdot2, t;
	int newHitValue = hitValue;

	__m128 eps_int = _mm_set1_ps(INTERSECT_EPSILON);	

	#ifdef _SIMD_SHOW_STATISTICS
	_debug_LeafIntersectCount++;
	_debug_LeafTriIntersectCount++;
	#endif
	
	__m128 nx = _mm_load1_ps(&LOD.m_n.e[0]);
	__m128 ny = _mm_load1_ps(&LOD.m_n.e[1]);
	__m128 nz = _mm_load1_ps(&LOD.m_n.e[2]);

	// is ray parallel to plane or a backface ?
	// vdot = dot(ray.direction(), LOD.m_n);
	vdot.v4 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(_mm_load_ps(rays.direction[0]), nx), 
	                                _mm_mul_ps(_mm_load_ps(rays.direction[1]), ny)),
									_mm_mul_ps(_mm_load_ps(rays.direction[2]), nz));
	
	// ...then cull it
	newHitValue |= _mm_movemask_ps(_mm_cmpgt_ps(vdot.v4, _mm_set1_ps(-EPSILON)));
	if (newHitValue == 15)
		return hitValue;
	
	// vdot2 = dot(ray.origin(),LOD.m_n);
	vdot2.v4 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(_mm_load_ps(rays.origin[0]), nx), 
									 _mm_mul_ps(_mm_load_ps(rays.origin[1]), ny)),
						 			 _mm_mul_ps(_mm_load_ps(rays.origin[2]), nz));

	// find parameter t of ray -> intersection point
	// t = (LOD.m_d - vdot2) / vdot;
	t.v4 = _mm_div_ps(_mm_sub_ps(_mm_load1_ps(&LOD.m_d), vdot2.v4), vdot.v4);

	// if either too near or further away than a previous hit, we stop
	//if (t < (tmin-INTERSECT_EPSILON) || t > (tmax + INTERSECT_EPSILON))
	//	return false;		
	newHitValue |= _mm_movemask_ps(_mm_or_ps(_mm_cmplt_ps(t.v4, _mm_sub_ps(tmin.v4, eps_int)), 
					                           _mm_cmpgt_ps(t.v4, _mm_add_ps(tmax.v4, eps_int))
												));
	if (newHitValue == 15)
		return hitValue;

	//
	// ok, we must have at least one hit:
	//

	// mask of rays that hit at this LOD
	newHitValue = (~newHitValue & ~hitValue) & 15;
	hitValue |= newHitValue;

	// new hit points found:	
	int bitval = 1;
	for (int rnum = 0; rnum < 4; rnum++, bitval <<= 1) {
			
		// ray already hit..
		if ((newHitValue & bitval) != bitval)
			continue;
			
		// Fill hitpoint structure:
		//									

		hits->n[0][rnum] = LOD.m_n[0];
		hits->n[1][rnum] = LOD.m_n[1];
		hits->n[2][rnum] = LOD.m_n[2];			
		hits->t[rnum] = t.v[rnum];
		hits->triIdx[rnum] = 0;

		hits->alpha[rnum] = 0.5f;  // fake barycentric coords
		hits->beta[rnum]  = 0.5f;

		// LOD info:
		hits->m_HitLODIdx[rnum] = idxList;
		hits->m_ErrBnd[rnum]    = LOD.m_Proj_ErrBnd;		
		hits->m_TraveledDist[rnum] = tmin.v[rnum];

		hits->objectPtr[rnum] = subObject; 

		#ifdef _USE_TRI_MATERIALS
		hits->m[rnum] = LOD.m_material;
		#else
		hits->m[rnum] = defaultMaterial;
		#endif
	}

	// Calculate hitpoints:
	// x = rays.origin + t*rays.direction;
	_mm_store_ps(hits->x[0].e, _mm_add_ps(_mm_load_ps(rays.origin[0]), _mm_mul_ps(_mm_load_ps(rays.direction[0]), t.v4)));
	_mm_store_ps(hits->x[1].e, _mm_add_ps(_mm_load_ps(rays.origin[1]), _mm_mul_ps(_mm_load_ps(rays.direction[1]), t.v4)));
	_mm_store_ps(hits->x[2].e, _mm_add_ps(_mm_load_ps(rays.origin[2]), _mm_mul_ps(_mm_load_ps(rays.direction[2]), t.v4)));
	

	/*
	Ray realRay;
	Hitpoint obj;
	realRay.setOrigin(rays.getOrigin(0));

	int bit = 1;
	for (int i = 0; i < 4; i++, bit <<= 1) {
		realRay.setDirection(rays.getDirection(i));
		if ((hitValue & bit) == 0 && RayLODIntersect(realRay, currentNode, &obj, tmax.v[i], tmin.v[i]) {
			hits->t[i] = obj.t;			
			hits->alpha[i] = obj.alpha;
			hits->beta[i] = obj.beta;
			hits->m[i] = obj.m;
    		hits->triIdx[i] = obj.triIdx;
			hits->x[0][i] = obj.x.x();
			hits->x[1][i] = obj.x.y();
			hits->x[2][i] = obj.x.z();
			hits->n[0][i] = obj.n.x();
			hits->n[1][i] = obj.n.y();
			hits->n[2][i] = obj.n.z();
			hits->m_HitLODIdx[i] = obj.m_HitLODIdx;
			hits->m_ErrBnd[i] = obj.m_ErrBnd;
			hits->m_TraveledDist[i] = obj.m_TraveledDist;
			hitValue |= bit;
		}
	}	*/	

	return hitValue;
}

FORCEINLINE int SIMDBSPTree::RayLODIntersect(const Ray &ray, ModelInstance *subObject, 
											 BSPArrayTreeNodePtr objList, HitPointPtr obj, float tmax, float tmin)   
{	
	float vdot, vdot2, t;		
	unsigned int idxList = GET_REAL_IDX(objList->lodIndex);	
	const LODNode & LOD = GET_LOD(subObject,idxList);
	const float Allowance = LOD.m_ErrBnd * 0.2;
	//const float Allowance = LOD.m_ErrBnd * 0.2;

	#ifdef _SIMD_SHOW_STATISTICS
	_debug_LeafIntersectCount++;
	_debug_LeafTriIntersectCount++;
	#endif

	// is ray parallel to plane or a backface ?
	vdot = dot(ray.direction(), LOD.m_n);

	// ...then cull it

	if (vdot > -EPSILON)
		return false;		

	// find parameter t of ray -> intersection point		
	vdot2 = dot(ray.origin(),LOD.m_n);		
	t = (LOD.m_d - vdot2) / vdot;

	const Vector3 &origin		= ray.data[0],
				  &invdirection = ray.data[2];

	// compute new allowances considering the expanded extents of R-LODs
	//float Max_allowance = (Allowance - origin[g_MinMaxDim [1]]) * invdirection[g_MinMaxDim [1]];						
	float Max_allowance = (Allowance) * invdirection[g_MinMaxDim [1]];						
	if (Max_allowance < 0)
		Max_allowance = - Max_allowance;

	// if either too near or further away than a previous hit, we stop
	if (t < (0-INTERSECT_EPSILON) || t > (tmax + INTERSECT_EPSILON + Max_allowance))
		return false;	
	
	// we have a hit:
	// fill hitpoint structure:
	//
	obj->alpha       = 0.5f;  // fake barycentric coords
	obj->beta        = 0.5f;

	// LOD info:
	obj->m_HitLODIdx = idxList;
	obj->m_ErrBnd = LOD.m_Proj_ErrBnd;

	// hitpoint:
	obj->x = ray.pointAtParameter(t);	
	obj->m_TraveledDist = t;
	
#ifdef _USE_TRI_MATERIALS	
	obj->m = LOD.m_material;
#else
	obj->m = defaultMaterial;
#endif
	obj->t = t;
	obj->triIdx = 0;
	obj->objectPtr = subObject; 

#ifdef _USE_VERTEX_NORMALS
	// interpolate vertex normals..
	obj->n = pTri.normals[0] + obj->alpha * pTri.normals[1] + obj->beta * pTri.normals[2];
#else	
	obj->n = LOD.m_n;
#endif

#ifdef _USE_TEXTURING
	// interpolate tex coords..
	obj->uv = pTri.uv[0] + obj->alpha * pTri.uv[1] + obj->beta * pTri.uv[2];
#endif

	return true;	
}
#endif


FORCEINLINE  float SIMDBSPTree::RayObjIntersectTarget(const Ray &ray, ModelInstance *subObject, BSPArrayTreeNodePtr objList, float target_t, float ErrorBnd)   
{
	float point[2];
	float vdot, vdot2;
	float alpha, beta;
	float t, u0, v0, u1, v1, u2, v2;
	int foundtri = -1;	
	int count = GETCHILDCOUNT(objList);
	unsigned int idxList = GETIDXOFFSET(objList);	

	#ifdef _SIMD_SHOW_STATISTICS
	_debug_LeafIntersectCount++;
	#endif
	
	for (int i=0; i<count; i++,idxList++) {	
		unsigned int triID = *MAKEIDX_PTR(subObject->indexlist,idxList);
		//assert(idxList < treeStats.sumTris);		
		const Triangle &tri = GETTRI(subObject,triID);
		assert(tri.i1 <= 2);
		assert(tri.i2 <= 2);

		#ifdef _SIMD_SHOW_STATISTICS
		_debug_LeafTriIntersectCount++;
		#endif

		// is ray parallel to plane or a backface ?
		vdot = dot(ray.direction(), tri.n);

		#ifndef NO_BACKFACE_CULLING
		if (fabs(vdot) < EPSILON)
			continue;
		#endif

		// find parameter t of ray -> intersection point
		vdot2 = dot(ray.origin(),tri.n);
		t = (tri.d - vdot2) / vdot;

		// In non-LOD case, ErrorBnd is just for numerical issue
		// For LOD case, this is surface deviation gap.
		if (t < ErrorBnd || t > target_t)		
			continue;

		// intersection point with plane
		point[0] = ray.data[0].e[tri.i1] + ray.data[1].e[tri.i1] * t;
		point[1] = ray.data[0].e[tri.i2] + ray.data[1].e[tri.i2] * t;

		// begin barycentric intersection algorithm 
		const Vector3 &tri_p0 = GETVERTEX(subObject,tri.p[0]); 
		float p0_1 = tri_p0.e[tri.i1], p0_2 = tri_p0.e[tri.i2]; 
		u0 = point[0] - p0_1; 
		v0 = point[1] - p0_2; 
		const Vector3 &tri_p1 = GETVERTEX(subObject,tri.p[1]); 
		u1 = tri_p1[tri.i1] - p0_1; 
		v1 = tri_p1[tri.i2] - p0_2; 
		const Vector3 &tri_p2 = GETVERTEX(subObject,tri.p[2]); 
		u2 = tri_p2[tri.i1] - p0_1; 
		v2 = tri_p2[tri.i2] - p0_2;

		beta = (v0 * u1 - u0 * v1) / (v2 * u1 - u2 * v1);
		if (beta < 0 || beta > 1)
			continue;
		alpha = (u0 - beta * u2) / u1;
		
		// not in triangle ?
		if (alpha < 0 || (alpha + beta) > 1.0f)
			continue;

		// we have a hit:
		return t;	
	}

	return 0.0f;	
}

FORCEINLINE  float SIMDBSPTree::RayLODIntersectTarget(const Ray &ray, ModelInstance *subObject, unsigned int idxList, float tmax, float tmin)   
{
	float vdot, vdot2, t;			
	const LODNode & LOD = GET_LOD(subObject,idxList);

	#ifdef _SIMD_SHOW_STATISTICS
	_debug_LeafIntersectCount++;
	_debug_LeafTriIntersectCount++;
	#endif

	// is ray parallel to plane or a backface ?
	vdot = dot(ray.direction(), LOD.m_n);

	// ...then cull it
	//if (vdot > -EPSILON)
	//	return false;		
 
	// find parameter t of ray -> intersection point		
	vdot2 = dot(ray.origin(),LOD.m_n);		
	t = (LOD.m_d - vdot2) / vdot;

	// if either too near or further away than a previous hit, we stop
	if (t < (tmin-INTERSECT_EPSILON) || t > (tmax + INTERSECT_EPSILON))
		return false;	

	return t;
}

#ifdef USE_LOD
FORCEINLINE int SIMDBSPTree::RayLODIntersectTarget(const SIMDRay &rays, ModelInstance *subObject, BSPArrayTreeNodePtr currentNode, SIMDHitpoint *hits, SIMDVec4 &tmax, SIMDVec4 &tmin, SIMDVec4 &target_t, int hitValue)
{

	unsigned int idxList = GET_REAL_IDX(currentNode->lodIndex);	
	const LODNode & LOD = GET_LOD(subObject,idxList);
	SIMDVec4 vdot, vdot2, t;
	int newHitValue = hitValue;

	#ifdef _SIMD_SHOW_STATISTICS
	_debug_LeafIntersectCount++;
	_debug_LeafTriIntersectCount++;
	#endif

	__m128 eps_int = _mm_set1_ps(INTERSECT_EPSILON);	

	__m128 nx = _mm_load1_ps(&LOD.m_n.e[0]);
	__m128 ny = _mm_load1_ps(&LOD.m_n.e[1]);
	__m128 nz = _mm_load1_ps(&LOD.m_n.e[2]);

	// is ray parallel to plane or a backface ?
	// vdot = dot(ray.direction(), LOD.m_n);
	vdot.v4 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(_mm_load_ps(rays.direction[0]), nx), 
									_mm_mul_ps(_mm_load_ps(rays.direction[1]), ny)),
									_mm_mul_ps(_mm_load_ps(rays.direction[2]), nz));

	// ...then cull it
	newHitValue |= _mm_movemask_ps(_mm_cmpgt_ps(vdot.v4, _mm_set1_ps(-EPSILON)));
	if (newHitValue == 15)
		return hitValue;

	// vdot2 = dot(ray.origin(),LOD.m_n);
	vdot2.v4 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(_mm_load_ps(rays.origin[0]), nx), 
		 							 _mm_mul_ps(_mm_load_ps(rays.origin[1]), ny)),
									 _mm_mul_ps(_mm_load_ps(rays.origin[2]), nz));

	// find parameter t of ray -> intersection point
	// t = (LOD.m_d - vdot2) / vdot;
	t.v4 = _mm_div_ps(_mm_sub_ps(_mm_load1_ps(&LOD.m_d), vdot2.v4), vdot.v4);

	// if either too near or further away than a previous hit, we stop
	//if (t < (tmin-INTERSECT_EPSILON) || t > (tmax + INTERSECT_EPSILON))
	//	return false;		
	__m128 newHitMask =_mm_or_ps(_mm_cmplt_ps(t.v4, _mm_sub_ps(tmin.v4, eps_int)), 
					    	     _mm_cmpgt_ps(t.v4, _mm_add_ps(tmax.v4, eps_int)));
	newHitValue |= _mm_movemask_ps(newHitMask);
	if (newHitValue == 15)
		return hitValue;	

	//
	// ok, we must have at least one hit:
	//

	newHitValue = (~newHitValue & ~hitValue) & 15;	
	newHitMask = _mm_load_ps((float *)maskLUTable[newHitValue]);

	// modify target_t, set new values
	target_t.v4 = _mm_or_ps(_mm_and_ps(newHitMask, t.v4), _mm_andnot_ps(newHitMask, target_t.v4));

	return hitValue | newHitValue;
}
#endif



void SIMDBSPTree::drawNode(BSPArrayTreeNodePtr node, Vector3 min, Vector3 max, int depth, unsigned int subObjectId) {
	// Find axis (encoded in the lower two bits)	
	ModelInstance *subObject = &objectList[subObjectId];
	int axisNr = AXIS(node);		
	int count = GETCHILDCOUNT(node);
	BSPArrayTreeNodePtr children = GETNODE(subObject->tree,GETLEFTCHILD(node));
	BSPArrayTreeNodePtr children2 = GETNODE(subObject->tree,GETRIGHTCHILD(node));
	
	/*
	if (ISLEAF(node) && count == treeStats.maxTriCountPerLeaf) {		
		glColor4f(0.7f, 0, 0, 0.3f);
		unsigned int *idxList = MAKEIDX_PTR(subObject->indexlist,GETIDXOFFSET(node));

		for (int i = 0; i < count; i++) {
			const Triangle &t = GETTRI(idxList[i]);

			glBegin(GL_LINE_LOOP);
			glVertex3fv(GETVERTEX(subObject,t.p[0]).e);
			glVertex3fv(GETVERTEX(subObject,t.p[1]).e);
			glVertex3fv(GETVERTEX(subObject,t.p[2]).e);
			glEnd();
		}
	}*/

	
	glColor4f(0.9f, 0, 0, 0.3f);
	if (axisNr == 0) {
		glBegin(GL_LINE_LOOP);
		glVertex3f(node->splitcoord, min[1], min[2]);
		glVertex3f(node->splitcoord, max[1], min[2]);
		glVertex3f(node->splitcoord, max[1], max[2]);
		glVertex3f(node->splitcoord, min[1], max[2]);
		glEnd();
	}
	else if (axisNr == 1) {
		glBegin(GL_LINE_LOOP);
		glVertex3f(min[0], node->splitcoord, min[2]);
		glVertex3f(max[0], node->splitcoord, min[2]);
		glVertex3f(max[0], node->splitcoord, max[2]);
		glVertex3f(min[0], node->splitcoord, max[2]);
		glEnd();
	}
	else if (axisNr == 2) {
		glBegin(GL_LINE_LOOP);
		glVertex3f(min[0], min[1], node->splitcoord);
		glVertex3f(max[0], min[1], node->splitcoord);
		glVertex3f(max[0], max[1], node->splitcoord);
		glVertex3f(min[0], max[1], node->splitcoord);
		glEnd();
	} 

#if HIERARCHY_TYPE == TYPE_KD_TREE
	// Inner node ? Then recurse
	if (ISNOLEAF(node)) {		
		Vector3 midpoint = max;
		midpoint[axisNr] = node->splitcoord;

		drawNode(children, min, midpoint, depth + 1, subObjectId);
		drawNode(children2, midpoint, max, depth + 1, subObjectId);
	}
#endif

#if HIERARCHY_TYPE == TYPE_BVH
	// Inner node ? Then recurse
	if (ISNOLEAF(node)) {		
		Vector3 midpoint = max;
		midpoint[axisNr] = node->splitcoord;

		drawNode(children, children->min, children->max, depth + 1, subObjectId);
		drawNode(children2, children2->min, children2->max, depth + 1, subObjectId);
	}
#endif
	
}


void SIMDBSPTree::renderSplit(int axisNr, float splitCoord, Vector3 min, Vector3 max,TriangleIndexList *newlists[2]) {	
	ModelInstance *subObject = &objectList[0];
	glMatrixMode( GL_PROJECTION );
	glLoadIdentity();
	glFrustum(-1,1, -1,1, 2, 4000 );
	glMatrixMode( GL_MODELVIEW );

	cout << "Render Split at: " << splitCoord << " Axis " << axisNr << " left: " << newlists[0]->size() << " right: " << newlists[1]->size() << endl;

	glLoadIdentity();

	// Position		
	Ray viewer;
	float exVScale = 0.8f;
	Vector3 exVecScale =  exVScale*(objectBB[1] - objectBB[0]);
	Vector3 viewPos = Vector3(objectBB[0][0] - exVecScale[0], objectBB[1][1] - exVecScale[1], objectBB[1][2] + exVecScale[2]);
	Vector3 viewDir = ((objectBB[1] + objectBB[0])/2.0f + Vector3(0, 0, 0)) - viewPos;		
	
	viewer.setOrigin(viewPos);
	viewer.setDirection(viewDir);

	//;
	//viewer.setOrigin(Vector3(310.40,223.50,-153.20));
	//viewer.setDirection(Vector3(-0.83,-0.39,0.41));
	Vector3 lookAt = viewer.origin() + viewer.direction();	
	gluLookAt(viewer.origin().x(), viewer.origin().y(), viewer.origin().z(),  
		      lookAt.x(), lookAt.y(), lookAt.z(), 
		      0, 1, 0);

	// clear image
	glClearColor(0,0,0,1);
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	
	// set OpenGL state
	glDisable(GL_DEPTH_TEST);		
	glDisable(GL_TEXTURE_2D);	
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_BLEND);	

	glColor4f(0.0f, 1.0f, 0.0, 1.0);
	if (axisNr == 0) {
		glBegin(GL_LINE_LOOP);
		glVertex3f(splitCoord, min[1], min[2]);
		glVertex3f(splitCoord, max[1], min[2]);
		glVertex3f(splitCoord, max[1], max[2]);
		glVertex3f(splitCoord, min[1], max[2]);
		glEnd();
	}
	else if (axisNr == 1) {
		glBegin(GL_LINE_LOOP);
		glVertex3f(min[0], splitCoord, min[2]);
		glVertex3f(max[0], splitCoord, min[2]);
		glVertex3f(max[0], splitCoord, max[2]);
		glVertex3f(min[0], splitCoord, max[2]);
		glEnd();
	}
	else if (axisNr == 2) {
		glBegin(GL_LINE_LOOP);
		glVertex3f(min[0], min[1], splitCoord);
		glVertex3f(max[0], min[1], splitCoord);
		glVertex3f(max[0], max[1], splitCoord);
		glVertex3f(min[0], max[1], splitCoord);
		glEnd();
	}

	glColor4f(0.0f, 1.0f, 0.0, 0.6f);
	glBegin(GL_LINE_LOOP); // bottom rectangle
	glVertex3f(min[0], min[1], min[2]);
	glVertex3f(max[0], min[1], min[2]);
	glVertex3f(max[0], min[1], max[2]);
	glVertex3f(min[0], min[1], max[2]);
	glEnd();

	glBegin(GL_LINE_LOOP); // top rectangle
	glVertex3f(min[0], max[1], min[2]);
	glVertex3f(max[0], max[1], min[2]);
	glVertex3f(max[0], max[1], max[2]);
	glVertex3f(min[0], max[1], max[2]);
	glEnd();

	glBegin(GL_LINES); // sides
	glVertex3f(min[0], min[1], min[2]);
	glVertex3f(min[0], max[1], min[2]);

	glVertex3f(max[0], min[1], min[2]);
	glVertex3f(max[0], max[1], min[2]);

	glVertex3f(min[0], min[1], max[2]);
	glVertex3f(min[0], max[1], max[2]);

	glVertex3f(max[0], min[1], max[2]);
	glVertex3f(max[0], max[1], max[2]);	
	glEnd();
	


	for (int curList = 0; curList < 2; curList++) {
		// wire color
		glColor4f(1.0f * curList, 0.0, 1.0f * (1-curList), 0.5);

		for (unsigned int i = 0; i < newlists[curList]->size(); i++) {
			const Triangle &t = GETTRI(subObject,newlists[curList]->at(i));

			glBegin(GL_LINE_LOOP);
			glVertex3fv(GETVERTEX(subObject,t.p[0]).e);
			glVertex3fv(GETVERTEX(subObject,t.p[1]).e);
			glVertex3fv(GETVERTEX(subObject,t.p[2]).e);			
			glEnd();
		}
	}
	
	glutSwapBuffers();	
	getchar();

	// restore state
	glDisable(GL_BLEND);
	glDisable(GL_DEPTH_TEST);	
	glEnable(GL_TEXTURE_2D);
}

void SIMDBSPTree::GLdrawTree(Ray &viewer, unsigned int subObjectId) {	
	ModelInstance *subObject = &objectList[subObjectId];
	glMatrixMode( GL_PROJECTION );
	glLoadIdentity();
	glFrustum(-1,1, -1,1, 2, 4000 );
	glMatrixMode( GL_MODELVIEW );

	glLoadIdentity();
	Vector3 lookAt = viewer.origin() + viewer.direction();
	gluLookAt(viewer.origin().x(), viewer.origin().y(), viewer.origin().z(),  
		lookAt.x(), lookAt.y(), lookAt.z(), 
		0, 1, 0);

	// clear image
	glClearColor(0,0,0,1);
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

	// set OpenGL state
	glEnable(GL_DEPTH_TEST);		
	glDisable(GL_TEXTURE_2D);	
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_BLEND);

	// wire color
	glColor4f(0.9, 0.1, 0.1, 0.7);

	// recursive draw of BSP tree
	drawNode(GETNODE(subObject->tree,0), objectBB[0], objectBB[1], 0, subObjectId);

	// restore state
	glDisable(GL_BLEND);
	glDisable(GL_DEPTH_TEST);	
	glEnable(GL_TEXTURE_2D);
}

void SIMDBSPTree::printNodeInArray(const char *LoggerName, unsigned int currentIdx, int depth, unsigned int subObjectId) {
	ModelInstance *subObject = &objectList[subObjectId];
	LogManager *log = LogManager::getSingletonPtr();
	char outStr[500];
	char indent[100];
	char *axis = "XYZL";
	int axisNr;
	int numTris = 0;
	BSPArrayTreeNodePtr current;
	unsigned int child_left, child_right;

	current = GETNODE(subObject->tree,currentIdx);

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
	else if (depth == 0) {
		//if (tree == NULL)
		//	return;		
	}

	// find axis (encoded in the lower two bits)	
	axisNr = AXIS(current);
	child_left = GETLEFTCHILD(current);
	child_right = GETRIGHTCHILD(current);

	if (ISLEAF(current)) { // leaf		
		numTris = GETCHILDCOUNT(current);
		sprintf(outStr, "%sLeaf %d Tris", indent, numTris);
		log->logMessage(outStr, LoggerName);	
	}
	else {
		sprintf(outStr, "%sNode %c (%.2f)", indent, axis[axisNr], current->splitcoord);
		log->logMessage(outStr, LoggerName);
	}

	if (ISNOLEAF(current)) {
		printNodeInArray(LoggerName, child_left, depth + 1, subObjectId);		
		printNodeInArray(LoggerName, child_right, depth + 1, subObjectId);
	}
}

void SIMDBSPTree::printHighLevelNode(const char *LoggerName, unsigned int currentIdx, int depth) {

	LogManager *log = LogManager::getSingletonPtr();
	char outStr[500];
	char indent[100];
	char *axis = "XYZL";
	int axisNr;
	int numTris = 0;
	BSPArrayTreeNodePtr current;
	unsigned int child_left, child_right;

	current = GETHLNODE(objectTree, currentIdx);

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
	else if (depth == 0) {
		//if (tree == NULL)
		//	return;		
	}

	// find axis (encoded in the lower two bits)	
	axisNr = AXIS(current);
	child_left = GETLEFTCHILD(current);
	child_right = GETRIGHTCHILD(current);

	if (ISLEAF(current)) { // leaf		
		numTris = GETCHILDCOUNT(current);
		sprintf(outStr, "%sLeaf %d Object(s)", indent, numTris);
		log->logMessage(outStr, LoggerName);
	}
	else {
		sprintf(outStr, "%sNode %c (%.2f)", indent, axis[axisNr], current->splitcoord);
		log->logMessage(outStr, LoggerName);
	}

	if (ISNOLEAF(current)) {
		printHighLevelNode(LoggerName, child_left, depth + 1);		
		printHighLevelNode(LoggerName, child_right, depth + 1);
	}

}

void SIMDBSPTree::printHighLevelTree(bool dumpTree, const char *LoggerName) {
	LogManager *log = LogManager::getSingletonPtr();
	char outputBuffer[2000];
	log->logMessage("-------------------------------------------", LoggerName);
	log->logMessage("High-Level Tree Statistics", LoggerName);
	log->logMessage("-------------------------------------------", LoggerName);
	sprintf(outputBuffer, "Objects:\t\t%d", objTreeStats.numTris);
	log->logMessage(outputBuffer, LoggerName);
	sprintf(outputBuffer, "Nodes:\t\t%d", objTreeStats.numNodes);
	log->logMessage(outputBuffer, LoggerName);
	sprintf(outputBuffer, "Leafs:\t\t%d", objTreeStats.numLeafs);
	log->logMessage(outputBuffer, LoggerName);
	sprintf(outputBuffer, "Max. leaf depth:\t%d (of %d)", objTreeStats.maxLeafDepth, objTreeStats.maxDepth);
	log->logMessage(outputBuffer, LoggerName);
	sprintf(outputBuffer, "Max. tri count/leaf:\t%d", objTreeStats.maxTriCountPerLeaf);
	log->logMessage(outputBuffer, LoggerName);
	if (objTreeStats.numLeafs > 0) {		
		sprintf(outputBuffer, "Tri refs total:\t\t%d", objTreeStats.sumTris);
		log->logMessage(outputBuffer, LoggerName);
	}

	if (dumpTree) {
		log->logMessage("-------------------------------------------", LoggerName);
		log->logMessage("High-Level Tree structure", LoggerName);
		log->logMessage("-------------------------------------------", LoggerName);		
		printHighLevelNode(LoggerName, NULL, 0);
		log->logMessage("-------------------------------------------", LoggerName);		
	}
}

void SIMDBSPTree::printTree(bool dumpTree, const char *LoggerName, unsigned int subObjectId) {
	LogManager *log = LogManager::getSingletonPtr();
	char outputBuffer[2000];
	log->logMessage("-------------------------------------------", LoggerName);
	log->logMessage("BSP Tree Statistics", LoggerName);
	log->logMessage("-------------------------------------------", LoggerName);
	sprintf(outputBuffer, "Time to build:\t%d seconds, %d milliseconds", 
		   (int)objectList[subObjectId].treeStats.timeBuild, 
		   (int)((objectList[subObjectId].treeStats.timeBuild - floor(objectList[subObjectId].treeStats.timeBuild)) * 1000));
	log->logMessage(outputBuffer, LoggerName);
	sprintf(outputBuffer, "Triangles:\t%d", objectList[subObjectId].treeStats.numTris);
	log->logMessage(outputBuffer, LoggerName);
	sprintf(outputBuffer, "Nodes:\t\t%d", objectList[subObjectId].treeStats.numNodes);
	log->logMessage(outputBuffer, LoggerName);
	sprintf(outputBuffer, "Leafs:\t\t%d", objectList[subObjectId].treeStats.numLeafs);
	log->logMessage(outputBuffer, LoggerName);
	sprintf(outputBuffer, "Max. leaf depth:\t%d (of %d)", objectList[subObjectId].treeStats.maxLeafDepth, objectList[subObjectId].treeStats.maxDepth);
	log->logMessage(outputBuffer, LoggerName);
	sprintf(outputBuffer, "Max. tri count/leaf:\t%d", objectList[subObjectId].treeStats.maxTriCountPerLeaf);
	log->logMessage(outputBuffer, LoggerName);
	if (objectList[subObjectId].treeStats.numLeafs > 0) {
		sprintf(outputBuffer, "Avg. leaf depth:\t%.2f", (float)objectList[subObjectId].treeStats.sumDepth / (float)objectList[subObjectId].treeStats.numLeafs);
		log->logMessage(outputBuffer, LoggerName);

		sprintf(outputBuffer, "Avg. tris/leaf:\t%.2f", (float)objectList[subObjectId].treeStats.sumTris / (float)objectList[subObjectId].treeStats.numLeafs);
		log->logMessage(outputBuffer, LoggerName);

		sprintf(outputBuffer, "Tri refs total:\t\t%d", objectList[subObjectId].treeStats.sumTris);
		log->logMessage(outputBuffer, LoggerName);

	}
	sprintf(outputBuffer, "Used memory:\t%d KB", (objectList[subObjectId].treeStats.numNodes*sizeof(BSPArrayTreeNode) + (objectList[subObjectId].treeStats.sumTris * sizeof(int))) / 1024);
	log->logMessage(outputBuffer, LoggerName);

	if (dumpTree) {
		log->logMessage("-------------------------------------------", LoggerName);
		log->logMessage("BSP Tree structure", LoggerName);
		log->logMessage("-------------------------------------------", LoggerName);		
		printNodeInArray(LoggerName, NULL, 0);
		log->logMessage("-------------------------------------------", LoggerName);		
	}
}
int SIMDBSPTree::getNumTris(unsigned int subObjectId) {
	return objectList[subObjectId].treeStats.numTris;
}

void SIMDBSPTree::dumpCounts() {
	cout << "Tree Stats after trace:\n";
	cout << "Tree Intersections:\t\t" << _debug_TreeIntersectCount << endl;
	cout << "Inner node intersections:\t" << _debug_NodeIntersections << endl;	
	cout << "Overhead:\t\t\t" << _debug_NodeIntersectionsOverhead << " (" << ((double)(_debug_NodeIntersectionsOverhead)/(double)_debug_NodeIntersections)*100.0f << "%)" << endl;

	// does not work at the moment.
	//cout << "Leaf node intersections:\t" << _debug_LeafIntersectCount << endl;	
	//cout << "Overhead:\t\t\t" << _debug_LeafIntersectOverhead << " (" << ((double)(_debug_LeafIntersectOverhead)/(double)_debug_LeafIntersectCount)*100.0f << "%)" << endl;

	cout << "Tri Intersections:\t\t" << _debug_LeafTriIntersectCount << endl;

	if (_debug_TreeIntersectCount > 0)
		cout << "Avg. Node Intersects / Ray:\t" << (double)_debug_LeafIntersectCount/(double)_debug_TreeIntersectCount << endl;
	if (_debug_TreeIntersectCount > 0)
		cout << "Avg. Tri Intersects / Ray:\t" << (double)_debug_LeafTriIntersectCount/(double)_debug_TreeIntersectCount << endl;	
	if (_debug_LeafIntersectCount > 0)
		cout << "Avg. Tri Intersects / Node:\t" << (double)_debug_LeafTriIntersectCount/(double)_debug_LeafIntersectCount << endl;	
}

FORCEINLINE void SIMDBSPTree::initializeObjectIntersection(int threadID) {	
	stackPtrHL[threadID] = 0;
	stacksHL[threadID][0].node = 0;	
	stacksHL[threadID][0].min.v4 = _mm_setzero_ps();
	stacksHL[threadID][0].min.v4 = _mm_setzero_ps();
	//cout << "initializeObjectIntersection()" << endl;
}

FORCEINLINE ModelInstance *SIMDBSPTree::getNextObjectIntersection(Ray &ray, int threadID, float *returnMin, float *returnMax) {
	StackElem *stack = stacksHL[threadID];	
	BSPArrayTreeNodePtr currentNode;
	unsigned int currentOffset;
	int currentAxis;
	float min, max;	
	const Vector3 &origin = ray.data[0], 
				  &invdirection = ray.data[2];			
	
	// are we still in a previous call?
	if (stackPtrHL[threadID] > 1) {
		// yes, get next node from stack
		stackPtrHL[threadID]--;
		currentOffset = stack[stackPtrHL[threadID]].node;
		currentNode = GETHLNODE(objectTree, currentOffset);
		min = stack[stackPtrHL[threadID]].min.v[0];
		max = stack[stackPtrHL[threadID]].max.v[0];
		ray.undoTransform();
	}
	else if (stackPtrHL[threadID] == 0){
		// does intersect whole scene?
		if (!RayBoxIntersect(ray, objectBB[0], objectBB[1], returnMin, returnMax)) 
			return NULL;

		currentOffset = INT_MAX;
		min = EPSILON;
		max = FLT_MAX;
		currentNode = GETHLNODE(objectTree,0);
		stackPtrHL[threadID] = 1;
	}
	else
		return false;
	
	// traverse BSP tree:
	while (currentOffset != 0) {		
		currentAxis = AXIS(currentNode);

		// while we are not a leaf..
		while ( ISNOLEAF(currentNode) ) {

			// calculate distance to splitting plane
			float dist = (currentNode->splitcoord - origin[currentAxis]) * invdirection[currentAxis];						

			int childSelect = ray.posneg[currentAxis];

			if (dist < min-BSP_EPSILON)  {
				currentNode = GETHLNODE(objectTree, GETCHILDNUM(currentNode, childSelect));
			}
			else if (dist > max+BSP_EPSILON) {
				currentNode = GETHLNODE(objectTree, GETCHILDNUM(currentNode, childSelect ^ 1));
			}
			else {
				stack[stackPtrHL[threadID]].node = GETCHILDNUM(currentNode, childSelect);
				stack[stackPtrHL[threadID]].min.v[0] = dist;
				stack[stackPtrHL[threadID]].max.v[0] = max;	
				stackPtrHL[threadID]++;				
				currentNode = GETHLNODE(objectTree, GETCHILDNUM(currentNode, childSelect ^ 1));
				max = dist;
			}

			currentAxis = AXIS(currentNode);
		}

		// intersect with the object's bounding box:		
		if (GETCHILDCOUNT(currentNode)) {
			int subObjectId = *MAKEHLIDX_PTR(objectIndexList, currentNode->indexOffset);
			ModelInstance *subObject = &objectList[subObjectId];

			if (RayBoxIntersect(ray, subObject->bb[0], subObject->bb[1], returnMin, returnMax)) {

				// transform ray to local coordinate system:
				ray.transform(&subObject->translate_world);

				return subObject; // the ray intersects this object
			}
		}
			
		// no hit or empty leaf, continue:
		stackPtrHL[threadID]--;
		currentOffset = stack[stackPtrHL[threadID]].node;
		currentNode = GETHLNODE(objectTree, currentOffset);
		min = stack[stackPtrHL[threadID]].min.v[0];
		max = stack[stackPtrHL[threadID]].max.v[0];
	} 

	// no further hit with tree
	return NULL;
}

FORCEINLINE ModelInstance *SIMDBSPTree::getNextObjectIntersection(SIMDRay &rays, int threadID, SIMDVec4 &returnMin, SIMDVec4 &returnMax) {
	StackElem *stack = stacksHL[threadID];	
	BSPArrayTreeNodePtr currentNode;
	unsigned int currentOffset;
	int currentAxis;
	SIMDVec4 min, max;	

	//cout << "getNextObjectIntersection()" << endl;
	
	// are we still in a previous call?
	if (stackPtrHL[threadID] > 1) {
		//cout << " --> still traversing!" << endl;
		// yes, get next node from stack
		stackPtrHL[threadID]--;
		currentOffset = stack[stackPtrHL[threadID]].node;
		currentNode = GETHLNODE(objectTree,currentOffset);		
		min.v4 = stack[stackPtrHL[threadID]].min.v4;
		max.v4 = stack[stackPtrHL[threadID]].max.v4;

		rays.undoTransform();
	}
	else if (stackPtrHL[threadID] == 0){
		//cout << " --> first node traverse" << endl;

		// does intersect whole scene?
		if (!RayBoxIntersect(rays, objectBB, returnMin, returnMax)) 
			return NULL;

		//cout << " --> hit scene!" << endl;

		currentOffset = INT_MAX;
		min.v4 = _mm_set1_ps(EPSILON);
		max.v4 = _mm_set1_ps(FLT_MAX);
		currentNode = GETHLNODE(objectTree,0);
		stackPtrHL[threadID] = 1;
	}
	else {
		//cout << " --> traverse end." << endl;
		return 0;
	}

	__m128 epsilon = _mm_set1_ps(BSP_EPSILON);

	// traverse BSP tree:	
	while (currentOffset != 0) {
		int axis;

		while (ISNOLEAF(currentNode)) {
			axis = AXIS(currentNode);

			const __m128 splitCoord = _mm_load1_ps(&currentNode->splitcoord);
			const __m128 rayOrigin = _mm_load_ps(rays.origin[axis]);
			const __m128 invDirection = _mm_load_ps(rays.invdirection[axis]);

			//
			// calculate dist to splitting plane:
			// dist = (currentNode->splitcoord - origin[currentAxis]) * invdirection[currentAxis];
			//

			__m128 dist = _mm_mul_ps(_mm_sub_ps(splitCoord, rayOrigin), invDirection);		

			// assume far child first
			int childSelect = rays.rayChildOffsets[axis];

			// need to take far child?
			if (_mm_movemask_ps(_mm_cmpgt_ps(dist, _mm_sub_ps(min.v4, epsilon))) == 0) {			
				currentNode = GETHLNODE(objectTree,GETCHILDNUM(currentNode, childSelect));				
				continue;
			} 
			// or just near child?
			else if (_mm_movemask_ps(_mm_cmple_ps(dist, _mm_add_ps(max.v4, epsilon))) == 0) { 						
				// node = near node
				currentNode = GETHLNODE(objectTree,GETCHILDNUM(currentNode, childSelect ^ 1));
				continue;
			} 
			else { // need to intersect both children:				
				const __m128 mask2 = _mm_cmpgt_ps(dist, min.v4);
				const __m128 mask3 = _mm_cmplt_ps(dist, max.v4);

				stack[stackPtrHL[threadID]].node = GETCHILDNUM(currentNode, childSelect);
				stack[stackPtrHL[threadID]].max.v4 = max.v4;

				// update max values:
				max.v4 = _mm_or_ps( _mm_and_ps( mask3, dist ), _mm_andnot_ps( mask3, max.v4 ) );
				stack[stackPtrHL[threadID]].min.v4 = _mm_or_ps( _mm_and_ps( mask2, dist ), _mm_andnot_ps( mask2, min.v4 ) );
				max.v4 = dist;

				stackPtrHL[threadID]++;

				// select near node
				currentNode = GETHLNODE(objectTree,GETCHILDNUM(currentNode, childSelect ^ 1));
			}
		}

		axis = AXIS(currentNode);

		if (GETCHILDCOUNT(currentNode)) {
			int subObjectId = *MAKEHLIDX_PTR(objectIndexList, currentNode->indexOffset);
			ModelInstance *subObject = &objectList[subObjectId];

			//cout << " --> found child." << subObjectId << endl;

			if (RayBoxIntersect(rays, subObject->bb, returnMin, returnMax)) {

				//cout << " --> ray hits child." << subObjectId << endl;

				// transform ray to local coordinate system:
				rays.transform(&subObject->translate_world);

				return subObject; // the ray intersects this object
			}
		}

		// no, at least one ray needs to be traced further

		// get far child from stack
		stackPtrHL[threadID]--;
		currentOffset = stack[stackPtrHL[threadID]].node;
		currentNode = GETHLNODE(objectTree,currentOffset);		
		min.v4 = stack[stackPtrHL[threadID]].min.v4;
		max.v4 = stack[stackPtrHL[threadID]].max.v4;
	}

	// no further hit with tree
	return NULL;
}