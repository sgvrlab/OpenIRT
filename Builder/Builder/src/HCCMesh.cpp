#include <iostream>
#include <stdio.h>
#include "HCCMesh.h"
#include <io.h>
#include "Files.h"
#include <direct.h>

#define GETNODE(object, offset) ((TREE_CLASS*)&((*object)[offset]))
#define GETTRI(object, offset) ((TrianglePtr)&((*object)[offset]))
#define GETCHILDCOUNT(node) ((node)->indexCount >> 2)
#define GETIDXOFFSET(node) ((node)->indexOffset)
#define GETINDEX(object, node) ((unsigned int *)&((*object)[(node)->triIndex >> 2]))
#define GETVERTEX(object, offset) ((Vertex*)&((*object)[offset]))

#define GETCLUSTER(node) ((node)->indexCount >> 10)
#define GETCHILDCOUNT2(node) (((node)->indexCount && 3FF) >> 2)

#ifdef USE_LOD
const unsigned int ERR_BITs = 5;
#define GETLOD(object, offset) ((LODNodePtr)&((*object)[offset]))
#define CONVERT_VEC3F_VECTOR3(s,d) {d.e [0] = s.x;d.e [1] = s.y; d.e[2] = s.z;}
#define CONVERT_VECTOR3_VEC3F(s,d) {d.x = s.e [0];;d.y = s.e [1];d.z = s.e [2];}

const int LOD_DEPTH_GRANUALITY = 1; 
const float MIN_VOLUME_REDUCTION_BW_LODS = pow ((float)2, (int)LOD_DEPTH_GRANUALITY);
const int MIN_NUM_TRIANGLES_LOD = pow ((float)2, (int)LOD_DEPTH_GRANUALITY);
const unsigned int MAX_NUM_LODs = UINT_MAX;
const unsigned int LOD_BIT = 1;
//#define HAS_LOD(idx) (idx & LOD_BIT)
#define HAS_LOD(idx) (idx != 0)
#define GET_REAL_IDX(idx) (idx >> ERR_BITs)
//#define GET_REAL_IDX(idx) (idx >> LOD_BIT)
#endif

#define GETROOT() 0
#define ISLEAF(node) (((node)->children & 3) == 3)
#define GETLEFTCHILD(node) ((node)->children >> 2)
#define GETRIGHTCHILD(node) (((node)->children >> 2) + 1)

#define CISLEAF(node) (((node)->data & 1) == 1)
#define CISLEAFOFPATCH(node) (((node)->data & 0x3) == 2)
#define CISINCOMPLETETREE(node) (((node)->data & 8) == 8)

#ifdef CONVERT_PHOTON
#undef GETROOT
#undef GETRIGHTCHILD
#undef CISLEAF
#undef CISLEAFOFPATCH
#define GETROOT() 1
#define GETPHOTON(object, offset) ((PhotonPtr)&((*object)[offset]))
#define GETRIGHTCHILD(node) ((node)->children2 >> 2)
#define CISLEAF(node) (((node)->plane & 0x4) == 0x4)
#define CISLEAFOFPATCH(node) (((node)->plane & 0xC) == 0x8)
#endif

//#include "stopwatch.hpp"
//Stopwatch T1("T1");
//Stopwatch T2("T2");
//Stopwatch T3("T3");
//Stopwatch T4("T4");
//Stopwatch T5("T5");
HCCMesh::HCCMesh()
{
	numClusters = 0;
	numLowLevelNodes = 0;
	maxNodesPerCluster = 2048;
	maxVertsPerCluster = 512;
	curCluster = 0;
	fileSize = fileSizeHeader = fileSizeNode = fileSizeSupp = fileSizeVert = fileSizeTris = 0;
	numAxis[0] = numAxis[1] = numAxis[2] = 0;
	nQuantize = 104;

/*////
#ifdef GENERATE_OUT_OF_CORE_REP
	fileSizeHeaderOut =	fileSizeHighNodeOut = fileSizeNodeOut = fileSizeSuppOut = fileSizeMeshOut =	fileSizeAddVertOut = 0;
	offset = bOffset = 0;
	offsetGeom = bOffsetGeom = 0;
#endif
*/////
}

HCCMesh::~HCCMesh()
{
}

int HCCMesh::getTreeStat(TreeStat &treeStat, unsigned int startIndex, int depth)
{
	TREE_CLASS* node = GETNODE(tree, startIndex);
	treeStat.numNodes++;

	if(ISLEAF(node))
	{
		if(treeStat.minDepth > depth) treeStat.minDepth = depth;
		if(treeStat.maxDepth < depth) treeStat.maxDepth = depth;
		treeStat.numLeafs++;
		treeStat.numTris += GETCHILDCOUNT(node);
		return 1;
	}

	getTreeStat(treeStat, GETLEFTCHILD(node), depth+1);
	getTreeStat(treeStat, GETRIGHTCHILD(node), depth+1);
	return 1;
}

int HCCMesh::convertHCCMesh(unsigned int nodeIndex, int &numNodes, VertexHash &v, int depth, unsigned int parentIndex, int type)
{
	static __int64 ss = 0;
	ss++;

	TREE_CLASS* node = GETNODE(tree, nodeIndex);
	if(ISLEAF(node))
	{
		numNodes = 1;
#ifdef CONVERT_PHOTON
		v.insert(std::pair<unsigned int, unsigned int>(nodeIndex, nodeIndex));
#else
		unsigned int idx = node->indexOffset;//*GETINDEX(indices, node);
		TrianglePtr tri = GETTRI(tris, idx);
		v.insert(std::pair<unsigned int, unsigned int>(tri->p[0], tri->p[0]));
		v.insert(std::pair<unsigned int, unsigned int>(tri->p[1], tri->p[1]));
		v.insert(std::pair<unsigned int, unsigned int>(tri->p[2], tri->p[2]));
		//numVertices = 3;
#endif
		return 1;
	}

	int numLeftNodes = 0, numRightNodes = 0;
	VertexHash rightVertices;

	int leftStat = convertHCCMesh(GETLEFTCHILD(node) , numLeftNodes, v, depth+1, nodeIndex, 1);
	int rightStat = convertHCCMesh(GETRIGHTCHILD(node), numRightNodes, rightVertices, depth+1, nodeIndex, 2);

	numNodes = numLeftNodes + numRightNodes + 1;

	if(leftStat == 0 || rightStat == 0)
	{
		TREE_CLASS curNode = *node;

#ifdef USE_LOD
		unsigned int lodIndex = curNode.lodindex;
#endif
		// just make sure
		curNode.children2 = GETRIGHTCHILD(node) << 2;

		unsigned int leftClusterFileSize, rightClusterFileSize;

		if(leftStat != 0) 
		{
			unsigned int clusterID = curCluster;
			leftClusterFileSize = makeCluster(GETLEFTCHILD(node), numLeftNodes, nodeIndex, 1);
			for(int i=0;i<numLeftNodes;i++) prog.step();
			int axis = curNode.children & 0x3;
			curNode.children = (clusterID << 2) | axis;
			curNode.children2 |= 0x2;
		}
		if(rightStat != 0) 
		{
			unsigned int clusterID = curCluster;
			if(nodeIndex == 4)
			{
				printf("!!!! %d %d 2\n", GETRIGHTCHILD(node), nodeIndex);
				printf("%u %u %f %f %f %f %f %f\n", node->children, node->children2, node->min.e[0], node->min.e[1], node->min.e[2], node->max.e[0], node->max.e[1], node->max.e[2]);
				exit(-1);
			}
			rightClusterFileSize = makeCluster(GETRIGHTCHILD(node), numRightNodes, nodeIndex, 2);
			for(int i=0;i<numRightNodes;i++) prog.step();
			int childStat = curNode.children2 & 0x3;
			curNode.children2 = (clusterID << 2) | childStat;
			curNode.children2 |= 0x1;
		}

		prog.step();
#ifdef USE_LOD
		highLODIndexHash.insert(std::pair<unsigned int, unsigned int>(nodeIndex, lodIndex));
#endif
		treeHighNodeHash.insert(std::pair<unsigned int, TREE_CLASS>(nodeIndex, curNode));
		return 0;
	}


	for(VertexHashIterator it = rightVertices.begin(); it != rightVertices.end(); ++it)
	{
		if(v.find(it->second) == v.end())
		{
			v.insert(std::pair<unsigned int, unsigned int>(it->second, it->second));
		}
	}
	
	if(numNodes > maxNodesPerCluster || v.size() > maxVertsPerCluster || depth <= 2)
	{
		TREE_CLASS curNode = *node;

#ifdef USE_LOD
		unsigned int lodIndex = curNode.lodindex;
#endif
		// just make sure
		curNode.children2 = GETRIGHTCHILD(node) << 2;

		unsigned int leftClusterFileSize, rightClusterFileSize;
		unsigned int leftClusterID, rightClusterID;

		leftClusterID = curCluster;
		leftClusterFileSize = makeCluster(GETLEFTCHILD(node), numLeftNodes, nodeIndex, 3);
		for(int i=0;i<numLeftNodes;i++) prog.step();
		rightClusterID = curCluster;
		rightClusterFileSize = makeCluster(GETRIGHTCHILD(node), numRightNodes, nodeIndex, 4);
		for(int i=0;i<numRightNodes;i++) prog.step();

		int axis = curNode.children & 0x3;
		curNode.children = (leftClusterID << 2) | axis;
		curNode.children2 = (rightClusterID << 2) | 0x3;

#ifdef USE_LOD
		highLODIndexHash.insert(std::pair<unsigned int, unsigned int>(nodeIndex, lodIndex));
#endif
		treeHighNodeHash.insert(std::pair<unsigned int, TREE_CLASS>(nodeIndex, curNode));
		return 0;
	}
	return 1;
}

unsigned int HCCMesh::makeCluster(unsigned int nodeIndex, int numNodes, unsigned int parentIndex, int type)
{
	//cout << "Cluster " << numClusters << " has " << numNodes << " nodes" << endl;
	if(processedClusters.find(nodeIndex) != processedClusters.end())
	{
		printf("Already made cluster! root index = %d\n", nodeIndex);
	}
	processedClusters.insert(nodeIndex);

	curCNodeIndex = 0;
	curCSuppIndex = 0;
	curCVertIndex = 0;
	curCTriIndex = 0;
	compTreeNode.clear();
	compTreeSupp.clear();
	clusterVert.clear();
	tempTri.clear();

	vClusterHash = new VertexHash();
	lClusterHash = new LinkHash();

	TreeStat treeStat;
	getTreeStat(treeStat, nodeIndex, 1);

	list<unsigned int> qBFS;
	vector<unsigned int> listCompleteTreeIndex;
	vector<unsigned int> listGeneralTreeIndex;

	int sizeCompleteTree = (1 << treeStat.minDepth) - 1;

	qBFS.push_back(nodeIndex);
	for(int i=0;i<sizeCompleteTree;i++)
	{
		unsigned int index = qBFS.front();
		listCompleteTreeIndex.push_back(index);
		TREE_CLASS* node = GETNODE(tree, index);
		qBFS.pop_front();
		if(!ISLEAF(node))
		{
			qBFS.push_back(GETLEFTCHILD(node));
			qBFS.push_back(GETRIGHTCHILD(node));
		}
	}

	// copy from queue to vector list
	for(list<unsigned int>::iterator it = qBFS.begin() ; it != qBFS.end(); ++it)
	{
		listGeneralTreeIndex.push_back(*it);
	}

/*////
#ifdef GENERATE_OUT_OF_CORE_REP
	compTreeNodeOut = new BitCompression((FILE*)0, 0);
#endif
*/////

#ifdef USE_COMPLETE_TREE
	StoreCompleteTreeNodes(listCompleteTreeIndex);
	StoreGeneralNodes(listGeneralTreeIndex, treeStat);
#else
	vector<unsigned int> nextFront;
	StoreGeneralNodes(nodeIndex, treeStat.maxDepth, nextFront);
#endif

	delete vClusterHash;
	delete lClusterHash;

	assert(curCNodeIndex == treeStat.numNodes);

	TravStat ts;
	unsigned int vList[9] = {0, };
	memset(&ts, 0, sizeof(TravStat));


	// find template type of root of this cluster
	rootType = -1;
	TREE_CLASS* localRootNode = GETNODE(tree, nodeIndex);
	if(ISLEAF(localRootNode))
	{
		rootType = numTemplates-1;
	}
	else
	{
		for(int i=0;i<numTemplates;i++)
		{
			if(isSameTree(nodeIndex, 0, 1, templates[i].tree))
			{
				rootType = i;
				break;
			}
		}
	}
	assert(rootType >= 0);

	ts.type = rootType;

	computeBB(0, vList, ts, 1);

	// check the tree structure
	TreeStat testTreeStat;
	memset(&testTreeStat, 0, sizeof(TreeStat));
	memset(&ts, 0, sizeof(TravStat));

	ts.type = rootType;

	testTraverse(0, testTreeStat, ts);
	assert(treeStat.numNodes == testTreeStat.numNodes);
#ifndef CONVERT_PHOTON
	assert(treeStat.numLeafs == testTreeStat.numLeafs);
#endif


	numLowLevelNodes += numNodes;
	numClusters++;


	unsigned int clusterNumNode = compTreeNode.size();
	unsigned int clusterNumSupp = compTreeSupp.size();
	unsigned int clusterNumVert = clusterVert.size();

	TREE_CLASS* node = GETNODE(tree, nodeIndex);
	unsigned int clusterFileSize = sizeof(CompClusterHeader) + clusterNumNode*sizeof(CTREE_CLASS) + clusterNumSupp*sizeof(CompTreeSupp) + clusterNumVert*sizeof(CompTreeVert);

	CompClusterHeader header;
	header.fileSize = clusterFileSize;
	header.numNode = clusterNumNode;
	header.numSupp = clusterNumSupp;
	header.numVert = clusterNumVert;
#ifndef CONVERT_PHOTON
	header.BBMin = localRootNode->min;
	header.BBMax = localRootNode->max;
#endif
	header.rootType = rootType;

#ifdef USE_LOD
	// compute lods
	compLOD.clear();
	ComputeSimpRep(header);
	QuantizeErr(header);

	unsigned int numLODs = compLOD.size();
	fwrite(&numLODs, sizeof(unsigned int), 1, fpLODCluster);
	for(int i=0;i<numLODs;i++) fwrite(&compLOD[i], sizeof(CompLOD), 1, fpLODCluster);
#endif

	fileSizeHeader += sizeof(CompClusterHeader);
	fileSizeNode += clusterNumNode*sizeof(CTREE_CLASS);
	fileSizeSupp += clusterNumSupp*sizeof(CompTreeSupp);
	fileSizeVert += clusterNumVert*sizeof(CompTreeVert);
	fileSize = fileSizeHeader + fileSizeNode + fileSizeSupp + fileSizeVert;

	/*
	char clusterFileName[255];
	sprintf(clusterFileName, "%s/Cluster%d", filePath, curCluster);
	FILE *fpCluster = fopen(clusterFileName, "wb");
	fwrite(&header, sizeof(CompClusterHeader), 1, fpCluster);
	for(int i=0;i<clusterNumNode;i++) fwrite(&compTreeNode[i], sizeof(CTREE_CLASS), 1, fpCluster);
	for(int i=0;i<clusterNumSupp;i++) fwrite(&compTreeSupp[i], sizeof(CompTreeSupp), 1, fpCluster);
	for(int i=0;i<clusterNumVert;i++) fwrite(&clusterVert[i], sizeof(CompTreeVert), 1, fpCluster);
	fclose(fpCluster);
	*/
	fwrite(&header, sizeof(CompClusterHeader), 1, fpCluster);
	for(int i=0;i<clusterNumNode;i++) fwrite(&compTreeNode[i], sizeof(CTREE_CLASS), 1, fpCluster);
	for(int i=0;i<clusterNumSupp;i++) fwrite(&compTreeSupp[i], sizeof(CompTreeSupp), 1, fpCluster);
	for(int i=0;i<clusterNumVert;i++) fwrite(&clusterVert[i], sizeof(CompTreeVert), 1, fpCluster);

/*////
#ifdef GENERATE_OUT_OF_CORE_REP

#ifdef CONVERT_PHOTON
	compressClusterHCCPhoton(curCluster, nodeIndex, header);
#else
	compressClusterHCCMesh(curCluster, nodeIndex, header);

#ifdef USE_LOD
	compressClusterHCCLOD();
#endif

#ifdef USE_GZIP
	char gzipClusterOut[255];
	sprintf(gzipClusterOut, "%s/cluster%d", gzipDir, curCluster);
	gzFile fpGzip = gzopen(gzipClusterOut, "wb");

	gzwrite(fpGzip, &header, sizeof(CompClusterHeader));
	for(int i=0;i<clusterNumNode;i++) gzwrite(fpGzip, &compTreeNode[i], sizeof(CTREE_CLASS));
	for(int i=0;i<clusterNumSupp;i++) gzwrite(fpGzip, &compTreeSupp[i], sizeof(CompTreeSupp));
	for(int i=0;i<clusterNumVert;i++) gzwrite(fpGzip, &clusterVert[i], sizeof(CompTreeVert));

	gzclose(fpGzip);
#endif

#endif

#endif
*////
	curCluster++;

	return clusterFileSize;
}

#ifdef USE_COMPLETE_TREE
int HCCMesh::StoreCompleteTreeNodes(vector<unsigned int> &listNodeIndex)
{
	int numNodes = listNodeIndex.size();
	int numLeafs = (numNodes+1)/2;
	int numInners = numNodes - numLeafs;

	// store inner nodes of complete tree
	for(int i=0;i<numInners;i++)
	{
		TREE_CLASS* nodeS = GETNODE(tree, listNodeIndex[i]);
		CTREE_CLASS nodeC;
		
		nodeC.data = (0x8) | (nodeS->children & 0x3);

		compTreeNode.push_back(nodeC);

		curCNodeIndex++;
	}

	// store leaf nodes of complete tree
	for(int i=numInners;i<numNodes;i++)
	{
		TREE_CLASS* nodeS = GETNODE(tree, listNodeIndex[i]);
		CTREE_CLASS nodeC;

		int axis = nodeS->children & 0x3;

		if(ISLEAF(nodeS))
		{
			// temporary store the three vertices of a triangle.
			unsigned int triIdx = *GETINDEX(indices, nodeS);
			TrianglePtr tri = GETTRI(tris, triIdx);

			VertexHashIterator it;
			unsigned int vIdx = -1;

			TempTri tTri;
			for(int p=0;p<3;p++)
			{
				it = vClusterHash->find(tri->p[p]);
				if(it == vClusterHash->end())
				{
					// new vertex of this cluster
					vIdx = curCVertIndex;
					CompTreeVert cv;
					cv.vert.set((*GETVERTEX(verts, tri->p[p])).v.e);
					clusterVert.push_back(cv);
					vClusterHash->insert(std::pair<unsigned int, unsigned int>(tri->p[p], vIdx));
					curCVertIndex++;
				}
				else vIdx = it->second;
				assert(vIdx < (1<<16));
				tTri.p[p] = (unsigned short)vIdx;
			}

			tempTri.push_back(tTri);

			nodeC.data = curCTriIndex;
			nodeC.data <<= 4;
			nodeC.data |= axis;
			
			compTreeNode.push_back(nodeC);

			curCTriIndex++;
			assert(curCTriIndex < (1<<16));
		}
		else
		{
			lClusterHash->insert(std::pair<unsigned int, unsigned int>(GETLEFTCHILD(nodeS), (curCSuppIndex << 1) | 0x1));
			lClusterHash->insert(std::pair<unsigned int, unsigned int>(GETRIGHTCHILD(nodeS), (curCSuppIndex << 1) | 0x0));
			CompTreeSupp supp;
			supp.leftIndex = 0;
			supp.rightIndex = 0;
			supp.data = 0;

			compTreeSupp.push_back(supp);

			nodeC.data = (curCSuppIndex << 4);// | (nodeS->children & 0x3);
			nodeC.data |= 0xC;
			nodeC.data |= axis;

			compTreeNode.push_back(nodeC);

			curCSuppIndex++;
		}

		curCNodeIndex++;
	}
	return 1;
}
#endif

int HCCMesh::StoreGeneralNodes(vector<unsigned int> &listNodeIndex, TreeStat &treeStat)
{
	for(int i=0;i<listNodeIndex.size();i++)
	{
		vector<unsigned int> nextFront;
		StoreGeneralNodes(listNodeIndex[i], treeStat.maxDepth - treeStat.minDepth, nextFront);
	}
	return 1;
}

int HCCMesh::StoreGeneralNodes(unsigned int startIndex, int maxDepth, vector<unsigned int> &nextFront)
{
	if(maxDepth <= 4)
	{
		TREE_CLASS* rootNode = GETNODE(tree, startIndex);

		int templateType = -1;
		if(ISLEAF(rootNode))
		{
			templateType = numTemplates-1;
		}
		else
		{
			for(int i=0;i<numTemplates;i++)
			{
				if(isSameTree(startIndex, 0, 1, templates[i].tree))
				{
					templateType = i;
					break;
				}
			}
		}
		assert(templateType >= 0);

		// make link between parent and child links
		LinkHashIterator itLink = lClusterHash->find(startIndex);
		if(itLink != lClusterHash->end())
		{
			int isLeftChild = itLink->second & 0x1;
			unsigned int suppIndex = itLink->second >> 1;

			CompTreeSupp &supp = compTreeSupp[suppIndex];
			if(isLeftChild)
			{
				/*
				supp.leftIndex = curCNodeIndex;
				supp.data |= ((unsigned int)templateType) << 27;
				*/
				assert(curCNodeIndex < 2048);
				supp.leftIndex = (curCNodeIndex << 5) | templateType;
			}
			else
			{
				/*
				supp.rightIndex = curCNodeIndex;
				supp.data |= ((unsigned int)templateType) << 22;
				*/
				assert(curCNodeIndex < 2048);
				supp.rightIndex = (curCNodeIndex << 5) | templateType;
			}
		}
		else
		{
			//cout << "This node is a root of this cluster" << endl;
		}

		TemplateTable t = templates[templateType];
		unsigned int map[MAX_SIZE_TEMPLATE];
		findCorrIndex(startIndex, 0, t.tree, map);

		// store tree structure
		for(int i=0;i<t.numNodes;i++)
		{
			unsigned int nodeIdx = map[i];
			TREE_CLASS* nodeS = GETNODE(tree, nodeIdx);
			TREE_CLASS* nodeT = GETNODE(&t.tree, i);
			CTREE_CLASS nodeC;
#ifdef CONVERT_PHOTON
			nodeC.pos = nodeS->pos;
			nodeC.phi = nodeS->phi;
			nodeC.theta = nodeS->theta;
			nodeC.plane = nodeS->plane;
			memcpy(nodeC.power, nodeS->power, sizeof(unsigned short)*4);
#endif

			if(ISLEAF(nodeS))
			{
#ifdef CONVERT_PHOTON
				nodeC.plane |= (0x3 << 2);
				
				compTreeNode.push_back(nodeC);
#else
				// temporary store the three vertices of a triangle.
				unsigned int triIdx = nodeS->indexOffset;//*GETINDEX(indices, nodeS);
				TrianglePtr tri = GETTRI(tris, triIdx);

				VertexHashIterator it;
				unsigned int vIdx = -1;

				TempTri tTri;
				for(int p=0;p<3;p++)
				{
					it = vClusterHash->find(tri->p[p]);
					if(it == vClusterHash->end())
					{
						// new vertex of this cluster
						vIdx = curCVertIndex;
						CompTreeVert cv;
						Vertex ov = *GETVERTEX(verts, tri->p[p]);
#ifdef USE_VERTEX_QUANTIZE
						pqVert.EnQuantize(ov.v.e, cv.qV);
#else
						cv.vert.set(ov.v.e);
#endif
						unsigned int data = 0;
						unsigned int r, g, b;
						r = ov.c.e[0] * 31.0f;
						g = ov.c.e[1] * 63.0f;
						b = ov.c.e[2] * 31.0f;
						int nIdx = quantizeVector(ov.n);
						
						//assert(nIdx >= 0 && nIdx < 65536);
						if(!(nIdx >= 0 && nIdx < 65536))
						{
							nIdx = 0;
						}

						Vector3 qN = quantizedNormals[nIdx];

						data |= (nIdx & 0xFFFF) << 16;
						/*
						data |= (r & 0x1F) << 11;
						data |= (g & 0x3F) << 5;
						data |= (b & 0x1F);
						*/
						data |= tri->material;
#ifdef USE_VERTEX_QUANTIZE
						cv.data = data;
#else
						cv.vert.m_alpha = *((float *)&data);
#endif
						clusterVert.push_back(cv);
						vClusterHash->insert(std::pair<unsigned int, unsigned int>(tri->p[p], vIdx));
						curCVertIndex++;
					}
					else vIdx = it->second;
					assert(vIdx < (1<<16));
					tTri.p[p] = (unsigned short)vIdx;
				}

				tempTri.push_back(tTri);

				nodeC.data = curCTriIndex;
				nodeC.data <<= 5;
				nodeC.data |= 0x3;
				
				compTreeNode.push_back(nodeC);

				curCTriIndex++;
				assert(curCTriIndex < (1<<16));
#endif

/*////
#ifdef GENERATE_OUT_OF_CORE_REP
				compTreeNodeOut->encode(2, 0x3);
#endif
*/////
			}
			else if(ISLEAF(nodeT))
			{
				// leaf node of this patch
				lClusterHash->insert(std::pair<unsigned int, unsigned int>(GETLEFTCHILD(nodeS), (curCSuppIndex << 1) | 0x1));
				lClusterHash->insert(std::pair<unsigned int, unsigned int>(GETRIGHTCHILD(nodeS), (curCSuppIndex << 1) | 0x0));

				CompTreeSupp supp;
				supp.leftIndex = 0;
				supp.rightIndex = 0;
				supp.data = 0;

				compTreeSupp.push_back(supp);

#ifdef CONVERT_PHOTON
				nodeC.plane |= (curCSuppIndex << 5);
				nodeC.plane |= 0x2 << 2;
#else
				nodeC.data = (curCSuppIndex << 5);
				nodeC.data |= 0x2;
#endif

				compTreeNode.push_back(nodeC);

/*////
#ifdef GENERATE_OUT_OF_CORE_REP
				compTreeNodeOut->encode(2, 0x2);
#endif
*/////
				curCSuppIndex++;
			}
			else
			{
				// inner node of this patch
#ifdef CONVERT_PHOTON
				nodeC.plane |= 0x0 << 2;
#else
				nodeC.data = 0;//nodeS->children & 0x3;
#endif

				compTreeNode.push_back(nodeC);
/*////
#ifdef GENERATE_OUT_OF_CORE_REP
				compTreeNodeOut->encode(2, 0x0);
#endif
*/////
			}

			curCNodeIndex++;
		}

		// generate next front
		for(int i=0;i<t.numLeafs;i++)
		{
			TREE_CLASS* node = GETNODE(tree, map[t.listLeaf[i]]);
			if(!ISLEAF(node))
			{
				nextFront.push_back(GETLEFTCHILD(node));
				nextFront.push_back(GETRIGHTCHILD(node));
			}
		}
		return 1;
	}

	int midDepth = ceil(float(maxDepth / 2) / 4) * 4;

	vector<unsigned int> nextTopFront;

	StoreGeneralNodes(startIndex, midDepth, nextTopFront);

	vector<unsigned int> nextSubFront;
	for(int i=0;i<nextTopFront.size();i++)
	{
		unsigned int startIndex = nextTopFront[i];
		nextSubFront.clear();
		StoreGeneralNodes(startIndex, maxDepth - midDepth, nextSubFront);

		for(int j=0;j<nextSubFront.size();j++)
		{
			nextFront.push_back(nextSubFront[j]);
		}
	}
	return 1;
}

////#include "Stopwatch.hpp"
int HCCMesh::convertHCCMesh(const char* filepath, unsigned int maxNodesPerCluster, unsigned int maxVertsPerCluster)
{
	//Stopwatch T("Compression Time");
	//T.Start();
	strcpy(filePath, filepath);
	this->maxNodesPerCluster = maxNodesPerCluster;
	this->maxVertsPerCluster = maxVertsPerCluster;

	OptionManager *opt = OptionManager::getSingletonPtr();

	/*
	LogManager *log = LogManager::getSingletonPtr();	

	// read tree stats
	char bspFileName[MAX_PATH];
	char header_[100];
	char bspfilestring[50];
	char output[1000];
	size_t ret;

	sprintf(bspFileName, "%s/BVH", filepath);

	FILE *fp = fopen(bspFileName, "rb");
	if (fp == NULL) {
		sprintf(output, "Could not open BSP tree file '%s'!", bspFileName);
		log->logMessage(LOG_WARNING, output);
		return false;
	}

	
	sprintf(output, "Loading BSP tree from files ('%s')...", bspFileName);
	log->logMessage(LOG_INFO, output);

	ret = fread(header_, 1, BSP_FILEIDSTRINGLEN + 1, fp);
	if (ret != (BSP_FILEIDSTRINGLEN + 1)) {
		sprintf(output, "Could not read header from BSP tree file '%s', aborting. (empty file?)", bspFileName);
		log->logMessage(LOG_ERROR, output);
		return false;
	}

	// test header format:
	strcpy(bspfilestring, BSP_FILEIDSTRING);
	for (int i = 0; i < BSP_FILEIDSTRINGLEN; i++) {
		if (header_[i] != bspfilestring[i]) {
			printf(output, "Invalid BSP tree header, aborting. (expected:'%c', found:'%c')", bspfilestring[i], header_[i]);
			log->logMessage(LOG_ERROR, output);
			return false;		
		}
	}

	// test file version:
	if (header_[BSP_FILEIDSTRINGLEN] != BSP_FILEVERSION) {
		printf(output, "Wrong BSP tree file version (expected:%d, found:%d)", BSP_FILEVERSION, header_[BSP_FILEIDSTRINGLEN]);
		log->logMessage(LOG_ERROR, output);
		return false;		
	}

	// format correct, read in full BSP tree info structure:

	// write count of nodes and tri indices:
	ret = fread(&m_treeStats, sizeof(BSPTreeInfo), 1, fp);
	if (ret != 1) {
		sprintf(output, "Could not read tree info header!");
		log->logMessage(LOG_ERROR, output);
		return false;
	}
	fclose(fp);
	*/

	char srcTreeName[255];
	char srcTrisName[255];
	char srcIndexName[255];
	char srcVertsName[255];
	char dstName[255];

#ifdef CONVERT_PHOTON
	sprintf(srcTreeName, "%s/photons.ooc", filepath);
	sprintf(srcTrisName, "%s/tris.ooc", filepath);
	sprintf(srcIndexName, "%s/BVH.idx", filepath);
	sprintf(srcVertsName, "%s/vertex.ooc", filepath);
#else
	sprintf(srcTreeName, "%s/BVH.node", filepath);
	sprintf(srcTrisName, "%s/tris.ooc", filepath);
	sprintf(srcIndexName, "%s/BVH.idx", filepath);
	sprintf(srcVertsName, "%s/vertex.ooc", filepath);
#endif

#ifdef CONVERT_PHOTON
	sprintf(dstName, "%s/photons.hccmesh", filepath);
#else
#ifdef USE_VERTEX_QUANTIZE
	sprintf(dstName, "%s/data.hccmesh2", filepath);
#else
	sprintf(dstName, "%s/data.hccmesh", filepath);
#endif
#endif

	tree = new OOCFile6464<TREE_CLASS>(srcTreeName, 
							1024*1024*512,
							1024*1024*4);
	tris = new OOCFile6464<Triangle>(srcTrisName, 
							1024*1024*256,
							1024*1024*4);
	indices = new OOCFile6464<unsigned int>(srcIndexName, 
							1024*1024*64,
							1024*1024*4);
	verts = new OOCFile6464<Vertex>(srcVertsName, 
							1024*1024*128,
							1024*1024*4);

#ifdef USE_LOD
	char srcLODsName[255];
	char dstLODName[255];
	char clusterLODName[255];
	char clusterLODNameOut[255];
	sprintf(srcLODsName, "%s/BVH.lod", filepath);
	sprintf(clusterLODName, "%s/Cluster.lod", filepath);
	sprintf(clusterLODNameOut, "%s/lod.hccmeshOut", filepath);
	LODs = new OOCFile64<LODNode>(srcLODsName, 
							1024*1024*opt->getOptionAsInt("ooc", "maxCacheMemLODsMB", 64),
							1024*opt->getOptionAsInt("ooc", "cacheEntrySizeLODsKB", 1024*4));
	sprintf(dstLODName, "%s/lod.hccmesh", filepath);
	fpLOD = fopen(dstLODName, "wb");
	fpLODCluster = fopen(clusterLODName, "wb");
	fpLODClusterOut = fopen(clusterLODNameOut, "wb");

	char fileNameQErr[255];
	sprintf(fileNameQErr, "%s/BVH.QErr", filepath);
	FILE *fpQErr = fopen(fileNameQErr, "rb");
	fread(&m_QuanErr, sizeof(CErrorQuan), 1, fpQErr);
	fclose(fpQErr);
#endif
	fpComp = fopen(dstName, "wb");

	TREE_CLASS* root = GETNODE(tree, GETROOT());

#ifdef USE_VERTEX_QUANTIZE
	pqVert.SetMinMax(root->min.e, root->max.e);
	pqVert.SetPrecision(16);
	pqVert.SetupQuantizer();
#endif

	//char triName[255];
	//sprintf(triName, "%s/tri.tmp", filepath);
	//fpTri = fopen(triName, "wb");

	//
	calculateQNormals();
	//
	
	char clusterFileName[255];
	sprintf(clusterFileName, "%s/Cluster", filePath);
	fpCluster = fopen(clusterFileName, "wb");

/*////
#ifdef GENERATE_OUT_OF_CORE_REP
	char dstNameOut[255];
#ifdef CONVERT_PHOTON
	sprintf(dstNameOut, "%s/photons.hccmeshOut", filepath);
#else
#ifdef USE_RANGE_ENCODER
	sprintf(dstNameOut, "%s/data_range.hccmeshOut", filepath);
#else
	sprintf(dstNameOut, "%s/data.hccmeshOut", filepath);
#endif
#endif
	fpCompOut = fopen(dstNameOut, "wb");

	char clusterOutFileName[255], clusterOutGeomFileName[255];
	sprintf(clusterOutFileName, "%s/ClusterOut", filePath);
	sprintf(clusterOutGeomFileName, "%s/dataGeom.hccmeshOut", filePath);
	fpClusterOut = fopen(clusterOutFileName, "wb");
	fpClusterOutGeom = fopen(clusterOutGeomFileName, "wb");

#ifdef CONVERT_PHOTON
	// calculate global bounding box
	cout << "Calcuate global bounding box of photons" << endl;
	globalBBMin.e[0] = FLT_MAX;
	globalBBMin.e[1] = FLT_MAX;
	globalBBMin.e[2] = FLT_MAX;
	globalBBMax.e[0] = -FLT_MAX;
	globalBBMax.e[1] = -FLT_MAX;
	globalBBMax.e[2] = -FLT_MAX;
	__int64 numPhotons = tree->m_fileSize.QuadPart/sizeof(TREE_CLASS);
	for(__int64 i=0;i<numPhotons;i++)
	{
		TREE_CLASS *photon = GETNODE(tree, i);
		updateBB(globalBBMin, globalBBMax, photon->pos);
	}
#else
	globalBBMin = root->min;
	globalBBMax = root->max;
#endif

#ifdef USE_GZIP
	sprintf(gzipDir, "%s/gzipCluster", filepath);
	if(!dirExists(gzipDir))
	{
		mkdir(gzipDir);
	}
#endif

#endif
*/////
	int numNodes;
	VertexHash v;
	curCluster = 1;

	FILE *fpTemp = fopen(srcTreeName, "r");
	numNodes = _filelengthi64(fileno(fpTemp))/sizeof(TREE_CLASS);
	fclose(fpTemp);

	prog.init("Converting", numNodes, 100);
	int numProcessedNodes = 0;
	convertHCCMesh(GETROOT(), numProcessedNodes, v, 1, -1, 0);
	if(numNodes != numProcessedNodes)
	{
		printf("Error! Node %d->%d\n", numNodes, numProcessedNodes);
	}

	reassignHighNodeStruct(GETROOT(), -1, -1);

	// store templates
	fwrite(&numTemplates, sizeof(int), 1, fpComp);
	fwrite(templates, sizeof(TemplateTable), numTemplates, fpComp);

	// store number of clusters
	fwrite(&curCluster, sizeof(unsigned int), 1, fpComp);

	// store high level tree node
	unsigned int numHighNode = treeHighNode.size();
	CompClusterHeader header;
	header.fileSize = sizeof(CompClusterHeader) + sizeof(TREE_CLASS)*numHighNode;
	header.numNode = numHighNode;
	header.numSupp = 0;
	header.numVert = 0;
#ifndef CONVERT_PHOTON
	header.BBMin = root->min;
	header.BBMax = root->max;
#endif

	fwrite(&header, sizeof(CompClusterHeader), 1, fpComp);
	for(int i=0;i<numHighNode;i++) 
		fwrite(&treeHighNode[i], sizeof(TREE_CLASS), 1, fpComp);

	char buf[4096];
	unsigned int size = 4096;
	__int64 totalSize = 0;
	__int64 stepSize = 0;

	fclose(fpCluster);
	fpCluster = fopen(clusterFileName, "rb");

	while(size > 0)
	{
		size = fread(buf, 1, size, fpCluster);
		totalSize += size;
		stepSize += size;
		if(stepSize >= 1024*1024*100)
		{
			cout << totalSize << " bytes included" << endl;
			stepSize = 0;
		}
		fwrite(buf, 1, size, fpComp);
	}
	cout << totalSize << " bytes included" << endl;
	fclose(fpCluster);
	unlink(clusterFileName);

/*////
#ifdef GENERATE_OUT_OF_CORE_REP
	bOffset = 0;

	// store templates
	fwrite(&numTemplates, sizeof(int), 1, fpCompOut);
	fwrite(templates, sizeof(TemplateTable), numTemplates, fpCompOut);

	// store number of clusters
	fwrite(&curCluster, sizeof(unsigned int), 1, fpCompOut);
	offset = _ftelli64(fpCompOut);
	fileSizeHeaderOut += offset-bOffset;
	bOffset = offset;

	// store high level tree node
	header.fileSize = sizeof(CompClusterHeader) + sizeof(TREE_CLASS)*numHighNode;
	header.numNode = numHighNode;
	header.numSupp = 0;
	header.numVert = 0;
	header.sizeTris = 0;
#ifndef CONVERT_PHOTON
	header.BBMin = root->min;
	header.BBMax = root->max;
#endif

	fwrite(&header, sizeof(CompClusterHeader), 1, fpCompOut);
	for(int i=0;i<numHighNode;i++) fwrite(&treeHighNode[i], sizeof(TREE_CLASS), 1, fpCompOut);
	offset = _ftelli64(fpCompOut);
	fileSizeHighNodeOut += offset-bOffset;
	bOffset = offset;

	// store file size of each cluster (out)
	for(int i=0;i<clusterFileSizeOut.size();i++)
	{
		fwrite(&clusterFileSizeOut[i], sizeof(unsigned int), 1, fpCompOut);
	}
	for(int i=0;i<clusterGeomFileSizeOut.size();i++)
	{
		fwrite(&clusterGeomFileSizeOut[i], sizeof(unsigned int), 1, fpCompOut);
	}

	// store file size of each cluster (in)
	for(int i=0;i<clusterFileSizeIn.size();i++)
	{
		fwrite(&clusterFileSizeIn[i], sizeof(unsigned int), 1, fpCompOut);
	}

	fclose(fpClusterOut);
	fclose(fpClusterOutGeom);
	fpClusterOut = fopen(clusterOutFileName, "rb");
	size = 4096;
	while(size > 0)
	{
		size = fread(buf, 1, size, fpClusterOut);
		fwrite(buf, 1, size, fpCompOut);
	}
	fclose(fpClusterOut);
	unlink(clusterOutFileName);
	fclose(fpCompOut);

#ifdef USE_GZIP

	char gzipHeaderOut[255];
	sprintf(gzipHeaderOut, "%s/header", gzipDir);
	gzFile fpGzipHeader = gzopen(gzipHeaderOut, "wb");

	// store templates
	gzwrite(fpGzipHeader, &numTemplates, sizeof(int));
	gzwrite(fpGzipHeader, templates, sizeof(TemplateTable)*numTemplates);

	// store number of clusters
	gzwrite(fpGzipHeader, &curCluster, sizeof(unsigned int));

	// store high level tree node
	header.fileSize = sizeof(CompClusterHeader) + sizeof(TREE_CLASS)*numHighNode;
	header.numNode = numHighNode;
	header.numSupp = 0;
	header.numVert = 0;
	header.numTris = 0;
	header.BBMin = root->min;
	header.BBMax = root->max;

	gzwrite(fpGzipHeader, &header, sizeof(CompClusterHeader));
	for(int i=0;i<numHighNode;i++) gzwrite(fpGzipHeader, &treeHighNode[i], sizeof(TREE_CLASS));

	//
	//// store file size of each cluster (out)
	//for(int i=0;i<clusterFileSizeOut.size();i++)
	//{
	//	gzwrite(fpGzipHeader, &clusterFileSizeOut[i], sizeof(unsigned int));
	//}
	//for(int i=0;i<clusterGeomFileSizeOut.size();i++)
	//{
	//	gzwrite(fpGzipHeader, &clusterGeomFileSizeOut[i], sizeof(unsigned int));
	//}

	//// store file size of each cluster (in)
	//for(int i=0;i<clusterFileSizeIn.size();i++)
	//{
	//	gzwrite(fpGzipHeader, &clusterFileSizeIn[i], sizeof(unsigned int));
	//}
	//

	gzclose(fpGzipHeader);
#endif

#endif
*/////
	cout << "numClusters = " << numClusters << endl;
	cout << "numLowLevelNodes = " << numLowLevelNodes << endl;
	cout << "file size = " << fileSize << endl;
	cout << "  header = " << fileSizeHeader << endl;
#ifdef CONVERT_PHOTON
	cout << "  photon = " << fileSizeNode << endl;
	cout << "  supplementary data = " << fileSizeSupp << endl;
#else
	cout << "  node = " << fileSizeNode << endl;
	cout << "  supplementary data = " << fileSizeSupp << endl;
	cout << "  vertex = " << fileSizeVert << endl;
	printf("compression ratio = %.1f : 1\n", (tree->m_fileSize.QuadPart + tris->m_fileSize.QuadPart + verts->m_fileSize.QuadPart) / (float)fileSize);
	printf("compression ratio of node = %.1f : 1\n", (tree->m_fileSize.QuadPart) / (float)(fileSizeHeader + fileSizeNode + fileSizeSupp));
#endif

/*////
#ifdef GENERATE_OUT_OF_CORE_REP
	cout<< "Out-of-core Representation" << endl;
	cout << "file size = " << fileSizeHeaderOut + fileSizeHighNodeOut + fileSizeNodeOut + fileSizeSuppOut + fileSizeMeshOut + fileSizeAddVertOut << endl;
	//
	//cout << " header = " << fileSizeHeaderOut << endl;
	//cout << " high level node = " << fileSizeHighNodeOut << endl;
	//cout << " node = " << fileSizeNodeOut << endl;
	//cout << " supplement data for hierarchy = " << fileSizeSuppOut << endl;
	//cout << " mesh = " << fileSizeMeshOut << endl;
	//cout << " additional vertex information(normal, color, texture coordinate) = " << fileSizeAddVertOut << endl;
	//
	printf("%15lld : header\n", fileSizeHeaderOut);
#ifdef CONVERT_PHOTON
	printf("%15lld : high level photon\n", fileSizeHighNodeOut);
	printf("%15lld : photon\n", fileSizeNodeOut);
	printf("%15lld : supplement data for hierarchy\n", fileSizeSuppOut);
#else
	printf("%15lld : high level node\n", fileSizeHighNodeOut);
	printf("%15lld : node\n", fileSizeNodeOut);
	printf("%15lld : supplement data for hierarchy\n", fileSizeSuppOut);
	printf("%15lld : mesh\n", fileSizeMeshOut);
	printf("%15lld : additional vertex information(normal, color, texture coordinate)\n", fileSizeAddVertOut);
#endif
#endif
*/////

#ifdef USE_LOD
	unsigned int numLODs = highLOD.size();
	fwrite(&numLODs, sizeof(unsigned int), 1, fpLOD);
	for(int i=0;i<highLODIndexList.size();i++)
		fwrite(&highLODIndexList[i], sizeof(unsigned int), 1, fpLOD);
	for(int i=0;i<highLOD.size();i++)
		fwrite(&highLOD[i], sizeof(LODNode), 1, fpLOD);
	fclose(fpLODCluster);
	fclose(fpLODClusterOut);
	fpLODCluster = fopen(clusterLODName, "rb");

	size = 4096;
	totalSize = 0;
	stepSize = 0;

	while(size > 0)
	{
		size = fread(buf, 1, size, fpLODCluster);
		totalSize += size;
		stepSize += size;
		if(stepSize >= 1024*1024*100)
		{
			cout << totalSize << " bytes included (LOD)" << endl;
			stepSize = 0;
		}
		fwrite(buf, 1, size, fpLOD);
	}

	fclose(fpLODCluster);
	unlink(clusterLODName);
	fclose(fpLOD);
#endif

	delete tree;
	delete tris;
	delete indices;
	delete verts;
	fclose(fpComp);

	//fclose(fpTri);
	//T.Stop();
	//cout << T << endl;
	return 1;
}

int HCCMesh::convertHCCMesh2(unsigned int nodeIndex, VertexSet &vs)
{
	TREE_CLASS *node = GETNODE(tree, nodeIndex);

	if(ISLEAF(node))
	{
		int numTris = GETCHILDCOUNT(node);
		for(int i=0;i<numTris;i++)
		{
			Triangle *tri = GETTRI(tris, GETIDXOFFSET(node) + i);
			vs.insert(tri->p[0]);
			vs.insert(tri->p[1]);
			vs.insert(tri->p[2]);
		}
		return 1;
	}

	VertexSet vsl, vsr;

	int lStat = convertHCCMesh2(GETLEFTCHILD(node), vsl);
	int rStat = convertHCCMesh2(GETRIGHTCHILD(node), vsr);
	if(lStat) vs.insert(vsl.begin(), vsl.end());
	if(rStat) vs.insert(vsr.begin(), vsr.end());

	if(vs.size() > 1024 || lStat + rStat == 1)
	{
		if(lStat)
		{
			vertOffsetTable.push_back(0);
			vertOffsetTable[makeCluster2(GETLEFTCHILD(node))] = curVertOffset;
			curVertOffset += vsl.size();
		}
		if(rStat)
		{
			vertOffsetTable.push_back(0);
			vertOffsetTable[makeCluster2(GETRIGHTCHILD(node))] = curVertOffset;
			curVertOffset += vsr.size();
		}
		return 0;
	}

	return lStat && rStat;
}

int HCCMesh::makeCluster2(unsigned int nodeIndex, VertexHash &vh)
{
	TREE_CLASS *node = GETNODE(tree, nodeIndex);
	if(ISLEAF(node))
	{
		int numTris = GETCHILDCOUNT(node);
		for(int i=0;i<numTris;i++)
		{
			unsigned triIdx = GETIDXOFFSET(node) + i;
			Triangle *tri = GETTRI(tris, triIdx);
			unsigned int newTri = 0;
			for(int j=0;j<3;j++)
			{
				int vert = tri->p[j];
				if(vh.find(vert) == vh.end())
				{
					Vert16 v16;
					v16.v = GETVERTEX(verts, vert)->v;
					v16.m = tri->material;
					fwrite(&v16, sizeof(Vert16), 1, fpVertsH2);

					vh[vert] = vh.size();
				}

				newTri |= vh[vert] << (j*10+2);
			}
			newTri |= ((tri->i1 << 1) & 0x2) | ((tri->i2 - 1) & 0x1);
			_fseeki64(fpTrisH2, ((__int64)triIdx)*sizeof(unsigned int), SEEK_SET);
			fwrite(&newTri, sizeof(unsigned int), 1, fpTrisH2);
		}
		TREE_CLASS newNode = *node;
		newNode.indexCount = (numClusters << 10) | node->indexCount;
		_fseeki64(fpTreeH2, ((__int64)nodeIndex)*sizeof(TREE_CLASS), SEEK_SET);
		fwrite(&newNode, sizeof(TREE_CLASS), 1, fpTreeH2);
		return 1;
	}

	makeCluster2(GETLEFTCHILD(node), vh);
	makeCluster2(GETRIGHTCHILD(node), vh);
	return 0;
}

int HCCMesh::makeCluster2(unsigned int nodeIndex)
{
	int ret = numClusters;

	VertexHash vh;
	makeCluster2(nodeIndex, vh);

	numClusters++;
	return ret;
}

int HCCMesh::convertHCCMesh2(const char* fileName)
{
	//Stopwatch T("Compression Time");
	//T.Start();

	char srcDirName[255];
	char srcTreeName[255];
	char srcTrisName[255];
	char srcVertsName[255];
	char dstDirName[255];
	char dstTreeName[255];
	char dstTrisName[255];
	char dstVertsName[255];
	char dstHeaderName[255];

	OptionManager *opt = OptionManager::getSingletonPtr();

	const char *baseDirName = opt->getOption("global", "scenePath", "");

	sprintf(srcDirName, "%s%s.ooc", baseDirName, fileName);
	sprintf(dstDirName, "%s%s.hccmesh2", baseDirName, fileName);
	mkdir(dstDirName);

	sprintf(srcTreeName, "%s/BVH.node", srcDirName);
	sprintf(srcTrisName, "%s/tris.ooc", srcDirName);
	sprintf(srcVertsName, "%s/vertex.ooc", srcDirName);

	sprintf(dstTreeName, "%s/BVH.hccmesh2", dstDirName);
	sprintf(dstTrisName, "%s/tris.hccmesh2", dstDirName);
	sprintf(dstVertsName, "%s/vertex.hccmesh2", dstDirName);
	sprintf(dstHeaderName, "%s/header", dstDirName);

	// copy BVH
	FILE *fp1, *fp2;
	fp1 = fopen(srcTreeName, "rb");
	fp2 = fopen(dstTreeName, "wb");
	int read;
	char buf[1024];
	while(read = fread(buf, 1, 1024, fp1))
		fwrite(buf, 1, read, fp2);
	fclose(fp1);
	fclose(fp2);

	tree = new OOCFile6464<TREE_CLASS>(srcTreeName, 
							1024*1024*512,
							1024*1024*4);
	tris = new OOCFile6464<Triangle>(srcTrisName, 
							1024*1024*256,
							1024*1024*4);
	verts = new OOCFile6464<Vertex>(srcVertsName, 
							1024*1024*128,
							1024*1024*4);

	fpTreeH2 = fopen(dstTreeName, "r+b");
	fpTrisH2 = fopen(dstTrisName, "wb");
	fpVertsH2 = fopen(dstVertsName, "wb");
	
	curVertOffset = 0;
	convertHCCMesh2(0, VertexSet());

	fclose(fpTreeH2);
	fclose(fpTrisH2);
	fclose(fpVertsH2);

	delete tree;
	delete tris;
	delete verts;

	FILE *fpHeader = fopen(dstHeaderName, "wb");
	for(int i=0;i<vertOffsetTable.size();i++)
	{
		fwrite(&vertOffsetTable[i], sizeof(unsigned int), 1, fpHeader);
	}
	fclose(fpHeader);

	return 1;
}

int HCCMesh::generateTemplates()
{
	static const int n = 26;
	static const int nodes[n][30] = {
		{15, 1, 2, 3, 4, 9, 10, 5, 6, 7, 8, 0,  0,  0,  0, 11, 12, 13, 14, 0, 0, 0, 0,},
		{7 , 1, 2, 3, 4, 0,  5, 6, 0, 0, 0,},
		{7 , 1, 2, 3, 4, 0,  0, 5, 6, 0, 0,}, 
		{9 , 1, 2, 3, 4, 0,  5, 6, 7, 8, 0, 0,  0,  0,}, 
		{9 , 1, 2, 3, 4, 7,  8, 5, 6, 0, 0, 0,  0,  0,}, 
		{9 , 1, 2, 3, 4, 7,  8, 0, 5, 6, 0, 0,  0,  0,}, 
		{11, 1, 2, 3, 4, 9, 10, 5, 6, 7, 8, 0,  0,  0,  0,  0,  0,}, 
		{11, 1, 2, 3, 4, 7,  8, 5, 6, 0, 0, 0,  9, 10,  0,  0,  0,}, 
		{11, 1, 2, 3, 4, 7,  8, 0, 5, 6, 0, 0,  9, 10,  0,  0,  0,}, 
		{13, 1, 2, 3, 4, 9, 10, 5, 6, 7, 8, 0,  0,  0,  0, 11, 12,  0,  0, 0,}, 
		{11, 1, 2, 3, 4, 7,  8, 5, 6, 0, 0, 0,  0,  9, 10,  0,  0,}, 
		{11, 1, 2, 3, 4, 7,  8, 0, 5, 6, 0, 0,  0,  9, 10,  0,  0,}, 
		{13, 1, 2, 3, 4, 9, 10, 5, 6, 7, 8, 0,  0,  0,  0,  0, 11, 12,  0, 0,}, 
		{13, 1, 2, 3, 4, 7,  8, 5, 6, 0, 0, 0,  9, 10, 11, 12,  0,  0,  0, 0,}, 
		{13, 1, 2, 3, 4, 7,  8, 0, 5, 6, 0, 0,  9, 10, 11, 12,  0,  0,  0, 0,}, 
		{7 , 1, 2, 0, 3, 4,  5, 6, 0, 0, 0,}, 
		{7 , 1, 2, 0, 3, 4,  0, 5, 6, 0, 0,},
		{9 , 1, 2, 0, 3, 4,  5, 6, 7, 8, 0, 0,  0,  0,}, 
		{9 , 1, 2, 3, 4, 5,  6, 0, 0, 7, 8, 0,  0,  0,},
		{9 , 1, 2, 3, 4, 5,  6, 0, 0, 0, 7, 8,  0,  0,},
		{11, 1, 2, 3, 4, 5,  6, 0, 0, 7, 8, 9, 10,  0,  0,  0,  0,}, 
		{7 , 1, 2, 3, 4, 5,  6, 0, 0, 0, 0,},
		{5 , 1, 2, 3, 4, 0,  0, 0,},
		{5 , 1, 2, 0, 3, 4,  0, 0,},
		{3 , 1, 2, 0, 0,},
		{1 , 0,}};

	numTemplates = n;
	for(int i=0;i<n;i++)
	{
		int numNodes = nodes[i][0];
		templates[i].numNodes = numNodes;
		templates[i].numLeafs = 0;

		int pos = 1;

		for(int j=0;j<numNodes;j++)
		{
			TREE_CLASS &node = templates[i].tree[j];
			unsigned int left, right;
			left = nodes[i][pos++];
			if(left == 0)
			{
				// leaf node
				node.children = 7;
				templates[i].listLeaf[templates[i].numLeafs++] = j;
			}
			else
			{
				// inner node
				right = nodes[i][pos++];
				node.children = left << 2;
				node.children2 = right << 2;
			}
		}
	}

	return 1;
}

int HCCMesh::generateTemplates(const char* srcFileName)
{
	FILE *fp = fopen(srcFileName, "r");
	int n;

	fscanf(fp, "%d", &n);
	numTemplates = n;
	for(int i=0;i<n;i++)
	{
		int numNodes;
		fscanf(fp, "%d", &numNodes);
		templates[i].numNodes = numNodes;
		templates[i].numLeafs = 0;
		for(int j=0;j<numNodes;j++)
		{
			TREE_CLASS &node = templates[i].tree[j];
			unsigned int left, right;
			fscanf(fp, "%d", &left);
			if(left == 0)
			{
				// leaf node
				node.children = 7;
				templates[i].listLeaf[templates[i].numLeafs++] = j;
			}
			else
			{
				// inner node
				fscanf(fp, "%d", &right);
				node.children = left << 2;
				node.children2 = right << 2;
			}
		}
	}

	fclose(fp);
	return 1;
}

int HCCMesh::isSameTree(unsigned int startIndexA, unsigned int startIndexB, int depth, TREE_CLASS t[])
{
	TREE_CLASS *nodeA, *nodeB;
	nodeA = GETNODE(tree, startIndexA);
	if(t)
		nodeB = GETNODE(&t, startIndexB);
	else
		nodeB = GETNODE(tree, startIndexB);

	if(t)
	{
		if(depth >= 4) return true;
	}

	if(ISLEAF(nodeA) || ISLEAF(nodeB))
	{
		if(ISLEAF(nodeA) && ISLEAF(nodeB)) return true;
		return false;
	}

	return isSameTree(GETLEFTCHILD(nodeA), GETLEFTCHILD(nodeB), depth+1, t) && isSameTree(GETRIGHTCHILD(nodeA), GETRIGHTCHILD(nodeB), depth+1, t);
}

int HCCMesh::findCorrIndex(unsigned int startIndexS, unsigned int startIndexT, TREE_CLASS t[], unsigned int map[])
{
	map[startIndexT] = startIndexS;
	TREE_CLASS *nodeS, *nodeT;
	nodeS = GETNODE(tree, startIndexS);
	nodeT = GETNODE(&t, startIndexT);
	if(ISLEAF(nodeS) || ISLEAF(nodeT))
	{
		return 1;
	}
	findCorrIndex(GETLEFTCHILD(nodeS), GETLEFTCHILD(nodeT), t, map);
	findCorrIndex(GETRIGHTCHILD(nodeS), GETRIGHTCHILD(nodeT), t, map);
	return 1;
}

#if 0
unsigned int HCCMesh::CGETLEFTCHILD(CTREE_CLASS* node, TravStat &ts)
{
	unsigned int leftChild;
	if(CISLEAFOFPATCH(node))
	{
		// leaf of a patch or complete tree
		unsigned int suppIndex = (node->data >> 5) & BIT_MASK_9;
		CompTreeSuppPtr supp = &compTreeSupp[suppIndex];
		/*
		leftChild = supp->leftIndex;
		ts.type = (supp->data >> 27) & BIT_MASK_5;
		*/
		leftChild = supp->leftIndex >> 5;
		ts.type = supp->data & BIT_MASK_5;
		ts.rootTemplate = leftChild;
		ts.index = leftChild;
		ts.isLeft = 1;
		return leftChild;
	}
#ifdef USE_COMPLETE_TREE
	if(CISINCOMPLETETREE(node))
	{
		leftChild = ts.index * 2 + 1;
		ts.index = leftChild;
		ts.isLeft = 1;
		return leftChild;
	}
#endif
	unsigned int offset = ts.index - ts.rootTemplate;
	leftChild = GETLEFTCHILD(&templates[ts.type].tree[offset]) + ts.rootTemplate;
	ts.index = leftChild;
	ts.isLeft = 1;
	return leftChild;
}

unsigned int HCCMesh::CGETRIGHTCHILD(CTREE_CLASS* node, TravStat &ts)
{
	unsigned int rightChild;
	if(CISLEAFOFPATCH(node))
	{
		// leaf of a patch or complete tree
		unsigned int suppIndex = (node->data >> 5) & BIT_MASK_9;
		CompTreeSuppPtr supp = &compTreeSupp[suppIndex];
		/*
		rightChild = supp->rightIndex;
		ts.type = (supp->data >> 22) & BIT_MASK_5;
		*/
		rightChild = supp->rightIndex >> 5;
		ts.type = supp->rightIndex & BIT_MASK_5;
		ts.rootTemplate = rightChild;
		ts.index = rightChild;
		ts.isLeft = 0;
		return rightChild;
	}
#ifdef USE_COMPLETE_TREE
	if(CISINCOMPLETETREE(node))
	{
		rightChild = ts.index * 2 + 2;
		ts.index = rightChild;
		ts.isLeft = 0;
		return rightChild;
	}
#endif
	unsigned int offset = ts.index - ts.rootTemplate;
	rightChild = GETRIGHTCHILD(&templates[ts.type].tree[offset]) + ts.rootTemplate;
	ts.index = rightChild;
	ts.isLeft = 0;
	return rightChild;
}
#endif

unsigned int HCCMesh::CGETLEFTCHILD(CTREE_CLASS* node, TravStat &ts, unsigned int &minBB)
{
	unsigned int leftChild;

	if(CISLEAFOFPATCH(node))
	{
		// leaf of a patch or complete tree
#ifdef CONVERT_PHOTON
		unsigned int suppIndex = (node->plane >> 5) & BIT_MASK_9;
#else
		unsigned int suppIndex = (node->data >> 5) & BIT_MASK_9;
#endif
		CompTreeSuppPtr supp = &(compTreeSupp[suppIndex]);
		/*
		leftChild = supp->leftIndex;
		ts.type = ts.type = (supp->data >> 27) & BIT_MASK_5;
		*/
		leftChild = supp->leftIndex >> 5;
		ts.type = supp->leftIndex & BIT_MASK_5;
		ts.rootTemplate = leftChild;
	}
#ifdef USE_COMPLETE_TREE
	else if(CISINCOMPLETETREE(node))
	{
		leftChild = ts.index * 2 + 1;
	}
#endif
	else
	{
		unsigned int offset = ts.index - ts.rootTemplate;
		TREE_CLASS* tNode = &templates[ts.type].tree[offset];
		leftChild = GETLEFTCHILD(tNode) + ts.rootTemplate;
	}
	CTREE_CLASS* lChild = &compTreeNode[leftChild];
#ifndef CONVERT_PHOTON
	minBB = lChild->data;
	if(CISLEAFOFPATCH(lChild))
	{
		CompTreeSuppPtr supp = &(compTreeSupp[(lChild->data >> 5) & BIT_MASK_9]);
		minBB &= ~0x3FE0;
		minBB |= ((supp->data >> 5) & BIT_MASK_9) << 5;
	}
#endif

	ts.index = leftChild;
	ts.isLeft = 1;

	return leftChild;
}

unsigned int HCCMesh::CGETRIGHTCHILD(CTREE_CLASS* node, TravStat &ts, unsigned int &maxBB)
{
	unsigned int rightChild;

	if(CISLEAFOFPATCH(node))
	{
		// leaf of a patch or complete tree
#ifdef CONVERT_PHOTON
		unsigned int suppIndex = (node->plane >> 5) & BIT_MASK_9;
#else
		unsigned int suppIndex = (node->data >> 5) & BIT_MASK_9;
#endif
		CompTreeSuppPtr supp = &(compTreeSupp[suppIndex]);
		/*
		rightChild = supp->rightIndex;
		ts.type = (supp->data >> 22) & BIT_MASK_5;
		*/
		rightChild = supp->rightIndex >> 5;
		ts.type = supp->rightIndex & BIT_MASK_5;
		ts.rootTemplate = rightChild;
	}
#ifdef USE_COMPLETE_TREE
	else if(CISINCOMPLETETREE(node))
	{
		rightChild = ts.index * 2 + 2;
	}
#endif
	else 
	{
		unsigned int offset = ts.index - ts.rootTemplate;
		TREE_CLASS* tNode = &templates[ts.type].tree[offset];
		rightChild = GETRIGHTCHILD(tNode) + ts.rootTemplate;
	}
	CTREE_CLASS* rChild = &compTreeNode[rightChild];
#ifndef CONVERT_PHOTON
	maxBB = rChild->data;
	if(CISLEAFOFPATCH(rChild))
	{
		CompTreeSuppPtr supp = &(compTreeSupp[(rChild->data >> 5) & BIT_MASK_9]);
		maxBB &= ~0x3FE0;
		maxBB |= ((supp->data >> 5) & BIT_MASK_9) << 5;
	}
#endif
	ts.index = rightChild;
	ts.isLeft = 0;
	return rightChild;
}

TREE_CLASS* HCCMesh::CGETNODE(unsigned int index, TravStat &ts, unsigned int minBB, unsigned int maxBB)
{
#ifndef CONVERT_PHOTON
	if(CISLEAF(&ts.node))
	{
		unsigned int vi[3] = {
			(ts.node.data >> 23) & BIT_MASK_9,
			(ts.node.data >> 14) & BIT_MASK_9,
			(ts.node.data >>  5) & BIT_MASK_9};

#ifdef USE_VERTEX_QUANTIZE
		Vector3 v[3];
		pqVert.DeQuantize(clusterVert[vi[0]].qV, v[0].e);
		pqVert.DeQuantize(clusterVert[vi[1]].qV, v[1].e);
		pqVert.DeQuantize(clusterVert[vi[2]].qV, v[2].e);
		float xv[3] = {v[0].e[0], v[1].e[0], v[2].e[0]};
		float yv[3] = {v[0].e[1], v[1].e[1], v[2].e[1]};
		float zv[3] = {v[0].e[2], v[1].e[2], v[2].e[2]};
#else
		_Vector4 *v[3] = {&clusterVert[vi[0]].vert, &clusterVert[vi[1]].vert, &clusterVert[vi[2]].vert};
		float xv[3] = {v[0]->e[0], v[1]->e[0], v[2]->e[0]};
		float yv[3] = {v[0]->e[1], v[1]->e[1], v[2]->e[1]};
		float zv[3] = {v[0]->e[2], v[1]->e[2], v[2]->e[2]};
#endif
		ts.node.min.e[0] = (xv[0] < xv[1] && xv[0] < xv[2]) ? xv[0] : (xv[1] < xv[2] ? xv[1] : xv[2]);
		ts.node.min.e[1] = (yv[0] < yv[1] && yv[0] < yv[2]) ? yv[0] : (yv[1] < yv[2] ? yv[1] : yv[2]);
		ts.node.min.e[2] = (zv[0] < zv[1] && zv[0] < zv[2]) ? zv[0] : (zv[1] < zv[2] ? zv[1] : zv[2]);
		ts.node.max.e[0] = (xv[0] > xv[1] && xv[0] > xv[2]) ? xv[0] : (xv[1] > xv[2] ? xv[1] : xv[2]);
		ts.node.max.e[1] = (yv[0] > yv[1] && yv[0] > yv[2]) ? yv[0] : (yv[1] > yv[2] ? yv[1] : yv[2]);
		ts.node.max.e[2] = (zv[0] > zv[1] && zv[0] > zv[2]) ? zv[0] : (zv[1] > zv[2] ? zv[1] : zv[2]);

		float diff[3] = {
			ts.node.max.e[0] - ts.node.min.e[0],
			ts.node.max.e[1] - ts.node.min.e[1],
			ts.node.max.e[2] - ts.node.min.e[2]};
		ts.axis = (diff[0] > diff[1] && diff[0] > diff[2]) ? 0 : (diff[1] > diff[2] ? 1 : 2);
		
		return &ts.node;
	}

	unsigned int vi[6] = {
		(minBB >> 23) & BIT_MASK_9,
		(minBB >> 14) & BIT_MASK_9,
		(minBB >>  5) & BIT_MASK_9,
		(maxBB >> 23) & BIT_MASK_9,
		(maxBB >> 14) & BIT_MASK_9,
		(maxBB >>  5) & BIT_MASK_9};
	unsigned int flag = (minBB >> 2) & BIT_MASK_3;
	flag = (flag << 3) | ((maxBB >> 2) & BIT_MASK_3);
	if(ts.isLeft == 0) flag = ~flag;

	unsigned int curPos = 0x20;

	float v[6];
#ifdef USE_VERTEX_QUANTIZE
	pqVert.DeQuantize(clusterVert[vi[0]].qV, &v[0], 0);
	pqVert.DeQuantize(clusterVert[vi[1]].qV, &v[1], 1);
	pqVert.DeQuantize(clusterVert[vi[2]].qV, &v[2], 2);
	pqVert.DeQuantize(clusterVert[vi[3]].qV, &v[3], 0);
	pqVert.DeQuantize(clusterVert[vi[4]].qV, &v[4], 1);
	pqVert.DeQuantize(clusterVert[vi[5]].qV, &v[5], 2);
#else
	v[0] = clusterVert[vi[0]].vert.e[0];
	v[1] = clusterVert[vi[1]].vert.e[1];
	v[2] = clusterVert[vi[2]].vert.e[2];
	v[3] = clusterVert[vi[3]].vert.e[0];
	v[4] = clusterVert[vi[4]].vert.e[1];
	v[5] = clusterVert[vi[5]].vert.e[2];
#endif
	ts.node.min.e[0] = (flag & curPos) ? v[0] : ts.node.min.e[0];
	curPos >>= 1;
	ts.node.min.e[1] = (flag & curPos) ? v[1] : ts.node.min.e[1];
	curPos >>= 1;
	ts.node.min.e[2] = (flag & curPos) ? v[2] : ts.node.min.e[2];
	curPos >>= 1;
	ts.node.max.e[0] = (flag & curPos) ? v[3] : ts.node.max.e[0];
	curPos >>= 1;
	ts.node.max.e[1] = (flag & curPos) ? v[4] : ts.node.max.e[1];
	curPos >>= 1;
	ts.node.max.e[2] = (flag & curPos) ? v[5] : ts.node.max.e[2];

	float diff[3] = {
		ts.node.max.e[0] - ts.node.min.e[0],
		ts.node.max.e[1] - ts.node.min.e[1],
		ts.node.max.e[2] - ts.node.min.e[2]};
	ts.axis = (diff[0] > diff[1] && diff[0] > diff[2]) ? 0 : (diff[1] > diff[2] ? 1 : 2);
#endif
	return &ts.node;
}

int HCCMesh::reassignHighNodeStruct(unsigned int srcNodeIndex, int isLeft, unsigned int parentIndex)
{
	TREE_CLASS &srcNode = (treeHighNodeHash.find(srcNodeIndex)->second);
	TREE_CLASS node = srcNode;
	unsigned int nodeIndex = treeHighNode.size();
	treeHighNode.push_back(node);
#ifdef USE_LOD
	unsigned int lodIndex = (highLODIndexHash.find(srcNodeIndex)->second);
	highLODIndexList.push_back(lodIndex);
	unsigned int idxList = GET_REAL_IDX(lodIndex);
	highLOD.push_back(*GETLOD(LODs, idxList));
#endif

	if(parentIndex != -1)
	{
		// assign child index of parent node to current node index;
		TREE_CLASS &parentNode = treeHighNode[parentIndex];
		int axis = parentNode.children & 0x3;
		int childStat = parentNode.children2 & 0x3;
		if(isLeft) parentNode.children = (nodeIndex << 2) | axis;
		else parentNode.children2 = (nodeIndex << 2) | childStat;
	}

	if((node.children2 & 0x2) != 0x2) reassignHighNodeStruct(srcNode.children >> 2, 1, nodeIndex);
	if((node.children2 & 0x1) != 0x1) reassignHighNodeStruct(srcNode.children2 >> 2, 0, nodeIndex);
	return 1;
}

/*
int HCCMesh::reassignHighNodeStruct2(unsigned int srcNodeIndex, int isLeft, unsigned int parentIndex)
{
	TREE_CLASS &srcNode = (treeHighNodeHash.find(srcNodeIndex)->second);
	TREE_CLASS node = srcNode;
	unsigned int nodeIndex = treeHighNode.size();
	treeHighNode.push_back(node);

	if(parentIndex != -1)
	{
		// assign child index of parent node to current node index;
		TREE_CLASS &parentNode = treeHighNode[parentIndex];
		int axis = parentNode.children & 0x3;
		if(isLeft) parentNode.children = (nodeIndex << 2) | axis;
	}

	if(node.children2) return 1;

	reassignHighNodeStruct(srcNode.children >> 2, 1, nodeIndex);
	reassignHighNodeStruct(srcNode.children2 >> 2, 0, nodeIndex);
	return 1;
}
*/

int HCCMesh::testTraverse(unsigned int startIndex, TreeStat &treeStat, TravStat ts)
{
	CTREE_CLASS* node = &compTreeNode[startIndex];
	treeStat.numNodes++;
	if(CISLEAF(node))
	{
		// leaf node
		treeStat.numLeafs++;
		return 1;
	}

	unsigned int leftChild, rightChild;
	TravStat leftTS, rightTS;
	/*
	leftTS.type = ts.type; rightTS.type = ts.type;
	leftTS.rootTemplate = ts.rootTemplate; rightTS.rootTemplate = ts.rootTemplate;
	if(CISLEAFOFPATCH(node))
	{
		// leaf of a patch or complete tree
		unsigned int suppIndex = (node->data >> 4) & BIT_MASK_14;
		CompTreeSuppPtr supp = &compTreeSupp[suppIndex];
		leftChild = supp->leftIndex;
		rightChild = supp->rightIndex;
		leftTS.type = ts.type = (supp->data >> 27) & BIT_MASK_5;
		rightTS.type = (supp->data >> 22) & BIT_MASK_5;
		leftTS.rootTemplate = leftChild;
		rightTS.rootTemplate = rightChild;
	}
#ifdef USE_COMPLETE_TREE
	else if(CISINCOMPLETETREE(node))
	{
		leftChild = startIndex * 2 + 1;
		rightChild = startIndex * 2 + 2;
	}
#endif
	else
	{
		unsigned int offset = startIndex - ts.rootTemplate;
		leftChild = GETLEFTCHILD(&templates[ts.type].tree[offset]) + ts.rootTemplate;
		rightChild = GETRIGHTCHILD(&templates[ts.type].tree[offset]) + ts.rootTemplate;
	}
	*/

	leftTS = ts; rightTS = ts;
	unsigned int cminBB = 0, cmaxBB = 0;
	leftChild = CGETLEFTCHILD(node, leftTS, cminBB);
	rightChild = CGETRIGHTCHILD(node, rightTS, cmaxBB);
	testTraverse(leftChild, treeStat, leftTS);
	testTraverse(rightChild, treeStat, rightTS);
	return 1;
}

int HCCMesh::computeBB(unsigned int nodeIndex, unsigned int vList[9], TravStat ts, int encode)
{
#ifndef CONVERT_PHOTON
	CTREE_CLASS* node = &compTreeNode[nodeIndex];

	if(CISLEAF(node))
	{
		unsigned short triIndex;
		if(encode) triIndex = node->data >> 5;
		else triIndex = curTempTri++;

		TempTriPtr tri = &tempTri[triIndex];
		unsigned int vi[3] = {tri->p[0], tri->p[1], tri->p[2]};
#ifdef USE_VERTEX_QUANTIZE
		Vector3 v[3];
		pqVert.DeQuantize(clusterVert[vi[0]].qV, v[0].e);
		pqVert.DeQuantize(clusterVert[vi[1]].qV, v[1].e);
		pqVert.DeQuantize(clusterVert[vi[2]].qV, v[2].e);
		float xv[3] = {v[0].e[0], v[1].e[0], v[2].e[0]};
		float yv[3] = {v[0].e[1], v[1].e[1], v[2].e[1]};
		float zv[3] = {v[0].e[2], v[1].e[2], v[2].e[2]};
#else
		_Vector4 *v[3] = {&clusterVert[vi[0]].vert, &clusterVert[vi[1]].vert, &clusterVert[vi[2]].vert};
		float xv[3] = {v[0]->e[0], v[1]->e[0], v[2]->e[0]};
		float yv[3] = {v[0]->e[1], v[1]->e[1], v[2]->e[1]};
		float zv[3] = {v[0]->e[2], v[1]->e[2], v[2]->e[2]};
#endif
		vList[0] = vi[argmin(xv, 3)];
		vList[1] = vi[argmin(yv, 3)];
		vList[2] = vi[argmin(zv, 3)];
		vList[3] = vi[argmax(xv, 3)];
		vList[4] = vi[argmax(yv, 3)];
		vList[5] = vi[argmax(zv, 3)];
		vList[6] = vi[0];
		vList[7] = vi[1];
		vList[8] = vi[2];

		assert((vi[0] & BIT_MASK_9) == vi[0]);
		assert((vi[1] & BIT_MASK_9) == vi[1]);
		assert((vi[2] & BIT_MASK_9) == vi[2]);
		node->data  = vi[0] << 23;
		node->data |= vi[1] << 14;
		node->data |= vi[2] << 5;
		node->data |= 0x3;

/*////
#ifdef GENERATE_OUT_OF_CORE_REP
		if(encode)
		{
			TempTri tri;

			tri.p[0] = vi[0];
			tri.p[1] = vi[1];
			tri.p[2] = vi[2];
			tempTri2.push_back(tri);
		}
#endif
*/////
		if(ts.isLeft)
		{
			node->data |= 0x7 << 2;
		}
		return 1;
	}

	unsigned int leftVList[9], rightVList[9];
	TravStat leftTS = ts, rightTS = ts;
	unsigned int cminBB = 0, cmaxBB = 0;
	unsigned int leftChild = CGETLEFTCHILD(node, leftTS, cminBB);
	unsigned int rightChild = CGETRIGHTCHILD(node, rightTS, cmaxBB);

	int leftStat = computeBB(leftChild, leftVList, leftTS, encode);
	int rightStat = computeBB(rightChild, rightVList, rightTS, encode);

#ifdef USE_VERTEX_QUANTIZE
	Vector3 leftV[6], rightV[6];
	for(int i=0;i<6;i++)
	{
		pqVert.DeQuantize(clusterVert[leftVList[i]].qV, leftV[i].e);
		pqVert.DeQuantize(clusterVert[rightVList[i]].qV, rightV[i].e);
	}
#else
	Vector3 leftV[6] = {
		clusterVert[leftVList[0]].vert, clusterVert[leftVList[1]].vert, 
		clusterVert[leftVList[2]].vert, clusterVert[leftVList[3]].vert, 
		clusterVert[leftVList[4]].vert, clusterVert[leftVList[5]].vert};
	Vector3 rightV[6] = {
		clusterVert[rightVList[0]].vert, clusterVert[rightVList[1]].vert, 
		clusterVert[rightVList[2]].vert, clusterVert[rightVList[3]].vert, 
		clusterVert[rightVList[4]].vert, clusterVert[rightVList[5]].vert};
#endif

	// find boundary vertices
	unsigned int childVList[6];
	unsigned int flag = 0;
	for(int i=0;i<3;i++)
	{
		if(leftV[i].e[i] < rightV[i].e[i])
		{
			vList[i] = leftVList[i];
			childVList[i] = rightVList[i];
		}
		else
		{
			vList[i] = rightVList[i];
			childVList[i] = leftVList[i];
			flag |= 1 << (5-i);
		}
	}
	for(int i=3;i<6;i++)
	{
		if(leftV[i].e[i-3] > rightV[i].e[i-3])
		{
			vList[i] = leftVList[i];
			childVList[i] = rightVList[i];
		}
		else
		{
			vList[i] = rightVList[i];
			childVList[i] = leftVList[i];
			flag |= 1 << (5-i);
		}
	}

	CTREE_CLASS* leftNode = &compTreeNode[leftChild];
	CTREE_CLASS* rightNode = &compTreeNode[rightChild];

	// store boundary vertices
	if(!CISLEAF(leftNode))
	{
		leftNode->data |= childVList[0] << 23;
		leftNode->data |= childVList[1] << 14;
		//leftNode->data |= childVList[2] << 5;
		assert(((leftNode->data >> 2) & BIT_MASK_3) == 0);
		leftNode->data |= ((flag >> 3) & BIT_MASK_3) << 2;

		if(CISLEAFOFPATCH(leftNode))
		{
			unsigned int suppIndex = (leftNode->data >> 5) & BIT_MASK_9;
			CompTreeSuppPtr supp = &compTreeSupp[suppIndex];
			supp->data |= childVList[2] << 5;
		}
		else 
		{
			leftNode->data |= childVList[2] << 5;
		}
	}

	if(!CISLEAF(rightNode))
	{
		rightNode->data |= childVList[3] << 23;
		rightNode->data |= childVList[4] << 14;
		//rightNode->data |= childVList[5] << 5;
		assert(((rightNode->data >> 2) & BIT_MASK_3) == 0);
		rightNode->data |= (flag & BIT_MASK_3) << 2;

		if(CISLEAFOFPATCH(rightNode))
		{
			unsigned int suppIndex = (rightNode->data >> 5) & BIT_MASK_9;
			CompTreeSuppPtr supp = &compTreeSupp[suppIndex];
			supp->data |= childVList[5] << 5;
		}
		else 
		{
			rightNode->data |= childVList[5] << 5;
		}
	}

#endif
	return 0;
}

int HCCMesh::argmin(float x[], int size)
{
	float min = x[0];
	int minIndex = 0;
	for(int i=0;i<size;i++)
	{
		if(min > x[i])
		{
			min = x[i];
			minIndex = i;
		}
	}
	return minIndex;
}
int HCCMesh::argmax(float x[], int size)
{
	float max = x[0];
	int maxIndex = 0;
	for(int i=0;i<size;i++)
	{
		if(max < x[i])
		{
			max = x[i];
			maxIndex = i;
		}
	}
	return maxIndex;

}

int HCCMesh::quantizeVector(Vector3 &x)
{
	int plane = 0;
	float u, v;
	float sign = -1;
	int i0, i1, i2;

	if (fabs(x.e[0]) > fabs(x.e[1]) && fabs(x.e[0]) > fabs(x.e[2])) {								
		i0 = 0;
	}
	else if (fabs(x.e[1]) > fabs(x.e[2])) {								
		i0 = 1;
	}
	else {
		i0 = 2;
	}
	
	i1 = (i0+1) % 3;
	i2 = (i0+2) % 3;

	sign = x.e[i0] >= 0 ? 1.0f : -1.0f;

	//assert(x.e[i0] != 0);
	if(x.e[i0] == 0) return 0;
	u = sign*x.e[i1]/x.e[i0];
	v = sign*x.e[i2]/x.e[i0];
	if(u < -1 || u > 1 || v < -1 || v > 1)
	{
		cout << "Wrong u, v [" << u << ", " << v << "]" << endl;
	}

	plane = i0 * 2;
	if(sign >= 0) plane++;

	float delta = 2.0f/(nQuantize-1);
	int uidx = (int)((u + 1.0f)/delta + 0.5f);
	int vidx = (int)((v + 1.0f)/delta + 0.5f);
	return nQuantize*nQuantize*plane + vidx * nQuantize + uidx;
}

void HCCMesh::calculateQNormals()
{
	// calculate quantized normals;
	float delta = 2.0f/(nQuantize-1);
	float u, v;
	int i0, i1, i2;
	float sign = -1.0f;

	for(int plane=0;plane<6;plane++)
	{
		i0 = plane/2;
		i1 = (i0 + 1) % 3;
		i2 = (i0 + 2) % 3;
		v = -1.0f;
		for(int i=0;i<nQuantize;i++)
		{
			u = -1.0f;
			for(int j=0;j<nQuantize;j++)
			{
				Vector3 &n = quantizedNormals[i*nQuantize+j + nQuantize*nQuantize*plane];
				n.e[i0] = sign;
				n.e[i1] = u;
				n.e[i2] = v;
				n.makeUnitVector();
				u += delta;
			}
			v += delta;
		}
		sign *= -1.0f;
	}
}

/*////
#ifdef GENERATE_OUT_OF_CORE_REP
int HCCMesh::OutRepToInRep(const char *outFileName, const char *inFileName)
{
	//T1.Start();
	FILE *fpOut = fopen(outFileName, "rb");
	FILE *fpOutGeom = fopen(outFileName, "rb"); // FIX_ME
	FILE *fpIn = fopen(inFileName, "wb");

	// copy header
	CompClusterHeader header;

	fread(&numTemplates, sizeof(int), 1, fpOut);
	fread(templates, sizeof(TemplateTable), numTemplates, fpOut);
	fread(&curCluster, sizeof(unsigned int), 1, fpOut);
	fread(&header, sizeof(CompClusterHeader), 1, fpOut);

	fwrite(&numTemplates, sizeof(int), 1, fpIn);
	fwrite(templates, sizeof(TemplateTable), numTemplates, fpIn);
	fwrite(&curCluster, sizeof(unsigned int), 1, fpIn);
	fwrite(&header, sizeof(CompClusterHeader), 1, fpIn);

	// copy high level tree node
	TREE_CLASS node;
	for(int i=0;i<header.numNode;i++) 
	{
		fread(&node, sizeof(TREE_CLASS), 1, fpOut);
		fwrite(&node, sizeof(TREE_CLASS), 1, fpIn);
	}

	// load file size of each cluster
	clusterFileSizeOut.clear();
	clusterGeomFileSizeOut.clear();
	for(int i=1;i<curCluster;i++)
	{
		unsigned int size;
		fread(&size, sizeof(unsigned int), 1, fpOut);
		clusterFileSizeOut.push_back(size);
	}
	for(int i=1;i<curCluster;i++)
	{
		unsigned int size;
		fread(&size, sizeof(unsigned int), 1, fpOut);
		clusterGeomFileSizeOut.push_back(size);
	}

	Progression prog("Decode Out-of-core Representation", curCluster-1, 100);

	// prepare buffer for decompression
	unsigned char buffer[2048*32];

	// decompress and store each cluster
	for(int i=1;i<curCluster;i++)
	{
		fread(&header, sizeof(CompClusterHeader), 1, fpOut);
		fwrite(&header, sizeof(CompClusterHeader), 1, fpIn);

		// Tree node
		compTreeNode.clear();
		int sizeTreeNodeOut;
		fread(&sizeTreeNodeOut, sizeof(int), 1, fpOut);
		fread(buffer, sizeTreeNodeOut, 1, fpOut);
		BitCompression bitTreeNodeOut(buffer, sizeTreeNodeOut, 1);
		for(int j=0, curSuppIndex = 0;j<header.numNode;j++)
		{
			CTREE_CLASS node;
#ifdef CONVERT_PHOTON
			node.plane = bitTreeNodeOut.decode(2) << 2;
			if(CISLEAFOFPATCH(&node))
			{
				node.plane |= (curSuppIndex << 5);
				curSuppIndex++;
			}
#else
			node.data = bitTreeNodeOut.decode(2);
			if(node.data == 0x2)
			{
				node.data |= (curSuppIndex << 5);
				curSuppIndex++;
			}
#endif
			compTreeNode.push_back(node);
		}

		// Tree supp
		compTreeSupp.clear();
		int sizeTreeSuppOut;
		fread(&sizeTreeSuppOut, sizeof(int), 1, fpOut);
		fread(buffer, sizeTreeSuppOut, 1, fpOut);
		BitCompression bitTreeSuppOut(buffer, sizeTreeSuppOut, 1);
		int bitsNumNode = getBits(header.numNode);
		for(int j=0;j<header.numSupp;j++)
		{
			CompTreeSupp supp;
			//
			//supp.leftIndex = bitTreeSuppOut.decode(bitsNumNode);
			//supp.rightIndex = bitTreeSuppOut.decode(bitsNumNode);
			//supp.data  = bitTreeSuppOut.decode(5) << 27;
			//supp.data |= bitTreeSuppOut.decode(5) << 22;
			//
			supp.leftIndex = bitTreeSuppOut.decode(bitsNumNode) << 5;
			supp.rightIndex = bitTreeSuppOut.decode(bitsNumNode) << 5;
			supp.leftIndex |= bitTreeSuppOut.decode(5);
			supp.rightIndex |= bitTreeSuppOut.decode(5);
			supp.data = 0;
			compTreeSupp.push_back(supp);
		}
		//
		//for(int j=0;j<header.numSupp;j++) 
		//{
		//	CompTreeSupp supp;
		//	fread(&supp, sizeof(CompTreeSupp), 1, fpOut);
		//	compTreeSupp.push_back(supp);
		//}
		//

		// Mesh
		//T2.Start();
		tempTri.clear();
		clusterVert.clear();

#ifdef USE_RANGE_ENCODER
		smreader = new SMreader_smc_v();
		((SMreader_smc_v*)smreader)->open(fpOut);
#else
		smreader = new SMreader_smc_v_d();
		((SMreader_smc_v_d*)smreader)->open(fpOut, fpOutGeom);
#endif

		SMevent event;
		while (event = smreader->read_element())
		{
			switch (event)
			{
			case SM_VERTEX:
				{
					CompTreeVert vert;
#ifdef USE_VERTEX_QUANTIZE
					pqVert.EnQuantize(smreader->v_pos_f, vert.qV);
#else
					vert.vert.set(smreader->v_pos_f);
#endif
					clusterVert.push_back(vert);
				}
				break;
			case SM_TRIANGLE:
				{
					TempTri tri;
					tri.p[0] = smreader->t_idx[0];
					tri.p[1] = smreader->t_idx[1];
					tri.p[2] = smreader->t_idx[2];
					tempTri.push_back(tri);
				}
				break;
			case SM_FINALIZED:
				break;
			default:
				break;
			}
		}
		smreader->close();
		delete smreader;
		//T2.Stop();

		// Additional vertex data
		//
		//for(int j=0;j<header.numVert;j++) 
		//{
		//	fread(&clusterVert[j].vert.m_alpha, sizeof(unsigned int), 1, fpOut);
		//}
		//
		// vertex normal
		for(int j=0;j<header.numVert;j++) 
		{
			unsigned short normal = 0;
			unsigned int data = 0;
			fread(&normal, sizeof(unsigned short), 1, fpOut);
			data = normal;
			data = data << 16;
#ifdef USE_VERTEX_QUANTIZE
			clusterVert[j].data = data;
#else
			clusterVert[j].vert.m_alpha = *((float *)&data);
#endif
		}
		// vertex color
		unsigned int numColors;
		fread(&numColors, sizeof(unsigned int), 1, fpOut);
		vector<unsigned short> colorList;
		for(int j=0;j<numColors;j++) 
		{
			unsigned short color;
			fread(&color, sizeof(unsigned short), 1, fpOut);
			colorList.push_back(color);
		}
		int sizeVertColorOut;
		fread(&sizeVertColorOut, sizeof(int), 1, fpOut);
		fread(buffer, sizeVertColorOut, 1, fpOut);
		BitCompression bitVertColorOut(buffer, sizeVertColorOut, 1);
		int bitsNumColor = 0;
		if(numColors > 1) bitsNumColor = getBits(numColors-1);
		for(int j=0;j<header.numVert;j++)
		{
			unsigned short color = colorList[bitVertColorOut.decode(bitsNumColor)];
#ifdef USE_VERTEX_QUANTIZE
			clusterVert[j].data |= color & BIT_MASK_16;
#else
			unsigned int data = *((unsigned int*)&clusterVert[j].vert.m_alpha);
			data |= color & BIT_MASK_16;
			clusterVert[j].vert.m_alpha = *((float *)&data);
#endif
		}

		// Compute bounding boxes
		TravStat ts;
		unsigned int vList[9] = {0, };
		memset(&ts, 0, sizeof(TravStat));

		ts.type = header.rootType;
		curTempTri = 0;
		computeBB(0, vList, ts, 0);

		//
		//// check the tree structure
		//TreeStat testTreeStat;
		//memset(&testTreeStat, 0, sizeof(TreeStat));
		//memset(&ts, 0, sizeof(TravStat));

		//ts.type = header.rootType;

		//testTraverse(0, testTreeStat, ts);
		//assert(header.numNode == testTreeStat.numNodes);
		//assert(tempTri.size() == testTreeStat.numLeafs);
		//

		// store cluster
		for(int i=0;i<compTreeNode.size();i++) fwrite(&compTreeNode[i], sizeof(CTREE_CLASS), 1, fpIn);
		for(int i=0;i<compTreeSupp.size();i++) fwrite(&compTreeSupp[i], sizeof(CompTreeSupp), 1, fpIn);
		for(int i=0;i<clusterVert.size();i++) fwrite(&clusterVert[i], sizeof(CompTreeVert), 1, fpIn);

		prog.step();
	}

	fclose(fpOut);
	fclose(fpOutGeom);
	fclose(fpIn);
	//T1.Stop();

	//cout << T1 << ", Total : " << T1.GetTime() << endl;
	//cout << T2 << ", Total : " << T2.GetTime() << endl;
	//cout << T3 << ", Total : " << T3.GetTime() << endl;
	//cout << T4 << ", Total : " << T4.GetTime() << endl;
	//cout << T5 << ", Total : " << T5.GetTime() << endl;
	return 1;
}
#endif
*/////

int HCCMesh::test()
{
	DictionaryCompression dic(16);
	dic.encode(5);
	dic.encode(2);
	dic.encode(3);
	dic.encode(5);
	dic.encode(3);
	dic.encode(5);
	dic.encode(5);
	dic.encode(3);
	dic.encode(1);
	dic.encode(4);
	dic.done();
	return 1;
}

Vector3 g_BBMin, g_BBMax;

#ifdef USE_LOD

bool HCCMesh::ComputeSimpRep (CompClusterHeader &header)
{
	// traverse the kd-tree from top-down
	// if the BB of the current node is much smaller than a pivot node, 
	//		we compute simplification representation
	//		RQ1: what is optimal condition for that given runtime performance?
	// if we decided to have a simplification representation, we tag it.
	// we continue this process until we hit the leaf
	// Then, we collect all the vertex, and run PCA to compute a plane or two triangles
	//		RQ2: how do we compute good representative representation for irregular meshes?
	//		we use weight according to area of face and use weighted vertex for PCA
	//		one possible way of doing it is to use centroid of the triangle.
	//		RQ3: which representation is faster for intersection and ray-differentials?
	//		RQ4: we also want to extend this method for out-of-core to handle massive models.
	//		RQ5: what is runtime error metric?

	assert (sizeof (LODNode) == 32);

	m_NumLODs = 0;
	//m_StatErr.init();
	//m_StatErrRatio.init();
	//m_DepthStats.init();

	COOCPCAwoExtent RootPCA;

	Vector3 BBMin = header.BBMin, BBMax = header.BBMax;
	Vector3 Diff;

	g_BBMin = BBMin;
	g_BBMax = BBMax;

	Diff = BBMax - BBMin;

	float VolumeBB = Diff.e [0] * Diff.e [1] * Diff.e [2];

	if (VolumeBB < 1) {
		//printf ("\n");
		//printf ("Warning: current impl. has numerical issues. Please increase scale of the model.\n");
		//printf ("\n");
	}

	// start from the root (0)
	//ComputeSimpRep (0, BBMin, BBMax, VolumeBB/ MIN_VOLUME_REDUCTION_BW_LODS, RootPCA);
	Vec3f RealBB [2];
	RealBB [0].Set ( 1e15, 1e15, 1e15);
	RealBB [1].Set (-1e15,-1e15,-1e15);


	TravStat ts;
	memset(&ts, 0, sizeof(TravStat));
	ts.type = header.rootType;
	ts.node.min = header.BBMin;
	ts.node.max = header.BBMax;
	m_curLODIndexTrav = 0;
	m_LODIndexMap.clear();
	m_LODIndexMap.resize(2048);
	ComputeSimpRep (0, BBMin, BBMax, VolumeBB, RootPCA, RealBB, ts);
	
	return true;

}



// BBMin and BBMax is a volume of previous LOD in the search path in the tree.
// RealBB [2] are conservative BB that cover all the geometry that are contained in the voxel.
bool HCCMesh::ComputeSimpRep (unsigned int NodeIdx, Vector3 & BBMin, Vector3 & BBMax,
									float MinVolume, COOCPCAwoExtent & PCA, Vec3f * RealBB, TravStat &ts)
{
	// detect whether we need a LOD representation.
	//		- if there is big volume change or depth change, we need to use a LOD at runtime.
	//		- we take care of depth change at runtime, so here we only consider volume change.

	/*
	if (NoLODs > MIN_NO_LODS) {
	*/
	int curLODIndexTrav = m_curLODIndexTrav;
	int i;
	CTREE_CLASS* pCurNode = &compTreeNode[NodeIdx];

	if (CISLEAF (pCurNode)) {
		// collect data

		unsigned int p[3];
		p[0] = (pCurNode->data >> 23) & BIT_MASK_9;
		p[1] = (pCurNode->data >> 14) & BIT_MASK_9;
		p[2] = (pCurNode->data >>  5) & BIT_MASK_9;

#ifdef USE_VERTEX_QUANTIZE
		Vector3 V0, V1, V2;
		pqVert.DeQuantize(clusterVert[p[0]].qV, V0.e);
		pqVert.DeQuantize(clusterVert[p[1]].qV, V1.e);
		pqVert.DeQuantize(clusterVert[p[2]].qV, V2.e);
#else
		const Vector3 V0 = clusterVert[p[0]].vert;
		const Vector3 V1 = clusterVert[p[1]].vert;
		const Vector3 V2 = clusterVert[p[2]].vert;
#endif

		Vector3 triN = cross(V1-V0, V2-V0);
		triN.makeUnitVector();

		Vec3f V [4];
		V [0].x = V0.e [0];V [0].y = V0.e [1];V [0].z = V0.e [2]; // position
		V [1].x = V1.e [0];V [1].y = V1.e [1];V [1].z = V1.e [2];
		V [2].x = V2.e [0];V [2].y = V2.e [1];V [2].z = V2.e [2];
		V [3].x = triN.e [0];	// normal
		V [3].y = triN.e [1];
		V [3].z = triN.e [2];

		for (int j = 0;j < 3;j++)
			V [j].UpdateMinMax (RealBB [0], RealBB [1]);

		// Note: we need to generalize this later
		//rgb color (0.7, 0.7, 0.7);
#ifdef USE_VERTEX_QUANTIZE
		unsigned int data = clusterVert[p[0]].data;
#else
		unsigned int data = *((unsigned int*)&clusterVert[p[0]].vert.m_alpha);
#endif

		rgb color = rgb(((data >> 11) & 0x1F) / 31.0f, ((data >> 5) & 0x3F) / 63.0f, ((data >> 0) & 0x1F) / 31.0f);
		PCA.InsertTriangle (V, V[3], color);

		return true;
	}

	Vector3 Diff = BBMax - BBMin;
	float VolumeBB = Diff.e [0] * Diff.e [1] * Diff.e [2];
	float TargetMinVolume = MinVolume;

	if(CISLEAFOFPATCH(pCurNode))
	{
		// leaf of a patch or complete tree
		unsigned int suppIndex = (pCurNode->data >> 5) & BIT_MASK_9;
		CompTreeSuppPtr supp = &(compTreeSupp[suppIndex]);

		supp->data &= ~(BIT_MASK_16 << 16);
		
		if (VolumeBB <= MinVolume) 
		{
			TargetMinVolume = VolumeBB/MIN_VOLUME_REDUCTION_BW_LODS;
			supp->data |= LOD_BIT << 16;
			m_curLODIndexTrav++;
		}
	}

	// compute BB of two child nodes
	Vector3 LBB [2], RBB [2];
	TravStat leftTS = ts, rightTS = ts;
	unsigned int cminBB = 0, cmaxBB = 0;
	unsigned int leftChild = CGETLEFTCHILD(pCurNode, leftTS, cminBB);
	unsigned int rightChild = CGETRIGHTCHILD(pCurNode, rightTS, cmaxBB);
	TREE_CLASS* lChild = CGETNODE(leftChild, leftTS, cminBB, cmaxBB);
	TREE_CLASS* rChild = CGETNODE(rightChild, rightTS, cminBB, cmaxBB);
	LBB [0] = lChild->min; LBB [1] = lChild->max;
	RBB [0] = rChild->min; RBB [1] = rChild->max;

	Diff = BBMax - BBMin;
	
	// continue top-down process
	COOCPCAwoExtent LPCA, RPCA;
	Vec3f LRealBB [2], RRealBB [2];
	
	LRealBB [0].Set ( 1e15, 1e15, 1e15);
	LRealBB [1].Set (-1e15,-1e15,-1e15);
	RRealBB [0].Set ( 1e15, 1e15, 1e15);
	RRealBB [1].Set (-1e15,-1e15,-1e15);

	ComputeSimpRep (leftChild, LBB [0], LBB [1], TargetMinVolume, LPCA, LRealBB, leftTS);
	ComputeSimpRep (rightChild, RBB [0], RBB [1], TargetMinVolume, RPCA, RRealBB, rightTS);

	PCA = LPCA + RPCA;		// compute PCA for parent one with constant complexity

	// compute new conservative BB
	for (i = 0;i < 2;i++)
	{
		LRealBB [i].UpdateMinMax (RealBB [0], RealBB [1]);
		RRealBB [i].UpdateMinMax (RealBB [0], RealBB [1]);
	}


	// continue bottom-up process
	// if we detect LOD tag, perform (linear time complexity) PCA in an out-of-core manner
	ComputeRLOD (NodeIdx, curLODIndexTrav, PCA, BBMin, BBMax, RealBB);
	return true;

}


float HCCMesh::ComputeRLOD (unsigned int NodeIdx, unsigned int NodeIdxTrav, COOCPCAwoExtent & PCA,
								Vector3 & BBMin, Vector3 & BBMax, Vec3f * RealBB)
{
	CTREE_CLASS* pCurNode = &compTreeNode[NodeIdx];

	if(NodeIdx == 0)
	{
		// root of this cluster

		// compute LOD
		Vec3f Center,  Extents [3];
		//Vec3f ShadingNormal;

		CExtLOD LOD;
		PCA.ComputePC (Center, Extents);
		PCA.SetLOD (LOD, BBMin, BBMax, RealBB);
	

		//m_StatErr.update (LOD.m_ErrBnd);

		rgb color = PCA.GetMeanColor (); 

		assert (LOD.m_i1 <= 2 && LOD.m_i2 <= 2);

		// store LOD into a file
		//m_pLODBuf->appendElement (LOD);
		CompLOD cLOD;
		cLOD.lod = LOD;

		m_LODIndexMap[NodeIdxTrav] = compLOD.size();
		
		compLOD.push_back(cLOD);
		assert (m_NumLODs < MAX_NUM_LODs);

		m_NumLODs++;
	
		return true;
	}

	if(CISLEAFOFPATCH(pCurNode))
	{
		// leaf of a patch or complete tree
		unsigned int suppIndex = (pCurNode->data >> 5) & BIT_MASK_9;
		CompTreeSuppPtr supp = &(compTreeSupp[suppIndex]);

		if (HAS_LOD ((supp->data >> 16))) {
			
			if (PCA.IsEmpty ()) {	// empty node
				supp->data &= ~(BIT_MASK_16 << 16);
				return true;
			}

 			if (PCA.GetNumContainedTriangle () < MIN_NUM_TRIANGLES_LOD) {	// contained too smal triangles
				supp->data &= ~(BIT_MASK_16 << 16);
				return true;
			}

			// compute LOD
			Vec3f Center,  Extents [3];
			//Vec3f ShadingNormal;

			CExtLOD LOD;
			PCA.ComputePC (Center, Extents);
			PCA.SetLOD (LOD, BBMin, BBMax, RealBB);
		

			//m_StatErr.update (LOD.m_ErrBnd);

			rgb color = PCA.GetMeanColor (); 

			assert (LOD.m_i1 <= 2 && LOD.m_i2 <= 2);

			// store LOD into a file
			//m_pLODBuf->appendElement (LOD);
			CompLOD cLOD;
			cLOD.lod = LOD;

			m_LODIndexMap[NodeIdxTrav] = compLOD.size();

			compLOD.push_back(cLOD);
			assert (m_NumLODs < MAX_NUM_LODs);

			supp->data &= ~(BIT_MASK_16 << 16);
			supp->data |= (m_NumLODs << ERR_BITs) << 16;
			m_NumLODs++;
		
			return true;
		}
	}
	return true;
}

bool HCCMesh::QuantizeErr(CompClusterHeader &header)
{
	int ForcedBit = 5;
	
	//m_QuanErr.Set (m_StatErr.getMin (), m_StatErr.getMax (), m_StatErrRatio.getAvg (), ForcedBit);

	TravStat ts;
	memset(&ts, 0, sizeof(TravStat));
	ts.type = header.rootType;
	ts.node.min = header.BBMin;
	ts.node.max = header.BBMax;
	QuantizeErr (0, ts);

	return true;
}

bool HCCMesh::QuantizeErr (int NodeIdx, TravStat &ts)
{
	CTREE_CLASS* pCurNode = &compTreeNode[NodeIdx];

	if(CISLEAFOFPATCH(pCurNode))
	{
		// leaf of a patch or complete tree
		unsigned int suppIndex = (pCurNode->data >> 5) & BIT_MASK_9;
		CompTreeSuppPtr supp = &(compTreeSupp[suppIndex]);
		unsigned int curLODIndex = supp->data >> 16;

		if (HAS_LOD(curLODIndex)) 
		{
			int lodIndex = GET_REAL_IDX(curLODIndex);		
			const LODNode & LOD = compLOD[lodIndex].lod;

			unsigned int QuanErr;
			m_QuanErr.Fit (LOD.m_ErrBnd, QuanErr);

			assert (QuanErr < pow ((float) 2, (float) 5));
			assert (curLODIndex >= m_QuanErr.m_MaxNumUp);
			supp->data |= (QuanErr << 16);

		}
	}
	
	if (CISLEAF (pCurNode))
		return true;

	TravStat leftTS = ts, rightTS = ts;
	unsigned int cminBB = 0, cmaxBB = 0;
	unsigned int child_left = CGETLEFTCHILD(pCurNode, leftTS, cminBB);
	unsigned int child_right = CGETRIGHTCHILD(pCurNode, rightTS, cmaxBB);

	QuantizeErr (child_left, leftTS);
	QuantizeErr (child_right, rightTS);

	return true;
}
#endif

int HCCMesh::compressClusterHCCMesh(unsigned int clusterID, unsigned int localRootIdx, CompClusterHeader &header)
{
/*////
#ifdef GENERATE_OUT_OF_CORE_REP
	compTreeNodeOut->done();

	__int64 clusterOffsetStart, clusterOffsetEnd;
	__int64 clusterGeomOffsetStart, clusterGeomOffsetEnd;
	clusterOffsetStart = _ftelli64(fpClusterOut);
	clusterGeomOffsetStart = _ftelli64(fpClusterOutGeom);

	int sizeTreeNodeOut = compTreeNodeOut->getNumberChars();

	// header
	fwrite(&header, sizeof(CompClusterHeader), 1, fpClusterOut);
	offset = _ftelli64(fpClusterOut);
	fileSizeHeaderOut += offset-bOffset;
	bOffset = offset;

	// node
	fwrite(&sizeTreeNodeOut, sizeof(int), 1, fpClusterOut);
	fwrite(compTreeNodeOut->getChars(), sizeTreeNodeOut, 1, fpClusterOut);
	offset = _ftelli64(fpClusterOut);
	fileSizeNodeOut += offset-bOffset;
	bOffset = offset;

	// supp
	compTreeSuppOut = new BitCompression((FILE*)0, 0);
	int bitsNumNode = getBits(header.numNode);
	for(int i=0;i<compTreeSupp.size();i++)
	{
		CompTreeSupp supp = compTreeSupp[i];
		//
		//compTreeSuppOut->encode(bitsNumNode, supp.leftIndex);
		//compTreeSuppOut->encode(bitsNumNode, supp.rightIndex);
		//compTreeSuppOut->encode(5, (supp.data >> 27) & BIT_MASK_5);
		//compTreeSuppOut->encode(5, (supp.data >> 22) & BIT_MASK_5);
		//
		compTreeSuppOut->encode(bitsNumNode, supp.leftIndex >> 5);
		compTreeSuppOut->encode(bitsNumNode, supp.rightIndex >> 5);
		compTreeSuppOut->encode(5, supp.leftIndex & BIT_MASK_5);
		compTreeSuppOut->encode(5, supp.rightIndex & BIT_MASK_5);
	}
	compTreeSuppOut->done();
	int sizeTreeSuppOut = compTreeSuppOut->getNumberChars();
	fwrite(&sizeTreeSuppOut, sizeof(int), 1, fpClusterOut);
	fwrite(compTreeSuppOut->getChars(), sizeTreeSuppOut, 1, fpClusterOut);
	delete compTreeSuppOut;
	//
	//int sizeTreeSuppOut = compTreeSupp.size()*sizeof(CompTreeSupp);
	//fwrite(&sizeTreeSuppOut, sizeof(int), 1, fpClusterOut);
	//for(int i=0;i<compTreeSupp.size();i++) fwrite(&compTreeSupp[i], sizeof(CompTreeSupp), 1, fpClusterOut);
	//
	offset = _ftelli64(fpClusterOut);
	fileSizeSuppOut += offset-bOffset;
	bOffset = offset;

	// Mesh
	compNumTrisOut = new BitCompression((FILE*)0, 0);
	int bitsNumTris = getBits(maxNumTrisPerLeaf);
	compNumTrisOut->encode(5, bitsNumTris);
	compNumTrisOut->encode(16, triList2.size());
	for(int i=0;i<triList2.size();i++)
	{
		TriList &tList = triList2[i];
		compNumTrisOut->encode(bitsNumTris, tList.numTris);
	}
	compNumTrisOut->done();
	int sizeNumTrisOut = compNumTrisOut->getNumberChars();
	fwrite(&sizeNumTrisOut, sizeof(int), 1, fpClusterOut);
	fwrite(compNumTrisOut->getChars(), sizeNumTrisOut, 1, fpClusterOut);
	delete compNumTrisOut;

#ifdef USE_RANGE_ENCODER
	smwriter = new SMwriter_smc_v();
	((SMwriter_smc_v*)smwriter)->open(fpClusterOut);
#else
	smwriter = new SMwriter_smc_v_d();
	((SMwriter_smc_v_d*)smwriter)->open(fpClusterOut, fpClusterOutGeom, 16);
#endif

	TREE_CLASS* localRoot = GETNODE(tree, localRootIdx);
	smwriter->set_nverts(clusterVert.size());
	smwriter->set_nfaces(numClusterTris);
	// global quantization
	smwriter->set_boundingbox(globalBBMin.e, globalBBMax.e);
	// local quantization
	//smwriter_smc->set_boundingbox(localRoot->min.e, localRoot->max.e);

	int tidxs[3] = {0, 0, 0};
	bool finals[3] = {false, false, false};
	vector<int> vidxList;

	for(int i=0;i<clusterVert.size();i++)
	{
#ifdef USE_VERTEX_QUANTIZE
		unsigned short normal = clusterVert[i].data >> 16;
#else
		unsigned short normal = (*((unsigned int*)&clusterVert[i].vert.m_alpha)) >> 16;
#endif
		int plane = normal / (nQuantize*nQuantize);
		int uv = normal % (nQuantize*nQuantize);
		int v = uv / nQuantize;
		int u = uv % nQuantize;
		int vidx[3] = {0, 0, 0};
		switch(plane)
		{
		case 0 : vidx[0] = 0; vidx[1] = u; vidx[2] = v; break;
		case 1 : vidx[0] = nQuantize-1; vidx[1] = u; vidx[2] = v; break;
		case 2 : vidx[0] = v; vidx[1] = 0; vidx[2] = u; break;
		case 3 : vidx[0] = v; vidx[1] = nQuantize-1; vidx[2] = u; break;
		case 4 : vidx[0] = u; vidx[1] = v; vidx[2] = 0; break;
		case 5 : vidx[0] = u; vidx[1] = v; vidx[2] = nQuantize-1; break;
		}
#ifdef USE_VERTEX_QUANTIZE
		Vector3 tempV;
		pqVert.DeQuantize(clusterVert[i].qV, tempV.e);
		smwriter->write_vertex(tempV.e, vidx);
#else
		smwriter->write_vertex(clusterVert[i].vert.e, vidx);
#endif
	}
	//
	//for(int i=0;i<tempTri2.size();i++)
	//{
	//	tidxs[0] = tempTri2[i].p[0];
	//	tidxs[1] = tempTri2[i].p[1];
	//	tidxs[2] = tempTri2[i].p[2];
	//	smwriter->write_triangle(tidxs, finals, &vidxList);
	//}
	//
	for(int i=0;i<triList2.size();i++)
	{
		TriList &tList = triList2[i];
		for(int j=0;j<tList.tris.size();j++)
		{
			TempTri &tri = tList.tris[j];
			tidxs[0] = tri.p[0];
			tidxs[1] = tri.p[1];
			tidxs[2] = tri.p[2];
			smwriter->write_triangle(tidxs, finals, &vidxList);
		}
	}

	smwriter->close();
	delete smwriter;

	offset = _ftelli64(fpClusterOut);
	offsetGeom = _ftelli64(fpClusterOutGeom);
	fileSizeMeshOut += offset-bOffset;
	fileSizeMeshOut += offsetGeom-bOffsetGeom;
	bOffset = offset;
	bOffsetGeom = offsetGeom;

	//for(int i=0;i<clusterVert.size();i++) fwrite(&clusterVert[vidxList[i]].vert.m_alpha, sizeof(unsigned int), 1, fpClusterOut);

	// vertex normal
	for(int i=0;i<clusterVert.size();i++) 
	{
#ifdef USE_VERTEX_QUANTIZE
		unsigned short normal = clusterVert[vidxList[i]].data >> 16;
#else
		unsigned short normal = (*((unsigned int*)&clusterVert[vidxList[i]].vert.m_alpha)) >> 16;
#endif
		fwrite(&normal, sizeof(unsigned short), 1, fpClusterOut);
	}

	// vertex color
	vertColorHash.clear();
	for(int i=0;i<clusterVert.size();i++) 
	{
#ifdef USE_VERTEX_QUANTIZE
		unsigned int data = clusterVert[vidxList[i]].data;
#else
		unsigned int data = *((unsigned int*)&clusterVert[vidxList[i]].vert.m_alpha);
#endif
		unsigned int color = data & BIT_MASK_16;
		if(vertColorHash.find(color) == vertColorHash.end())
		{
			vertColorHash.insert(std::pair<unsigned int, unsigned int>(color, vertColorHash.size()));
		}
	}

	unsigned int vertColorHashSize = vertColorHash.size();
	fwrite(&vertColorHashSize, sizeof(unsigned int), 1, fpClusterOut);
	for(VertColorHashIterator it = vertColorHash.begin(); it != vertColorHash.end(); ++it)
	{
		unsigned short color = (it->first) & BIT_MASK_16;
		fwrite(&color, sizeof(unsigned short), 1, fpClusterOut);
	}
	compVertColorOut = new BitCompression((FILE*)0, 0);
	int bitsNumColor = 0;
	if(vertColorHashSize > 1) bitsNumColor = getBits(vertColorHashSize-1);
	for(int i=0;i<clusterVert.size();i++)
	{
#ifdef USE_VERTEX_QUANTIZE
		unsigned int color = clusterVert[vidxList[i]].data & BIT_MASK_16;
#else
		unsigned int color = (*((unsigned int*)&clusterVert[vidxList[i]].vert.m_alpha)) & BIT_MASK_16;
#endif
		unsigned int colIndex = vertColorHash.find(color)->second;
		compVertColorOut->encode(bitsNumColor, colIndex);
	}
	compVertColorOut->done();
	int sizeVertColorOut = compVertColorOut->getNumberChars();
	fwrite(&sizeVertColorOut, sizeof(int), 1, fpClusterOut);
	fwrite(compVertColorOut->getChars(), sizeVertColorOut, 1, fpClusterOut);
	delete compVertColorOut;
	offset = _ftelli64(fpClusterOut);
	fileSizeAddVertOut += offset-bOffset;
	bOffset = offset;

	delete compTreeNodeOut;

	clusterOffsetEnd = _ftelli64(fpClusterOut);
	clusterGeomOffsetEnd = _ftelli64(fpClusterOutGeom);
	clusterFileSizeOut.push_back(clusterOffsetEnd-clusterOffsetStart);
	clusterGeomFileSizeOut.push_back(clusterGeomOffsetEnd-clusterGeomOffsetStart);
	clusterFileSizeIn.push_back(header.numNode*sizeof(CTREE_CLASS) + header.numSupp*sizeof(CompTreeSupp) + header.numVert*sizeof(CompTreeVert));
	return 1;
}

#ifdef USE_LOD
int HCCMesh::compressClusterHCCLOD()
{
	unsigned int numLODs = compLOD.size();
	fwrite(&numLODs, sizeof(unsigned int), 1, fpLODClusterOut);
	BitCompression *compLODOut = new BitCompression((FILE*)0, 0);
	int nbitsErrBnd = 16;
	int nbitsNormal = 16;
	int nbitsDistance = 16;
	int nbitsMaterial = 16;
	int nbitsProjErrBnd = 16;
	PositionQuantizerNew pqErrBnd;
	PositionQuantizerNew pqDistance;
	PositionQuantizerNew pqProjErrBnd;

	// calculate range of float values;
	float minErrBnd, maxErrBnd;
	float minDistance, maxDistance;
	float minProjErrBnd, maxProjErrBnd;

	minErrBnd = minDistance = minProjErrBnd = FLT_MAX;
	maxErrBnd = maxDistance = maxProjErrBnd = -FLT_MAX;
	for(int i=0;i<compLOD.size();i++)
	{
		LODNode &lod = compLOD[i].lod;
		if(minErrBnd > lod.m_ErrBnd) minErrBnd = lod.m_ErrBnd;
		if(maxErrBnd < lod.m_ErrBnd) maxErrBnd = lod.m_ErrBnd;
		if(minDistance > lod.m_d) minDistance = lod.m_d;
		if(maxDistance < lod.m_d) maxDistance = lod.m_d;
		if(minProjErrBnd > lod.m_Proj_ErrBnd) minProjErrBnd = lod.m_Proj_ErrBnd;
		if(maxProjErrBnd < lod.m_Proj_ErrBnd) maxProjErrBnd = lod.m_Proj_ErrBnd;
	}

	float mulErrBnd = ((1 << nbitsErrBnd) - 1) / (maxErrBnd - minErrBnd);
	float mulDistance = ((1 << nbitsDistance) - 1) / (maxDistance - minDistance);
	float mulProjErrBnd = ((1 << nbitsProjErrBnd) - 1) / (maxProjErrBnd - minProjErrBnd);

	DictionaryCompression *dic[5];
	dic[0] = new DictionaryCompression(nbitsErrBnd);
	dic[1] = new DictionaryCompression(nbitsNormal);
	dic[2] = new DictionaryCompression(nbitsDistance);
	dic[3] = new DictionaryCompression(nbitsMaterial);
	dic[4] = new DictionaryCompression(nbitsProjErrBnd);
	dic[0]->setEncoder(compLODOut);
	dic[1]->setEncoder(compLODOut);
	dic[2]->setEncoder(compLODOut);
	dic[3]->setEncoder(compLODOut);
	dic[4]->setEncoder(compLODOut);
	for(int i=0;i<numLODs;i++)
	{
		LODNode &lod = compLOD[i].lod;
		dic[0]->encodeNone((lod.m_ErrBnd - minErrBnd) * mulErrBnd);
		dic[1]->encodeNone(quantizeVector(lod.m_n));
		dic[2]->encodeNone((lod.m_d - minDistance) * mulDistance);
		compLODOut->encode(2, lod.m_i1 & 0x3);
		compLODOut->encode(2, lod.m_i2 & 0x3);
		dic[3]->encodeNone(lod.m_material);
		dic[4]->encodeNone((lod.m_Proj_ErrBnd - minProjErrBnd) * mulProjErrBnd);
	}
	dic[0]->done();
	dic[1]->done();
	dic[2]->done();
	dic[3]->done();
	dic[4]->done();
	compLODOut->done();
	delete dic[0];
	delete dic[1];
	delete dic[2];
	delete dic[3];
	delete dic[4];
	int sizeLODOut = compLODOut->getNumberChars();
	fwrite(&sizeLODOut, sizeof(int), 1, fpLODClusterOut);
	fwrite(compLODOut->getChars(), sizeLODOut, 1, fpLODClusterOut);
	delete compLODOut;

#endif
*/////
	return 1;
}

#ifdef CONVERT_PHOTON
int HCCMesh::compressClusterHCCPhoton(unsigned int clusterID, unsigned int localRootIdx, CompClusterHeader &header)
{
	compTreeNodeOut->done();

	__int64 clusterOffsetStart, clusterOffsetEnd;
	clusterOffsetStart = _ftelli64(fpClusterOut);

	int sizeTreeNodeOut = compTreeNodeOut->getNumberChars();

	// header
	fwrite(&header, sizeof(CompClusterHeader), 1, fpClusterOut);
	offset = _ftelli64(fpClusterOut);
	fileSizeHeaderOut += offset-bOffset;
	bOffset = offset;

	// node
	fwrite(&sizeTreeNodeOut, sizeof(int), 1, fpClusterOut);
	fwrite(compTreeNodeOut->getChars(), sizeTreeNodeOut, 1, fpClusterOut);
	delete compTreeNodeOut;
	offset = _ftelli64(fpClusterOut);
	fileSizeNodeOut += offset-bOffset;
	bOffset = offset;

	// supp
	compTreeSuppOut = new BitCompression((FILE*)0, 0);
	int bitsNumNode = getBits(header.numNode);
	for(int i=0;i<compTreeSupp.size();i++)
	{
		CompTreeSupp supp = compTreeSupp[i];
		compTreeSuppOut->encode(bitsNumNode, supp.leftIndex >> 5);
		compTreeSuppOut->encode(bitsNumNode, supp.rightIndex >> 5);
		compTreeSuppOut->encode(5, supp.leftIndex & BIT_MASK_5);
		compTreeSuppOut->encode(5, supp.rightIndex & BIT_MASK_5);
	}
	compTreeSuppOut->done();
	int sizeTreeSuppOut = compTreeSuppOut->getNumberChars();
	fwrite(&sizeTreeSuppOut, sizeof(int), 1, fpClusterOut);
	fwrite(compTreeSuppOut->getChars(), sizeTreeSuppOut, 1, fpClusterOut);
	delete compTreeSuppOut;
	offset = _ftelli64(fpClusterOut);
	fileSizeSuppOut += offset-bOffset;
	bOffset = offset;

	TREE_CLASS* localRoot = GETNODE(tree, localRootIdx);
	// photon
	BitCompression *compPhotonOut = new BitCompression((FILE*)0, 0);
	CTREE_CLASS *photon;
	int nbitsPos = 16;
	int nbitsAngle = 8;
	int nbitsPower = 16;
	PositionQuantizerNew pqPos;
	pqPos.SetMinMax(globalBBMin.e, globalBBMax.e);
	pqPos.SetPrecision(nbitsPos);
	pqPos.SetupQuantizer();
	I32 qPos[3];
	StaticDictionary *dic[6];
	dic[0] = new StaticDictionary(nbitsPos);
	dic[1] = new StaticDictionary(nbitsAngle);
	dic[2] = new StaticDictionary(nbitsPower);
	dic[3] = new StaticDictionary(nbitsPower);
	dic[4] = new StaticDictionary(nbitsPower);
	dic[5] = new StaticDictionary(nbitsPower);
	dic[0]->setEncoder(compPhotonOut);
	dic[1]->setEncoder(compPhotonOut);
	dic[2]->setEncoder(compPhotonOut);
	dic[3]->setEncoder(compPhotonOut);
	dic[4]->setEncoder(compPhotonOut);
	dic[5]->setEncoder(compPhotonOut);
	/*
	for(int i=0;i<compTreeNode.size();i++)
	{
		photon = &compTreeNode[i];
		pqPos.EnQuantize(photon->pos.e, qPos);
		//dicPos->encodeNone(qPos[0]);
		//dicPos->encodeNone(qPos[1]);
		//dicPos->encodeNone(qPos[2]);
		compPhotonOut->encode(8, photon->phi);
		compPhotonOut->encode(8, photon->theta);
		compPhotonOut->encode(2, photon->plane);
		compPhotonOut->encode(16, photon->power[0]);
		compPhotonOut->encode(16, photon->power[1]);
		compPhotonOut->encode(16, photon->power[2]);
		compPhotonOut->encode(16, photon->power[3]);
	}
	*/
	TravStat ts;
	memset(&ts, 0, sizeof(TravStat));
	ts.type = rootType;
	I32 qGlobalBBMin[3] = {0, 0, 0};
	I32 qGlobalBBMax[3] = {pqPos.m_aiQuantRange[0], pqPos.m_aiQuantRange[1], pqPos.m_aiQuantRange[2]};
	compressPhoton(compPhotonOut, dic, pqPos, 0, -1, qGlobalBBMin, qGlobalBBMax, ts);
	dic[0]->done();
	dic[1]->done();
	dic[2]->done();
	dic[3]->done();
	dic[4]->done();
	dic[5]->done();
	compPhotonOut->done();
	delete dic[0];
	delete dic[1];
	delete dic[2];
	delete dic[3];
	delete dic[4];
	delete dic[5];
	int sizePhotonOut = compPhotonOut->getNumberChars();
	fwrite(&sizePhotonOut, sizeof(int), 1, fpClusterOut);
	fwrite(compPhotonOut->getChars(), sizePhotonOut, 1, fpClusterOut);
	delete compPhotonOut;
	offset = _ftelli64(fpClusterOut);
	fileSizePhotonOut += offset-bOffset;
	bOffset = offset;

	clusterOffsetEnd = _ftelli64(fpClusterOut);
	clusterFileSizeOut.push_back(clusterOffsetEnd-clusterOffsetStart);
	clusterFileSizeIn.push_back(header.numNode*sizeof(CTREE_CLASS) + header.numSupp*sizeof(CompTreeSupp) + header.numVert*sizeof(CompTreeVert));
	return 1;
}

int HCCMesh::compressPhoton(
							BitCompression *encoder, StaticDictionary *dic[], PositionQuantizerNew &pq, 
							unsigned int nodeIndex, unsigned int parentIndex, 
							I32 qBBMin[], I32 qBBMax[], TravStat &ts)
{
	CTREE_CLASS* photon = &compTreeNode[nodeIndex];

	I32 predPos[3];
	unsigned char predPhi, predTheta;
	unsigned short predPower[4];

	int axis = photon->plane & 0x3;

	if(parentIndex == -1)
	{
		// root node
		predPos[0] = pq.m_aiQuantRange[0]/2;
		predPos[1] = pq.m_aiQuantRange[1]/2;
		predPos[2] = pq.m_aiQuantRange[2]/2;
		predPhi = 0;
		predTheta = 0;
		predPower[0] = USHRT_MAX/2;
		predPower[1] = USHRT_MAX/2;
		predPower[2] = USHRT_MAX/2;
		predPower[3] = USHRT_MAX/2;
	}
	else
	{
		// non root node : predict each values
		CTREE_CLASS *parent = &compTreeNode[parentIndex];
		pq.EnQuantize(parent->pos.e, predPos);
		predPos[axis] = (qBBMax[axis] - qBBMin[axis]) / 2;
		predPhi = parent->phi;
		predTheta = parent->theta;
		predPower[0] = parent->power[0];
		predPower[1] = parent->power[1];
		predPower[2] = parent->power[2];
		predPower[3] = parent->power[3];
	}

	I32 qPos[3];
	pq.EnQuantize(photon->pos.e, qPos);
	dic[0]->encodeLast(predPos[0], qPos[0]);
	dic[0]->encodeLast(predPos[1], qPos[1]);
	dic[0]->encodeLast(predPos[2], qPos[2]);
	dic[1]->encodeLast(predPhi, photon->phi);
	dic[1]->encodeLast(predTheta, photon->theta);
	dic[2]->encodeLast(predPower[0], photon->power[0]);
	dic[3]->encodeLast(predPower[1], photon->power[1]);
	dic[4]->encodeLast(predPower[2], photon->power[2]);
	dic[5]->encodeLast(predPower[3], photon->power[3]);
	encoder->encode(2, photon->plane & 0x3);

	if(CISLEAF(photon))
	{
		return 0;
	}

	I32 qLeftBBMin[3], qLeftBBMax[3], qRightBBMin[3], qRightBBMax[3];
	memcpy(qLeftBBMin, qBBMin, sizeof(I32)*3);
	memcpy(qLeftBBMax, qBBMax, sizeof(I32)*3);
	memcpy(qRightBBMin, qBBMin, sizeof(I32)*3);
	memcpy(qRightBBMax, qBBMax, sizeof(I32)*3);
	qLeftBBMax[axis] = qPos[axis];
	qRightBBMin[axis] = qPos[axis];

	TravStat leftTS, rightTS;
	unsigned int leftChild, rightChild;
	leftTS = ts; rightTS = ts;
	unsigned int cminBB = 0, cmaxBB = 0;
	leftChild = CGETLEFTCHILD(photon, leftTS, cminBB);
	rightChild = CGETRIGHTCHILD(photon, rightTS, cmaxBB);
	compressPhoton(encoder, dic, pq, leftChild, nodeIndex, qLeftBBMin, qLeftBBMax, leftTS);
	compressPhoton(encoder, dic, pq, rightChild, nodeIndex, qRightBBMin, qRightBBMax, rightTS);
}
#endif

int HCCMesh::convertSimple(const char* filepath)
{
	char srcTreeName[255];
	char srcTrisName[255];
	char srcVertsName[255];
	char dstTreeName[255];
	char dstTrisName[255];
	char dstVertsName[255];

	OptionManager *opt = OptionManager::getSingletonPtr();

	sprintf(srcTreeName, "%s/BVH.node", filepath);
	sprintf(srcTrisName, "%s/tris.ooc", filepath);
	sprintf(srcVertsName, "%s/vertex.ooc", filepath);
	sprintf(dstTreeName, "%s/SBVH.node", filepath);
	sprintf(dstTrisName, "%s/Stris.ooc", filepath);
	sprintf(dstVertsName, "%s/Svertex.ooc", filepath);

	tree = new OOCFile6464<TREE_CLASS>(srcTreeName, 
		1024*1024*512,
		1024*1024*4);
	tris = new OOCFile6464<Triangle>(srcTrisName, 
		1024*1024*256,
		1024*1024*4);
	verts = new OOCFile6464<Vertex>(srcVertsName, 
		1024*1024*128,
		1024*1024*4);

	FILE *fpDstTree = fopen(dstTreeName, "wb");
	FILE *fpDstTris = fopen(dstTrisName, "wb");
	FILE *fpDstVerts = fopen(dstVertsName, "wb");

	BSPArrayTreeNodePtr root = GETNODE(tree, GETROOT());

	PositionQuantizerNew pqTree;
	PositionQuantizerNew pqVerts;

	pqTree.SetMinMax(root->min.e, root->max.e);
	pqTree.SetPrecision(16);
	pqTree.SetupQuantizer();
	pqVerts.SetMinMax(root->min.e, root->max.e);
	pqVerts.SetPrecision(16);
	pqVerts.SetupQuantizer();

	#pragma pack(push, 1)
	typedef struct QTreeNode_t
	{
		unsigned int left;
		unsigned int right;
		unsigned short QMin[3];	// 16 bit quantized
		unsigned short QMax[3];	// 16 bit quantized
	} QTreeNode, *QTreeNodePtr;

	typedef struct STri_t
	{
		unsigned int p[3];
	} STri, *StriPtr;

	typedef struct QVert_t
	{
		unsigned short v[3]; // 16 bit quantized
		unsigned short qN;
		unsigned short c;
	} QVert, *QVertPtr;
	#pragma pack(pop)

	assert(sizeof(QTreeNode) == 20);
	assert(sizeof(STri) == 12);
	assert(sizeof(QVert) == 10);

	unsigned int numNodes;
	unsigned int numTris;
	unsigned int numVerts;
	FILE *fpTemp;
	fpTemp = fopen(srcTreeName, "r");
	numNodes = _filelengthi64(fileno(fpTemp))/sizeof(TREE_CLASS);
	fclose(fpTemp);
	fpTemp = fopen(srcTrisName, "r");
	numTris = _filelengthi64(fileno(fpTemp))/sizeof(Triangle);
	fclose(fpTemp);
	fpTemp = fopen(srcVertsName, "r");
	numVerts = _filelengthi64(fileno(fpTemp))/sizeof(Vertex);
	fclose(fpTemp);

	for(unsigned int i=0;i<numNodes;i++)
	{
		TREE_CLASS *srcNode = GETNODE(tree, i);
		QTreeNode dstNode;
		dstNode.left = srcNode->children;
		dstNode.right = srcNode->children2;
		pqTree.EnQuantize(srcNode->min.e, dstNode.QMin);
		pqTree.EnQuantize(srcNode->max.e, dstNode.QMax);
		fwrite(&dstNode, sizeof(QTreeNode), 1, fpDstTree);
	}

	for(unsigned int i=0;i<numTris;i++)
	{
		TrianglePtr srcTri = GETTRI(tris, i);
		STri dstTri;
		dstTri.p[0] = srcTri->p[0];
		dstTri.p[1] = srcTri->p[1];
		dstTri.p[2] = srcTri->p[2];
		fwrite(&dstTri, sizeof(STri), 1, fpDstTris);
	}

	//calculateQNormals();
	for(unsigned int i=0;i<numVerts;i++)
	{
		VertexPtr srcVert = GETVERTEX(verts, i);
		QVert dstVert;
		unsigned int r, g, b;
		pqVerts.EnQuantize(srcVert->v.e, dstVert.v);
		r = srcVert->c.e[0] * 31.0f;
		g = srcVert->c.e[1] * 63.0f;
		b = srcVert->c.e[2] * 31.0f;
		dstVert.qN = (unsigned short)quantizeVector(srcVert->n);
		dstVert.c = 0;
		dstVert.c |= (r & 0x1F) << 11;
		dstVert.c |= (g & 0x3F) << 5;
		dstVert.c |= (b & 0x1F);
		fwrite(&dstVert, sizeof(QVert), 1, fpDstVerts);
	}

	delete tree;
	delete tris;
	delete verts;

	fclose(fpDstTree);
	fclose(fpDstTris);
	fclose(fpDstVerts);

	return 0;
}
