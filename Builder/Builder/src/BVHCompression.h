#pragma once

#include <iostream>
using namespace std;
#include "io.h"
#include "mydefs.h"
#include "rangeencoder.h"
#include "D:/Projects/RT_Vis09/common/BitCompression.hpp"
#include "D:/Projects/RT_Vis09/common/rangedecoder_file.h"
#include "D:/Projects/RT_Vis09/common/stopwatch.hpp"
#include "D:/Projects/RT_Vis09/common/SimpleDictionary.hpp"
#include "D:/Projects/RT_Vis09/common/LZW.hpp"
#include "D:/Projects/RT_Vis09/common/StaticDictionary.hpp"

#define GETROOT() (1)
#define FLOOR 0
#define CEIL 1

#define _USE_DUMMY_NODE
#ifdef _USE_DUMMY_NODE
#define DUMMY_NODE 0
#endif

#define _USE_BIT_ENCODER

#ifdef _USE_BIT_ENCODER
#define Encoder BitCompression
#else
#define Encoder RangeEncoder
#endif

#define COMPRESS_TYPE_ARITHMETIC 1
#define COMPRESS_TYPE_INCREMENTAL 2
#define COMPRESS_TYPE_ZERO_OR_ALL 3
#define COMPRESS_TYPE_STATIC_DICTIONARY 4
#define COMPRESS_TYPE_STATIC_DICTIONARY_NEW 5
#define COMPRESS_TYPE_QIC_LZW 6
#define COMPRESS_TYPE COMPRESS_TYPE_STATIC_DICTIONARY_NEW

#define SIZE_DICTIONARY 64


//#define DEBUG_CODEC     // enable extra codec info to verify correctness

#define SIZE_BASE_PAGE 4096
#define SIZE_BASE_PAGE_POWER 12

//#define STATISTICS

#define _USE_CONTI_NODE
#define _USE_ONE_TRI_PER_LEAF

#define COMPRESS_TRI

#ifdef COMPRESS_TRI
//#define USE_TRI_3_TYPE_ENCODING
#define USE_TRI_DELTA_ENCODING
#endif

#ifdef STATISTICS
//#define STAT_ERRORS
#endif

#define AXIS(node) ((node)->children & 3)
#define ISLEAF(node) (((node)->children & 3) == 3)
#define GETNODEOFFSET(idx) ((idx >> 2) << 3)
#ifndef _USE_ONE_TRI_PER_LEAF
#define GETCHILDCOUNT(node) ((node)->indexCount >> 2)
#else
#define GETCHILDCOUNT(node) (1)
#endif

#define TEST_SIZE 60000

class BVHCompression
{
public:
	BVHCompression(int isComp);
public:
	~BVHCompression(void);
public:
	int isComp;

	void storeParentIndex(FILE *fpo, FILE *fpi, unsigned int index, unsigned int numNodes);
	void storeParentIndex(FILE *fpo, FILE *fpi, unsigned int numNodes);

	// Compress using RACBVH
	int compress(const char* filepath);

	// Compress using gzip
	int compressgzipBVH(const char* filepath);

	// Compress using quantize
	int compressQBVH(const char* filepath);

	int test(const char* filepath)
	{
		FILE *fpWrite, *fpRead;
		fpWrite = fopen("tt", "w");
		{
			BitCompression cw(fpWrite, 0);
			LZW dicTestW(2, 25);
			dicTestW.setEncoder(&cw);
			dicTestW.encode(0);
			dicTestW.encode(0);
			dicTestW.encode(0);
			dicTestW.encode(1);
			dicTestW.encode(0);
			dicTestW.encode(0);
			dicTestW.encode(1);
			dicTestW.encode(1);
			dicTestW.encode(1);
			dicTestW.done();
			/*
			LZW dicTestW(5, 25);
			dicTestW.setEncoder(&cw);
			dicTestW.encode(4);
			dicTestW.encode(1);
			dicTestW.encode(2);
			dicTestW.encode(2);
			dicTestW.encode(1);
			dicTestW.encode(0);

			dicTestW.encode(4);
			dicTestW.encode(1);
			dicTestW.encode(2);
			dicTestW.encode(2);
			dicTestW.encode(1);
			dicTestW.encode(0);

			dicTestW.encode(4);
			dicTestW.encode(1);
			dicTestW.encode(2);
			dicTestW.encode(2);
			dicTestW.encode(1);
			dicTestW.encode(0);

			dicTestW.encode(4);
			dicTestW.encode(1);
			dicTestW.encode(2);
			dicTestW.encode(2);
			dicTestW.encode(1);
			dicTestW.encode(0);

			dicTestW.encode(4);
			dicTestW.encode(3);
			dicTestW.encode(3);
			dicTestW.encode(0);

			dicTestW.encode(4);
			dicTestW.encode(3);
			dicTestW.encode(3);
			dicTestW.encode(0);

			dicTestW.encode(4);
			dicTestW.encode(3);
			dicTestW.encode(3);

			dicTestW.done();
			*/
			/*
			dicTestW.push(1);
			dicTestW.push(9);
			dicTestW.push(7);
			dicTestW.push(5);
			dicTestW.push(5);
			dicTestW.push(7);
			dicTestW.push(7);
			dicTestW.push(1);
			dicTestW.push(8);
			dicTestW.buildDictionary();
			dicTestW.sortByFrequency();
			dicTestW.encodeDicTable();
			dicTestW.encode(7);
			dicTestW.encode(5);
			dicTestW.encode(1);
			dicTestW.encode(9);
			dicTestW.encode(1);
			dicTestW.encode(5);
			dicTestW.encode(2);
			dicTestW.encode(8);
			dicTestW.encode(8);
			dicTestW.encode(5);
			dicTestW.encode(7);
			dicTestW.done();
			*/
		}
		fclose(fpWrite);
		fpRead = fopen("tt", "r");
		{
			BitCompression cr(fpRead, 1);
			LZW dicTestR(2, 25);
			dicTestR.setDecoder(&cr);
			//dicTestR.decodeDicTable();
			for(int i=0;i<9;i++)
					cout << dicTestR.decode() << endl;;
		}
		fclose(fpRead);
		/*
		char filename1[256], filename2[256];
		sprintf(filename1, "%s/test1", filepath);
		sprintf(filename2, "%s/test2", filepath);
		FILE *fp1 = fopen(filename1, "wb"), *fp2 = fopen(filename2, "wb");
		RangeEncoder re(fp1);
		BitCompression bc(fp2);
		int bitSet[TEST_SIZE];
		int testSet[TEST_SIZE];
		int i, j;
		for(i=0;i<TEST_SIZE;i++)
		{
			testSet[i] = rand()%0x10000;
			bitSet[i] = getBits(testSet[i]);
			re.encode(testSet[i]+1, testSet[i]);
			bc.encode(bitSet[i], testSet[i]);
		}

		re.done();
		bc.done();
		fclose(fp1);
		fclose(fp2);
		
		FILE *fp3 = fopen(filename1, "rb"), *fp4 = fopen(filename2, "rb");
		int filesize1 = filelength(fileno(fp3)), filesize2 = filelength(fileno(fp4));
		unsigned char *mem1 = new unsigned char[filesize1], *mem2 = new unsigned char[filesize2];
		fread(mem1, 1, filesize1, fp3);
		fread(mem2, 1, filesize2, fp4);
		RangeDecoderFile rd(mem1, filesize1);
		BitCompression bcd(mem2, filesize2);
		int results1[TEST_SIZE];
		int results2[TEST_SIZE];
		Stopwatch t1("RangeDecoder"), t2("BitDecoder");
		t1.Start();
		for(i=0;i<TEST_SIZE;i++)
		{
			results1[i] = rd.decode(testSet[i]+1);
		}
		t1.Stop();
		t2.Start();
		for(i=0;i<TEST_SIZE;i++)
		{
			results2[i] = bcd.decode(bitSet[i]);
			//results2[i] = bcd.bitDecoder(getBits(testSet[i]));
		}
		t2.Stop();
		cout << t1 << endl;
		cout << t2 << endl;
		fclose(fp3);
		fclose(fp4);
		for(i=0;i<TEST_SIZE;i++)
		{
			if(results1[i] != results2[i])
			{
				cout << "Not match!" << endl;
				exit(1);
			}
		}
		cout << "All match!" << endl;
		*/
		return true;
	}

	int encodeDeltaForChildIndexInOutCluster(unsigned int delta, unsigned int numBoundary, unsigned int *listBoundary, unsigned int *listBoundaryBits, BitCompression *e);
	int encodeDeltaForChildIndexInOutCluster(unsigned int delta, unsigned int numBoundary, unsigned int *listBoundary, RangeModel **rmDeltaChildIndex, RangeEncoder *e);

	int getBiggestAxis(int *minQ, int *maxQ)
	{
		I32 diffQ[3] = {maxQ[0]-minQ[0], maxQ[1]-minQ[1], maxQ[2]-minQ[2]};
		return (diffQ[0]> diffQ[1] && diffQ[0]> diffQ[2]) ? 0 : (diffQ[1] > diffQ[2] ? 1 : 2);
	}

	int getBits(int x)
	{
		int bits = 1;
		while(x >>= 1 != 0) bits++;
		return bits;
	}

	unsigned int *childCounts;
	void getAllChildCount(FILE *fp);
	void getAllChildCountRec(FILE *fp, unsigned int nodeIndex);
};
